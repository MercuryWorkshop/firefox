/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 *
 * In-process WebAssembly interpreter for the wasm32-emscripten SpiderMonkey
 * build, which has no in-process wasm JIT backend. Selected at runtime by the
 * GECKO_WASM_INTERP env flag (see wasm::UseInterp), as an alternative to the
 * host-passthrough path in WasmJS.cpp. Decodes a module in-process, exposes the
 * real linear memory to content as a USER_OWNED ArrayBuffer (no mirror), and
 * runs a bytecode interpreter. Covers Core wasm (MVP + sign-extension +
 * non-trapping float->int + bulk-memory + reference types + i64/BigInt); SIMD,
 * threads/atomics, exceptions and GC are rejected at validation.
 *
 * This header is #included exactly once, from WasmJS.cpp, so it relies on that
 * file's includes and lives in its translation unit (no moz.build change).
 */

#ifndef wasm_WasmInterp_h
#define wasm_WasmInterp_h

#include "mozilla/ScopeExit.h"
#include "mozilla/Vector.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "js/AllocPolicy.h"
#include "js/ArrayBuffer.h"
#include "js/Conversions.h"
#include "js/friend/StackLimits.h"  // AutoCheckRecursionLimit (recursive interp)
#include "js/TracingAPI.h"
#include "js/UniquePtr.h"
#include "js/Utility.h"
#include "gc/Tracer.h"
#include "vm/BigIntType.h"
#include "wasm/WasmConstants.h"

#if defined(__EMSCRIPTEN__)
#  include <emscripten.h>
#endif

namespace js {
namespace wasm {
namespace interp {

using mozilla::Vector;

// ---- Value types -----------------------------------------------------------

enum class VT : uint8_t { I32, I64, F32, F64, FuncRef, ExternRef, Void };

static inline bool IsRefVT(VT t) {
  return t == VT::FuncRef || t == VT::ExternRef;
}

// Decode a single value-type byte (Core only). Returns false for SIMD/GC/etc.
static inline bool DecodeVT(uint8_t b, VT* out) {
  switch (b) {
    case uint8_t(TypeCode::I32): *out = VT::I32; return true;
    case uint8_t(TypeCode::I64): *out = VT::I64; return true;
    case uint8_t(TypeCode::F32): *out = VT::F32; return true;
    case uint8_t(TypeCode::F64): *out = VT::F64; return true;
    case uint8_t(TypeCode::FuncRef): *out = VT::FuncRef; return true;
    case uint8_t(TypeCode::ExternRef): *out = VT::ExternRef; return true;
    default: return false;
  }
}

// Operand-stack / local cell. Refs are stored as bits: externref = a
// JS::Value's raw bits; funcref = a packed (Instance*<<32 | funcIndex), 0=null.
union Cell {
  int32_t i32;
  uint32_t u32;
  int64_t i64;
  uint64_t u64;
  float f32;
  double f64;
};

// ---- Module structures (immutable after decode) ----------------------------

struct FuncType {
  Vector<VT, 4, SystemAllocPolicy> params;
  Vector<VT, 1, SystemAllocPolicy> results;
  bool equals(const FuncType& o) const {
    if (params.length() != o.params.length() ||
        results.length() != o.results.length()) {
      return false;
    }
    for (size_t i = 0; i < params.length(); i++) {
      if (params[i] != o.params[i]) return false;
    }
    for (size_t i = 0; i < results.length(); i++) {
      if (results[i] != o.results[i]) return false;
    }
    return true;
  }
};

enum class ExternKind : uint8_t { Func = 0, Table = 1, Memory = 2, Global = 3 };

struct Import {
  UniqueChars module;
  UniqueChars field;
  ExternKind kind;
  uint32_t index;  // typeIndex (func) / table / memory / global desc index
};

struct Export {
  UniqueChars name;
  ExternKind kind;
  uint32_t index;
};

struct GlobalDesc {
  VT type;
  bool isMutable;
  bool isImport;
  // For defined globals: a constant init expr.
  uint8_t initKind;  // Op byte: I32Const/.../GlobalGet/RefNull/RefFunc
  uint64_t initImm;  // const bits OR global index OR func index
};

struct TableDesc {
  VT elem;  // FuncRef or ExternRef
  uint32_t initial;
  uint32_t maximum;  // UINT32_MAX if none
  bool isImport;
};

struct MemoryDesc {
  uint32_t initialPages;
  uint32_t maximumPages;  // UINT32_MAX if none
  bool shared;
  bool isImport;
};

// A control-flow target resolved by the pre-pass, keyed by the body-relative pc
// of a block/loop/if opcode.
struct CtrlMeta {
  uint32_t endOpPc;  // pc of the matching `end` opcode
  uint32_t elsePc;   // pc just after the matching `else` opcode (0 if none)
};

struct FuncDef {
  uint32_t typeIndex;
  Vector<VT, 8, SystemAllocPolicy> localTypes;  // params then locals
  uint32_t numParams;
  const uint8_t* body;     // first opcode
  const uint8_t* bodyEnd;  // one past the final `end`
  // Raw Code-section entry range [codeStart, codeEnd) (locals decl + expr).
  // PrepareFunc (locals + control-flow side-table) is deferred to first call so
  // a huge module (gecko.wasm: 362K functions) doesn't build every side-table up
  // front; these locate the body for that lazy prepare.
  const uint8_t* codeStart = nullptr;
  const uint8_t* codeEnd = nullptr;
  // Branch side-tables, keyed by body-relative pc of the control opcode.
  std::unordered_map<uint32_t, CtrlMeta> ctrl;
  bool prepared = false;
};

struct DataSeg {
  bool active;
  uint32_t memIndex;
  uint8_t offKind;  // I32Const or GlobalGet
  uint64_t offImm;
  Vector<uint8_t, 0, SystemAllocPolicy> bytes;
};

struct ElemSeg {
  bool active;
  bool declared;
  uint32_t tableIndex;
  uint8_t offKind;
  uint64_t offImm;
  VT elem;
  Vector<uint32_t, 0, SystemAllocPolicy> funcIndices;  // UINT32_MAX = null
};

struct Module {
  Vector<uint8_t, 0, SystemAllocPolicy> bytecode;  // owned copy
  Vector<FuncType, 0, SystemAllocPolicy> types;
  Vector<Import, 0, SystemAllocPolicy> imports;
  Vector<Export, 0, SystemAllocPolicy> exports;
  Vector<uint32_t, 0, SystemAllocPolicy> funcTypes;  // index space (imp+def)
  Vector<FuncDef, 0, SystemAllocPolicy> defined;     // defined funcs only
  Vector<GlobalDesc, 0, SystemAllocPolicy> globals;
  Vector<TableDesc, 0, SystemAllocPolicy> tables;
  Vector<MemoryDesc, 0, SystemAllocPolicy> memories;
  Vector<DataSeg, 0, SystemAllocPolicy> datas;
  Vector<ElemSeg, 0, SystemAllocPolicy> elems;
  uint32_t numImportFuncs = 0;
  uint32_t numImportGlobals = 0;
  uint32_t numImportTables = 0;
  uint32_t numImportMemories = 0;
  bool hasStart = false;
  uint32_t startFunc = 0;
  // Pinned == shared cross-thread via structured clone (another thread's Module
  // wrapper references this struct). Skip the per-thread finalizer free (we
  // intentionally leak shared engine modules rather than cross-thread refcount).
  bool pinned = false;
  // Lazy PrepareFunc is thread-safe for a shared (cross-thread) module: each
  // defined function's prepared-flag is an atomic; the first caller prepares
  // under prepareLock and release-stores the flag, later callers acquire-load it
  // (so the FuncDef.ctrl/localTypes writes are visible). One flag per defined
  // function, allocated once at decode (stable address; atomics aren't movable).
  std::unique_ptr<std::atomic<uint8_t>[]> preparedFlags;
  std::mutex prepareLock;

  const FuncType& funcType(uint32_t fi) const { return types[funcTypes[fi]]; }
};

// ---- Raw LEB readers (bounds-checked; module already passed Validate) -------

struct Reader {
  const uint8_t* p;
  const uint8_t* end;
  bool ok = true;

  Reader(const uint8_t* b, const uint8_t* e) : p(b), end(e) {}

  uint8_t u8() {
    if (p >= end) { ok = false; return 0; }
    return *p++;
  }
  uint32_t u32leb() {
    uint32_t r = 0; unsigned s = 0; uint8_t b;
    do {
      if (p >= end) { ok = false; return r; }
      b = *p++;
      r |= uint32_t(b & 0x7f) << s;
      s += 7;
    } while (b & 0x80);
    return r;
  }
  uint64_t u64leb() {
    uint64_t r = 0; unsigned s = 0; uint8_t b;
    do {
      if (p >= end) { ok = false; return r; }
      b = *p++;
      r |= uint64_t(b & 0x7f) << s;
      s += 7;
    } while (b & 0x80);
    return r;
  }
  int32_t s32leb() {
    int32_t r = 0; unsigned s = 0; uint8_t b;
    do {
      if (p >= end) { ok = false; return r; }
      b = *p++;
      r |= int32_t(b & 0x7f) << s;
      s += 7;
    } while (b & 0x80);
    if (s < 32 && (b & 0x40)) r |= int32_t(-1) << s;
    return r;
  }
  int64_t s64leb() {
    int64_t r = 0; unsigned s = 0; uint8_t b;
    do {
      if (p >= end) { ok = false; return r; }
      b = *p++;
      r |= int64_t(b & 0x7f) << s;
      s += 7;
    } while (b & 0x80);
    if (s < 64 && (b & 0x40)) r |= int64_t(-1) << s;
    return r;
  }
  float f32() {
    float v = 0;
    if (end - p < 4) { ok = false; return 0; }
    memcpy(&v, p, 4); p += 4; return v;
  }
  double f64() {
    double v = 0;
    if (end - p < 8) { ok = false; return 0; }
    memcpy(&v, p, 8); p += 8; return v;
  }
  void skip(size_t n) {
    if (size_t(end - p) < n) { ok = false; p = end; return; }
    p += n;
  }
};

// Linear memory backing, owned by a MemoryObject (see WasmInterpObj-inl.h).
struct LinearMemory {
  uint8_t* base = nullptr;
  uint32_t size = 0;       // current bytes
  uint32_t maxBytes = 0;   // cap
  bool shared = false;
  // For shared memory: the SharedArrayRawBuffer* (as void* to avoid pulling the
  // internal header into this widely-included file). base aliases its data, so
  // all threads that reconstruct a SAB over this rawbuf see the same bytes. Set
  // when shared; null for non-shared.
  void* rawbuf = nullptr;
  // Pinned == shared cross-thread via structured clone: another thread's wrapper
  // references this struct, so the per-thread finalizer must NOT free it (we
  // intentionally leak shared engine memory rather than cross-thread refcount).
  bool pinned = false;
};

// Table backing, owned by a TableObject. funcref tables use `funcs` (packed
// Instance*<<32 | funcIndex; 0 = null); externref tables use `refs` (Values).
struct InstTable {
  VT elem = VT::FuncRef;
  uint32_t maxLen = UINT32_MAX;
  Vector<uint64_t, 0, SystemAllocPolicy> funcs;
  Vector<JS::Value, 0, SystemAllocPolicy> refs;
  InstTable* nextExtern = nullptr;     // intrusive list of externref tables (GC)
  InstTable* nextFuncTable = nullptr;  // intrusive list of funcref tables (GC)
  uint32_t length() const {
    return elem == VT::FuncRef ? funcs.length() : refs.length();
  }
};

// ---- Forward decls ---------------------------------------------------------

class Instance;

// Defined in WasmInterpObj-inl.h, used by the interpreter core:
LinearMemory* MemoryFromObject(JSObject* memObj);
InstTable* TableFromObject(JSObject* tableObj);
// Grow memObj's memory by deltaPages; returns the previous size in pages, or
// -1 on failure (no exception set; the caller pushes -1 per spec).
int64_t GrowMemoryObject(JSContext* cx, JS::HandleObject memObj,
                         uint32_t deltaPages);
int64_t GrowTableObject(JSContext* cx, JS::HandleObject tableObj,
                        uint32_t delta, uint64_t initFunc, JS::HandleValue initRef);
JSObject* GetOrCreateFuncWrapper(JSContext* cx, Instance* inst,
                                 uint32_t funcIndex);
// If obj is an interp export-function wrapper, write its packed funcref and
// return true; otherwise return false.
bool GetFuncWrapperPacked(JSObject* obj, uint64_t* packedOut);

// Public API used by WasmJS.cpp:
JSObject* CompileBytes(JSContext* cx, const uint8_t* bytes, size_t len);
bool IsModuleObject(JSObject* obj);
bool IsInstanceObject(JSObject* obj);
JSObject* InstantiateModuleObject(JSContext* cx, HandleObject moduleObj,
                                  HandleObject importObj);
JSObject* NewMemoryObjectJS(JSContext* cx, uint32_t initialPages,
                            uint32_t maxPages, bool shared);
JSObject* NewTableObjectJS(JSContext* cx, VT elem, uint32_t initial,
                           uint32_t maximum);
JSObject* NewGlobalObjectJS(JSContext* cx, VT type, bool isMutable,
                            const Cell* initVal, JS::HandleValue initRef);

}  // namespace interp
}  // namespace wasm
}  // namespace js

#include "wasm/WasmInterpDecode-inl.h"
#include "wasm/WasmInterpRun-inl.h"
#include "wasm/WasmInterpObj-inl.h"

#endif  // wasm_WasmInterp_h
