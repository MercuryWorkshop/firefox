/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// WasmJitBackend.cpp -- MIR->wasm-bytecode back-end. Walks an OptimizeMIR'd Warp
// graph and emits a wasm function body in "value-per-local" form: every used
// MDefinition gets its own wasm local; each node computes its value (pushing
// operands via local.get) then local.set's its own local. The host engine
// re-runs real register allocation on the result. Control flow lowers to a
// single br_table dispatch loop. JS::Values are NUNBOX32 (low32 payload, high32
// tag).

#include "wasm/WasmJitBackend.h"

#include <algorithm>
#include <stdio.h>
#include <unordered_map>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "jit/CompileInfo.h"
#include "jit/IonTypes.h"  // IsResumeAfter
#include "jit/MIR.h"
#include "jit/MIRGenerator.h"
#include "jit/MIRGraph.h"
#include "jit/WarpSnapshot.h"
#include "js/Value.h"
#include "jsfriendapi.h"  // js::IdToValue
#include "js/shadow/Object.h"
#include "vm/BytecodeUtil.h"  // GetNextPc
#include "vm/JSFunction.h"
#include "vm/JSObject.h"
#include "vm/NativeObject.h"
#include "gc/Heap.h"      // gc::Arena::thingSize, NurseryCellHeader (inline alloc)
#include "gc/Nursery.h"   // Nursery::nurseryCellHeaderSize/offsetOfCurrentEndFromPosition
#include "wasm/WasmBinary.h"

using namespace js;
using namespace js::jit;
using namespace js::wasm;

namespace {

static int gWJDbg = -1;
static inline bool WJDbg() {
  if (gWJDbg < 0) gWJDbg = getenv("GECKO_WJWARP_DUMP") ? 1 : 0;
  return gWJDbg;
}
#define WJBAIL(...) (WJDbg() ? (fprintf(stderr, "[wb-bail] " __VA_ARGS__), false) : false)

// Bail-reason tally: under GECKO_WJ_BAILDBG, count how often each MIR opcode
// forces a function to stay in PBL, and dump the histogram at process exit.
static const char* WJOpName(jit::MDefinition::Opcode op) {
  switch (op) {
#define WJ_OPNAME(o) \
  case jit::MDefinition::Opcode::o: \
    return #o;
    MIR_OPCODE_LIST(WJ_OPNAME)
#undef WJ_OPNAME
  }
  return "?";
}
static void WJTallyBail(jit::MDefinition::Opcode op) {
  js::wasm::gWJBailReason = WJOpName(op);  // record for GECKO_WJ_CDBG
  static int on = -1;
  if (on < 0) on = getenv("GECKO_WJ_BAILDBG") ? 1 : 0;
  if (!on) return;
  fprintf(stderr, "[wb-baildbg] %s\n", WJOpName(op));
}

// wasm valtype byte for an MIR type, or 0 if this node carries no wasm value.
static uint8_t WJValType(MIRType t) {
  switch (t) {
    case MIRType::Int32:
    case MIRType::Boolean:
      return uint8_t(TypeCode::I32);
    case MIRType::Int64:
    case MIRType::Value:  // boxed JS::Value (NUNBOX32 -> 64 bits)
      return uint8_t(TypeCode::I64);
    case MIRType::Double:
      return uint8_t(TypeCode::F64);
    case MIRType::Float32:
      return uint8_t(TypeCode::F32);
    case MIRType::Object:
    case MIRType::String:
    case MIRType::Symbol:
    case MIRType::BigInt:
    case MIRType::Slots:
    case MIRType::Elements:
      return uint8_t(TypeCode::I32);  // wasm32 pointer (low 32 bits)
    default:
      return 0;
  }
}

struct WJBackend {
  std::vector<int32_t> localOf;  // def id -> wasm local index (-1 = none)
  std::vector<uint8_t> localTy;  // valtype per declared local (index order)
  uint32_t paramCount = 1;       // wasm param 0 = scratch base address (f64)
  uint32_t scratchBase = 0;      // local holding the scratch base as i32
  uint32_t unboxScratch = 0;     // i64 local for unboxing fused slot loads
  uint32_t callScratch = 0;      // i32 local: callee ptr for the call-site IC
  uint32_t callArgBase = 0;      // first of (1+kWJMaxArgs) i64 locals: boxed this+args
  uint32_t callFlagLocal = 0;    // f64 local: deopt flag returned by a call
  uint32_t callResultLocal = 0;  // i64 local: boxed result returned by a call
  uint32_t envLocal = 0;         // i32 local: this invocation's environment object,
                                 // saved from gWJCurrentEnv in the prologue (sound
                                 // even across nested calls that clobber the global)
  bool useEnvRoot = false;       // true if any FunctionEnvironment node exists -> the
                                 // env object must be GC-rooted for the function's
                                 // whole lifetime (it's a movable nursery CallObject
                                 // for closures; a stale raw ptr reads garbage ->
                                 // navier/earley/splay "X is not a function"/OOB)
  uint32_t envRootIdx = 0;       // i32 local: index of the PERSISTENT GC-shadow-stack
                                 // slot (gWJCallRoots) holding the boxed env for the
                                 // whole invocation. FunctionEnvironment rematerializes
                                 // from it on every use (traced every GC -> always
                                 // current, even across non-call GC points). Pushed in
                                 // the prologue, popped at every exit.
  const CompileInfo* info = nullptr;  // slot layout for resume points
  MResumePoint* curRp = nullptr;      // resume point for the instruction being emitted
  uint32_t curOp = 0;                 // op() of the instruction being emitted (deopt histogram)
  uint32_t deoptCallLine = 0;         // __LINE__ of the EmitDeopt call (deopt-line histogram)
  uint32_t propReadIdx = 0;           // per-compile mega-read counter (stable prop-site key)
  WarpSnapshot* snapshot = nullptr;   // for resolving NurseryObject references

  // Out-of-line deopt (the dispatch-loop path). Each guard miss inside a hot
  // block body branches to a single out-of-line dispatcher (the `block $D` that
  // wraps the dispatch loop) instead of inlining the big spill+resume sequence --
  // keeping the hot loop body tiny so the host VM register-allocates/optimizes it
  // (inline deopt bloat made navier's inner loop ~90% never-taken spill code and
  // ~3.6x slower). The dispatcher, emitted after all bodies, switches on
  // deoptSiteLocal and runs the matching resume. Only active on the dispatch path.
  bool oolDeopt = false;          // out-of-line mode enabled
  bool forceMega = false;         // megamorphic recompile: GuardShape-guarded
                                  // property reads -> multi-shape EmitPropIC (no
                                  // deopt) instead of a monomorphic GuardShape
                                  // that storms on a polymorphic receiver.
  bool inDispatchBody = false;    // currently emitting a hot dispatch block body
  uint32_t bodyLoopIdx = 0;       // br-index of loop $L from the current body base
  uint32_t deoptExtraNest = 0;    // extra `if`/block levels enclosing a deopt vs the
                                  // usual single guard-`if` (so the OOL br reaches $D)
  uint32_t deoptSiteLocal = 0;    // i32 local: which deopt site fired
  uint32_t propObjLocal = 0;      // i32 local: prop-IC receiver pointer
  uint32_t propShapeLocal = 0;    // i32 local: prop-IC receiver shape
  uint32_t propTaggedLocal = 0;   // i32 local: prop-IC tagged slot offset
  uint32_t allocPosLocal = 0;     // i32 local: inline-alloc old nursery position
  uint32_t allocObjLocal = 0;     // i32 local: inline-alloc result object pointer
  uint32_t rootBaseLocal = 0;     // i32 local: GC-root spill slot base (sp*8+base), computed once per call
  // (rp, curOp, isShadow) per out-of-line deopt site. isShadow => the frame
  // operands were spilled to gWJResumeVals in the HOT path (shadow frame), so the
  // dispatcher must NOT re-read locals (that read pins them live-out -> the loop's
  // hot values get register-spilled every iteration; the navier/splay 1x ceiling).
  std::vector<std::tuple<MResumePoint*, uint32_t, bool>> deoptSites;

  // Shadow frame: maintain the (single-frame) resume operands in gWJResumeVals via
  // hot-path stores at each resume point, so deopts read MEMORY instead of locals.
  bool shadowDeopt = false;

  int32_t local(const MDefinition* d) const {
    uint32_t id = d->id();
    return id < localOf.size() ? localOf[id] : -1;
  }
  void ensureSize(uint32_t id) {
    if (id >= localOf.size()) localOf.resize(id + 1, -1);
  }
  int32_t reserve(uint8_t ty) {
    int32_t idx = int32_t(paramCount + localTy.size());
    localTy.push_back(ty);
    return idx;
  }
  int32_t assign(const MDefinition* d) {
    ensureSize(d->id());
    if (localOf[d->id()] >= 0) return localOf[d->id()];
    uint8_t ty = WJValType(d->type());
    int32_t idx = reserve(ty);
    localOf[d->id()] = idx;
    return idx;
  }
};

static bool GetOp(Encoder& e, WJBackend& be, const MDefinition* d) {
  // Slots/Elements are RAW INTERIOR pointers into an object's heap storage. A
  // minor GC that moves the owning object invalidates them, so they must NEVER
  // be read from a cached local that may predate a GC-causing call -- that stale
  // pointer reads garbage (the raytrace `ray`-goes-garbage crash). Rematerialize
  // them on every use from the owning object's local (an Object, GC-rooted and
  // reloaded across calls, hence always current). This mirrors Ion, which treats
  // MElements/MSlots as rematerializable rather than keeping them live in regs.
  // Slots/Elements are RAW INTERIOR pointers; a minor GC moving the owning object
  // invalidates a cached copy. Rematerialize from the (GC-rooted) owning object's
  // local on every use -- sound, and measured no slower than caching (the array
  // overhead is bounds-checks/unboxing, not this load).
  if (d->type() == MIRType::Slots) {
    // Read the owning object via GetOp too (it may be an Unbox/GuardShape whose
    // cached raw ptr is stale across a GC-ing call) -- rematerialize the whole
    // chain from the rooted source.
    if (!GetOp(e, be, d->toSlots()->object())) return false;
    return e.writeOp(Op::I32Load) && e.writeVarU32(2) &&
           e.writeVarU32(uint32_t(js::NativeObject::offsetOfSlots()));
  }
  if (d->type() == MIRType::Elements) {
    if (!GetOp(e, be, d->toElements()->object())) return false;
    return e.writeOp(Op::I32Load) && e.writeVarU32(2) &&
           e.writeVarU32(uint32_t(js::NativeObject::offsetOfElements()));
  }
  // FunctionEnvironment: rematerialize the env from its PERSISTENT GC-shadow-stack
  // slot (gWJCallRoots[envRootIdx]) on every use -- the slot is traced on every GC
  // so this is always the GC-current env, even across non-call GC points (where a
  // cached raw ptr would go stale: navier/earley closure bugs). Unbox the boxed
  // ObjectValue (low 32 bits = object payload, NUNBOX32).
  if (d->op() == MDefinition::Opcode::FunctionEnvironment && be.useEnvRoot) {
    uintptr_t rootsBase = uintptr_t(static_cast<void*>(&gWJCallRoots[0]));
    return e.writeOp(Op::LocalGet) && e.writeVarU32(be.envRootIdx) &&
           e.writeOp(Op::I32Const) && e.writeVarS32(8) && e.writeOp(Op::I32Mul) &&
           e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(rootsBase)) &&
           e.writeOp(Op::I32Add) && e.writeOp(Op::I64Load) && e.writeVarU32(3) &&
           e.writeVarU32(0) && e.writeOp(Op::I32WrapI64);
  }
  // Parameter: read the GC-ROOTED COPY local, NOT the raw wasm param. The raw
  // param holds a boxed object payload that goes STALE when a minor GC moves the
  // arg object (e.g. earley/nboyer pass freshly-consed nursery lists, then `new
  // sc_Pair` triggers a minor GC, then read `.car` on the now-moved arg -> memory
  // out of bounds). The raw param is never reloaded, but the copy local IS
  // reloaded across every call/GC point by WJCollectRoots, and it's initialized
  // at the top of the prologue (before any block-entry spill or deopt), so it is
  // always GC-current. Fall back to the raw param only if there's no copy local
  // (an unused param, never actually read here).
  if (d->op() == MDefinition::Opcode::Parameter) {
    const MParameter* p = d->toParameter();
    if (p->index() != MParameter::THIS_SLOT &&
        uint32_t(p->index()) >= kWJMaxArgs)
      return false;
    int32_t cl = be.local(d);
    if (cl >= 0) return e.writeOp(Op::LocalGet) && e.writeVarU32(uint32_t(cl));
    uint32_t pidx = (p->index() == MParameter::THIS_SLOT)
                        ? 1
                        : (2 + uint32_t(p->index()));
    return e.writeOp(Op::LocalGet) && e.writeVarU32(pidx);
  }
  // Unbox to a GC-POINTER type (Object/String/Symbol/BigInt): the unboxed value
  // is a RAW heap pointer. Caching it in a local makes it go STALE across a
  // GC-causing call (the boxed source is rooted+reloaded, but the cached raw
  // pointer is not) -- e.g. earley sc_reverse: `res=sc_cons(l1.car,res); l1=l1.cdr`
  // re-used the pre-call Unbox(l1) pointer after sc_cons's nursery alloc moved
  // l1, reading l1.cdr from freed memory -> "car of undefined". So REMATERIALIZE
  // by re-unboxing the (rooted, GC-current) source on every use, mirroring how
  // Slots/Elements are rematerialized. The fallible type guard already ran when
  // the Unbox node was first emitted; here we only need the (current) pointer.
  if (d->op() == MDefinition::Opcode::Unbox &&
      (d->type() == MIRType::Object || d->type() == MIRType::String ||
       d->type() == MIRType::Symbol || d->type() == MIRType::BigInt)) {
    MDefinition* src = d->toUnbox()->input();
    if (src->type() == MIRType::Value || src->type() == MIRType::Int64) {
      return GetOp(e, be, src) && e.writeOp(Op::I32WrapI64);  // boxed -> ptr
    }
    if (src->type() == MIRType::Object || src->type() == MIRType::String ||
        src->type() == MIRType::Symbol || src->type() == MIRType::BigInt) {
      return GetOp(e, be, src);  // src already a (rooted/reloaded) raw ptr
    }
    // other src types: fall through to the cached local.
  }
  // GuardShape (and friends) are PASSTHROUGHs: their value is the guarded object.
  // Rematerialize via GetOp(object) so a post-call use re-derives from the rooted
  // source (same stale-pointer hazard as Unbox -- earley sc_reverse l1.cdr). The
  // guard itself already ran when the node was emitted.
  if (d->op() == MDefinition::Opcode::GuardShape)
    return GetOp(e, be, d->toGuardShape()->object());
  if (d->op() == MDefinition::Opcode::GuardShapeList)
    return GetOp(e, be, d->toGuardShapeList()->object());
  // Object passthroughs (return their operand 0 unchanged after a guard): also
  // rematerialize from the rooted source so a post-call use isn't a stale ptr.
  if (d->op() == MDefinition::Opcode::GuardSpecificFunction ||
      d->op() == MDefinition::Opcode::GuardToFunction ||
      d->op() == MDefinition::Opcode::GuardFunctionScript ||
      d->op() == MDefinition::Opcode::ConstantProto)
    return GetOp(e, be, d->getOperand(0));
  if (d->op() == MDefinition::Opcode::GuardObjectIdentity)
    return GetOp(e, be, d->toGuardObjectIdentity()->object());
  int32_t l = be.local(d);
  if (l < 0) return false;
  return e.writeOp(Op::LocalGet) && e.writeVarU32(uint32_t(l));
}

// Sound deopt: spill the current resume point's live state and continue in PBL
// (defined below). Returns false if it can't be emitted (caller bails the fn).
static bool EmitDeoptResume(Encoder& e, WJBackend& be);
static bool EmitValueTruthy(Encoder& e, WJBackend& be, uint32_t v);
static bool EmitEnvRootPop(Encoder& e, WJBackend& be);

// Emit a deopt at the current instruction's resume point. All guard misses route
// here; if there's no usable resume point the whole function bails at compile.
static bool EmitDeoptImpl(Encoder& e, WJBackend& be) { return EmitDeoptResume(e, be); }
#define EmitDeopt(e, be) (((be).deoptCallLine = __LINE__), EmitDeoptImpl((e), (be)))

// NUNBOX32 tag word (high 32 bits) for a non-double value type.
static uint32_t TagWord(JSValueType t) {
  return uint32_t(JSVAL_TAG_CLEAR) | uint32_t(t);
}

// Box a typed value already on the stack into an i64 JS::Value.
static bool EmitBoxFromStack(Encoder& e, MIRType from) {
  if (from == MIRType::Double) {
    return e.writeOp(Op::I64ReinterpretF64);
  }
  if (from == MIRType::Int32 || from == MIRType::Boolean ||
      from == MIRType::Object || from == MIRType::String ||
      from == MIRType::Symbol || from == MIRType::BigInt) {
    JSValueType vt = from == MIRType::Int32     ? JSVAL_TYPE_INT32
                     : from == MIRType::Boolean ? JSVAL_TYPE_BOOLEAN
                     : from == MIRType::String  ? JSVAL_TYPE_STRING
                     : from == MIRType::Symbol  ? JSVAL_TYPE_SYMBOL
                     : from == MIRType::BigInt  ? JSVAL_TYPE_BIGINT
                                                : JSVAL_TYPE_OBJECT;
    // payload(i32) -> i64 zero-extended, OR'd with (tag << 32).
    if (!e.writeOp(Op::I64ExtendI32U)) return false;
    if (!e.writeOp(Op::I64Const) ||
        !e.writeVarS64(int64_t(uint64_t(TagWord(vt)) << 32))) {
      return false;
    }
    return e.writeOp(Op::I64Or);
  }
  return false;
}

// Push the i64 value held in wasm local `localIdx`.
static bool GetLocal(Encoder& e, uint32_t localIdx) {
  return e.writeOp(Op::LocalGet) && e.writeVarU32(localIdx);
}

// Guard that the i64 value in wasm local `localIdx` has tag word `tag`; deopt if not.
static bool EmitTagGuardLocal(Encoder& e, WJBackend& be, uint32_t localIdx,
                              uint32_t tag) {
  if (!GetLocal(e, localIdx)) return false;
  if (!e.writeOp(Op::I64Const) || !e.writeVarS64(32)) return false;
  if (!e.writeOp(Op::I64ShrU)) return false;
  if (!e.writeOp(Op::I32WrapI64)) return false;
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(tag))) return false;
  if (!e.writeOp(Op::I32Ne)) return false;
  static int noBranch = getenv("GECKO_WJ_GUARD_NOBRANCH") ? 1 : 0;
  if (noBranch) return e.writeOp(Op::Drop);  // PROBE: compute cond, no exit
  if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
  if (!EmitDeopt(e, be)) return false;
  return e.writeOp(Op::End);
}

// Guard that the i64 value in local `localIdx` is a double; deopt if not.
static bool EmitDoubleGuardLocal(Encoder& e, WJBackend& be, uint32_t localIdx) {
  if (!GetLocal(e, localIdx)) return false;
  if (!e.writeOp(Op::I64Const) || !e.writeVarS64(32)) return false;
  if (!e.writeOp(Op::I64ShrU)) return false;
  if (!e.writeOp(Op::I32WrapI64)) return false;
  if (!e.writeOp(Op::I32Const) ||
      !e.writeVarS32(int32_t(uint32_t(JSVAL_TAG_CLEAR)))) {
    return false;
  }
  if (!e.writeOp(Op::I32GtU)) return false;  // tag > CLEAR => not a double
  static int noBranch = getenv("GECKO_WJ_GUARD_NOBRANCH") ? 1 : 0;
  if (noBranch) return e.writeOp(Op::Drop);  // PROBE: compute cond, no exit
  if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
  if (!EmitDeopt(e, be)) return false;
  return e.writeOp(Op::End);
}

// Unbox the i64 JS::Value held in `localIdx` to `out`, leaving the unboxed value
// on the stack. Fallible mode emits a tag guard (deopt on mismatch).
static bool EmitUnboxLocal(Encoder& e, WJBackend& be, uint32_t localIdx,
                           MIRType out, bool fallible) {
  switch (out) {
    case MIRType::Int32:
      if (fallible &&
          !EmitTagGuardLocal(e, be, localIdx, TagWord(JSVAL_TYPE_INT32)))
        return false;
      return GetLocal(e, localIdx) && e.writeOp(Op::I32WrapI64);
    case MIRType::Boolean:
      if (fallible &&
          !EmitTagGuardLocal(e, be, localIdx, TagWord(JSVAL_TYPE_BOOLEAN)))
        return false;
      return GetLocal(e, localIdx) && e.writeOp(Op::I32WrapI64);
    case MIRType::Object:
      if (fallible &&
          !EmitTagGuardLocal(e, be, localIdx, TagWord(JSVAL_TYPE_OBJECT)))
        return false;
      return GetLocal(e, localIdx) && e.writeOp(Op::I32WrapI64);
    case MIRType::String:
      if (fallible &&
          !EmitTagGuardLocal(e, be, localIdx, TagWord(JSVAL_TYPE_STRING)))
        return false;
      return GetLocal(e, localIdx) && e.writeOp(Op::I32WrapI64);
    case MIRType::Symbol:
      if (fallible &&
          !EmitTagGuardLocal(e, be, localIdx, TagWord(JSVAL_TYPE_SYMBOL)))
        return false;
      return GetLocal(e, localIdx) && e.writeOp(Op::I32WrapI64);
    case MIRType::BigInt:
      if (fallible &&
          !EmitTagGuardLocal(e, be, localIdx, TagWord(JSVAL_TYPE_BIGINT)))
        return false;
      return GetLocal(e, localIdx) && e.writeOp(Op::I32WrapI64);
    case MIRType::Double: {
      static int infDbl = getenv("GECKO_WJ_INFALLIBLE_DBL") ? 1 : 0;
      if (!fallible || infDbl) {
        return GetLocal(e, localIdx) && e.writeOp(Op::F64ReinterpretI64);
      }
      static int noNum = getenv("GECKO_WJ_NONUMUNBOX") ? 1 : 0;
      if (noNum) {  // old: double-only guard, deopt on int32 (-> PBL fallback)
        if (!EmitDoubleGuardLocal(e, be, localIdx)) return false;
        return GetLocal(e, localIdx) && e.writeOp(Op::F64ReinterpretI64);
      }
      // NUMBER unbox: a Double-typed value that is actually an Int32 is still a
      // number -- convert it inline (i32->f64) instead of deopting. navier's
      // arrays/args hold int 0/1/4 mixed with doubles; deopting on those made the
      // hot loop resume into PBL every call. Only a genuine non-number deopts.
      // The deopt MUST sit at nesting==1 (its own single `if`) -- the OOL deopt
      // br-depth (bodyLoopIdx+2) assumes that; a deopt nested inside the f64
      // select would br to the wrong scope. So do the guard (deopt) FIRST in its
      // own `if`, THEN a separate (deopt-free) select.
      const uint8_t kF64bt = 0x7C;  // wasm f64 blocktype
      auto pushTag = [&]() -> bool {
        return GetLocal(e, localIdx) && e.writeOp(Op::I64Const) &&
               e.writeVarS64(32) && e.writeOp(Op::I64ShrU) &&
               e.writeOp(Op::I32WrapI64);
      };
      // guard: deopt unless number == (tag u<= CLEAR) | (tag == INT32).
      if (!pushTag() || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(uint32_t(JSVAL_TAG_CLEAR))) ||
          !e.writeOp(Op::I32LeU))
        return false;  // isDouble
      if (!pushTag() || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_INT32))) ||
          !e.writeOp(Op::I32Eq))
        return false;  // isInt32
      if (!e.writeOp(Op::I32Or) || !e.writeOp(Op::I32Eqz)) return false;  // !number
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;  // void if
      if (!EmitDeopt(e, be)) return false;  // nesting==1
      if (!e.writeOp(Op::End)) return false;
      // select (no deopt): double -> reinterpret, else int32 -> convert.
      if (!pushTag() || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(uint32_t(JSVAL_TAG_CLEAR))) ||
          !e.writeOp(Op::I32LeU) || !e.writeOp(Op::If) || !e.writeFixedU8(kF64bt))
        return false;
      if (!GetLocal(e, localIdx) || !e.writeOp(Op::F64ReinterpretI64)) return false;
      if (!e.writeOp(Op::Else)) return false;
      if (!GetLocal(e, localIdx) || !e.writeOp(Op::I32WrapI64) ||
          !e.writeOp(Op::F64ConvertI32S))
        return false;
      if (!e.writeOp(Op::End)) return false;
      return true;
    }
    default:
      return false;
  }
}

// Push a boxed i64 JS::Value for resume-point operand `v`. Constants are baked
// (GC-thing constants via the traced pool); live values are loaded + boxed.
static bool EmitSpillValue(Encoder& e, WJBackend& be, MDefinition* v) {
  if (v->isConstant()) {
    MConstant* c = v->toConstant();
    MIRType t = c->type();
    if (t == MIRType::Object || t == MIRType::String || t == MIRType::Symbol ||
        t == MIRType::BigInt) {
      uintptr_t slot = WJInternConstant(c->toJSValue().asRawBits());
      if (!slot) return false;
      return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(slot)) &&
             e.writeOp(Op::I64Load) && e.writeVarU32(3) && e.writeVarU32(0);
    }
    uint64_t bits;
    switch (t) {
      case MIRType::Int32:
      case MIRType::Double:
      case MIRType::Boolean:
      case MIRType::Null:
      case MIRType::Undefined:
        bits = c->toJSValue().asRawBits();
        break;
      case MIRType::MagicOptimizedOut:
        bits = JS::MagicValue(JS_OPTIMIZED_OUT).asRawBits();
        break;
      default:
        return WJBAIL("spill: const type %s\n", StringFromMIRType(t));
    }
    return e.writeOp(Op::I64Const) && e.writeVarS64(int64_t(bits));
  }
  // Parameter: read the GC-rooted COPY local (boxed i64), reloaded across calls
  // by WJCollectRoots and initialized at the top of the prologue. Reading the raw
  // wasm param would spill a STALE object payload after a minor GC moves the arg
  // (earley/nboyer). Fall back to the raw param only if there's no copy local.
  if (v->op() == MDefinition::Opcode::Parameter) {
    const MParameter* p = v->toParameter();
    if (p->index() != MParameter::THIS_SLOT &&
        uint32_t(p->index()) >= kWJMaxArgs)
      return false;
    int32_t cl = be.local(v);
    if (cl >= 0) return GetLocal(e, uint32_t(cl));  // GC-current boxed copy
    uint32_t pidx = (p->index() == MParameter::THIS_SLOT)
                        ? 1
                        : (2 + uint32_t(p->index()));
    return GetLocal(e, pidx);  // boxed i64 param, no re-boxing needed
  }
  if (be.local(v) < 0) return WJBAIL("spill: op#%u type %s has no local\n",
                                     unsigned(v->op()), StringFromMIRType(v->type()));
  // Read via GetOp (rematerializes Unbox/GuardShape/Slots/Elements object ptrs
  // from their rooted source) so a spilled/staged object ptr isn't stale across a
  // GC-ing call (earley). GetOp falls back to the cached local for plain nodes.
  if (!GetOp(e, be, v)) return false;
  if (v->type() == MIRType::Value || v->type() == MIRType::Int64) return true;
  if (v->type() == MIRType::Object || v->type() == MIRType::Int32 ||
      v->type() == MIRType::Boolean || v->type() == MIRType::Double ||
      v->type() == MIRType::String || v->type() == MIRType::Symbol ||
      v->type() == MIRType::BigInt) {
    return EmitBoxFromStack(e, v->type());
  }
  return WJBAIL("spill: type %s not boxable\n", StringFromMIRType(v->type()));
}

// Leave a raw i32 object pointer for `v` on the stack (used for the resume env
// chain). Handles object locals, boxed-Value locals, and object constants.
static bool EmitObjPtr(Encoder& e, WJBackend& be, MDefinition* v) {
  if (v->isConstant()) {
    MConstant* c = v->toConstant();
    if (c->type() == MIRType::Object) {
      JSObject* obj = c->toObjectOrNull();
      if (!obj) return e.writeOp(Op::I32Const) && e.writeVarS32(0);
      uintptr_t slot = WJInternConstant(JS::ObjectValue(*obj).asRawBits());
      if (!slot) return false;
      return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(slot)) &&
             e.writeOp(Op::I64Load) && e.writeVarU32(3) && e.writeVarU32(0) &&
             e.writeOp(Op::I32WrapI64);
    }
    return false;
  }
  // Rematerialize via GetOp (Unbox/GuardShape/etc. re-derive from their rooted
  // source) -- a cached object local goes stale across a GC-ing call (the callee
  // ptr for a 2nd call, the resume env ptr). GetOp falls back to be.local for
  // plain nodes, so this is a safe superset.
  if (v->type() == MIRType::Object) return GetOp(e, be, v);
  if (v->type() == MIRType::Value) {
    return GetOp(e, be, v) && e.writeOp(Op::I32WrapI64);
  }
  if (be.local(v) < 0) return false;
  return GetLocal(e, uint32_t(be.local(v)));
}

// Sound deopt: box the resume point's live state [this,args,locals,stack] into
// gWJResumeVals, record pc + stack depth, and call wjhelp(WJH_RESUME), which
// rebuilds a PBL frame and finishes the function there.
static bool EmitDeoptResumeInline(Encoder& e, WJBackend& be,
                                  bool skipFrameSpill = false) {
  if (!be.curRp || !be.info) return WJBAIL("resume: no rp/info\n");
  // Build the inline frame chain: index 0 = INNERMOST (the deopt point, possibly
  // in an inlined callee), then its callers out to the OUTERMOST (the compiled
  // function). WJH_RESUME runs them innermost->outermost in PBL, threading each
  // frame's return into the next outer frame's call-result slot. Each frame uses
  // its OWN CompileInfo (script/nargs/nlocals/slots) -- crucial because an
  // inlined frame's operands index its inlinee's script, not be.info's.
  std::vector<MResumePoint*> frames;
  for (MResumePoint* rp = be.curRp; rp; rp = rp->caller()) {
    frames.push_back(rp);
    if (frames.size() > kWJMaxResumeFrames) {
    js::wasm::gWJBailReason = "resume-too-deep";
    return false;  // too deep: bail
  }
  }
  uint32_t nframes = uint32_t(frames.size());
  // Multi-frame (inlined) deopt reconstruction: WJH_RESUME rebuilds each inline
  // frame and resumes the outer (caller) frames AT the call (ResumeAt), so PBL
  // simply re-executes the inlined call -- which is correct. This is the key to
  // keeping hot inlined functions compiled instead of bailing them whole to PBL
  // (verified: richards/deltablue compile fully + run correctly + ~2x faster with
  // this on). Gate OFF with GECKO_WJ_NOINLINERESUME=1 for debugging only.
  static int noInlineResume = getenv("GECKO_WJ_NOINLINERESUME") ? 1 : 0;
  if (nframes > 1 && noInlineResume) return false;

  // Debug: gWJDeoptByOp[curOp]++ at the deopt path (which guard kind deopts).
  // GECKO_WJ_DEOPTLINE: index by the EmitDeopt call's source line instead, so a
  // stale curOp can't mis-attribute (line%kWJNumOps; print maps back).
  static int deoptLine = getenv("GECKO_WJ_DEOPTLINE") ? 1 : 0;
  static int deoptHist = (getenv("GECKO_WJ_DEOPTHIST") || deoptLine) ? 1 : 0;
  uint32_t histIdx = deoptLine ? (be.deoptCallLine / 8) : be.curOp;
  if (deoptHist && histIdx < kWJNumOps) {
    uintptr_t a = uintptr_t(static_cast<void*>(&gWJDeoptByOp[histIdx]));
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(a)) ||
        !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(a)) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
        !e.writeOp(Op::I32Const) || !e.writeVarS32(1) || !e.writeOp(Op::I32Add) ||
        !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
      return false;
  }
  uintptr_t valsBase = uintptr_t(static_cast<void*>(&gWJResumeVals[0]));
  auto storeI32 = [&](uintptr_t addr, int32_t val) -> bool {
    return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(addr)) &&
           e.writeOp(Op::I32Const) && e.writeVarS32(val) &&
           e.writeOp(Op::I32Store) && e.writeVarU32(2) && e.writeVarU32(0);
  };

  uint32_t valsIdx = 0;  // running index into gWJResumeVals across all frames
  for (uint32_t f = 0; f < nframes; f++) {
    MResumePoint* rp = frames[f];
    const CompileInfo& info = rp->block()->info();
    uint32_t nargs = info.nargs();
    uint32_t nlocals = info.nlocals();
    uint32_t firstStack = info.firstStackSlot();
    if (rp->numOperands() < firstStack) return false;
    uint32_t stackDepth = uint32_t(rp->numOperands()) - firstStack;
    uint32_t frameValsOff = valsIdx;

    auto spill = [&](MDefinition* v) -> bool {
      // skipFrameSpill: the hot-path shadow already wrote this frame's operands to
      // their (fixed) gWJResumeVals slots, so the cold dispatcher must NOT re-read
      // the locals here (that read is exactly what pins them live-out across the
      // loop). Just advance valsIdx to keep the metadata layout consistent.
      if (skipFrameSpill) {
        valsIdx++;
        return true;
      }
      if (!e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(valsBase + valsIdx * 8)))
        return false;
      if (!EmitSpillValue(e, be, v)) return false;
      if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
        return false;
      valsIdx++;
      return true;
    };
    if (!spill(rp->getOperand(info.thisSlot()))) return false;
    for (uint32_t i = 0; i < nargs; i++) {
      if (!spill(rp->getOperand(info.argSlot(i)))) return false;
    }
    for (uint32_t i = 0; i < nlocals; i++) {
      if (!spill(rp->getOperand(info.localSlot(i)))) return false;
    }
    for (uint32_t i = 0; i < stackDepth; i++) {
      if (!spill(rp->getOperand(info.stackSlot(i)))) return false;
    }

    // Resume-after modes: the op at rp->pc() already produced its result (on the
    // captured stack); resume at the NEXT bytecode. This applies to the INNERMOST
    // (deopt) frame. For OUTER (inlined-call caller) frames, the captured stack is
    // the PRE-call inputs [callee, this, args]; we resume AT the call so PBL
    // re-executes it (re-running the inlined callee normally) -- so do NOT advance.
    jsbytecode* resumePc = rp->pc();
    if (f == 0 && IsResumeAfter(rp->mode())) resumePc = GetNextPc(resumePc);
    uint32_t pcOff = uint32_t(resumePc - info.script()->code());
    if (getenv("GECKO_WJ_RPDBG")) {
      fprintf(stderr, "[wj-rpc] f=%u %s:%u mode=%d rawpc=%u op=%s resumepc=%u depth=%u\n",
              f, info.script()->filename(), unsigned(info.script()->lineno()),
              int(rp->mode()), unsigned(rp->pc() - info.script()->code()),
              js::CodeName(JSOp(*rp->pc())), pcOff, stackDepth);
    }
    if (!storeI32(uintptr_t(&gWJResumePc[f]), int32_t(pcOff))) return false;
    if (!storeI32(uintptr_t(&gWJResumeStackDepth[f]), int32_t(stackDepth)))
      return false;
    if (!storeI32(uintptr_t(&gWJResumeScriptPtr[f]),
                  int32_t(uintptr_t(info.script()))))
      return false;
    if (!storeI32(uintptr_t(&gWJResumeNArgs[f]), int32_t(nargs))) return false;
    if (!storeI32(uintptr_t(&gWJResumeNLocals[f]), int32_t(nlocals))) return false;
    if (!storeI32(uintptr_t(&gWJResumeValsOff[f]), int32_t(frameValsOff)))
      return false;
    // Env chain: spill the RP's env operand (the frame's actual environment at
    // the deopt point -- the function's own CallObject if one was created, else
    // the captured enclosing env). 0 -> WJH_RESUME falls back to the frame's
    // function environment.
    if (!e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(uintptr_t(&gWJResumeEnvPtr[f]))))
      return false;
    if (!EmitObjPtr(e, be, rp->getOperand(info.environmentChainSlot()))) {
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(0)) return false;
    }
    if (!e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
      return false;

    // Enclosing env for the resumed frame's PBL prologue. The OUTERMOST frame
    // (f == nframes-1) is THIS compiled function; its enclosing (captured) env is
    // envLocal (= gWJCurrentEnv at entry = closure->environment()). The PBL
    // resume uses this to rebuild the frame env correctly instead of the canonical
    // script->function()->environment() (wrong for a re-instantiated closure).
    // Inlined frames: store 0 (keep canonical fallback -- they re-execute anyway).
    if (!e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(uintptr_t(&gWJResumeEnclosingEnv[f]))))
      return false;
    if (f == nframes - 1 && be.useEnvRoot) {
      // GC-current env from the persistent root slot (not the possibly-stale
      // envLocal snapshot): gWJCallRoots[envRootIdx] boxed -> unbox to i32.
      uintptr_t rootsBase = uintptr_t(static_cast<void*>(&gWJCallRoots[0]));
      if (!GetLocal(e, be.envRootIdx) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(8) || !e.writeOp(Op::I32Mul) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(rootsBase)) ||
          !e.writeOp(Op::I32Add) || !e.writeOp(Op::I64Load) ||
          !e.writeVarU32(3) || !e.writeVarU32(0) || !e.writeOp(Op::I32WrapI64))
        return false;
    } else if (f == nframes - 1) {
      if (!GetLocal(e, be.envLocal)) return false;
    } else if (!e.writeOp(Op::I32Const) || !e.writeVarS32(0)) {
      return false;
    }
    if (!e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
      return false;

    if (valsIdx + 64 >= 1024) return false;  // gWJResumeVals overflow guard
  }
  if (!storeI32(uintptr_t(&gWJResumeNFrames), int32_t(nframes))) return false;

  // call wjhelp(WJH_RESUME, 0) -> f64 flag (0 ok / 1 threw), then push a dummy
  // i64 result so the return matches the [f64 flag, i64 result] signature.
  if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(double(WJH_RESUME)) ||
      !e.writeOp(Op::F64Const) || !e.writeFixedF64(0.0) ||
      !e.writeOp(Op::Call) || !e.writeVarU32(0)) {
    return false;
  }
  // WJH_RESUME ran the rest of the function in PBL and left the boxed result in
  // gWJScratch[kWJResultOff]; return it (so a deopt mid-function still yields the
  // correct result in the register-result ABI).
  if (!GetLocal(e, be.scratchBase) || !e.writeOp(Op::I64Load) ||
      !e.writeVarU32(3) || !e.writeVarU32(kWJResultOff)) {
    return false;
  }
  if (!EmitEnvRootPop(e, be)) return false;  // pop the persistent env root slot
  return e.writeOp(Op::Return);
}

// Guard-miss deopt. In out-of-line mode (the dispatch path's hot block bodies),
// branch to the shared dispatcher (`block $D` wrapping the loop) after recording
// the site's resume point -- this keeps the hot loop body tiny. Otherwise inline
// the full spill+resume sequence (single-block / reloop / terminator paths).
static bool EmitDeoptResume(Encoder& e, WJBackend& be) {
  static int noResume = getenv("GECKO_WJ_NORESUME") ? 1 : 0;
  if (noResume) return false;  // bisect: bail any function needing a resume
  static int tinyDeopt = getenv("GECKO_WJ_TINYDEOPT") ? 1 : 0;
  if (tinyDeopt) return e.writeOp(Op::Unreachable);  // PROBE: measure code-bloat cost
  static int constExit = getenv("GECKO_WJ_CONSTEXIT") ? 1 : 0;
  if (constExit) {  // PROBE: minimal exit reading NO locals (isolates resume liveness)
    return e.writeOp(Op::F64Const) && e.writeFixedF64(0.0) &&
           e.writeOp(Op::I64Const) &&
           e.writeVarS64(int64_t(JS::UndefinedValue().asRawBits())) &&
           e.writeOp(Op::Return);
  }
  if (!be.curRp || !be.info) return WJBAIL("resume: no rp/info\n");
  if (be.oolDeopt && be.inDispatchBody) {
    uint32_t site = uint32_t(be.deoptSites.size());
    // Shadow-eligible iff single-frame (the hot-path shadow only mirrors frame 0).
    bool isShadow = be.shadowDeopt && be.curRp->caller() == nullptr;
    if (getenv("GECKO_WJ_DEOPTSITEDBG")) {
      JSScript* ms = be.info ? be.info->script() : nullptr;
      fprintf(stderr, "[wj-deoptsite] %s:%u curOp=%u(%s) callLine=%u\n",
              ms && ms->filename() ? ms->filename() : "?",
              ms ? unsigned(ms->lineno()) : 0, be.curOp,
              be.curOp < kWJNumOps ? "" : "?", be.deoptCallLine);
    }
    be.deoptSites.push_back({be.curRp, be.curOp, isShadow});
    // deoptSiteLocal = site; br to `block $D` (one scope past loop $L; +1 again
    // for the guard's own enclosing `if`, where every in-body deopt sits).
    return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(site)) &&
           e.writeOp(Op::LocalSet) && e.writeVarU32(be.deoptSiteLocal) &&
           e.writeOp(Op::Br) &&
           e.writeVarU32(be.bodyLoopIdx + 2 + be.deoptExtraNest);
  }
  return EmitDeoptResumeInline(e, be);
}

// Shadow-frame spill (hot path). Write the single-frame resume point's operands
// (this/args/locals/stack) to their FIXED gWJResumeVals slots [this@0, args@1..,
// locals@1+nargs.., stack@1+nargs+nlocals..] -- matching EmitDeoptResumeInline's
// layout for frame 0 (valsOff=0). Only the slots whose operand CHANGED vs prevRp
// are re-written (block entry passes prevRp=null -> full spill). After this, a
// deopt at this RP reads the frame from MEMORY, so its locals are NOT live-out at
// the guard exit -- freeing the loop's registers. No-op for multi-frame RPs.
static bool EmitFrameSpillDelta(Encoder& e, WJBackend& be, MResumePoint* rp,
                                MResumePoint* prevRp) {
  if (!rp || rp->caller()) return true;  // multi-frame: dispatcher spills inline
  const CompileInfo& info = rp->block()->info();
  uint32_t nargs = info.nargs(), nlocals = info.nlocals();
  uint32_t firstStack = info.firstStackSlot();
  if (rp->numOperands() < firstStack) return true;
  uint32_t stackDepth = uint32_t(rp->numOperands()) - firstStack;
  uintptr_t valsBase = uintptr_t(static_cast<void*>(&gWJResumeVals[0]));
  static int full = getenv("GECKO_WJ_SHADOWFULL") ? 1 : 0;
  if (full) prevRp = nullptr;  // PROBE: spill every slot every time (no delta)
  // Loop-invariant hoisting: a slot whose operand is DEFINED at a shallower loop
  // depth than this block is loop-invariant here -- it was already spilled in its
  // own (shallower) scope, which dominates and re-runs once per its iteration. So
  // SKIP it in deeper loops: that's what keeps invariants (this/args/invC/outer
  // induction vars) OUT of the hot inner loop, so V8 stops re-reading/pinning
  // them every iteration (the navier/splay 1x ceiling). Constants are bakeable
  // (no pin) and Parameters are handled by the prologue once-spill.
  static int noHoist = getenv("GECKO_WJ_NOHOIST") ? 1 : 0;
  uint32_t blockDepth = rp->block()->loopDepth();
  auto invariant = [&](MDefinition* v) -> bool {
    if (noHoist) return false;
    if (v->isConstant() || v->op() == MDefinition::Opcode::Parameter) return false;
    MBasicBlock* db = v->block();
    return db && db->loopDepth() < blockDepth;
  };
  auto changed = [&](uint32_t op) -> bool {
    if (!prevRp || op >= prevRp->numOperands()) return true;
    return rp->getOperand(op) != prevRp->getOperand(op);
  };
  auto spillSlot = [&](uint32_t idx, MDefinition* v) -> bool {
    return e.writeOp(Op::I32Const) &&
           e.writeVarS32(int32_t(valsBase + idx * 8)) &&
           EmitSpillValue(e, be, v) && e.writeOp(Op::I64Store) &&
           e.writeVarU32(3) && e.writeVarU32(0);
  };
  // this/args were spilled ONCE at the prologue. Re-spill a slot here only if it
  // no longer holds its Parameter (i.e. `this`/an arg was reassigned in the body),
  // which is rare. Skipping the common (unchanged) case keeps these invariants out
  // of the hot loop so they are not re-read/re-pinned every iteration.
  MDefinition* thisOp = rp->getOperand(info.thisSlot());
  if (thisOp->op() != MDefinition::Opcode::Parameter &&
      !spillSlot(0, thisOp))
    return false;
  for (uint32_t i = 0; i < nargs; i++) {
    MDefinition* argOp = rp->getOperand(info.argSlot(i));
    if (argOp->op() != MDefinition::Opcode::Parameter && changed(info.argSlot(i)) &&
        !spillSlot(1 + i, argOp))
      return false;
  }
  for (uint32_t i = 0; i < nlocals; i++) {
    MDefinition* lo = rp->getOperand(info.localSlot(i));
    if (!invariant(lo) && changed(info.localSlot(i)) &&
        !spillSlot(1 + nargs + i, lo))
      return false;
  }
  for (uint32_t i = 0; i < stackDepth; i++) {
    MDefinition* so = rp->getOperand(info.stackSlot(i));
    if (!invariant(so) && changed(info.stackSlot(i)) &&
        !spillSlot(1 + nargs + nlocals + i, so))
      return false;
  }
  return true;
}

// Emit a call to wjhelp(kind, site) for a VM op whose boxed operands the caller
// has already staged in gWJScratch[0..]. Roots the resume point's live
// Object/Value locals across the call (the helper may GC), propagates a thrown
// exception (return [1.0, 0]), reloads the roots, and leaves the boxed i64
// result (gWJScratch[kWJResultOff]) on the stack. `ins` is the node producing
// the result (excluded from rooting). Returns false => can't emit (bail fn).
// Collect the GC roots that must survive a (possibly-GCing) call at `ins`:
// the resume point's live Object/Value operands PLUS every Object/Value value
// defined earlier in `ins`'s block (so it dominates and definitely ran) that
// still has a use at/after `ins`. The latter is essential because our resume
// points are frequently the stale block-entry RP, which records mid-function
// slots as their entry constants -- missing live temporaries (a call's callee,
// a freshly-constructed object, an operand held across a callee that allocates).
static void WJCollectRoots(WJBackend& be, MInstruction* ins,
                           std::vector<uint32_t>& rootLocal,
                           std::vector<uint8_t>& rootIsObj) {
  auto add = [&](MDefinition* d, bool isObj) {
    int32_t l = be.local(d);
    if (l < 0) return;
    for (uint32_t r = 0; r < rootLocal.size(); r++) {
      if (rootLocal[r] == uint32_t(l)) return;
    }
    rootLocal.push_back(uint32_t(l));
    rootIsObj.push_back(isObj ? 1 : 0);
  };
  auto liveAfterIns = [&](MDefinition* d) -> bool {
    for (MUseIterator u = d->usesBegin(); u != d->usesEnd(); u++) {
      MNode* consumer = u->consumer();
      if (consumer->isDefinition()) {
        MDefinition* cd = consumer->toDefinition();
        if (cd->id() > ins->id()) return true;
        MBasicBlock* ub = cd->block();
        if (ub && ub != ins->block() && !ub->dominates(ins->block())) return true;
      }
    }
    return false;
  };
  auto consider = [&](MDefinition* d) {
    bool isObj = d->type() == MIRType::Object;
    bool isVal = d->type() == MIRType::Value;
    if ((isObj || isVal) && be.local(d) >= 0 && liveAfterIns(d)) add(d, isObj);
  };
  // Root values whose def dominates `ins` and are live across it, unioned with
  // the call's resume-point operands (the precise abstract-interpreter live
  // state, covering loop-carried values the dominance walk misses). Interior
  // pointers are kept fresh by GetOp rematerialization, not rooting.
  MBasicBlock* insBlock = ins->block();
  MIRGraph& graph = insBlock->graph();
  for (MBasicBlockIterator b = graph.begin(); b != graph.end(); b++) {
    MBasicBlock* blk = *b;
    bool same = (blk == insBlock);
    if (!same && !blk->dominates(insBlock)) continue;
    for (MPhiIterator p = blk->phisBegin(); p != blk->phisEnd(); p++) consider(*p);
    for (MInstructionIterator it = blk->begin(); it != blk->end(); it++) {
      MInstruction* d = *it;
      if (same && d->id() >= ins->id()) break;
      consider(d);
    }
  }
  for (MResumePoint* rp = be.curRp; rp; rp = rp->caller()) {
    uint32_t n = rp->numOperands();
    for (uint32_t i = 0; i < n; i++) {
      MDefinition* d = rp->getOperand(i);
      bool isObj = d->type() == MIRType::Object;
      bool isVal = d->type() == MIRType::Value;
      if ((isObj || isVal) && be.local(d) >= 0) add(d, isObj);
    }
  }
  // NB: the closure env is NOT rooted here -- it lives in a PERSISTENT GC-shadow
  // slot (gWJCallRoots[envRootIdx]) pushed in the prologue for the whole
  // invocation, and FunctionEnvironment rematerializes from it (always GC-current,
  // even across non-call GC points). See WJEmitBody prologue + GetOp.
}

static bool EmitHelperCallResult(Encoder& e, WJBackend& be, MInstruction* ins,
                                 int kind, uint32_t site) {
  std::vector<uint32_t> rootLocal;
  std::vector<uint8_t> rootIsObj;
  WJCollectRoots(be, ins, rootLocal, rootIsObj);
  static int noRoot = getenv("GECKO_WJ_NOROOT") ? 1 : 0;
  if (noRoot) {
    rootLocal.clear();
    rootIsObj.clear();
  }
  uint32_t nRoots = uint32_t(rootLocal.size());
  uintptr_t rootsBase = uintptr_t(static_cast<void*>(&gWJCallRoots[0]));
  uintptr_t spAddr = uintptr_t(static_cast<void*>(&gWJRootSP));
  auto emitSPAdjust = [&](int32_t delta) -> bool {
    return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(spAddr)) &&
           e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(spAddr)) &&
           e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(0) &&
           e.writeOp(Op::I32Const) && e.writeVarS32(delta) &&
           e.writeOp(Op::I32Add) && e.writeOp(Op::I32Store) && e.writeVarU32(2) &&
           e.writeVarU32(0);
  };
  // Compute the spill slot BASE once: rootBaseLocal = gWJRootSP*8 + rootsBase. SP is
  // unchanged between spill and reload (bumped after spill, restored before reload),
  // so the same base serves both; slot k is base + k*8 via the wasm store/load
  // immediate offset -- avoids reloading gWJRootSP and recomputing *8+base per root
  // (was ~8 ops/root x2). splay is call+root-heavy: this is its single biggest lever
  // (GC-root spill = +14% NOROOT). base is 8-aligned (rootsBase + sp*8 both are).
  if (nRoots) {
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(spAddr)) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
        !e.writeOp(Op::I32Const) || !e.writeVarS32(8) || !e.writeOp(Op::I32Mul) ||
        !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(rootsBase)) ||
        !e.writeOp(Op::I32Add) || !e.writeOp(Op::LocalSet) ||
        !e.writeVarU32(be.rootBaseLocal))
      return false;
  }
  auto emitSlotBase = [&]() -> bool { return GetLocal(e, be.rootBaseLocal); };
  for (uint32_t k = 0; k < nRoots; k++) {
    if (!emitSlotBase()) return false;
    if (!GetLocal(e, rootLocal[k])) return false;
    if (rootIsObj[k] && !EmitBoxFromStack(e, MIRType::Object)) return false;
    if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(k * 8))
      return false;
  }
  if (nRoots && !emitSPAdjust(int32_t(nRoots))) return false;
  // wjhelp(kind, site) -> f64 flag
  if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(double(kind)) ||
      !e.writeOp(Op::F64Const) || !e.writeFixedF64(double(site)) ||
      !e.writeOp(Op::Call) || !e.writeVarU32(0))
    return false;
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.callFlagLocal)) return false;
  if (nRoots && !emitSPAdjust(-int32_t(nRoots))) return false;
  // exception -> return [1.0, 0]
  if (!GetLocal(e, be.callFlagLocal) || !e.writeOp(Op::F64Const) ||
      !e.writeFixedF64(0.0) || !e.writeOp(Op::F64Ne))
    return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
  if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(1.0) ||
      !e.writeOp(Op::I64Const) || !e.writeVarS64(0) || !e.writeOp(Op::Return))
    return false;
  if (!e.writeOp(Op::End)) return false;
  for (uint32_t k = 0; k < nRoots; k++) {
    if (!emitSlotBase()) return false;
    if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(k * 8))
      return false;
    if (rootIsObj[k] && !e.writeOp(Op::I32WrapI64)) return false;
    if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(rootLocal[k])) return false;
  }
  // boxed result
  return GetLocal(e, be.scratchBase) && e.writeOp(Op::I64Load) &&
         e.writeVarU32(3) && e.writeVarU32(kWJResultOff);
}

// Stage a boxed value operand into gWJScratch[slot] (byte offset slot*8).
static bool EmitStageScratch(Encoder& e, WJBackend& be, MDefinition* v,
                             uint32_t slot) {
  return GetLocal(e, be.scratchBase) && EmitSpillValue(e, be, v) &&
         e.writeOp(Op::I64Store) && e.writeVarU32(3) && e.writeVarU32(slot * 8);
}

// Stage a compile-time-constant boxed Value into gWJScratch[slot]. GC-thing
// values are baked via the traced constant pool (loaded live at runtime);
// others are immediate.
static bool EmitStageConstBoxed(Encoder& e, WJBackend& be, uint64_t bits,
                                uint32_t slot) {
  if (!GetLocal(e, be.scratchBase)) return false;
  JS::Value v = JS::Value::fromRawBits(bits);
  if (v.isGCThing()) {
    uintptr_t pslot = WJInternConstant(bits);
    if (!pslot) return false;
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(pslot)) ||
        !e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0))
      return false;
  } else if (!e.writeOp(Op::I64Const) || !e.writeVarS64(int64_t(bits))) {
    return false;
  }
  return e.writeOp(Op::I64Store) && e.writeVarU32(3) && e.writeVarU32(slot * 8);
}

// Allocate a prop-IC site for the CURRENT mega read/store, REUSING the same site
// across recompiles via a stable (script, per-compile read-index) key. This keeps
// the site pool bounded by the number of distinct read sites (not by recompile
// count) so it never exhausts -- the exhaustion was why a late mega-recompile's
// read couldn't convert and fell back to a wrong fixed-slot load (deltablue 741).
static uint32_t WJPropSite(WJBackend& be) {
  JSScript* s = be.info ? be.info->script() : nullptr;
  uint64_t key = (uint64_t(uint32_t(uintptr_t(static_cast<void*>(s)))) << 20) |
                 (uint64_t(be.propReadIdx++) & 0xFFFFF);
  return js::wasm::WJAllocPropSiteKeyed(key);
}

// Inline property-load IC for an own data property with a compile-time-known
// name (MegamorphicLoadSlot, named GetPropertyCache). `object` must be a
// receiver of MIRType::Object. Loads obj->shape(), compares to up to kWJPropWays
// per-site cached shapes; on a match loads the slot inline (fixed or dynamic) --
// no C++ hop. On a miss it calls wjhelp(WJH_PROPIC, site), which does a pure
// lookup, fills the IC, and returns the value. Leaves the boxed value (i64) on
// the stack. `site` must be a valid (nonzero) pre-allocated prop-site. Mirrors
// Ion's megamorphic-cache inline load; the per-site cache avoids hash collisions.
static bool EmitPropIC(Encoder& e, WJBackend& be, MInstruction* ins,
                       MDefinition* object, uint64_t keyBits, uint32_t site,
                       js::Shape* bakedShape = nullptr, uint32_t bakedByteOff = 0,
                       bool bakedFixed = false) {
  uint32_t base = site * kWJPropWays;
  // Sites are REUSED across recompiles (WJPropSite, keyed by script+read-index). If
  // a warmer recompile shifts which reads convert, a site could be repurposed for a
  // DIFFERENT property -- so reset the key AND the cached (shape->offset) ways here
  // each compile. A stale shape entry from the old property would otherwise match a
  // same-shaped receiver and load the OLD property's offset (wrong slot).
  if (gWJPropKey[site] != keyBits) {
    for (uint32_t w = 0; w < kWJPropWays; w++) gWJPropShape[base + w] = 0;
  }
  gWJPropKey[site] = keyBits;
  uintptr_t shapeArr = uintptr_t(static_cast<void*>(&gWJPropShape[0]));
  uintptr_t offArr = uintptr_t(static_cast<void*>(&gWJPropOff[0]));
  // propObj = receiver pointer (rematerialized, GC-current).
  if (!GetOp(e, be, object)) return false;
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propObjLocal)) return false;
  // propShape = I32Load(propObj + offsetOfShape).
  if (!GetLocal(e, be.propObjLocal)) return false;
  if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
      !e.writeVarU32(uint32_t(offsetof(JS::shadow::Object, shape))))
    return false;
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propShapeLocal)) return false;

  // Load the slot value for way `w` (tagged offset at gWJPropOff[base+w]).
  auto emitSlotLoad = [&](uint32_t w) -> bool {
    if (!e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(offArr + (base + w) * 4)) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
        !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propTaggedLocal))
      return false;
    // baseAddr = (tagged & 1) ? propObj : I32Load(propObj + offsetOfSlots)
    if (!GetLocal(e, be.propTaggedLocal) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(1) || !e.writeOp(Op::I32And))
      return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32))) return false;
    if (!GetLocal(e, be.propObjLocal)) return false;
    if (!e.writeOp(Op::Else)) return false;
    if (!GetLocal(e, be.propObjLocal) || !e.writeOp(Op::I32Load) ||
        !e.writeVarU32(2) ||
        !e.writeVarU32(uint32_t(js::NativeObject::offsetOfSlots())))
      return false;
    if (!e.writeOp(Op::End)) return false;
    // + (tagged >> 1)  (byte offset)
    if (!GetLocal(e, be.propTaggedLocal) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(1) || !e.writeOp(Op::I32ShrU) || !e.writeOp(Op::I32Add))
      return false;
    return e.writeOp(Op::I64Load) && e.writeVarU32(3) && e.writeVarU32(0);
  };

  // Runtime-cache ways + helper miss path (the polymorphic fallback). Each way:
  // gWJPropShape[base+w] == propShape ? load that way's (runtime) tagged offset.
  auto emitWaysAndHelper = [&]() -> bool {
    for (uint32_t w = 0; w < kWJPropWays; w++) {
      if (!e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(shapeArr + (base + w) * 4)) ||
          !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
          !GetLocal(e, be.propShapeLocal) || !e.writeOp(Op::I32Eq))
        return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I64)))
        return false;  // both arms yield i64
      if (!emitSlotLoad(w)) return false;
      if (!e.writeOp(Op::Else)) return false;
    }
    if (!EmitStageScratch(e, be, object, 0)) return false;
    if (!EmitHelperCallResult(e, be, ins, WJH_PROPIC, site)) return false;
    for (uint32_t w = 0; w < kWJPropWays; w++) {
      if (!e.writeOp(Op::End)) return false;
    }
    return true;
  };

  // BAKED MONOMORPHIC FAST PATH (Ion-like): for a read whose receiver shape is
  // known at compile time (a GuardShape site), guard obj->shape == the baked
  // (interned, relocation-safe) shape and load at a COMPILE-TIME-CONSTANT offset --
  // no runtime offset load, no fixed/dynamic tagged-bit branch (those are the
  // EmitPropIC overhead that made mega ~10% slower than a plain guarded load on
  // monomorphic benches like splay). On a shape mismatch (polymorphic receiver, or
  // a moved-then-different shape) fall to the runtime cache + helper -- NO deopt,
  // so still correct + fast for deltablue's polymorphic constraint methods.
  uint32_t bakedSlot = bakedShape ? WJInternShape(uintptr_t(bakedShape)) : 0;
  if (bakedShape && bakedSlot) {
    if (!GetLocal(e, be.propShapeLocal)) return false;
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(bakedSlot)) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0))
      return false;  // live (relocated) baked shape from the pool
    if (!e.writeOp(Op::I32Eq)) return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I64))) return false;
    // then: baked constant-offset load.
    if (!GetLocal(e, be.propObjLocal)) return false;
    if (!bakedFixed) {  // dynamic slot: base = obj->slots
      if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
          !e.writeVarU32(uint32_t(js::NativeObject::offsetOfSlots())))
        return false;
    }
    if (bakedByteOff) {
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(bakedByteOff)) ||
          !e.writeOp(Op::I32Add))
        return false;
    }
    if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0))
      return false;
    if (!e.writeOp(Op::Else)) return false;
    if (!emitWaysAndHelper()) return false;
    return e.writeOp(Op::End);
  }
  return emitWaysAndHelper();
}

// GetName IC fast path. Loads the per-site cached holder (gWJNameHolder[site], a
// traced realm-singleton global object/lexical); if present and its shape still
// matches, loads the cached slot directly (the value may have mutated, the slot
// location hasn't). On holder==0 or shape mismatch, stages env+name and calls
// wjhelp(WJH_GETNAME, site) which resolves + fills the cache + returns the value.
// Leaves the boxed value (i64) on the stack. `nameBits` = StringValue(name) bits.
static bool EmitNameIC(Encoder& e, WJBackend& be, MInstruction* ins,
                       MDefinition* envOp, uint64_t nameBits, uint32_t site) {
  gWJNameKey[site] = nameBits;
  uintptr_t holderAddr = uintptr_t(static_cast<void*>(&gWJNameHolder[site]));
  uintptr_t shapeAddr = uintptr_t(static_cast<void*>(&gWJNameShape[site]));
  uintptr_t offAddr = uintptr_t(static_cast<void*>(&gWJNameOff[site]));
  // holder = load gWJNameHolder[site] -> propObjLocal (reuse prop-IC scratch).
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(holderAddr)) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
      !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propObjLocal))
    return false;
  // valid = holder!=0 ? (holder->shape == gWJNameShape[site]) : 0
  if (!GetLocal(e, be.propObjLocal) || !e.writeOp(Op::I32Eqz)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32))) return false;
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(0)) return false;  // holder==0
  if (!e.writeOp(Op::Else)) return false;
  if (!GetLocal(e, be.propObjLocal) || !e.writeOp(Op::I32Load) ||
      !e.writeVarU32(2) ||
      !e.writeVarU32(uint32_t(offsetof(JS::shadow::Object, shape))) ||
      !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(shapeAddr)) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
      !e.writeOp(Op::I32Eq))
    return false;
  if (!e.writeOp(Op::End)) return false;
  // if (valid) load cached slot else miss-helper. Both arms yield i64.
  if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I64))) return false;
  // hit: tagged = gWJNameOff[site]; base = (tagged&1)?holder:holder->slots_;
  //      value = I64Load(base + (tagged>>1)).
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(offAddr)) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
      !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propTaggedLocal))
    return false;
  if (!GetLocal(e, be.propTaggedLocal) || !e.writeOp(Op::I32Const) ||
      !e.writeVarS32(1) || !e.writeOp(Op::I32And))
    return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32))) return false;
  if (!GetLocal(e, be.propObjLocal)) return false;
  if (!e.writeOp(Op::Else)) return false;
  if (!GetLocal(e, be.propObjLocal) || !e.writeOp(Op::I32Load) ||
      !e.writeVarU32(2) ||
      !e.writeVarU32(uint32_t(js::NativeObject::offsetOfSlots())))
    return false;
  if (!e.writeOp(Op::End)) return false;
  if (!GetLocal(e, be.propTaggedLocal) || !e.writeOp(Op::I32Const) ||
      !e.writeVarS32(1) || !e.writeOp(Op::I32ShrU) || !e.writeOp(Op::I32Add))
    return false;
  if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0))
    return false;
  if (!e.writeOp(Op::Else)) return false;
  // miss: stage env + name, call WJH_GETNAME(site) (resolves + fills cache).
  if (!EmitStageScratch(e, be, envOp, 0)) return false;
  if (!EmitStageConstBoxed(e, be, nameBits, 1)) return false;
  if (!EmitHelperCallResult(e, be, ins, WJH_GETNAME, site)) return false;
  return e.writeOp(Op::End);
}

// Emit a guarded incremental-GC pre-write barrier on an OLD value about to be
// overwritten. `emitOldValue` must push the old boxed Value (i64) on the stack.
// Fast path (the common case, flag clear): load+test the zone marking flag and
// skip. Only when an incremental GC is in progress do we stash the old value in
// gWJHelpVal and call WJH_PREBARRIER (gc::ValuePreWriteBarrier). Lets object/
// unknown-typed slot+element stores stay in JIT instead of bailing to PBL.
template <typename F>
static bool EmitGuardedValuePreBarrier(Encoder& e, WJBackend& be, F emitOldValue) {
  uintptr_t flagAddr = gWJMarkBarrierAddr;
  if (!flagAddr) return false;  // address not baked -> caller bails (sound)
  uintptr_t valAddr = uintptr_t(static_cast<void*>(&gWJHelpVal));
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(flagAddr)) ||
      !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
      !e.writeOp(Op::I32Const) || !e.writeVarS32(1) || !e.writeOp(Op::I32And))
    return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(valAddr))) return false;
  if (!emitOldValue()) return false;  // pushes the old i64 value
  if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
    return false;
  if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(double(WJH_PREBARRIER)) ||
      !e.writeOp(Op::F64Const) || !e.writeFixedF64(0.0) ||
      !e.writeOp(Op::Call) || !e.writeVarU32(0) || !e.writeOp(Op::Drop))
    return false;
  return e.writeOp(Op::End);
}

// Inline property-STORE IC for a writable own data property (SetPropertyCache /
// named set). Mirrors EmitPropIC: matches obj->shape() against per-site cached
// shapes; on a hit stores the value into the slot inline, with a guarded
// incremental pre-write barrier on the old value and a (self-guarded) whole-cell
// post-write barrier when the stored value may be a GC pointer. On a miss it
// calls wjhelp(WJH_SETPROPIC), which converts idval, fills the IC for a writable
// own data property, and stores (or falls back to the generic set). This is an
// EFFECT: it leaves nothing on the stack. `site` must be nonzero.
static bool EmitPropStoreIC(Encoder& e, WJBackend& be, MInstruction* ins,
                            MDefinition* object, MDefinition* idval,
                            MDefinition* value, bool strict, uint32_t site) {
  gWJPropStrict[site] = strict ? 1 : 0;
  uint32_t base = site * kWJPropWays;
  uintptr_t shapeArr = uintptr_t(static_cast<void*>(&gWJPropShape[0]));
  uintptr_t offArr = uintptr_t(static_cast<void*>(&gWJPropOff[0]));
  // propObj = receiver pointer (rematerialized).
  if (!GetOp(e, be, object)) return false;
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propObjLocal)) return false;
  // propShape = I32Load(propObj + offsetOfShape).
  if (!GetLocal(e, be.propObjLocal)) return false;
  if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
      !e.writeVarU32(uint32_t(offsetof(JS::shadow::Object, shape))))
    return false;
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propShapeLocal)) return false;
  // boxed value -> unboxScratch (reused for store + post-barrier inspection).
  if (!EmitSpillValue(e, be, value)) return false;
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.unboxScratch)) return false;

  MIRType vt = value->type();
  bool valMaybeGC = (vt == MIRType::Object || vt == MIRType::Value ||
                     vt == MIRType::String || vt == MIRType::Symbol ||
                     vt == MIRType::BigInt);

  auto emitStore = [&](uint32_t w) -> bool {
    // propTagged = gWJPropOff[base+w]; slotAddr -> propTaggedLocal repurposed.
    if (!e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(offArr + (base + w) * 4)) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
        !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propTaggedLocal))
      return false;
    // slotBase = (tagged & 1) ? propObj : I32Load(propObj + offsetOfSlots)
    if (!GetLocal(e, be.propTaggedLocal) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(1) || !e.writeOp(Op::I32And))
      return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32))) return false;
    if (!GetLocal(e, be.propObjLocal)) return false;
    if (!e.writeOp(Op::Else)) return false;
    if (!GetLocal(e, be.propObjLocal) || !e.writeOp(Op::I32Load) ||
        !e.writeVarU32(2) ||
        !e.writeVarU32(uint32_t(js::NativeObject::offsetOfSlots())))
      return false;
    if (!e.writeOp(Op::End)) return false;
    // slotAddr = slotBase + (tagged >> 1)  -> propShapeLocal (free in this arm)
    if (!GetLocal(e, be.propTaggedLocal) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(1) || !e.writeOp(Op::I32ShrU) || !e.writeOp(Op::I32Add) ||
        !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propShapeLocal))
      return false;
    // Guarded incremental pre-write barrier on the OLD slot value.
    if (!EmitGuardedValuePreBarrier(e, be, [&]() {
          return GetLocal(e, be.propShapeLocal) && e.writeOp(Op::I64Load) &&
                 e.writeVarU32(3) && e.writeVarU32(0);
        }))
      return false;
    // Store: *(slotAddr) = value.
    if (!GetLocal(e, be.propShapeLocal) || !GetLocal(e, be.unboxScratch) ||
        !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
      return false;
    // Inline guarded whole-cell post-write barrier. Buffer obj ONLY when obj is
    // tenured AND the stored value is a nursery GC cell (mirrors Ion). The nursery
    // test loads chunk(ptr)->storeBuffer (nonzero == nursery); chunkBase = ptr &
    // ~ChunkMask. The payload deref is short-circuited behind the GC-thing check
    // so a non-GC Value never dereferences a non-pointer payload.
    if (valMaybeGC) {
      const uint32_t chunkMaskInv = uint32_t(~js::gc::ChunkMask);
      const uint32_t sbOff = uint32_t(js::gc::ChunkStoreBufferOffset);
      // objTenured = (chunk(obj)->storeBuffer == 0)
      if (!GetLocal(e, be.propObjLocal) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(chunkMaskInv)) || !e.writeOp(Op::I32And) ||
          !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(sbOff) ||
          !e.writeOp(Op::I32Eqz))
        return false;
      // isGCThing
      if (vt == MIRType::Value) {
        // (unboxScratch >> 32) u>= JSVAL_TAG_STRING
        if (!GetLocal(e, be.unboxScratch) || !e.writeOp(Op::I64Const) ||
            !e.writeVarS64(32) || !e.writeOp(Op::I64ShrU) ||
            !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(uint32_t(JSVAL_TAG_STRING))) ||
            !e.writeOp(Op::I32GeU))
          return false;
      } else if (!e.writeOp(Op::I32Const) || !e.writeVarS32(1)) {
        return false;  // Object/String/Symbol/BigInt: always a GC thing
      }
      if (!e.writeOp(Op::I32And)) return false;  // objTenured && isGCThing
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      // valNursery = chunk(payload)->storeBuffer (nonzero => nursery)
      if (!GetLocal(e, be.unboxScratch) || !e.writeOp(Op::I32WrapI64) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(chunkMaskInv)) ||
          !e.writeOp(Op::I32And) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
          !e.writeVarU32(sbOff))
        return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      uintptr_t objAddr = uintptr_t(static_cast<void*>(&gWJHelpObj));
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(objAddr)) ||
          !GetLocal(e, be.propObjLocal) || !e.writeOp(Op::I32Store) ||
          !e.writeVarU32(2) || !e.writeVarU32(0))
        return false;
      if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(double(WJH_POSTBARRIER)) ||
          !e.writeOp(Op::F64Const) || !e.writeFixedF64(0.0) ||
          !e.writeOp(Op::Call) || !e.writeVarU32(0) || !e.writeOp(Op::Drop))
        return false;
      if (!e.writeOp(Op::End) || !e.writeOp(Op::End)) return false;
    }
    return true;
  };

  for (uint32_t w = 0; w < kWJPropWays; w++) {
    if (!e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(shapeArr + (base + w) * 4)) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
        !GetLocal(e, be.propShapeLocal) || !e.writeOp(Op::I32Eq))
      return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;  // void
    if (!emitStore(w)) return false;
    if (!e.writeOp(Op::Else)) return false;
  }
  // Miss: stage obj, idval, value; call WJH_SETPROPIC (fills + stores); drop.
  if (!EmitStageScratch(e, be, object, 0)) return false;
  if (!EmitStageScratch(e, be, idval, 1)) return false;
  if (!EmitStageScratch(e, be, value, 2)) return false;
  if (!EmitHelperCallResult(e, be, ins, WJH_SETPROPIC, site)) return false;
  if (!e.writeOp(Op::Drop)) return false;
  for (uint32_t w = 0; w < kWJPropWays; w++) {
    if (!e.writeOp(Op::End)) return false;
  }
  return true;
}

// Megamorphic recompile (be.forceMega) support. Find the property KEY for a
// fixed slot on a shape (the inverse of the cold-IC monomorphic GuardShape: we
// re-derive the name so a multi-shape EmitPropIC can self-guard + helper-fall-
// back instead of deopting on a polymorphic receiver). Native shapes only.
static bool WJDerivePropKey(js::Shape* shape, uint32_t slot, uint64_t* keyBits) {
  if (!shape || !shape->isShared()) return false;  // shared native shapes only
  for (js::SharedShapePropertyIter<js::NoGC> it(&shape->asShared()); !it.done();
       it++) {
    if (it->hasSlot() && it->slot() == slot) {
      // EmitPropIC/WJH_PROPIC expect a PropertyKey (jsid) raw bits, NOT a Value.
      *keyBits = it->key().asRawBits();
      return true;
    }
  }
  return false;
}
// A representative shared shape for a shape guard (GuardShape or, after the IC
// warms polymorphic, GuardShapeList). For a list all shapes share the property's
// slot (Warp only folds same-slot shapes), so shapes()[0] derives the same key.
static js::Shape* WJGuardRepShape(jit::MDefinition* g) {
  if (g->isGuardShape()) return g->toGuardShape()->shape();
  if (g->isGuardShapeList()) {
    const auto& shapes = g->toGuardShapeList()->shapeList()->shapes();
    for (size_t i = 0; i < 4; i++)
      if (shapes[i]) return shapes[i];
  }
  return nullptr;
}
static bool WJIsShapeGuard(jit::MDefinition* g) {
  return g && (g->isGuardShape() || g->isGuardShapeList());
}
// A shape guard (GuardShape/GuardShapeList) is mega-convertible iff every SHAPE-
// DEPENDENT use is a fixed/dynamic-slot READ or derivable STORE. Then the guard
// becomes a no-op passthrough (no deopt -- crucially, GuardShapeList no longer
// deopts on a shape outside its captured set, which was deltablue 414's storm)
// and reads become self-guarding multi-shape EmitPropIC. Shape-AGNOSTIC uses
// (calls/compares/returns) are fine unguarded.
static bool WJMegaConvertibleGuard(jit::MDefinition* gg) {
  using Op = jit::MDefinition::Opcode;
  js::Shape* shape = WJGuardRepShape(gg);
  if (!shape || !shape->isShared()) return false;
  uint32_t nfixed = shape->asShared().numFixedSlots();
  for (jit::MUseIterator u = gg->usesBegin(); u != gg->usesEnd(); u++) {
    jit::MNode* c = u->consumer();
    if (!c->isDefinition()) continue;  // resume points etc. are shape-agnostic
    jit::MDefinition* d = c->toDefinition();
    Op op = d->op();
    uint64_t k;
    if (op == Op::LoadFixedSlot) {
      if (!WJDerivePropKey(shape, uint32_t(d->toLoadFixedSlot()->slot()), &k))
        return false;
    } else if (op == Op::LoadFixedSlotAndUnbox) {
      if (!WJDerivePropKey(shape, uint32_t(d->toLoadFixedSlotAndUnbox()->slot()), &k))
        return false;
    } else if (op == Op::StoreFixedSlot) {
      // store converts to a baked-key store IC; require a derivable key + that the
      // object operand is THIS guard (operand 0), not the stored value.
      auto* s = d->toStoreFixedSlot();
      if (s->object() != gg) return false;  // gg is the stored VALUE -> bail
      if (!WJDerivePropKey(shape, uint32_t(s->slot()), &k)) return false;
    } else if (op == Op::Slots) {
      // dynamic slots: every use of the Slots must be a derivable dynamic READ.
      for (jit::MUseIterator su = d->usesBegin(); su != d->usesEnd(); su++) {
        if (!su->consumer()->isDefinition()) continue;
        jit::MDefinition* sd = su->consumer()->toDefinition();
        Op so = sd->op();
        uint32_t dynSlot;
        if (so == Op::LoadDynamicSlot)
          dynSlot = uint32_t(sd->toLoadDynamicSlot()->slot());
        else if (so == Op::LoadDynamicSlotAndUnbox)
          dynSlot = uint32_t(sd->toLoadDynamicSlotAndUnbox()->slot());
        else
          return false;  // dynamic store / other -> keep guard
        if (!WJDerivePropKey(shape, nfixed + dynSlot, &k)) return false;
      }
    } else if (op == Op::Elements || op == Op::StoreDynamicSlot ||
               op == Op::LoadFixedSlotFromOffset ||
               op == Op::StoreFixedSlotFromOffset || op == Op::GuardShape) {
      return false;  // shape-dependent, not converting -> keep the guard
    }
    // else: shape-agnostic use -> safe with an unguarded object.
  }
  return true;  // no disallowed shape-dependent use -> safe to passthrough
}
// The object operand guarded by a shape guard (GuardShape/GuardShapeList).
static jit::MDefinition* WJGuardObject(jit::MDefinition* g) {
  if (g->isGuardShape()) return g->toGuardShape()->object();
  if (g->isGuardShapeList()) return g->toGuardShapeList()->object();
  return nullptr;
}
// If `slots` is MSlots(obj) where obj is a mega-convertible shape guard, return
// that guard (dynamic reads through it convert to EmitPropIC); else null.
static jit::MDefinition* WJMegaSlotsGuard(jit::MDefinition* slots) {
  if (!slots || !slots->isSlots()) return nullptr;
  jit::MDefinition* obj = slots->getOperand(0);
  if (!WJIsShapeGuard(obj) || !WJMegaConvertibleGuard(obj)) return nullptr;
  return obj;
}
// GuardShapeListToOffset (polymorphic read where the property sits at DIFFERENT
// slots per shape -> shape->offset table + deopt on an unlisted shape, deltablue
// 414's storm). For mega: derive the property key (all shapes hold the same
// property; shapes()[0]+offsets()[0] suffices -- offsets are dynamic-slot BYTE
// offsets, so abs slot = numFixedSlots + offset/8). Returns the guarded object +
// key iff convertible.
static bool WJMegaGSLTO(jit::MDefinition* g, jit::MDefinition** objOut,
                        uint64_t* keyOut) {
  if (!g->isGuardShapeListToOffset()) return false;
  auto* gg = g->toGuardShapeListToOffset();
  const auto& shapes = gg->shapeList()->shapes();
  const auto& offsets = gg->shapeList()->offsets();
  js::Shape* s0 = nullptr;
  uint32_t off0 = 0;
  for (size_t i = 0; i < 4; i++) {
    if (shapes[i]) { s0 = shapes[i]; off0 = offsets[i]; break; }
  }
  if (!s0 || !s0->isShared()) return false;
  if (off0 % sizeof(JS::Value) != 0) return false;
  uint32_t slot = s0->asShared().numFixedSlots() + off0 / uint32_t(sizeof(JS::Value));
  if (!WJDerivePropKey(s0, slot, keyOut)) return false;
  *objOut = gg->object();
  return true;
}

// A generic-IC helper (WJH_BINARYARITH/UNARYARITH/...) leaves a BOXED i64 result
// on the stack. If Warp TYPED the node (Int32/Double/Boolean), coerce the box to
// that wasm type so it matches the node's local (else host rejects the module).
static bool EmitHelperResultAsType(Encoder& e, WJBackend& be, jit::MIRType type) {
  if (type == jit::MIRType::Value) return true;  // box already matches i64 local
  if (type == jit::MIRType::Double || type == jit::MIRType::Float32) {
    // A numeric box may be int32-tagged (e.g. 2+3=5) or double-tagged. Convert
    // both to f64 WITHOUT deopting (arith of a Double-typed node is always
    // numeric): int32 -> f64.convert_i32_s; else -> f64.reinterpret. (Raw
    // reinterpret alone mangles an int32 box -> garbage f64 -> OOB/trap.)
    uint32_t s = be.unboxScratch;
    if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(s)) return false;
    // tag == INT32 ?
    if (!GetLocal(e, s) || !e.writeOp(Op::I64Const) || !e.writeVarS64(32) ||
        !e.writeOp(Op::I64ShrU) || !e.writeOp(Op::I32WrapI64) ||
        !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_INT32))) || !e.writeOp(Op::I32Eq))
      return false;
    if (!e.writeOp(Op::If) || !e.writeFixedU8(0x7C)) return false;  // result f64
    if (!GetLocal(e, s) || !e.writeOp(Op::I32WrapI64) ||
        !e.writeOp(Op::F64ConvertI32S))
      return false;
    if (!e.writeOp(Op::Else)) return false;
    if (!GetLocal(e, s) || !e.writeOp(Op::F64ReinterpretI64)) return false;
    return e.writeOp(Op::End);
  }
  // Int32 / Boolean: the payload is the low 32 bits of the NUNBOX box.
  return e.writeOp(Op::I32WrapI64);
}

// Compute a node's value, leaving exactly one wasm value on the stack. (For
// effectful/void nodes EmitEffect is used instead.) Returns false => unsupported.
static bool EmitValue(Encoder& e, WJBackend& be, MInstruction* ins) {
  js::wasm::gWJBailReason = WJOpName(ins->op());  // attribute an internal bail to this op
  switch (ins->op()) {
    case MDefinition::Opcode::Parameter: {
      MParameter* p = ins->toParameter();
      // Register ABI: this = wasm param 1; arg i = wasm param 2+i.
      if (p->index() == MParameter::THIS_SLOT) {
        return e.writeOp(Op::LocalGet) && e.writeVarU32(1);
      }
      if (uint32_t(p->index()) >= kWJMaxArgs) {
        js::wasm::gWJBailReason = "too-many-args";
        return false;  // too many args: bail
      }
      return e.writeOp(Op::LocalGet) && e.writeVarU32(2 + uint32_t(p->index()));
    }
    case MDefinition::Opcode::Constant: {
      MConstant* c = ins->toConstant();
      switch (c->type()) {
        case MIRType::Int32:
          return e.writeOp(Op::I32Const) && e.writeVarS32(c->toInt32());
        case MIRType::Boolean:
          return e.writeOp(Op::I32Const) && e.writeVarS32(c->toBoolean() ? 1 : 0);
        case MIRType::Double:
          return e.writeOp(Op::F64Const) && e.writeFixedF64(c->toDouble());
        case MIRType::Null:
          return e.writeOp(Op::I64Const) &&
                 e.writeVarS64(int64_t(JS::NullValue().asRawBits()));
        case MIRType::Undefined:
          return e.writeOp(Op::I64Const) &&
                 e.writeVarS64(int64_t(JS::UndefinedValue().asRawBits()));
        case MIRType::Object: {
          // Bake the object via the traced GC-constant pool, then load the live
          // pointer (low 32 bits of the boxed Value) at runtime.
          JSObject* obj = c->toObjectOrNull();
          if (!obj) return false;
          uintptr_t slot = WJInternConstant(JS::ObjectValue(*obj).asRawBits());
          if (!slot) return false;
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(slot)))
            return false;
          if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0))
            return false;
          return e.writeOp(Op::I32WrapI64);  // -> object pointer
        }
        case MIRType::String:
        case MIRType::Symbol:
        case MIRType::BigInt: {
          // GC-thing constant: bake the boxed Value via the traced pool, load
          // the live pointer (low 32 bits) at runtime.
          uintptr_t slot = WJInternConstant(c->toJSValue().asRawBits());
          if (!slot) return false;
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(slot)))
            return false;
          if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0))
            return false;
          return e.writeOp(Op::I32WrapI64);  // -> GC pointer
        }
        default:
          return false;
      }
    }
    case MDefinition::Opcode::Elements: {
      int32_t objLocal = be.local(ins->toElements()->object());
      if (objLocal < 0) return false;
      uint32_t off = uint32_t(js::NativeObject::offsetOfElements());
      if (!GetLocal(e, uint32_t(objLocal))) return false;
      return e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(off);
    }
    case MDefinition::Opcode::Box: {
      MDefinition* in = ins->toBox()->input();
      if (in->type() == MIRType::Value) return GetOp(e, be, in);  // already boxed
      // Magic values have no payload -- bake the boxed magic constant directly.
      JSWhyMagic why;
      bool isMagic = true;
      switch (in->type()) {
        case MIRType::MagicOptimizedOut: why = JS_OPTIMIZED_OUT; break;
        case MIRType::MagicHole: why = JS_ELEMENTS_HOLE; break;
        case MIRType::MagicIsConstructing: why = JS_IS_CONSTRUCTING; break;
        case MIRType::MagicUninitializedLexical:
          why = JS_UNINITIALIZED_LEXICAL;
          break;
        default: isMagic = false; break;
      }
      if (isMagic) {
        return e.writeOp(Op::I64Const) &&
               e.writeVarS64(int64_t(JS::MagicValue(why).asRawBits()));
      }
      // Payload-less values: bake the boxed constant.
      if (in->type() == MIRType::Undefined) {
        return e.writeOp(Op::I64Const) &&
               e.writeVarS64(int64_t(JS::UndefinedValue().asRawBits()));
      }
      if (in->type() == MIRType::Null) {
        return e.writeOp(Op::I64Const) &&
               e.writeVarS64(int64_t(JS::NullValue().asRawBits()));
      }
      if (!GetOp(e, be, in)) return false;
      return EmitBoxFromStack(e, in->type());
    }
    case MDefinition::Opcode::BoxNonStrictThis: {
      // `this` in a sloppy function. For an object receiver this is identity;
      // primitives go through ToObject (rare) -> deopt. Guard object, extract ptr.
      int32_t inLocal = be.local(ins->getOperand(0));
      if (inLocal < 0) return false;
      return EmitUnboxLocal(e, be, uint32_t(inLocal), MIRType::Object, /*fallible=*/true);
    }
    case MDefinition::Opcode::Unbox: {
      MUnbox* u = ins->toUnbox();
      int32_t inLocal = be.local(u->input());
      if (inLocal < 0) return false;
      return EmitUnboxLocal(e, be, uint32_t(inLocal), u->type(),
                            u->mode() == MUnbox::Fallible);
    }
    case MDefinition::Opcode::GuardShape: {
      // Passthrough the object; deopt if obj->shape() != the recorded Shape*.
      // The expected shape is loaded from a GC-traced pool slot (NOT baked as a
      // raw immediate): a compacting GC relocates the shape and updates the slot,
      // so the guard keeps comparing against the live pointer. Baking it raw made
      // a moved shape ALWAYS fail -> permanent deopt storm (crypto am3: ~475k).
      MGuardShape* g = ins->toGuardShape();
      // Megamorphic recompile: if every use is a derivable fixed-slot read, the
      // reads become self-guarding EmitPropIC (handled in LoadFixedSlot), so this
      // guard is a no-op passthrough -- NO deopt on a polymorphic receiver.
      if (be.forceMega) {
        bool conv = WJMegaConvertibleGuard(g);
        if (getenv("GECKO_WJ_MEGADBG")) {
          js::Shape* sh = g->shape();
          const char* badop = "none";
          if (conv) badop = "CONVERTED";
          else if (!sh || !sh->isShared()) badop = "not-shared-shape";
          else {
            for (jit::MUseIterator u = g->usesBegin(); u != g->usesEnd(); u++) {
              if (!u->consumer()->isDefinition()) continue;
              badop = WJOpName(u->consumer()->toDefinition()->op());
              jit::MDefinition::Opcode o = u->consumer()->toDefinition()->op();
              if (o == jit::MDefinition::Opcode::LoadFixedSlot ||
                  o == jit::MDefinition::Opcode::LoadFixedSlotAndUnbox)
                continue;
              break;
            }
          }
          JSScript* ms = be.info ? be.info->script() : nullptr;
          fprintf(stderr, "[wj-mega] %s:%u GuardShape conv=%d firstbad=%s\n",
                  ms && ms->filename() ? ms->filename() : "?",
                  ms ? unsigned(ms->lineno()) : 0, conv, badop);
        }
        if (conv) return GetOp(e, be, g->object());
        if (getenv("GECKO_WJ_MEGAFAIL")) {
          js::Shape* sh = g->shape();
          const char* badop = (!sh || !sh->isShared()) ? "not-shared" : "?";
          if (sh && sh->isShared()) {
            for (jit::MUseIterator u = g->usesBegin(); u != g->usesEnd(); u++) {
              if (!u->consumer()->isDefinition()) continue;
              jit::MDefinition::Opcode o = u->consumer()->toDefinition()->op();
              if (o == jit::MDefinition::Opcode::LoadFixedSlot ||
                  o == jit::MDefinition::Opcode::LoadFixedSlotAndUnbox)
                continue;
              badop = WJOpName(o);
              break;
            }
          }
          JSScript* ms = be.info ? be.info->script() : nullptr;
          fprintf(stderr, "[wj-megafail] %s:%u GuardShape NOTCONV firstbad=%s\n",
                  ms && ms->filename() ? ms->filename() : "?",
                  ms ? unsigned(ms->lineno()) : 0, badop);
        }
      }
      int32_t objLocal = be.local(g->object());
      if (objLocal < 0) return false;
      uint32_t shapeOff = uint32_t(offsetof(JS::shadow::Object, shape));
      // BAKESHAPE: compare obj->shape_ to the shape pointer baked as an i32
      // immediate (no GC-pool load). A compacting GC that relocates the shape
      // makes the guard fail -> deopt -> the recompile-on-storm valve rebakes the
      // new shape. Saves one memory load per guard on the hot path. Default off.
      static int bakeShape = getenv("GECKO_WJ_BAKESHAPE") ? 1 : 0;
      if (bakeShape) {
        if (!GetLocal(e, uint32_t(objLocal))) return false;  // obj
        if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
            !e.writeVarU32(shapeOff))
          return false;  // obj->shape_
        if (!e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(uint32_t(uintptr_t(g->shape())))))
          return false;  // baked expected shape
        if (!e.writeOp(Op::I32Ne)) return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
        if (!EmitDeopt(e, be)) return false;
        if (!e.writeOp(Op::End)) return false;
        return GetLocal(e, uint32_t(objLocal));
      }
      uintptr_t slot = WJInternShape(uintptr_t(g->shape()));
      if (!slot) return false;
      // Read the object via GetOp (rematerialize from the rooted source) so the
      // shape check + passthrough see the GC-CURRENT object, not a stale cached
      // pointer from before a GC-ing call (earley sc_reverse).
      if (!GetOp(e, be, g->object())) return false;  // obj (current)
      if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(shapeOff))
        return false;  // obj->shape_
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(slot)) ||
          !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0))
        return false;  // current (relocated) expected shape from the pool
      if (!e.writeOp(Op::I32Ne)) return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetOp(e, be, g->object());  // result = the (current) object
    }
    case MDefinition::Opcode::GuardShapeList: {
      // Polymorphic shape guard: deopt unless obj->shape() is one of the recorded
      // shapes (up to 4). Each shape goes through the GC-traced pool (relocation-
      // safe). Passthrough the object. Lets polymorphic property access (e.g.
      // splay `this.root_` Node-or-null, deltablue constraint subtypes) compile.
      MGuardShapeList* g = ins->toGuardShapeList();
      // Megamorphic recompile: a GuardShapeList DEOPTS on any shape outside its
      // captured set -> storms when a new subtype appears (deltablue 414). If its
      // uses are convertible reads, passthrough (no deopt) and let the reads
      // self-guard via multi-shape EmitPropIC (helper fallback handles ANY shape).
      if (be.forceMega) {
        bool conv = WJMegaConvertibleGuard(ins);
        if (getenv("GECKO_WJ_MEGADBG")) {
          js::Shape* sh = WJGuardRepShape(ins);
          const char* bad = conv ? "CONV" : (!sh || !sh->isShared()) ? "notshared" : "?";
          if (!conv && sh && sh->isShared()) {
            for (jit::MUseIterator u = g->usesBegin(); u != g->usesEnd(); u++) {
              if (!u->consumer()->isDefinition()) continue;
              jit::MDefinition* d = u->consumer()->toDefinition();
              jit::MDefinition::Opcode o = d->op();
              if (o == jit::MDefinition::Opcode::LoadFixedSlot ||
                  o == jit::MDefinition::Opcode::LoadFixedSlotAndUnbox) {
                uint64_t k;
                if (!WJDerivePropKey(sh, o == jit::MDefinition::Opcode::LoadFixedSlot
                    ? uint32_t(d->toLoadFixedSlot()->slot())
                    : uint32_t(d->toLoadFixedSlotAndUnbox()->slot()), &k)) {
                  bad = "nokey-read"; break;
                }
                continue;
              }
              bad = WJOpName(o); break;
            }
          }
          JSScript* ms = be.info ? be.info->script() : nullptr;
          fprintf(stderr, "[wj-mega] %s:%u GuardShapeList conv=%d bad=%s\n",
                  ms && ms->filename() ? ms->filename() : "?",
                  ms ? unsigned(ms->lineno()) : 0, conv, bad);
        }
        if (conv) return GetOp(e, be, g->object());
      }
      int32_t objLocal = be.local(g->object());
      if (objLocal < 0) return false;
      uint32_t shapeOff = uint32_t(offsetof(JS::shadow::Object, shape));
      const auto& shapes = g->shapeList()->shapes();
      bool any = false;
      for (size_t i = 0; i < 4; i++) {  // ShapeListSnapshot::NumShapes
        js::Shape* s = shapes[i];
        if (!s) continue;
        uintptr_t slot = WJInternShape(uintptr_t(s));
        if (!slot) return false;
        // push (obj->shape == shapes[i])
        if (!GetLocal(e, uint32_t(objLocal)) || !e.writeOp(Op::I32Load) ||
            !e.writeVarU32(2) || !e.writeVarU32(shapeOff) ||
            !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(slot)) ||
            !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
            !e.writeOp(Op::I32Eq))
          return false;
        if (any && !e.writeOp(Op::I32Or)) return false;
        any = true;
      }
      if (!any) return false;  // empty list shouldn't happen; bail to be safe
      // deopt unless matched: !matched -> deopt
      if (!e.writeOp(Op::I32Eqz)) return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetLocal(e, uint32_t(objLocal));
    }
    case MDefinition::Opcode::GuardShapeListToOffset: {
      // Polymorphic property access -> slot offset. deopt unless obj->shape() is
      // one of the recorded (shape, offset) pairs; result is the matched shape's
      // byte offset (Int32), consumed by LoadDynamicSlotFromOffset. Mirrors Ion's
      // visitGuardShapeListToOffset. Two passes: a guard at nesting==1 (OOL-deopt
      // safe), then a deopt-free select for the offset. Relocation-safe interned
      // shapes (GC-traced pool). (splay insert/remove `node.left/right`.)
      auto* g = ins->toGuardShapeListToOffset();
      // Megamorphic recompile: if every use is a LoadDynamicSlotFromOffset that
      // converts to EmitPropIC (which self-guards + helper-falls-back, ignoring
      // this offset), drop the DEOPT and emit a dummy offset. This kills the
      // GuardShapeListToOffset storm on an unlisted shape (deltablue 414).
      if (be.forceMega) {
        jit::MDefinition* o = nullptr;
        uint64_t k = 0;
        bool allConv = g->hasUses() && WJMegaGSLTO(ins, &o, &k) &&
                       o->type() == MIRType::Object;
        for (jit::MUseIterator u = g->usesBegin(); allConv && u != g->usesEnd();
             u++) {
          if (!u->consumer()->isDefinition()) continue;
          if (!u->consumer()->toDefinition()->isLoadDynamicSlotFromOffset())
            allConv = false;
        }
        if (allConv) return e.writeOp(Op::I32Const) && e.writeVarS32(0);
      }
      int32_t objLocal = be.local(g->object());
      if (objLocal < 0) return false;
      uint32_t shapeOff = uint32_t(offsetof(JS::shadow::Object, shape));
      const auto& shapes = g->shapeList()->shapes();
      const auto& offsets = g->shapeList()->offsets();
      uintptr_t shapeSlot[4];
      uint32_t offs[4];
      int count = 0;
      for (size_t i = 0; i < 4; i++) {  // ShapeListSnapshot::NumShapes
        js::Shape* s = shapes[i];
        if (!s) continue;
        uintptr_t slot = WJInternShape(uintptr_t(s));
        if (!slot) return false;
        shapeSlot[count] = slot;
        offs[count] = offsets[i];
        count++;
      }
      if (count == 0) return false;
      // obj->shape -> propShapeLocal (reused across guard + select).
      if (!GetLocal(e, uint32_t(objLocal)) || !e.writeOp(Op::I32Load) ||
          !e.writeVarU32(2) || !e.writeVarU32(shapeOff) ||
          !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.propShapeLocal))
        return false;
      auto pushShapeEq = [&](int k) -> bool {
        return GetLocal(e, be.propShapeLocal) && e.writeOp(Op::I32Const) &&
               e.writeVarS32(int32_t(shapeSlot[k])) && e.writeOp(Op::I32Load) &&
               e.writeVarU32(2) && e.writeVarU32(0) && e.writeOp(Op::I32Eq);
      };
      // guard: deopt unless OR(propShape == shape[k]).
      for (int k = 0; k < count; k++) {
        if (!pushShapeEq(k)) return false;
        if (k && !e.writeOp(Op::I32Or)) return false;
      }
      if (!e.writeOp(Op::I32Eqz) || !e.writeOp(Op::If) || !e.writeFixedU8(0x40))
        return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      // offset select (deopt-free; guard guarantees a match -> last is the else).
      for (int k = 0; k + 1 < count; k++) {
        if (!pushShapeEq(k)) return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32)))
          return false;
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(offs[k])))
          return false;
        if (!e.writeOp(Op::Else)) return false;
      }
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(offs[count - 1])))
        return false;
      for (int k = 0; k + 1 < count; k++) {
        if (!e.writeOp(Op::End)) return false;
      }
      return true;
    }
    case MDefinition::Opcode::GuardSpecificFunction: {
      // Passthrough operand 0 (the function); deopt if it != the expected
      // function (operand 1, an object pointer). Both are i32 object ptrs.
      MDefinition* obj = ins->getOperand(0);
      MDefinition* expected = ins->getOperand(1);
      int32_t ol = be.local(obj), xl = be.local(expected);
      if (ol < 0 || xl < 0) return false;
      if (getenv("GECKO_WJ_GSFPASS")) return GetLocal(e, uint32_t(ol));
      if (!GetLocal(e, uint32_t(ol)) || !GetLocal(e, uint32_t(xl)) ||
          !e.writeOp(Op::I32Ne)) {
        return false;
      }
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetLocal(e, uint32_t(ol));
    }
    case MDefinition::Opcode::GuardObjectIdentity: {
      // Passthrough operand 0; deopt if (obj==expected) when bailOnEquality, else
      // if (obj!=expected). Both are i32 object pointers.
      MGuardObjectIdentity* g = ins->toGuardObjectIdentity();
      int32_t ol = be.local(g->object()), xl = be.local(g->expected());
      if (ol < 0 || xl < 0) return false;
      if (!GetLocal(e, uint32_t(ol)) || !GetLocal(e, uint32_t(xl))) return false;
      if (!e.writeOp(g->bailOnEquality() ? Op::I32Eq : Op::I32Ne)) return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetLocal(e, uint32_t(ol));
    }
    case MDefinition::Opcode::ConstantProto: {
      // Wraps a constant prototype object; its value is just that constant.
      int32_t l = be.local(ins->getOperand(0));
      if (l < 0) return false;
      return GetLocal(e, uint32_t(l));
    }
    case MDefinition::Opcode::GuardToFunction: {
      // Passthrough the object operand; function-ness is established by the
      // following GuardFunctionScript / call-site guards.
      int32_t l = be.local(ins->getOperand(0));
      if (l < 0) return false;
      return GetLocal(e, uint32_t(l));
    }
    case MDefinition::Opcode::GuardNullOrUndefined: {
      // Deopt unless the value's tag is NULL or UNDEFINED; passthrough the value.
      // (splay `x == null` fast paths.) tag = high 32 bits of the NUNBOX box.
      int32_t l = be.local(ins->getOperand(0));
      if (l < 0) return false;
      auto pushTag = [&]() -> bool {
        return GetLocal(e, uint32_t(l)) && e.writeOp(Op::I64Const) &&
               e.writeVarS64(32) && e.writeOp(Op::I64ShrU) &&
               e.writeOp(Op::I32WrapI64);
      };
      if (!pushTag() || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_NULL))) || !e.writeOp(Op::I32Eq))
        return false;
      if (!pushTag() || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_UNDEFINED))) ||
          !e.writeOp(Op::I32Eq))
        return false;
      if (!e.writeOp(Op::I32Or) || !e.writeOp(Op::I32Eqz)) return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetLocal(e, uint32_t(l));  // passthrough the (null/undefined) value
    }
    case MDefinition::Opcode::GuardFunctionScript: {
      // Deopt unless the function's BaseScript matches the expected one. This is
      // the real guard behind polymorphic-call inlining (each inlined target has
      // a distinct script). Load func->(jitInfoOrScript) and compare.
      int32_t fl = be.local(ins->getOperand(0));
      if (fl < 0) return false;
      uintptr_t expected =
          uintptr_t(static_cast<void*>(ins->toGuardFunctionScript()->expected()));
      uint32_t off = JSFunction::offsetOfJitInfoOrScript();
      if (!GetLocal(e, uint32_t(fl))) return false;
      if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(off))
        return false;
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(expected)) ||
          !e.writeOp(Op::I32Ne)) {
        return false;
      }
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetLocal(e, uint32_t(fl));
    }
    case MDefinition::Opcode::Callee: {
      // The running function object. Use the FRAME's canonical function (per the
      // block's CompileInfo, so an inlined frame yields its inlinee's function).
      // Correct for non-closure functions (prototype methods, top-level fns) --
      // the common case; closures sharing a script would differ. Bake via the
      // GC-traced const pool and load the live pointer like an object constant.
      JSScript* cs = ins->block()->info().script();
      JSFunction* fun = cs ? cs->function() : nullptr;
      if (!fun) return false;
      uintptr_t slot = WJInternConstant(JS::ObjectValue(*fun).asRawBits());
      if (!slot) return false;
      return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(slot)) &&
             e.writeOp(Op::I64Load) && e.writeVarU32(3) && e.writeVarU32(0) &&
             e.writeOp(Op::I32WrapI64);
    }
    case MDefinition::Opcode::FunctionEnvironment: {
      // result = function->environment(). Mirrors Ion's LFunctionEnvironment,
      // which loads the env from the SPECIFIC function operand -- crucial for
      // INLINED frames, whose function differs from the compiled (outer) one.
      // For the outer running function the operand is MCallee; its runtime
      // closure env isn't a compile-time value, so use envLocal (snapshotted
      // from gWJCurrentEnv at entry = the caller-provided runtime env). For any
      // other function operand (an inlined closure, materialized as a constant
      // or value) load ->environment() directly, like Ion.
      // CORRECTNESS: the only sound env we have is envLocal (snapshotted from
      // gWJCurrentEnv at entry = the caller-provided runtime env), valid ONLY for
      // the outer running function (operand == MCallee, not inlined). For inlined
      // frames or any other operand the env value is unsound -- loading
      // fn->environment() off a non-Callee operand read a wild pointer (splay OOB)
      // and mis-resolved closure vars (navier set_bnd, deltablue c.execute). So
      // BAIL those to PBL. Proper fix: env-PARAM ABI (pass each frame's env).
      if (getenv("GECKO_WJ_NOFE")) return false;
      MDefinition* fn = ins->toFunctionEnvironment()->function();
      bool inlined = ins->block()->info().script() != be.info->script();
      if (getenv("GECKO_WJ_FEDBG")) {
        JSScript* s = be.info->script();
        fprintf(stderr, "[wb-fedbg] FE fnOp=%s inlined=%d %s:%u\n",
                WJOpName(fn->op()), inlined, s ? s->filename() : "?",
                s ? s->lineno() : 0);
      }
      // Outer running function: rematerialize from the persistent GC-rooted env
      // slot (GetOp handles the same; this value-node store is mostly dead since
      // GetOp intercepts every use, but kept correct). Inlined/non-Callee env:
      // a different frame's env we don't persistently root -> bail to PBL.
      if (fn->isCallee() && !inlined) return GetOp(e, be, ins);
      return false;  // inlined / non-Callee env: not sound yet -> PBL
    }
    case MDefinition::Opcode::NurseryObject: {
      // A reference to a specific snapshot-held object; pool it (traced) and load
      // the live pointer like an object constant.
      if (!be.snapshot) return false;
      uint32_t idx = ins->toNurseryObject()->nurseryObjectIndex();
      if (idx >= be.snapshot->nurseryObjects().length()) return false;
      JSObject* obj = be.snapshot->nurseryObjects()[idx];
      if (!obj) return false;
      uintptr_t slot = WJInternConstant(JS::ObjectValue(*obj).asRawBits());
      if (!slot) return false;
      return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(slot)) &&
             e.writeOp(Op::I64Load) && e.writeVarU32(3) && e.writeVarU32(0) &&
             e.writeOp(Op::I32WrapI64);
    }
    case MDefinition::Opcode::ArrayLength: {
      int32_t off = js::ObjectElements::offsetOfLength();  // negative (elem-relative)
      if (!GetOp(e, be, ins->getOperand(0)) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(off) || !e.writeOp(Op::I32Add)) {
        return false;
      }
      return e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(0);
    }
    case MDefinition::Opcode::InitializedLength: {
      int32_t off = js::ObjectElements::offsetOfInitializedLength();
      if (!GetOp(e, be, ins->getOperand(0)) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(off) || !e.writeOp(Op::I32Add)) {
        return false;
      }
      return e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(0);
    }
    case MDefinition::Opcode::BoundsCheck: {
      MBoundsCheck* bc = ins->toBoundsCheck();
      int32_t il = be.local(bc->index()), ll = be.local(bc->length());
      if (il < 0 || ll < 0) return false;
      int32_t mn = bc->minimum(), mx = bc->maximum();
      if (mn == 0 && mx == 0) {
        // Common case: deopt unless (uint32)index < (uint32)length.
        if (!GetLocal(e, uint32_t(il)) || !GetLocal(e, uint32_t(ll)) ||
            !e.writeOp(Op::I32GeU))
          return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
        if (!EmitDeopt(e, be)) return false;
        if (!e.writeOp(Op::End)) return false;
        return GetLocal(e, uint32_t(il));
      }
      // Hoisted range check: deopt unless (index+min) >= 0 && (index+max) < length
      // (signed; length is non-negative). i.e. deopt if (index+min < 0) ||
      // (index+max >= length).
      if (!GetLocal(e, uint32_t(il)) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(mn) || !e.writeOp(Op::I32Add) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(0) || !e.writeOp(Op::I32LtS))
        return false;  // (index+min) < 0
      if (!GetLocal(e, uint32_t(il)) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(mx) || !e.writeOp(Op::I32Add) ||
          !GetLocal(e, uint32_t(ll)) || !e.writeOp(Op::I32GeS) ||
          !e.writeOp(Op::I32Or))
        return false;  // OR (index+max) >= length
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetLocal(e, uint32_t(il));  // passthrough index
    }
    case MDefinition::Opcode::LoadElement:
    case MDefinition::Opcode::LoadElementAndUnbox: {
      bool andUnbox = ins->op() == MDefinition::Opcode::LoadElementAndUnbox;
      MDefinition* elemsD = ins->getOperand(0);
      MDefinition* idxD = ins->getOperand(1);
      bool holeCheck = andUnbox ? false : ins->toLoadElement()->needsHoleCheck();
      int32_t il = be.local(idxD);
      if (il < 0) return false;
      // addr = elements + index * sizeof(Value); load the boxed element.
      if (!GetOp(e, be, elemsD) || !GetLocal(e, uint32_t(il)) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(sizeof(JS::Value))) ||
          !e.writeOp(Op::I32Mul) || !e.writeOp(Op::I32Add)) {
        return false;
      }
      if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0))
        return false;
      MIRType ty = ins->type();
      if (!holeCheck && ty == MIRType::Value) return true;  // boxed Value on stack
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.unboxScratch)) return false;
      if (holeCheck) {
        // Deopt if the element is a hole (MagicValue(JS_ELEMENTS_HOLE)).
        if (!GetLocal(e, be.unboxScratch) || !e.writeOp(Op::I64Const) ||
            !e.writeVarS64(int64_t(JS::MagicValue(JS_ELEMENTS_HOLE).asRawBits())) ||
            !e.writeOp(Op::I64Eq)) {
          return false;
        }
        if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
        if (!EmitDeopt(e, be)) return false;
        if (!e.writeOp(Op::End)) return false;
      }
      if (ty == MIRType::Value) return GetLocal(e, be.unboxScratch);
      return EmitUnboxLocal(e, be, be.unboxScratch, ty, /*fallible=*/true);
    }
    case MDefinition::Opcode::Slots: {
      int32_t objLocal = be.local(ins->toSlots()->object());
      if (objLocal < 0) return false;
      uint32_t slotsOff = uint32_t(js::NativeObject::offsetOfSlots());
      if (!GetLocal(e, uint32_t(objLocal))) return false;
      return e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(slotsOff);
    }
    case MDefinition::Opcode::LoadFixedSlot: {
      MLoadFixedSlot* l = ins->toLoadFixedSlot();
      // Megamorphic recompile: this read is guarded by a mega-convertible
      // GuardShape (now a passthrough). Re-derive the property key and emit a
      // multi-shape EmitPropIC (shape-cache + helper fallback, NO deopt) on the
      // real receiver -- so a polymorphic receiver (deltablue Equality vs Scale)
      // never deopts. The PropIC self-guards, so losing the GuardShape is sound.
      if (be.forceMega && WJIsShapeGuard(l->object()) &&
          WJMegaConvertibleGuard(l->object())) {
        // Guard removed -> read MUST convert by-name or BAIL (a fixed-slot read
        // under a removed guard is a correctness bug for a polymorphic receiver).
        jit::MDefinition* gs = l->object();
        uint64_t keyBits = 0;
        if (WJDerivePropKey(WJGuardRepShape(gs), uint32_t(l->slot()), &keyBits)) {
          uint32_t site = WJPropSite(be);
          if (site != 0 && WJGuardObject(gs)->type() == MIRType::Object) {
            if (!EmitPropIC(e, be, ins, WJGuardObject(gs), keyBits, site,
                            WJGuardRepShape(gs),
                            uint32_t(js::NativeObject::getFixedSlotOffset(l->slot())),
                            /*bakedFixed=*/true))
              return false;
            return EmitHelperResultAsType(e, be, l->type());
          }
        }
        return WJBAIL("mega read couldn't convert (guard removed)\n");
      }
      uint32_t off = uint32_t(js::NativeObject::getFixedSlotOffset(l->slot()));
      // Read the object via GetOp (rematerializes Unbox/GuardShape from the rooted
      // source) -- a cached object local goes stale across a GC-ing call.
      if (!GetOp(e, be, l->object())) return false;
      if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(off))
        return false;
      // Usually the result is a boxed Value (i64 local). But Warp can TYPE a
      // LoadFixedSlot (e.g. an Object base feeding a nested LoadFixedSlotAndUnbox,
      // as in navier's Field accessors): then the node's local is i32/f64, so the
      // raw i64 box must be converted (infallible -- Warp proved the type). Without
      // this the host rejects the module (local.set i32 <- i64.load).
      if (l->type() == MIRType::Value) return true;
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.unboxScratch)) return false;
      return EmitUnboxLocal(e, be, be.unboxScratch, l->type(), /*fallible=*/false);
    }
    case MDefinition::Opcode::LoadFixedSlotAndUnbox: {
      MLoadFixedSlotAndUnbox* l = ins->toLoadFixedSlotAndUnbox();
      // Megamorphic recompile (see LoadFixedSlot): self-guarding multi-shape load.
      if (be.forceMega && WJIsShapeGuard(l->object()) &&
          WJMegaConvertibleGuard(l->object())) {
        // Guard removed -> read MUST convert by-name or BAIL (a fixed-slot read
        // under a removed guard is a correctness bug for a polymorphic receiver).
        jit::MDefinition* gs = l->object();
        uint64_t keyBits = 0;
        if (WJDerivePropKey(WJGuardRepShape(gs), uint32_t(l->slot()), &keyBits)) {
          uint32_t site = WJPropSite(be);
          if (site != 0 && WJGuardObject(gs)->type() == MIRType::Object) {
            if (!EmitPropIC(e, be, ins, WJGuardObject(gs), keyBits, site,
                            WJGuardRepShape(gs),
                            uint32_t(js::NativeObject::getFixedSlotOffset(l->slot())),
                            /*bakedFixed=*/true))
              return false;
            return EmitHelperResultAsType(e, be, l->type());
          }
        }
        return WJBAIL("mega read couldn't convert (guard removed)\n");
      }
      uint32_t off = uint32_t(js::NativeObject::getFixedSlotOffset(l->slot()));
      if (!GetOp(e, be, l->object())) return false;  // rematerialized object
      if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(off))
        return false;
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.unboxScratch))
        return false;
      return EmitUnboxLocal(e, be, be.unboxScratch, l->type(),
                            l->mode() == MUnbox::Fallible);
    }
    case MDefinition::Opcode::LoadDynamicSlotAndUnbox: {
      MLoadDynamicSlotAndUnbox* l = ins->toLoadDynamicSlotAndUnbox();
      if (be.forceMega) {
        if (jit::MDefinition* gs = WJMegaSlotsGuard(l->slots())) {
          // The shape guard WILL be passthrough'd (removed -- WJMegaConvertibleGuard
          // matched). So this read MUST become a self-guarding by-name EmitPropIC;
          // a bare fixed-slot load would read the WRONG slot for a polymorphic
          // receiver that no longer hits the (removed) guard. If we can't convert
          // (no derivable key / site pool exhausted), BAIL the whole function --
          // emitting the fixed-slot load here is a CORRECTNESS bug (deltablue 741
          // "Cycle": late valve-recompile exhausted the prop-site pool, guard
          // removed but read stayed fixed-slot).
          uint32_t nf = WJGuardRepShape(gs)->asShared().numFixedSlots();
          uint64_t keyBits = 0;
          if (WJDerivePropKey(WJGuardRepShape(gs), nf + uint32_t(l->slot()), &keyBits) &&
              WJGuardObject(gs)->type() == MIRType::Object) {
            uint32_t site = WJPropSite(be);
            if (site != 0) {
              if (!EmitPropIC(e, be, ins, WJGuardObject(gs), keyBits, site,
                              WJGuardRepShape(gs),
                              uint32_t(l->slot() * sizeof(JS::Value)),
                              /*bakedFixed=*/false))
                return false;
              return EmitHelperResultAsType(e, be, l->type());
            }
          }
          return WJBAIL("mega read couldn't convert (guard removed)\n");
        }
      }
      uint32_t off = uint32_t(l->slot() * sizeof(JS::Value));
      if (!GetOp(e, be, l->slots())) return false;
      if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(off))
        return false;
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.unboxScratch))
        return false;
      return EmitUnboxLocal(e, be, be.unboxScratch, l->type(),
                            l->mode() == MUnbox::Fallible);
    }
    case MDefinition::Opcode::LoadDynamicSlot: {
      MLoadDynamicSlot* l = ins->toLoadDynamicSlot();
      if (be.forceMega) {
        if (jit::MDefinition* gs = WJMegaSlotsGuard(l->slots())) {
          // Guard removed -> read MUST convert by-name or BAIL (see
          // LoadDynamicSlotAndUnbox: a fixed-slot read under a removed guard is a
          // correctness bug on a polymorphic receiver).
          uint32_t nf = WJGuardRepShape(gs)->asShared().numFixedSlots();
          uint64_t keyBits = 0;
          if (WJDerivePropKey(WJGuardRepShape(gs), nf + uint32_t(l->slot()), &keyBits) &&
              WJGuardObject(gs)->type() == MIRType::Object) {
            uint32_t site = WJPropSite(be);
            if (site != 0) {
              if (!EmitPropIC(e, be, ins, WJGuardObject(gs), keyBits, site,
                              WJGuardRepShape(gs),
                              uint32_t(l->slot() * sizeof(JS::Value)),
                              /*bakedFixed=*/false))
                return false;
              return EmitHelperResultAsType(e, be, l->type());
            }
          }
          return WJBAIL("mega read couldn't convert (guard removed)\n");
        }
      }
      uint32_t off = uint32_t(l->slot() * sizeof(JS::Value));
      if (!GetOp(e, be, l->slots())) return false;
      if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(off))
        return false;
      // As LoadFixedSlot: convert when Warp typed the slot (else i64 box stays).
      if (l->type() == MIRType::Value) return true;
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.unboxScratch)) return false;
      return EmitUnboxLocal(e, be, be.unboxScratch, l->type(), /*fallible=*/false);
    }
    case MDefinition::Opcode::LoadDynamicSlotFromOffset: {
      // result = loadValue(slots + offset). offset is a runtime byte offset
      // (Ion: BaseIndex(slots, offset, TimesOne)). Result is a boxed Value.
      MLoadDynamicSlotFromOffset* l = ins->toLoadDynamicSlotFromOffset();
      // Megamorphic recompile: the offset comes from a GuardShapeListToOffset that
      // DEOPTS on an unlisted shape (deltablue 414 storm). Convert to a no-deopt
      // multi-shape EmitPropIC on the real receiver (helper fallback for any shape).
      if (be.forceMega) {
        jit::MDefinition* obj = nullptr;
        uint64_t keyBits = 0;
        if (WJMegaGSLTO(l->offset(), &obj, &keyBits) &&
            obj->type() == MIRType::Object) {
          uint32_t site = WJPropSite(be);
          if (site != 0) {
            if (!EmitPropIC(e, be, ins, obj, keyBits, site)) return false;
            return EmitHelperResultAsType(e, be, l->type());
          }
          // The GuardShapeListToOffset may have been converted to a DUMMY offset (0)
          // -- then GetOp(offset) reads slots+0 (the WRONG slot). Can't convert (no
          // site) -> BAIL rather than read the wrong slot. Conservative (also bails
          // the rare case the GSLTO kept its real offset), but always correct.
          return WJBAIL("mega GSLTO read couldn't convert\n");
        }
      }
      if (!GetOp(e, be, l->slots())) return false;       // i32 slots ptr
      if (!GetOp(e, be, l->offset())) return false;       // i32 byte offset
      if (!e.writeOp(Op::I32Add)) return false;
      return e.writeOp(Op::I64Load) && e.writeVarU32(3) && e.writeVarU32(0);
    }
    case MDefinition::Opcode::LoadFixedSlotFromOffset: {
      // result = loadValue(object + offset). Fixed slots are inline in the
      // object, so the base is the object pointer (not a separate slots buffer).
      MLoadFixedSlotFromOffset* l = ins->toLoadFixedSlotFromOffset();
      if (!GetOp(e, be, l->object())) return false;       // i32 object ptr
      if (!GetOp(e, be, l->offset())) return false;        // i32 byte offset
      if (!e.writeOp(Op::I32Add)) return false;
      if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0))
        return false;
      if (l->type() == MIRType::Value) return true;
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.unboxScratch)) return false;
      return EmitUnboxLocal(e, be, be.unboxScratch, l->type(), /*fallible=*/false);
    }
    case MDefinition::Opcode::StrictConstantCompareInt32: {
      // value === k (or !==): a strict int32 compare on a boxed Value. The value
      // may be stored int32-boxed OR double-boxed, so compare the raw i64 bits
      // against both representations of k (and -0.0 when k==0), mirroring Ion's
      // visitStrictConstantCompareInt32. Result is an i32 boolean.
      static int noScc = getenv("GECKO_WJ_NOSCC") ? 1 : 0;
      if (noScc) return false;
      if (getenv("GECKO_WJ_NOSCC")) return false;  // diagnostic gate (bisect only)
      MStrictConstantCompareInt32* c = ins->toStrictConstantCompareInt32();
      bool eq = c->jsop() == JSOp::StrictEq;
      int32_t k = c->constant();
      uint64_t intBits = JS::Int32Value(k).asRawBits();
      uint64_t dblBits = JS::DoubleValue(double(k)).asRawBits();
      auto cmp = [&](uint64_t bits) -> bool {
        return GetLocal(e, be.unboxScratch) && e.writeOp(Op::I64Const) &&
               e.writeVarS64(int64_t(bits)) &&
               e.writeOp(eq ? Op::I64Eq : Op::I64Ne);
      };
      if (!GetOp(e, be, c->value())) return false;  // boxed i64 value
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.unboxScratch)) return false;
      if (!cmp(intBits)) return false;
      if (!cmp(dblBits)) return false;
      if (!e.writeOp(eq ? Op::I32Or : Op::I32And)) return false;
      if (k == 0) {
        if (!cmp(JS::DoubleValue(-0.0).asRawBits())) return false;
        if (!e.writeOp(eq ? Op::I32Or : Op::I32And)) return false;
      }
      return true;
    }
    case MDefinition::Opcode::ToDouble:
    case MDefinition::Opcode::ToFloat32: {
      MDefinition* in = ins->getOperand(0);
      if (!GetOp(e, be, in)) return false;
      if (in->type() == MIRType::Int32) return e.writeOp(Op::F64ConvertI32S);
      if (in->type() == MIRType::Double) return true;  // no-op
      return false;
    }
    case MDefinition::Opcode::Add:
    case MDefinition::Opcode::Sub:
    case MDefinition::Opcode::Mul: {
      bool d = ins->type() == MIRType::Double;
      bool i = ins->type() == MIRType::Int32;
      if (d) {
        if (!GetOp(e, be, ins->getOperand(0)) ||
            !GetOp(e, be, ins->getOperand(1)))
          return false;
        return e.writeOp(ins->isAdd()   ? Op::F64Add
                         : ins->isSub() ? Op::F64Sub
                                        : Op::F64Mul);
      }
      if (!i) return false;
      // int32 +/-/* overflow SHOULD bail to a double (Ion does). The i64-overflow-
      // check version is correct (storerepro a[i]=a[i-1]*3+1 -> JIT==PBL) and the
      // lastRp resume fix made the bulk of it sound -- but there's a RESIDUAL
      // mid-statement resume issue (randtest Math.random: 1000693 -> 1000169 with
      // lastRp, still != 1000024) where a deopt WITHIN a statement (after partial
      // computation) doesn't perfectly reconstruct. So still reverted to wrapping
      // (correct for the masked/truncated common case; latently wrong only for a
      // non-truncated int32 op that overflows, not hit by current benches). RE-LAND
      // once the residual depth>0 mid-statement expr-stack resume is fixed.
      if (!GetOp(e, be, ins->getOperand(0)) ||
          !GetOp(e, be, ins->getOperand(1)))
        return false;
      return e.writeOp(ins->isAdd()   ? Op::I32Add
                       : ins->isSub() ? Op::I32Sub
                                      : Op::I32Mul);
    }
    case MDefinition::Opcode::Div: {
      if (ins->type() == MIRType::Double) {
        if (!GetOp(e, be, ins->getOperand(0)) ||
            !GetOp(e, be, ins->getOperand(1))) {
          return false;
        }
        return e.writeOp(Op::F64Div);
      }
      if (ins->type() == MIRType::Int32) {
        // Integer division. wasm i32.div_s traps on /0 and INT32_MIN/-1, and a
        // non-exact or negative-zero result must be a double -- deopt in all
        // those cases (matching Ion's MDiv bailouts) so we never produce a wrong
        // int32 or trap.
        MDiv* d = ins->toDiv();
        int32_t ll = be.local(d->lhs()), rl = be.local(d->rhs());
        if (ll < 0 || rl < 0) return false;
        auto gl = [&](int32_t l) { return GetLocal(e, uint32_t(l)); };
        auto deoptIf = [&]() -> bool {
          return e.writeOp(Op::If) && e.writeFixedU8(0x40) && EmitDeopt(e, be) &&
                 e.writeOp(Op::End);
        };
        // rhs == 0 -> deopt
        if (!gl(rl) || !e.writeOp(Op::I32Eqz) || !deoptIf()) return false;
        // lhs == INT32_MIN && rhs == -1 -> deopt (overflow)
        if (!gl(ll) || !e.writeOp(Op::I32Const) || !e.writeVarS32(INT32_MIN) ||
            !e.writeOp(Op::I32Eq) || !gl(rl) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(-1) || !e.writeOp(Op::I32Eq) || !e.writeOp(Op::I32And) ||
            !deoptIf())
          return false;
        // negative zero: lhs == 0 && rhs < 0 -> result is -0.0, must be double
        if (d->canBeNegativeZero()) {
          if (!gl(ll) || !e.writeOp(Op::I32Eqz) || !gl(rl) ||
              !e.writeOp(Op::I32Const) || !e.writeVarS32(0) ||
              !e.writeOp(Op::I32LtS) || !e.writeOp(Op::I32And) || !deoptIf())
            return false;
        }
        // non-exact (true quotient not integral) -> must be double
        if (!d->isTruncated()) {
          if (!gl(ll) || !gl(rl) || !e.writeOp(Op::I32RemS) ||
              !e.writeOp(Op::I32Const) || !e.writeVarS32(0) ||
              !e.writeOp(Op::I32Ne) || !deoptIf())
            return false;
        }
        return gl(ll) && gl(rl) && e.writeOp(Op::I32DivS);
      }
      return false;
    }
    case MDefinition::Opcode::Mod: {
      if (ins->type() == MIRType::Double) {
        // JS `a % b` on doubles == C fmod (result has sign of `a`, |result|<|b|).
        // wasm has no f64.rem, but fmod(a,b) == a - b*trunc(a/b) exactly (handles
        // inf/0/NaN edge cases identically: trunc(NaN)=NaN propagates). Compute
        // a/b once into a temp would need a local; recompute is fine (GetOp is
        // cheap for locals/consts).
        MDefinition* a = ins->getOperand(0);
        MDefinition* b = ins->getOperand(1);
        if (!GetOp(e, be, a)) return false;                 // a
        if (!GetOp(e, be, b)) return false;                 // a b
        if (!GetOp(e, be, a) || !GetOp(e, be, b)) return false;  // a b a b
        if (!e.writeOp(Op::F64Div)) return false;           // a b (a/b)
        if (!e.writeOp(Op::F64Trunc)) return false;         // a b trunc(a/b)
        if (!e.writeOp(Op::F64Mul)) return false;           // a (b*trunc)
        return e.writeOp(Op::F64Sub);                       // a - b*trunc
      }
      if (ins->type() == MIRType::Int32) {
        MMod* m = ins->toMod();
        int32_t ll = be.local(m->lhs()), rl = be.local(m->rhs());
        if (ll < 0 || rl < 0) return false;
        auto gl = [&](int32_t l) { return GetLocal(e, uint32_t(l)); };
        auto deoptIf = [&]() -> bool {
          return e.writeOp(Op::If) && e.writeFixedU8(0x40) && EmitDeopt(e, be) &&
                 e.writeOp(Op::End);
        };
        // rhs == 0 -> NaN result (double) -> deopt.
        if (!gl(rl) || !e.writeOp(Op::I32Eqz) || !deoptIf()) return false;
        // Negative-zero: a negative dividend with a zero remainder yields -0.0,
        // which must be a double -> deopt (unless the result is truncated to int).
        // i32.rem_s(INT32_MIN,-1)==0 in wasm (no trap), and that case has lhs<0 so
        // it's covered here too.
        if (!m->isTruncated() && m->canBeNegativeDividend()) {
          if (!gl(ll) || !e.writeOp(Op::I32Const) || !e.writeVarS32(0) ||
              !e.writeOp(Op::I32LtS) ||                       // lhs < 0
              !gl(ll) || !gl(rl) || !e.writeOp(Op::I32RemS) ||
              !e.writeOp(Op::I32Eqz) ||                       // rem == 0
              !e.writeOp(Op::I32And) || !deoptIf())
            return false;
        }
        return gl(ll) && gl(rl) && e.writeOp(Op::I32RemS);
      }
      return false;
    }
    case MDefinition::Opcode::Sqrt: {
      if (ins->type() != MIRType::Double) return false;
      if (!GetOp(e, be, ins->getOperand(0))) return false;
      return e.writeOp(Op::F64Sqrt);
    }
    case MDefinition::Opcode::Floor:
    case MDefinition::Opcode::Ceil:
    case MDefinition::Opcode::Trunc: {
      // Math.floor/ceil/trunc(double) -> int32 (Warp-typed Int32). Apply the
      // f64 rounding, then deopt unless the (integral) result is exactly
      // representable as int32 -- mirrors Ion's MFloor/MCeil/MTrunc, which bail
      // on out-of-range/NaN. The deopt MUST sit at nesting==1 (its own `if`).
      MDefinition* in = ins->getOperand(0);
      if (ins->type() != MIRType::Int32) return false;
      if (in->type() != MIRType::Double && in->type() != MIRType::Float32) {
        if (in->type() == MIRType::Int32) return GetOp(e, be, in);  // already int
        return false;
      }
      Op round = ins->isFloor()  ? Op::F64Floor
                 : ins->isCeil() ? Op::F64Ceil
                                 : Op::F64Trunc;
      auto pushRounded = [&]() -> bool {
        return GetOp(e, be, in) && e.writeOp(round);
      };
      // deopt unless (rounded >= -2147483648.0 && rounded <= 2147483647.0).
      // NaN makes both comparisons false -> deopts (correct).
      if (!pushRounded() || !e.writeOp(Op::F64Const) ||
          !e.writeFixedF64(-2147483648.0) || !e.writeOp(Op::F64Ge))
        return false;
      if (!pushRounded() || !e.writeOp(Op::F64Const) ||
          !e.writeFixedF64(2147483647.0) || !e.writeOp(Op::F64Le) ||
          !e.writeOp(Op::I32And) || !e.writeOp(Op::I32Eqz))
        return false;  // !(inRange)
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      // In range: convert the integral rounded value to int32 (exact, no trap).
      return pushRounded() && e.writeOp(Op::I32TruncF64S);
    }
    case MDefinition::Opcode::MinMax: {
      MMinMax* m = ins->toMinMax();
      if (!GetOp(e, be, m->lhs()) || !GetOp(e, be, m->rhs())) return false;
      if (m->type() == MIRType::Double) {
        return e.writeOp(m->isMax() ? Op::F64Max : Op::F64Min);
      }
      if (m->type() == MIRType::Int32) {
        // i32 has no min/max: select on a signed compare. Stack: [a, b].
        if (!GetOp(e, be, m->lhs()) || !GetOp(e, be, m->rhs())) return false;
        if (!e.writeOp(m->isMax() ? Op::I32GtS : Op::I32LtS)) return false;
        return e.writeOp(Op::SelectNumeric);  // a (gt/lt) b ? a : b
      }
      return false;
    }
    case MDefinition::Opcode::Not: {
      MDefinition* in = ins->toNot()->input();
      MIRType t = in->type();
      if (t == MIRType::Boolean || t == MIRType::Int32) {
        return GetOp(e, be, in) && e.writeOp(Op::I32Eqz);
      }
      if (t == MIRType::Object) {  // non-null object is always truthy
        return e.writeOp(Op::I32Const) && e.writeVarS32(0);
      }
      if (t == MIRType::Null || t == MIRType::Undefined) {
        return e.writeOp(Op::I32Const) && e.writeVarS32(1);
      }
      if (t == MIRType::Double) {  // !d == (d == 0 || isNaN(d))
        if (!GetOp(e, be, in) || !e.writeOp(Op::F64Const) ||
            !e.writeFixedF64(0.0) || !e.writeOp(Op::F64Ne))
          return false;  // d != 0
        if (!GetOp(e, be, in) || !GetOp(e, be, in) || !e.writeOp(Op::F64Eq))
          return false;  // d == d (false for NaN)
        return e.writeOp(Op::I32And) && e.writeOp(Op::I32Eqz);
      }
      if (t == MIRType::Value) {
        int32_t l = be.local(in);
        if (l < 0) return false;
        return EmitValueTruthy(e, be, uint32_t(l)) && e.writeOp(Op::I32Eqz);
      }
      return false;
    }
    case MDefinition::Opcode::StringLength: {
      int32_t sl = be.local(ins->toStringLength()->string());
      if (sl < 0) return false;
      if (!GetLocal(e, uint32_t(sl))) return false;
      return e.writeOp(Op::I32Load) && e.writeVarU32(2) &&
             e.writeVarU32(uint32_t(JSString::offsetOfLength()));
    }
    case MDefinition::Opcode::GetFrameArgument: {
      // Register ABI: actual arg i is wasm param 2+i (boxed Value). Only static,
      // in-range indices map; anything else bails.
      MDefinition* idx = ins->toGetFrameArgument()->index();
      if (!idx->isConstant() || idx->toConstant()->type() != MIRType::Int32)
        return false;
      int32_t i = idx->toConstant()->toInt32();
      if (i < 0 || uint32_t(i) >= kWJMaxArgs) return false;
      return e.writeOp(Op::LocalGet) && e.writeVarU32(2 + uint32_t(i));
    }
    case MDefinition::Opcode::StrictConstantCompareBoolean: {
      // `value (===|!==) <constant bool>`: a strict bit compare of the boxed
      // value against the canonical BooleanValue.
      MStrictConstantCompareBoolean* cc =
          ins->toStrictConstantCompareBoolean();
      if (!GetOp(e, be, cc->getOperand(0))) return false;  // boxed value (i64)
      uint64_t bits = JS::BooleanValue(cc->constant()).asRawBits();
      if (!e.writeOp(Op::I64Const) || !e.writeVarS64(int64_t(bits))) return false;
      JSOp jsop = cc->jsop();
      bool ne = (jsop == JSOp::Ne || jsop == JSOp::StrictNe);
      return e.writeOp(ne ? Op::I64Ne : Op::I64Eq);
    }
    case MDefinition::Opcode::BooleanToInt32: {
      // A Boolean is already an i32 0/1 in our unboxed representation, so the
      // Int32 result is the same value -- pass it through. (Absent, this forced
      // hot driver loops with a `cond|0`/`+true` idiom entirely into PBL.)
      MDefinition* in = ins->getOperand(0);
      if (in->type() != MIRType::Boolean) return false;
      return GetOp(e, be, in);
    }
    case MDefinition::Opcode::TruncateToInt32: {
      // JS ToInt32 (`x|0`, bitwise, asm.js): trunc toward zero, then modulo 2^32
      // as a signed int32 (WRAPPING, not clamping). Int32 passes through. For a
      // Double, i64.trunc_sat_f64_s truncates into i64 (saturating only past
      // +-2^63) and i32.wrap_i64 takes the low 32 bits = the modular ToInt32. NaN
      // -> 0. Mirrors Ion's MTruncateToInt32. Does NOT deopt (truncation IS the
      // defined semantics here; deopting was UNSOUND -- re-ran committed stores).
      MDefinition* in = ins->getOperand(0);
      if (in->type() == MIRType::Int32) return GetOp(e, be, in);
      if (in->type() != MIRType::Double) return false;
      return GetOp(e, be, in) && e.writeOp(MiscOp::I64TruncSatF64S) &&
             e.writeOp(Op::I32WrapI64);
    }
    case MDefinition::Opcode::ToNumberInt32: {
      // ToNumberInt32 is the SPECULATIVE int conversion Warp inserts when an op
      // is typed Int32 but an operand is a Double (e.g. `x[i] += dt*s[i]` typed
      // Int32 from early all-zero iterations). Unlike TruncateToInt32, it must
      // DEOPT when the double is NOT an exact int32 -- truncating would silently
      // corrupt (dt*s[i]=0.0001 -> 0, navier addFields checksum). Mirrors Ion's
      // convertDoubleToInt32 (bail on non-representable). The deopt then trips the
      // recompile valve, which re-specializes the op as Double. Sound now (lastRp
      // gives the nearest resume point; the only store here is after this op).
      MDefinition* in = ins->getOperand(0);
      if (in->type() == MIRType::Int32) return GetOp(e, be, in);
      if (in->type() != MIRType::Double) return false;
      // deopt unless f64.convert_i32_s(i32.trunc_sat_f64_s(d)) == d  (exact int32).
      if (!GetOp(e, be, in) || !e.writeOp(MiscOp::I32TruncSatF64S) ||
          !e.writeOp(Op::F64ConvertI32S) || !GetOp(e, be, in) ||
          !e.writeOp(Op::F64Ne))
        return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetOp(e, be, in) && e.writeOp(MiscOp::I32TruncSatF64S);
    }
    case MDefinition::Opcode::GetPropertyCache: {
      MGetPropertyCache* g = ins->toGetPropertyCache();
      if (!EmitStageScratch(e, be, g->value(), 0)) return false;
      if (!EmitStageScratch(e, be, g->idval(), 1)) return false;
      return EmitHelperCallResult(e, be, ins, WJH_GETPROP, 0);
    }
    case MDefinition::Opcode::GetNameCache: {
      // Global/lexical name lookup. envObj = operand 0; the name is the GETNAME/
      // GETGNAME bytecode's atom operand at the node's resume-point pc (baked as a
      // StringValue, like Ion's IonGetNameIC reading the pc). Helper does the env
      // walk (GetEnvironmentName). Result is a Value.
      if (getenv("GECKO_WJ_NOGETNAME")) return false;
      MResumePoint* rp = ins->resumePoint();
      JSScript* s = be.info ? be.info->script() : nullptr;
      if (!rp || !rp->pc() || !s) return false;
      js::PropertyName* name = s->getName(rp->pc());
      if (!name) return false;
      uint64_t nameBits = JS::StringValue(name).asRawBits();
      uint32_t site = WJAllocNameSite();
      if (site == 0) {  // out of IC sites: uncached helper (rare, still correct)
        if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
        if (!EmitStageConstBoxed(e, be, nameBits, 1)) return false;
        return EmitHelperCallResult(e, be, ins, WJH_GETNAME, 0);
      }
      return EmitNameIC(e, be, ins, ins->getOperand(0), nameBits, site);
    }
    case MDefinition::Opcode::ToPropertyKeyCache: {
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
      return EmitHelperCallResult(e, be, ins, WJH_TOPROPKEY, 0);  // Value
    }
    case MDefinition::Opcode::CharCodeAt: {
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // string
      if (!EmitStageScratch(e, be, ins->getOperand(1), 1)) return false;  // index
      if (!EmitHelperCallResult(e, be, ins, WJH_CHARCODEAT, 0)) return false;
      return EmitHelperResultAsType(e, be, ins->type());  // Int32
    }
    case MDefinition::Opcode::FromCharCode: {
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;  // code
      if (!EmitHelperCallResult(e, be, ins, WJH_FROMCHARCODE, 0)) return false;
      return EmitHelperResultAsType(e, be, ins->type());  // String
    }
    case MDefinition::Opcode::ToString: {
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_TOSTRING, 0)) return false;
      return EmitHelperResultAsType(e, be, ins->type());  // String
    }
    case MDefinition::Opcode::BinaryCache: {
      // The JSOp comes from the node's RESUME POINT pc (matching Ion's
      // visitBinaryValueCache -- NOT trackedSite, which points at a JumpTarget
      // here). Only the 12 arith ops reach a BinaryValueCache (comparisons use
      // MCompare).
      MResumePoint* rp = ins->resumePoint();
      if (!rp || !rp->pc()) return false;
      JSOp jsop = JSOp(*rp->pc());
      MIRType ty = ins->type();
      int32_t l0 = be.local(ins->getOperand(0)), l1 = be.local(ins->getOperand(1));

      // (A) SPECULATIVE INLINE f64 ARITH (default-on). A cold/unspecialized arith
      // IC yields an untyped MBinaryCache; the IC never warms in this PBL build
      // (CacheIR stubs can't attach -> Warp stays blind), so navier's hot double
      // loops were stuck on it. Instead of a C++ helper hop, unbox both operands
      // as NUMBERS (int32 -> convert, double -> reinterpret, else DEOPT to PBL --
      // correct for string concat / objects) and do the f64 op inline. This is
      // typed arith WITHOUT needing the IC to warm. +,-,*,/ map directly to f64
      // ops (JS number semantics); %/** and bitops fall through to (B).
      Op fop = Op::F64Add;
      bool f64op = true;
      switch (jsop) {
        case JSOp::Add: fop = Op::F64Add; break;
        case JSOp::Sub: fop = Op::F64Sub; break;
        case JSOp::Mul: fop = Op::F64Mul; break;
        case JSOp::Div: fop = Op::F64Div; break;
        default: f64op = false; break;
      }
      // ty==Double: operands are proven numbers -> inline f64 (the fallible unbox's
      // deopt never fires).
      if (f64op && l0 >= 0 && l1 >= 0 && ty == MIRType::Double &&
          !getenv("GECKO_WJ_NOSPECARITH")) {
        if (!EmitUnboxLocal(e, be, uint32_t(l0), MIRType::Double, /*fallible=*/true))
          return false;
        if (!EmitUnboxLocal(e, be, uint32_t(l1), MIRType::Double, /*fallible=*/true))
          return false;
        return e.writeOp(fop);  // f64 result matches the f64 local
      }
      // ty==Value Add/Sub/Mul/Div: the operands COULD be non-numbers (splay's string
      // concat `'..' + tag + '..'`). The old code unboxed them as Double and DEOPTED
      // on strings every time -> deopt-resume churn/corruption. Instead branch at
      // runtime with NO deopt: if BOTH operands are numbers, inline f64 (keeps
      // navier's Value-typed numeric arith fast); else call WJH_BINARYARITH
      // (AddValues/... -- correct for strings + mixed types). The result is a boxed
      // Value (i64) either way.
      if (f64op && l0 >= 0 && l1 >= 0 && ty == MIRType::Value &&
          !getenv("GECKO_WJ_NOSPECARITH")) {
        auto pushTag = [&](int32_t l) -> bool {
          return GetLocal(e, uint32_t(l)) && e.writeOp(Op::I64Const) &&
                 e.writeVarS64(32) && e.writeOp(Op::I64ShrU) &&
                 e.writeOp(Op::I32WrapI64);
        };
        auto pushIsNum = [&](int32_t l) -> bool {  // (tag u<= CLEAR) | (tag == INT32)
          return pushTag(l) && e.writeOp(Op::I32Const) &&
                 e.writeVarS32(int32_t(uint32_t(JSVAL_TAG_CLEAR))) &&
                 e.writeOp(Op::I32LeU) && pushTag(l) && e.writeOp(Op::I32Const) &&
                 e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_INT32))) &&
                 e.writeOp(Op::I32Eq) && e.writeOp(Op::I32Or);
        };
        auto pushF64 = [&](int32_t l) -> bool {  // int32 -> convert; else reinterpret
          if (!pushTag(l) || !e.writeOp(Op::I32Const) ||
              !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_INT32))) ||
              !e.writeOp(Op::I32Eq) || !e.writeOp(Op::If) || !e.writeFixedU8(0x7C))
            return false;
          if (!GetLocal(e, uint32_t(l)) || !e.writeOp(Op::I32WrapI64) ||
              !e.writeOp(Op::F64ConvertI32S) || !e.writeOp(Op::Else))
            return false;
          if (!GetLocal(e, uint32_t(l)) || !e.writeOp(Op::F64ReinterpretI64) ||
              !e.writeOp(Op::End))
            return false;
          return true;
        };
        if (!pushIsNum(l0) || !pushIsNum(l1) || !e.writeOp(Op::I32And)) return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I64)))
          return false;
        // both numbers: inline f64, box to a double Value.
        if (!pushF64(l0) || !pushF64(l1) || !e.writeOp(fop)) return false;
        if (!EmitBoxFromStack(e, MIRType::Double)) return false;
        if (!e.writeOp(Op::Else)) return false;
        // non-numbers: VM helper (no deopt).
        if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
        if (!EmitStageScratch(e, be, ins->getOperand(1), 1)) return false;
        if (!EmitHelperCallResult(e, be, ins, WJH_BINARYARITH, uint32_t(jsop)))
          return false;
        return e.writeOp(Op::End);
      }

      // (A2) INLINE i32 ARITH for an Int32-typed result. Bitwise ops directly;
      // Add/Sub/Mul in the i64 domain with an overflow deopt (a result that
      // doesn't fit int32 becomes a double, violating the Int32 type). Operands
      // unboxed as int32 (deopt if not). Mod/Pow/Ursh fall to (B).
      if (ty == MIRType::Int32 && l0 >= 0 && l1 >= 0 &&
          !getenv("GECKO_WJ_NOSPECARITH")) {
        Op bop = Op::I32Or;
        bool bit = true;
        switch (jsop) {
          case JSOp::BitOr: bop = Op::I32Or; break;
          case JSOp::BitAnd: bop = Op::I32And; break;
          case JSOp::BitXor: bop = Op::I32Xor; break;
          case JSOp::Lsh: bop = Op::I32Shl; break;
          case JSOp::Rsh: bop = Op::I32ShrS; break;
          default: bit = false; break;
        }
        if (bit) {
          if (!EmitUnboxLocal(e, be, uint32_t(l0), MIRType::Int32, true)) return false;
          if (!EmitUnboxLocal(e, be, uint32_t(l1), MIRType::Int32, true)) return false;
          return e.writeOp(bop);
        }
        if (jsop == JSOp::Add || jsop == JSOp::Sub || jsop == JSOp::Mul) {
          Op iop = jsop == JSOp::Add   ? Op::I64Add
                   : jsop == JSOp::Sub ? Op::I64Sub
                                       : Op::I64Mul;
          if (!EmitUnboxLocal(e, be, uint32_t(l0), MIRType::Int32, true)) return false;
          if (!e.writeOp(Op::I64ExtendI32S)) return false;
          if (!EmitUnboxLocal(e, be, uint32_t(l1), MIRType::Int32, true)) return false;
          if (!e.writeOp(Op::I64ExtendI32S)) return false;
          if (!e.writeOp(iop)) return false;  // exact i64 result
          if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.unboxScratch))
            return false;
          // deopt unless result == sign_extend(wrap32(result)) (i.e. fits int32).
          if (!GetLocal(e, be.unboxScratch) || !e.writeOp(Op::I32WrapI64) ||
              !e.writeOp(Op::I64ExtendI32S) || !GetLocal(e, be.unboxScratch) ||
              !e.writeOp(Op::I64Ne))
            return false;
          if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
          if (!EmitDeopt(e, be)) return false;
          if (!e.writeOp(Op::End)) return false;
          return GetLocal(e, be.unboxScratch) && e.writeOp(Op::I32WrapI64);
        }
      }

      // (A3) COMPARISON (Boolean-typed MBinaryCache): a cold/untyped ==,!=,<,<=,>,
      // >=,===,!== whose operands our inline Compare couldn't type. Route to the
      // WJH_COMPARE VM helper (correct for any operand types). It's a C++ hop
      // (~PBL speed for THIS op) but keeps the whole FUNCTION compiled in JIT rather
      // than bailing it to PBL (deltablue's constraint comparisons did the latter
      // -> whole functions ran in PBL -> slow). Default-on: a compile-coverage win.
      if (ins->type() == MIRType::Boolean) {
        switch (jsop) {
          case JSOp::Eq: case JSOp::Ne: case JSOp::StrictEq: case JSOp::StrictNe:
          case JSOp::Lt: case JSOp::Le: case JSOp::Gt: case JSOp::Ge:
            break;
          default: return false;
        }
        if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
        if (!EmitStageScratch(e, be, ins->getOperand(1), 1)) return false;
        if (!EmitHelperCallResult(e, be, ins, WJH_COMPARE, uint32_t(jsop)))
          return false;
        return EmitHelperResultAsType(e, be, MIRType::Boolean);
      }

      // (B) Generic VM-helper fallback (WJH_BINARYARITH = the *Values VM ops): %, **,
      // bitops, or non-number Add/Sub/Mul/Div not handled inline above. DEFAULT-ON
      // now (was ARITHCACHE-gated): a C++ hop but keeps the FUNCTION in JIT instead
      // of bailing to PBL (crypto bignum %/bitops, raytrace Math.pow). jsop from
      // rp->pc() is correct (the old crash was the trackedSite-pc bug).
      // GECKO_WJ_NOARITHHELPER reverts.
      if (getenv("GECKO_WJ_NOARITHHELPER")) return false;
      switch (jsop) {
        case JSOp::Add: case JSOp::Sub: case JSOp::Mul: case JSOp::Div:
        case JSOp::Mod: case JSOp::Pow: case JSOp::BitOr: case JSOp::BitAnd:
        case JSOp::BitXor: case JSOp::Lsh: case JSOp::Rsh: case JSOp::Ursh:
          break;
        default: return false;
      }
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
      if (!EmitStageScratch(e, be, ins->getOperand(1), 1)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_BINARYARITH, uint32_t(jsop)))
        return false;
      return EmitHelperResultAsType(e, be, ins->type());
    }
    case MDefinition::Opcode::UnaryCache: {
      MResumePoint* rp = ins->resumePoint();
      if (!rp || !rp->pc()) return false;
      JSOp jsop = JSOp(*rp->pc());
      MIRType ty = ins->type();
      int32_t l0 = be.local(ins->getOperand(0));

      // (A) SPECULATIVE INLINE f64 (default-on; see BinaryCache). Neg/Pos/ToNumeric
      // of a Value/Double: unbox the operand as a NUMBER (deopt on non-number) and
      // do the f64 op inline -- typed arith without the (never-warming) IC. Neg ->
      // f64.neg; Pos/ToNumeric on a number is just the unboxed value.
      if ((jsop == JSOp::Neg || jsop == JSOp::Pos || jsop == JSOp::ToNumeric) &&
          l0 >= 0 && (ty == MIRType::Value || ty == MIRType::Double) &&
          !getenv("GECKO_WJ_NOSPECARITH")) {
        if (!EmitUnboxLocal(e, be, uint32_t(l0), MIRType::Double, /*fallible=*/true))
          return false;
        if (jsop == JSOp::Neg && !e.writeOp(Op::F64Neg)) return false;
        if (ty == MIRType::Double) return true;
        return EmitBoxFromStack(e, MIRType::Double);
      }

      // Value-typed Inc/Dec (navier's `i++/j++` loop counters Warp didn't prove
      // Int32). Runtime branch, NO deopt: number -> inline f64 +/-1 (box as a double
      // Value -- semantically a number); non-number -> WJH_UNARYARITH helper. Keeps
      // the hot loop counter inline instead of a per-iteration C++ hop or a bail.
      if ((jsop == JSOp::Inc || jsop == JSOp::Dec) && l0 >= 0 &&
          ty == MIRType::Value && !getenv("GECKO_WJ_NOSPECARITH")) {
        auto pushTag = [&]() -> bool {
          return GetLocal(e, uint32_t(l0)) && e.writeOp(Op::I64Const) &&
                 e.writeVarS64(32) && e.writeOp(Op::I64ShrU) &&
                 e.writeOp(Op::I32WrapI64);
        };
        if (!pushTag() || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(uint32_t(JSVAL_TAG_CLEAR))) ||
            !e.writeOp(Op::I32LeU) || !pushTag() || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_INT32))) ||
            !e.writeOp(Op::I32Eq) || !e.writeOp(Op::I32Or))
          return false;  // isNumber
        if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I64)))
          return false;
        // number: int32 -> convert, else reinterpret; +/- 1.0; box double.
        if (!pushTag() || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_INT32))) ||
            !e.writeOp(Op::I32Eq) || !e.writeOp(Op::If) || !e.writeFixedU8(0x7C))
          return false;
        if (!GetLocal(e, uint32_t(l0)) || !e.writeOp(Op::I32WrapI64) ||
            !e.writeOp(Op::F64ConvertI32S) || !e.writeOp(Op::Else) ||
            !GetLocal(e, uint32_t(l0)) || !e.writeOp(Op::F64ReinterpretI64) ||
            !e.writeOp(Op::End))
          return false;
        if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(1.0) ||
            !e.writeOp(jsop == JSOp::Inc ? Op::F64Add : Op::F64Sub))
          return false;
        if (!EmitBoxFromStack(e, MIRType::Double)) return false;
        if (!e.writeOp(Op::Else)) return false;
        if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
        if (!EmitHelperCallResult(e, be, ins, WJH_UNARYARITH, uint32_t(jsop)))
          return false;
        return e.writeOp(Op::End);
      }

      // (A2) INLINE i32 for an Int32-typed result (cold loop counters i++/i--,
      // ~x). Operand unboxed as int32 (deopt if not). Inc/Dec overflow-checked;
      // Neg deopts on 0 (-> -0 double) and INT32_MIN (overflow); BitNot is exact.
      if (ty == MIRType::Int32 && l0 >= 0 && !getenv("GECKO_WJ_NOSPECARITH")) {
        if (jsop == JSOp::BitNot) {
          if (!EmitUnboxLocal(e, be, uint32_t(l0), MIRType::Int32, true)) return false;
          return e.writeOp(Op::I32Const) && e.writeVarS32(-1) &&
                 e.writeOp(Op::I32Xor);
        }
        if (jsop == JSOp::Inc || jsop == JSOp::Dec) {
          if (!EmitUnboxLocal(e, be, uint32_t(l0), MIRType::Int32, true)) return false;
          if (!e.writeOp(Op::I64ExtendI32S) || !e.writeOp(Op::I64Const) ||
              !e.writeVarS64(jsop == JSOp::Inc ? 1 : -1) || !e.writeOp(Op::I64Add))
            return false;
          if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.unboxScratch))
            return false;
          if (!GetLocal(e, be.unboxScratch) || !e.writeOp(Op::I32WrapI64) ||
              !e.writeOp(Op::I64ExtendI32S) || !GetLocal(e, be.unboxScratch) ||
              !e.writeOp(Op::I64Ne))
            return false;
          if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
          if (!EmitDeopt(e, be)) return false;
          if (!e.writeOp(Op::End)) return false;
          return GetLocal(e, be.unboxScratch) && e.writeOp(Op::I32WrapI64);
        }
        if (jsop == JSOp::Neg) {
          // deopt if operand == 0 (Neg(0) = -0, a double) or INT32_MIN (overflow).
          if (!EmitUnboxLocal(e, be, uint32_t(l0), MIRType::Int32, true)) return false;
          if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.unboxScratch))
            return false;
          // (op == 0) | (op == INT32_MIN)  -> deopt
          if (!GetLocal(e, be.unboxScratch) || !e.writeOp(Op::I32Eqz))
            return false;
          if (!GetLocal(e, be.unboxScratch) || !e.writeOp(Op::I32Const) ||
              !e.writeVarS32(INT32_MIN) || !e.writeOp(Op::I32Eq) ||
              !e.writeOp(Op::I32Or))
            return false;
          if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
          if (!EmitDeopt(e, be)) return false;
          if (!e.writeOp(Op::End)) return false;
          // 0 - op
          return e.writeOp(Op::I32Const) && e.writeVarS32(0) &&
                 GetLocal(e, be.unboxScratch) && e.writeOp(Op::I32Sub);
        }
      }

      // (B) Generic VM-helper fallback (WJH_UNARYARITH = DoUnaryArithFallback).
      // DEFAULT-ON now: an uncovered unary (e.g. a Value-typed Inc/Dec navier emits)
      // must COMPILE, not bail the whole function to PBL. jsop now comes from
      // rp->pc() (correct) -- the old ARITHCACHE crash was the trackedSite-pc bug
      // giving a wrong jsop. GECKO_WJ_NOARITHHELPER reverts.
      if (getenv("GECKO_WJ_NOARITHHELPER")) return false;
      switch (jsop) {
        case JSOp::BitNot: case JSOp::Pos: case JSOp::Neg:
        case JSOp::Inc: case JSOp::Dec: case JSOp::ToNumeric:
          break;
        default: return false;
      }
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_UNARYARITH, uint32_t(jsop)))
        return false;
      return EmitHelperResultAsType(e, be, ins->type());
    }
    case MDefinition::Opcode::ArrayPush: {
      MArrayPush* a = ins->toArrayPush();
      if (!EmitStageScratch(e, be, a->object(), 0)) return false;
      if (!EmitStageScratch(e, be, a->value(), 1)) return false;
      // result is the new length (Int32): unbox the helper's boxed Int32 result.
      return EmitHelperCallResult(e, be, ins, WJH_ARRAYPUSH, 0) &&
             e.writeOp(Op::I32WrapI64);
    }
    case MDefinition::Opcode::ArrayPopShift: {
      MArrayPopShift* a = ins->toArrayPopShift();
      if (!EmitStageScratch(e, be, a->object(), 0)) return false;
      // result is the removed element (Value); site = Pop(0)/Shift(1).
      return EmitHelperCallResult(e, be, ins, WJH_ARRAYPOPSHIFT,
                                  a->mode() == MArrayPopShift::Shift ? 1 : 0);
    }
    case MDefinition::Opcode::MegamorphicLoadSlot: {
      // obj.name where name is a compile-time PropertyKey. Inline the per-site
      // property IC (own data property -> direct slot load, no C++ hop) when the
      // receiver is an object and a site is available; else fall back to the
      // generic helper get.
      MMegamorphicLoadSlot* m = ins->toMegamorphicLoadSlot();
      static int noPropIC = getenv("GECKO_WJ_NOPROPIC") ? 1 : 0;
      if (!noPropIC && m->object()->type() == MIRType::Object) {
        uint32_t site = WJPropSite(be);
        if (site != 0) {
          return EmitPropIC(e, be, ins, m->object(), m->name().asRawBits(), site);
        }
      }
      if (!EmitStageScratch(e, be, m->object(), 0)) return false;
      uint64_t nameBits = js::IdToValue(m->name()).asRawBits();
      if (!EmitStageConstBoxed(e, be, nameBits, 1)) return false;
      return EmitHelperCallResult(e, be, ins, WJH_GETPROP, 0);
    }
    case MDefinition::Opcode::InstanceOfCache: {
      // `obj instanceof ctor`: ctor is operand 1. Result is a boolean (i32);
      // unbox the boxed BooleanValue the helper returns.
      if (!EmitStageScratch(e, be, ins->getOperand(0), 0)) return false;
      if (!EmitStageScratch(e, be, ins->getOperand(1), 1)) return false;
      if (!EmitHelperCallResult(e, be, ins, WJH_INSTANCEOF, 0)) return false;
      return e.writeOp(Op::I32WrapI64);
    }
    case MDefinition::Opcode::CreateThis: {
      MCreateThis* c = ins->toCreateThis();
      if (!EmitStageScratch(e, be, c->getOperand(0), 0)) return false;  // callee
      if (!EmitStageScratch(e, be, c->getOperand(1), 1)) return false;  // newTarget
      return EmitHelperCallResult(e, be, ins, WJH_CREATETHIS, 0);
    }
    case MDefinition::Opcode::NewObject:
    case MDefinition::Opcode::NewPlainObject:
    case MDefinition::Opcode::NewArrayObject:
    case MDefinition::Opcode::NewArray: {
      // Object/array allocation -> wjhelp(WJH_NEW*) which calls the same VM
      // fallback Ion uses (NewPlainObjectOptimizedFallback / NewArrayObject-
      // OptimizedFallback / NewArrayOperation). Keeps the allocating function in
      // JIT (was bailing whole to PBL: splay GeneratePayloadTree, navier reset,
      // raytrace Vectors, earley conses). Stage the alloc params into the (non-GC)
      // globals, call, unwrap the boxed Object result to a pointer. NB the alloc
      // itself is a C++ hop; inline nursery bump-alloc (Ion's createPlainGCObject)
      // is the follow-up for alloc-bound benches (splay). The helper rematerializes
      // the shape from the traced pool (GC-current).
      // DEFAULT-OFF (GECKO_WJ_NEWALLOC to enable): measured a REGRESSION on the
      // alloc-bound benches it targets -- splay 1.03x->0.87x (helper C++ hop +
      // fresh-array StoreElement deopts make the JIT'd GeneratePayloadTree SLOWER
      // than clean PBL), and raytrace allocates a Vector per vector-op (same).
      // Bailing those functions whole to PBL (return false) is faster than helper-
      // alloc. The REAL fix is INLINE nursery bump-allocation (Ion's
      // createPlainGCObject: load nursery position, bump, bounds-check vs
      // currentEnd, store NurseryCellHeader+shape+slots+elements, fill undefined;
      // this WJH_NEW* helper becomes the OOL nursery-full fallback). Kept gated.
      // DEFAULT-ON now (was gated): bailing the whole function to PBL is worse
      // than the helper hop per the "compile everything" directive, and inline
      // bump-alloc below handles the hot NewPlainObject case. GECKO_WJ_NONEWALLOC
      // reverts to bailing.
      static int doNewAlloc = getenv("GECKO_WJ_NONEWALLOC") ? 0 : 1;
      if (!doNewAlloc) return false;
      auto stG = [&](void* g, int32_t v) -> bool {
        return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(uintptr_t(g))) &&
               e.writeOp(Op::I32Const) && e.writeVarS32(v) &&
               e.writeOp(Op::I32Store) && e.writeVarU32(2) && e.writeVarU32(0);
      };
      int kind;
      // NewObject in ObjectLiteral mode carries NO compile-time template shape (the
      // shape is in the bytecode) -> NewObjectOperation(script, pc) via WJH_NEWOBJECT.
      if (ins->op() == MDefinition::Opcode::NewObject &&
          !ins->toNewObject()->templateObject()) {
        // Use the RESUME POINT pc, not trackedSite() -- the latter points at a
        // JumpTarget here (NewObjectOperation would read a garbage gcthing ->
        // wrong-shaped object -> OOB StoreFixedSlot -> heap corruption). The rp pc
        // is the JSOp::NewObject bytecode (matches Ion's visitNewObject site).
        MResumePoint* rp = ins->resumePoint();
        if (!rp || !rp->pc()) return false;
        JSScript* sc = rp->block()->info().script();
        if (!sc) return false;
        jsbytecode* pc = rp->pc();
        if (JSOp(*pc) != JSOp::NewObject && JSOp(*pc) != JSOp::NewInit &&
            JSOp(*pc) != JSOp::Object) {
          if (getenv("GECKO_WJ_NEWOBJDBG"))
            fprintf(stderr, "[wj-newobj] %s:%u UNEXPECTED op@rp=%s -> bail\n",
                    sc->filename() ? sc->filename() : "?", unsigned(sc->lineno()),
                    js::CodeName(JSOp(*pc)));
          return false;  // unexpected pc -> bail rather than risk corruption
        }
        uint32_t pcOff = uint32_t(pc - sc->code());
        if (!stG(&js::wasm::gWJNewObjScript,
                 int32_t(uintptr_t(static_cast<void*>(sc)))) ||
            !stG(&js::wasm::gWJNewObjPcOff, int32_t(pcOff)))
          return false;
        if (!EmitHelperCallResult(e, be, ins, WJH_NEWOBJECT, 0)) return false;
        return e.writeOp(Op::I32WrapI64);
      }
      // NewObject (object literal `{a:.., b:..}` -- splay GeneratePayloadTree) shares
      // the NewPlainObject inline bump-alloc, deriving the shape/layout from its
      // template object. Both allocate an object with a fixed shape, fixed slots
      // init'd to undefined (the bytecode's InitProp/StoreFixedSlot fills the real
      // values). Falls to the WJH_NEWPLAIN helper (any shared shape) on nursery-full.
      bool isPlainLike = ins->op() == MDefinition::Opcode::NewPlainObject ||
                         ins->op() == MDefinition::Opcode::NewObject;
      if (isPlainLike) {
        const js::Shape* pShape = nullptr;
        uint32_t pNfixed = 0, pNdynamic = 0;
        js::gc::AllocKind pAllocKind = js::gc::AllocKind::OBJECT0;
        gc::Heap pHeap = gc::Heap::Default;
        if (ins->op() == MDefinition::Opcode::NewPlainObject) {
          MNewPlainObject* n = ins->toNewPlainObject();
          pShape = n->shape();
          pNfixed = n->numFixedSlots();
          pNdynamic = n->numDynamicSlots();
          pAllocKind = n->allocKind();
          pHeap = n->initialHeap();
        } else {
          JSObject* t = ins->toNewObject()->templateObject();
          if (!t || !t->is<js::NativeObject>()) return false;  // VM-call mode: bail
          js::NativeObject* nt = &t->as<js::NativeObject>();
          pShape = nt->shape();
          pNfixed = nt->numFixedSlots();
          pNdynamic = nt->numDynamicSlots();
          pAllocKind = nt->asTenured().getAllocKind();
          pHeap = ins->toNewObject()->initialHeap();
        }
        uintptr_t shapeAddr = WJInternShape(uintptr_t(pShape));
        if (!shapeAddr) return false;
        // INLINE nursery bump-allocation (Ion createPlainGCObject), with the
        // WJH_NEWPLAIN helper as the nursery-full OOL fallback. Makes alloc-bound
        // benches (raytrace Vectors, splay payload objects) fast instead of a C++
        // hop. Gated GECKO_WJ_INLINEALLOC. Only ndynamic==0 (else helper). All
        // layout values (header size, currentEnd offset, thingSize) come from the
        // real runtime functions so wasm32 alignment is correct by construction.
        static int inlineAlloc = getenv("GECKO_WJ_NOINLINEALLOC") ? 0 : 1;
        if (inlineAlloc && pNdynamic == 0 && gWJNurseryPosAddr &&
            gWJObjHeaderWord) {
          uint32_t headerSize = uint32_t(js::Nursery::nurseryCellHeaderSize());
          int32_t endOff = js::Nursery::offsetOfCurrentEndFromPosition();
          uint32_t totalSize =
              uint32_t(js::gc::Arena::thingSize(pAllocKind)) + headerSize;
          uint32_t nfixed = pNfixed;
          uint32_t shapeOff = uint32_t(offsetof(JS::shadow::Object, shape));
          uint32_t slotsOff = uint32_t(js::NativeObject::offsetOfSlots());
          uint32_t elemsOff = uint32_t(js::NativeObject::offsetOfElements());
          int32_t emptySlots = int32_t(uintptr_t(js::emptyObjectSlots));
          int32_t emptyElems = int32_t(uintptr_t(js::emptyObjectElements));
          int32_t posAddr = int32_t(gWJNurseryPosAddr);
          int64_t undefBits = int64_t(JS::UndefinedValue().asRawBits());
          auto I32C = [&](int32_t v) {
            return e.writeOp(Op::I32Const) && e.writeVarS32(v);
          };
          auto store32 = [&](uint32_t off) {
            return e.writeOp(Op::I32Store) && e.writeVarU32(2) &&
                   e.writeVarU32(off);
          };
          // allocPos = *posAddr
          if (!I32C(posAddr) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
              !e.writeVarU32(0) || !e.writeOp(Op::LocalSet) ||
              !e.writeVarU32(be.allocPosLocal))
            return false;
          // (currentEnd = load posAddr+endOff) < (newpos = allocPos + totalSize) ?
          if (!I32C(posAddr) || !e.writeOp(Op::I32Load) || !e.writeVarU32(2) ||
              !e.writeVarU32(uint32_t(endOff)))
            return false;
          if (!GetLocal(e, be.allocPosLocal) || !I32C(int32_t(totalSize)) ||
              !e.writeOp(Op::I32Add) || !e.writeOp(Op::I32LtU))
            return false;
          if (!e.writeOp(Op::If) || !e.writeFixedU8(uint8_t(TypeCode::I32)))
            return false;
          // OOL (nursery full): helper alloc, unwrap boxed Object -> ptr.
          if (!stG(&gWJNewShapeSlot, int32_t(shapeAddr)) ||
              !stG(&gWJNewAux, int32_t(uint32_t(pAllocKind))) ||
              !stG(&gWJNewHeap, int32_t(uint32_t(pHeap))) ||
              !EmitHelperCallResult(e, be, ins, WJH_NEWPLAIN, 0) ||
              !e.writeOp(Op::I32WrapI64))
            return false;
          if (!e.writeOp(Op::Else)) return false;
          // INLINE: commit position, write header, shape, slots, elements, slots.
          if (!I32C(posAddr) || !GetLocal(e, be.allocPosLocal) ||
              !I32C(int32_t(totalSize)) || !e.writeOp(Op::I32Add) || !store32(0))
            return false;  // *posAddr = newpos
          if (!GetLocal(e, be.allocPosLocal) || !I32C(int32_t(gWJObjHeaderWord)) ||
              !store32(0))
            return false;  // header at allocPos
          if (!GetLocal(e, be.allocPosLocal) || !I32C(int32_t(headerSize)) ||
              !e.writeOp(Op::I32Add) || !e.writeOp(Op::LocalSet) ||
              !e.writeVarU32(be.allocObjLocal))
            return false;  // obj = allocPos + headerSize
          if (!GetLocal(e, be.allocObjLocal) || !I32C(int32_t(shapeAddr)) ||
              !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
              !store32(shapeOff))
            return false;  // obj->shape = *(shape pool slot) (relocated)
          if (!GetLocal(e, be.allocObjLocal) || !I32C(emptySlots) ||
              !store32(slotsOff))
            return false;
          if (!GetLocal(e, be.allocObjLocal) || !I32C(emptyElems) ||
              !store32(elemsOff))
            return false;
          for (uint32_t f = 0; f < nfixed; f++) {
            uint32_t off = uint32_t(js::NativeObject::getFixedSlotOffset(f));
            if (!GetLocal(e, be.allocObjLocal) || !e.writeOp(Op::I64Const) ||
                !e.writeVarS64(undefBits) || !e.writeOp(Op::I64Store) ||
                !e.writeVarU32(3) || !e.writeVarU32(off))
              return false;
          }
          if (!GetLocal(e, be.allocObjLocal) || !e.writeOp(Op::End)) return false;
          return true;  // obj pointer (i32) on stack
        }
        if (!stG(&gWJNewShapeSlot, int32_t(shapeAddr)) ||
            !stG(&gWJNewAux, int32_t(uint32_t(pAllocKind))) ||
            !stG(&gWJNewHeap, int32_t(uint32_t(pHeap))))
          return false;
        kind = WJH_NEWPLAIN;
      } else if (ins->op() == MDefinition::Opcode::NewArrayObject) {
        MNewArrayObject* n = ins->toNewArrayObject();
        if (!stG(&gWJNewAux, int32_t(n->length())) ||
            !stG(&gWJNewHeap, int32_t(uint32_t(n->initialHeap()))))
          return false;
        kind = WJH_NEWARROBJ;
      } else {
        MNewArray* n = ins->toNewArray();
        if (!stG(&gWJNewAux, int32_t(n->length()))) return false;
        kind = WJH_NEWARR;
      }
      if (!EmitHelperCallResult(e, be, ins, kind, 0)) return false;
      return e.writeOp(Op::I32WrapI64);  // boxed Object -> pointer
    }
    case MDefinition::Opcode::BitAnd:
    case MDefinition::Opcode::BitOr:
    case MDefinition::Opcode::BitXor:
    case MDefinition::Opcode::Lsh:
    case MDefinition::Opcode::Rsh:
    case MDefinition::Opcode::Ursh: {
      // JS bitwise/shift ops are i32 (shift counts are masked mod 32, which
      // wasm shifts already do). Ursh whose result exceeds INT32_MAX is typed
      // Double by Ion -- bail that case.
      if (ins->type() != MIRType::Int32) return false;
      if (ins->getOperand(0)->type() != MIRType::Int32 ||
          ins->getOperand(1)->type() != MIRType::Int32) {
        return false;
      }
      if (!GetOp(e, be, ins->getOperand(0)) || !GetOp(e, be, ins->getOperand(1))) {
        return false;
      }
      switch (ins->op()) {
        case MDefinition::Opcode::BitAnd: return e.writeOp(Op::I32And);
        case MDefinition::Opcode::BitOr: return e.writeOp(Op::I32Or);
        case MDefinition::Opcode::BitXor: return e.writeOp(Op::I32Xor);
        case MDefinition::Opcode::Lsh: return e.writeOp(Op::I32Shl);
        case MDefinition::Opcode::Rsh: return e.writeOp(Op::I32ShrS);
        default: return e.writeOp(Op::I32ShrU);  // Ursh
      }
    }
    case MDefinition::Opcode::Call: {
      MCall* call = ins->toCall();
      // Constructing calls: implemented (WJH_CONSTRUCT via
      // InternalConstructWithProvidedThis) and CORRECT for interp->JIT callers
      // (the construct produces valid objects; repro /tmp/ctortest2.js passes).
      // DISABLED because of an unresolved JIT->JIT issue: when a JIT'd caller
      // (e.g. blend) fast-call_indirects into a JIT'd construct-containing callee
      // (repro /tmp/ct5.js, /tmp/ctortest3.js), the FIRST construct succeeds
      // (verified: fval=function, valid result) but a subsequent throw surfaces
      // as `unreachable` -- it's a ReportValueError whose stack decompilation
      // (FrameIter -> baselineScriptAndPc -> retAddrEntryFromReturnAddress) traps
      // because this is JS_CODEGEN_NONE (no baseline frames) and FrameIter can't
      // walk our wasm-JIT frames during error reporting. NOT a GC-rooting issue
      // (NOROOT doesn't change it; the dominance-rooting fix in WJCollectRoots is
      // kept and benefits all other helpers). Re-enable after the error/frame
      // re-entrancy across nested wasm-JIT construct calls is resolved.
      if (call->isConstructing()) {
        static int noConstruct = getenv("GECKO_WJ_NOCONSTRUCT") ? 1 : 0;
        if (noConstruct) return false;
        uint32_t cargc = call->numActualArgs();
        if (cargc > kWJMaxArgs) return false;
        for (uint32_t i = 0; i < cargc; i++) {
          if (!EmitStageScratch(e, be, call->getArg(1 + i), i)) return false;
        }
        if (!EmitStageScratch(e, be, call->getArg(0), kWJThisSlot)) return false;
        uintptr_t cAddr = uintptr_t(static_cast<void*>(&gWJCallCallee));
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(cAddr)) ||
            !EmitSpillValue(e, be, call->getCallee()) ||
            !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
          return false;
        uintptr_t ntAddr = uintptr_t(static_cast<void*>(&gWJConstructNewTarget));
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(ntAddr)) ||
            !EmitSpillValue(e, be, call->getArg(call->numStackArgs() - 1)) ||
            !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
          return false;
        uintptr_t aAddr = uintptr_t(static_cast<void*>(&gWJCallArgc));
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(aAddr)) ||
            !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(cargc)) ||
            !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
          return false;
        return EmitHelperCallResult(e, be, ins, WJH_CONSTRUCT, 0);
      }
      uint32_t argc = call->numActualArgs();
      if (argc > kWJMaxArgs) return false;  // too many for the register ABI
      // Box `this` + actual args into the call-arg locals (reused by both the
      // fast register-call path and the slow scratch+wjhelp path). Pad unused
      // arg slots with undefined.
      if (!EmitSpillValue(e, be, call->getArg(0))) return false;  // this
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.callArgBase)) return false;
      for (uint32_t i = 0; i < kWJMaxArgs; i++) {
        if (i < argc) {
          if (!EmitSpillValue(e, be, call->getArg(1 + i))) return false;
        } else if (!e.writeOp(Op::I64Const) ||
                   !e.writeVarS64(int64_t(JS::UndefinedValue().asRawBits()))) {
          return false;
        }
        if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.callArgBase + 1 + i))
          return false;
      }

      // GC-root shadow stack: spill live object/Value locals (which the GC can't
      // see in wasm locals) so a moving minor GC inside the callee updates them.
      std::vector<uint32_t> rootLocal;
      std::vector<uint8_t> rootIsObj;
      WJCollectRoots(be, ins, rootLocal, rootIsObj);
      static int noRoot = getenv("GECKO_WJ_NOROOT") ? 1 : 0;
      if (noRoot) {
        rootLocal.clear();
        rootIsObj.clear();
      }
      uint32_t nRoots = uint32_t(rootLocal.size());
      uintptr_t rootsBase = uintptr_t(static_cast<void*>(&gWJCallRoots[0]));
      uintptr_t spAddr = uintptr_t(static_cast<void*>(&gWJRootSP));
      auto emitSPAdjust = [&](int32_t delta) -> bool {
        return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(spAddr)) &&
               e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(spAddr)) &&
               e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(0) &&
               e.writeOp(Op::I32Const) && e.writeVarS32(delta) &&
               e.writeOp(Op::I32Add) && e.writeOp(Op::I32Store) &&
               e.writeVarU32(2) && e.writeVarU32(0);
      };
      // Spill base computed ONCE (= gWJRootSP*8 + rootsBase); slot k = base + k*8 via
      // the store/load immediate offset. SP is unchanged between spill and reload, so
      // rootBaseLocal serves both. Avoids reloading gWJRootSP + *8+base per root (the
      // splay GC-root-spill lever; survives the call_indirect -- wasm locals do).
      if (nRoots) {
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(spAddr)) ||
            !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
            !e.writeOp(Op::I32Const) || !e.writeVarS32(8) || !e.writeOp(Op::I32Mul) ||
            !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(rootsBase)) ||
            !e.writeOp(Op::I32Add) || !e.writeOp(Op::LocalSet) ||
            !e.writeVarU32(be.rootBaseLocal))
          return false;
      }
      auto emitSlotBase = [&]() -> bool { return GetLocal(e, be.rootBaseLocal); };
      for (uint32_t k = 0; k < nRoots; k++) {
        if (!emitSlotBase()) return false;
        if (!GetLocal(e, rootLocal[k])) return false;
        if (rootIsObj[k] && !EmitBoxFromStack(e, MIRType::Object)) return false;
        if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(k * 8))
          return false;
      }
      if (nRoots && !emitSPAdjust(int32_t(nRoots))) return false;

      uint32_t site = WJAllocCallSite();
      uintptr_t calleeAddr = uintptr_t(static_cast<void*>(&gWJCallCallee));
      uintptr_t argcAddr = uintptr_t(static_cast<void*>(&gWJCallArgc));

      // callScratch = actual callee object pointer.
      if (!EmitObjPtr(e, be, call->getCallee())) return false;
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.callScratch)) return false;

      // Polymorphic inline cache. block (result f64) { for each way: if the
      // cached callee matches (and its table slot is in range) do a direct
      // wasm->wasm call_indirect and br out with the result; else fall to the
      // next way, then to the slow wjhelp hop }. Way 0 first => monomorphic
      // calls cost one compare; polymorphic dispatch (task.run) stays in wasm.
      if (!e.writeOp(Op::Block) || !e.writeFixedU8(0x40)) return false;  // void
      static int forceSlowCall = getenv("GECKO_WJ_FORCESLOWCALL") ? 1 : 0;
      for (uint32_t w = 0; w < kWJCallWays && !forceSlowCall; w++) {
        uintptr_t fnW =
            uintptr_t(static_cast<void*>(&gWJCallFn[site * kWJCallWays + w]));
        uintptr_t tblW =
            uintptr_t(static_cast<void*>(&gWJCallTblIdx[site * kWJCallWays + w]));
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(fnW)) ||
            !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0))
          return false;
        if (!GetLocal(e, be.callScratch) || !e.writeOp(Op::I32Eq)) return false;
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(tblW)) ||
            !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0))
          return false;
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(4096)) ||
            !e.writeOp(Op::I32LtU) || !e.writeOp(Op::I32And))
          return false;
        if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
        // gWJCurrentEnv = callee->environment(): the callee's prologue saves this
        // into its envLocal so its MFunctionEnvironment is correct (closures).
        {
          uintptr_t envAddr = uintptr_t(static_cast<void*>(&gWJCurrentEnv));
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(envAddr)))
            return false;
          if (!GetLocal(e, be.callScratch) || !e.writeOp(Op::I32Load) ||
              !e.writeVarU32(2) ||
              !e.writeVarU32(uint32_t(JSFunction::offsetOfEnvironment())))
            return false;
          if (!e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
            return false;
        }
        // Register call: push sb (f64) + this + args (i64) + table index, then
        // call_indirect the callee's main directly (no memory marshalling).
        if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(0)) return false;  // sb
        for (uint32_t a = 0; a <= kWJMaxArgs; a++) {  // this + kWJMaxArgs args
          if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(be.callArgBase + a))
            return false;
        }
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(tblW)) ||
            !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0))
          return false;  // table index
        if (!e.writeOp(Op::CallIndirect) || !e.writeVarU32(0) || !e.writeVarU32(0))
          return false;  // type 0 -> [f64 flag, i64 result]
        if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.callResultLocal))
          return false;  // pop result (top)
        if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.callFlagLocal))
          return false;  // pop flag
        if (!e.writeOp(Op::Br) || !e.writeVarU32(1)) return false;  // -> block end
        if (!e.writeOp(Op::End)) return false;                      // end if (way w)
      }
      {  // SLOW: store this+args to scratch (the callee's trampoline / JS::Call
        // reads them), set callee+argc, wjhelp(WJH_CALL, site).
        for (uint32_t a = 0; a < argc; a++) {
          if (!GetLocal(e, be.scratchBase) ||
              !GetLocal(e, be.callArgBase + 1 + a) || !e.writeOp(Op::I64Store) ||
              !e.writeVarU32(3) || !e.writeVarU32(a * 8))
            return false;
        }
        if (!GetLocal(e, be.scratchBase) || !GetLocal(e, be.callArgBase) ||
            !e.writeOp(Op::I64Store) || !e.writeVarU32(3) ||
            !e.writeVarU32(kWJThisOff))
          return false;
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(calleeAddr))) return false;
        if (!EmitSpillValue(e, be, call->getCallee())) return false;
        if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0)) return false;
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(argcAddr)) ||
            !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(argc)) ||
            !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0)) {
          return false;
        }
        if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(double(WJH_CALL)) ||
            !e.writeOp(Op::F64Const) || !e.writeFixedF64(double(site)) ||
            !e.writeOp(Op::Call) || !e.writeVarU32(0)) {
          return false;  // wjhelp(WJH_CALL, site) -> f64 flag
        }
        if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.callFlagLocal))
          return false;  // flag
        // wjhelp left the boxed result in gWJScratch[kWJResultOff]; load it.
        if (!GetLocal(e, be.scratchBase) || !e.writeOp(Op::I64Load) ||
            !e.writeVarU32(3) || !e.writeVarU32(kWJResultOff) ||
            !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.callResultLocal))
          return false;
      }
      if (!e.writeOp(Op::End)) return false;  // end block (void)
      // Pop the GC-root frame (runs on both the normal and exception paths).
      if (nRoots && !emitSPAdjust(-int32_t(nRoots))) return false;
      // Propagate a pending exception (flag != 0 -> return [1.0, dummy]).
      if (!GetLocal(e, be.callFlagLocal) || !e.writeOp(Op::F64Const) ||
          !e.writeFixedF64(0.0) || !e.writeOp(Op::F64Ne)) {
        return false;
      }
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(1.0) ||
          !e.writeOp(Op::I64Const) || !e.writeVarS64(0) || !e.writeOp(Op::Return)) {
        return false;
      }
      if (!e.writeOp(Op::End)) return false;
      // Reload the rooted locals (GC may have moved the objects).
      for (uint32_t k = 0; k < nRoots; k++) {
        if (!emitSlotBase()) return false;
        if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(k * 8))
          return false;
        if (rootIsObj[k] && !e.writeOp(Op::I32WrapI64)) return false;
        if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(rootLocal[k])) return false;
      }
      // The call's value is the boxed result returned in a register.
      return GetLocal(e, be.callResultLocal);
    }
    case MDefinition::Opcode::Compare: {
      MCompare* c = ins->toCompare();
      MDefinition* l = c->getOperand(0);
      MDefinition* r = c->getOperand(1);
      MCompare::CompareType ct0 = c->compareType();
      // `x == null` / `x == undefined`: a tag check on the tested Value operand
      // (the other operand is the null/undefined literal, which has no local).
      if (ct0 == MCompare::Compare_Null || ct0 == MCompare::Compare_Undefined) {
        MDefinition* tested = be.local(l) >= 0 ? l : r;
        if (be.local(tested) < 0) return false;
        if (tested->type() != MIRType::Value) {
          // A statically-typed non-null value: result is a constant.
          bool isEq = c->jsop() == JSOp::Eq || c->jsop() == JSOp::StrictEq;
          return e.writeOp(Op::I32Const) && e.writeVarS32(isEq ? 0 : 1);
        }
        if (!GetOp(e, be, tested)) return false;
        if (!e.writeOp(Op::I64Const) || !e.writeVarS64(32) ||
            !e.writeOp(Op::I64ShrU) || !e.writeOp(Op::I32WrapI64)) {
          return false;
        }
        bool strict = c->jsop() == JSOp::StrictEq || c->jsop() == JSOp::StrictNe;
        uint32_t litTag = ct0 == MCompare::Compare_Null
                              ? TagWord(JSVAL_TYPE_NULL)
                              : TagWord(JSVAL_TYPE_UNDEFINED);
        if (strict) {
          if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(litTag)) ||
              !e.writeOp(Op::I32Eq)) {
            return false;
          }
        } else {
          // loose: matches null OR undefined.
          if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.callScratch))
            return false;  // stash tag in a scratch i32
          if (!GetLocal(e, be.callScratch) ||
              !e.writeOp(Op::I32Const) ||
              !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_NULL))) ||
              !e.writeOp(Op::I32Eq) || !GetLocal(e, be.callScratch) ||
              !e.writeOp(Op::I32Const) ||
              !e.writeVarS32(int32_t(TagWord(JSVAL_TYPE_UNDEFINED))) ||
              !e.writeOp(Op::I32Eq) || !e.writeOp(Op::I32Or)) {
            return false;
          }
        }
        if (c->jsop() == JSOp::Ne || c->jsop() == JSOp::StrictNe) {
          if (!e.writeOp(Op::I32Eqz)) return false;  // negate
        }
        return true;
      }
      if (be.local(l) < 0 || be.local(r) < 0) {
        return WJBAIL("compare ct=%d lty=%s rty=%s (operand no local)\n",
                      int(ct0), StringFromMIRType(l->type()),
                      StringFromMIRType(r->type()));
      }
      if (!GetOp(e, be, l) || !GetOp(e, be, r)) return false;
      MCompare::CompareType ct = c->compareType();
      bool dbl = ct == MCompare::Compare_Double;
      // Int32/Boolean/Object(+Null/Undefined ptr) all compare as i32 here.
      bool i32 = ct == MCompare::Compare_Int32 || ct == MCompare::Compare_Object ||
                 ct == MCompare::Compare_Null || ct == MCompare::Compare_Undefined;
      if (!dbl && !i32) {
        if (getenv("GECKO_WJWARP_DUMP"))
          fprintf(stderr, "[wb-be] Compare type=%d lhsTy=%d unhandled\n",
                  int(ct), int(l->type()));
        return false;
      }
      // Object/ref identity only supports (strict)equality.
      bool refCmp = ct == MCompare::Compare_Object || ct == MCompare::Compare_Null ||
                    ct == MCompare::Compare_Undefined;
      switch (c->jsop()) {
        case JSOp::Lt: return !refCmp && e.writeOp(dbl ? Op::F64Lt : Op::I32LtS);
        case JSOp::Le: return !refCmp && e.writeOp(dbl ? Op::F64Le : Op::I32LeS);
        case JSOp::Gt: return !refCmp && e.writeOp(dbl ? Op::F64Gt : Op::I32GtS);
        case JSOp::Ge: return !refCmp && e.writeOp(dbl ? Op::F64Ge : Op::I32GeS);
        case JSOp::Eq:
        case JSOp::StrictEq: return e.writeOp(dbl ? Op::F64Eq : Op::I32Eq);
        case JSOp::Ne:
        case JSOp::StrictNe: return e.writeOp(dbl ? Op::F64Ne : Op::I32Ne);
        default: return false;
      }
    }
    default:
      WJTallyBail(ins->op());
      return false;
  }
}

// True if the node has a side effect we must emit even though it produces no
// wasm value (or its value is unused). Returns false from EmitEffect => skip.
enum class EffectKind { Skip, Emitted, Fail };

static EffectKind EmitEffect(Encoder& e, WJBackend& be, MInstruction* ins) {
  js::wasm::gWJBailReason = WJOpName(ins->op());  // attribute an internal bail to this op
  switch (ins->op()) {
    case MDefinition::Opcode::Bail:
    case MDefinition::Opcode::Unreachable:
    case MDefinition::Opcode::Throw: {
      // Throw: a cold error path (splay `throw new Error(...)`, raytrace asserts).
      // Deopt to PBL, which re-executes from the resume point and performs the real
      // throw with correct exception semantics. Lets the hot path stay JIT'd instead
      // of bailing the whole function to PBL. This is a TOP-LEVEL deopt (not inside a
      // guard's `if`), so clear inDispatchBody so EmitDeopt uses the inline-return
      // path -- the OOL `br $D` depth is calibrated for a guard's enclosing `if` and
      // would branch one scope too far here (invalid wasm: "expected 2 for branch").
      bool savedDispatch = be.inDispatchBody;
      be.inDispatchBody = false;
      bool ok = EmitDeopt(e, be);
      be.inDispatchBody = savedDispatch;
      return ok ? EffectKind::Emitted : EffectKind::Fail;
    }
    case MDefinition::Opcode::Start:
    case MDefinition::Opcode::CheckOverRecursed:
    case MDefinition::Opcode::InterruptCheck:
      return EffectKind::Skip;  // prologue / ignored (host stack guards depth)
    case MDefinition::Opcode::AddAndStoreSlot: {
      // Add an own property: transition obj->shape to the new shape (which has the
      // added slot) and store the value at slotOffset. Mirrors Ion's
      // visitAddAndStoreSlot. The shape pre-write barrier Ion emits is safely
      // ELIDED here: this is a property-ADDITION transition, so the new shape has
      // the OLD shape as its shape-tree parent -> the old shape stays reachable
      // and gets marked via the new shape (the SATB invariant holds without the
      // explicit barrier). The value store needs no pre-barrier (new init); any
      // post-write barrier is a separate MPostWriteBarrier node (handled).
      // (raytrace constructors / color+vector field init bail here otherwise.)
      auto* a = ins->toAddAndStoreSlot();
      if (a->preserveWrapper()) return EffectKind::Fail;  // proxy: rare, bail
      uintptr_t shapeSlot = WJInternShape(uintptr_t(a->shape()));
      if (!shapeSlot) return EffectKind::Fail;
      uint32_t shapeOff = uint32_t(offsetof(JS::shadow::Object, shape));
      uint32_t slotOffset = a->slotOffset();
      // obj->shape = relocated new shape
      if (!GetOp(e, be, a->object()) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(shapeSlot)) || !e.writeOp(Op::I32Load) ||
          !e.writeVarU32(2) || !e.writeVarU32(0) || !e.writeOp(Op::I32Store) ||
          !e.writeVarU32(2) || !e.writeVarU32(shapeOff))
        return EffectKind::Fail;
      // store value at the slot (fixed: obj+offset; dynamic: slots+offset).
      if (a->kind() == MAddAndStoreSlot::Kind::FixedSlot) {
        if (!GetOp(e, be, a->object()) || !EmitSpillValue(e, be, a->value()) ||
            !e.writeOp(Op::I64Store) || !e.writeVarU32(3) ||
            !e.writeVarU32(slotOffset))
          return EffectKind::Fail;
      } else {
        if (!GetOp(e, be, a->object()) || !e.writeOp(Op::I32Load) ||
            !e.writeVarU32(2) ||
            !e.writeVarU32(uint32_t(js::NativeObject::offsetOfSlots())) ||
            !EmitSpillValue(e, be, a->value()) || !e.writeOp(Op::I64Store) ||
            !e.writeVarU32(3) || !e.writeVarU32(slotOffset))
          return EffectKind::Fail;
      }
      return EffectKind::Emitted;
    }
    case MDefinition::Opcode::StoreFixedSlot: {
      MStoreFixedSlot* s = ins->toStoreFixedSlot();
      // Megamorphic recompile: this store is on a mega-convertible (passthrough)
      // GuardShape, so the receiver may be polymorphic -> a raw fixed-slot store
      // would corrupt. Route through the generic SETPROP helper (baked key, no
      // deopt, handles any shape). EFFECT -> leaves nothing on the stack.
      if (be.forceMega && WJIsShapeGuard(s->object()) &&
          WJMegaConvertibleGuard(s->object())) {
        jit::MDefinition* gs = s->object();
        uint64_t keyBits = 0;
        if (WJDerivePropKey(WJGuardRepShape(gs), uint32_t(s->slot()), &keyBits) &&
            WJGuardObject(gs)->type() == MIRType::Object) {
          // WJH_SETPROP does ToPropertyKey(scratch[1]) -> needs the key as a
          // VALUE (StringValue), not the jsid bits WJDerivePropKey returns.
          uint64_t keyVal = js::IdToValue(
              JS::PropertyKey::fromRawBits(uintptr_t(keyBits))).asRawBits();
          if (!EmitStageScratch(e, be, WJGuardObject(gs), 0)) return EffectKind::Fail;
          if (!EmitStageConstBoxed(e, be, keyVal, 1)) return EffectKind::Fail;
          if (!EmitStageScratch(e, be, s->value(), 2)) return EffectKind::Fail;
          if (!EmitHelperCallResult(e, be, ins, WJH_SETPROP, 0))
            return EffectKind::Fail;
          return e.writeOp(Op::Drop) ? EffectKind::Emitted : EffectKind::Fail;
        }
      }
      int32_t objLocal = be.local(s->object());
      if (objLocal < 0) return EffectKind::Fail;
      MDefinition* v = s->value();
      if (s->needsBarrier()) {
        // Route through wjhelp(WJH_SETSLOT) -> NativeObject::setSlot, which runs
        // the proper post-write barrier. gWJHelpObj/Slot/Val carry the operands.
        uintptr_t objAddr = uintptr_t(static_cast<void*>(&gWJHelpObj));
        uintptr_t slotAddr = uintptr_t(static_cast<void*>(&gWJHelpSlot));
        uintptr_t valAddr = uintptr_t(static_cast<void*>(&gWJHelpVal));
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(objAddr)) ||
            !GetOp(e, be, s->object()) ||
            !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
          return EffectKind::Fail;
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(slotAddr)) ||
            !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(s->slot())) ||
            !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
          return EffectKind::Fail;
        if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(valAddr)) ||
            !EmitSpillValue(e, be, v) ||
            !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
          return EffectKind::Fail;
        if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(double(WJH_SETSLOT)) ||
            !e.writeOp(Op::F64Const) || !e.writeFixedF64(0.0) ||
            !e.writeOp(Op::Call) || !e.writeVarU32(0) || !e.writeOp(Op::Drop))
          return EffectKind::Fail;
        return EffectKind::Emitted;
      }
      // Barrier-free (primitive) store: inline.
      uint32_t off = uint32_t(js::NativeObject::getFixedSlotOffset(s->slot()));
      if (getenv("GECKO_WJ_SLOTDBG2"))
        fprintf(stderr, "[wb-slot] StoreFixedSlot slot=%zu off=%u objOp=%s barrier=%d\n",
                s->slot(), off, WJOpName(s->object()->op()), s->needsBarrier());
      if (!GetOp(e, be, s->object())) return EffectKind::Fail;  // addr (current)
      if (!EmitSpillValue(e, be, v)) return EffectKind::Fail;         // boxed value
      if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(off))
        return EffectKind::Fail;
      return EffectKind::Emitted;
    }
    case MDefinition::Opcode::StoreFixedSlotFromOffset: {
      if (getenv("GECKO_WJ_NONEWSTORE")) return EffectKind::Fail;
      // storeValue(object + offset) = value. Fixed slots are inline in the
      // object. needsBarrier() => pre-write barrier on the OLD value (mirrors
      // StoreDynamicSlot); a separate MPostWriteBarrier handles the nursery side.
      MStoreFixedSlotFromOffset* s = ins->toStoreFixedSlotFromOffset();
      MDefinition* v = s->value();
      if (s->needsBarrier()) {
        if (!EmitGuardedValuePreBarrier(e, be, [&]() {
              return GetOp(e, be, s->object()) && GetOp(e, be, s->offset()) &&
                     e.writeOp(Op::I32Add) && e.writeOp(Op::I64Load) &&
                     e.writeVarU32(3) && e.writeVarU32(0);
            }))
          return EffectKind::Fail;
      }
      if (!GetOp(e, be, s->object())) return EffectKind::Fail;
      if (!GetOp(e, be, s->offset())) return EffectKind::Fail;
      if (!e.writeOp(Op::I32Add)) return EffectKind::Fail;
      if (!EmitSpillValue(e, be, v)) return EffectKind::Fail;
      if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
        return EffectKind::Fail;
      return EffectKind::Emitted;
    }
    case MDefinition::Opcode::StoreDynamicSlot: {
      if (getenv("GECKO_WJ_NONEWSTORE")) return EffectKind::Fail;
      // slots[slot] = value. The barrier-free case (primitive value, or a
      // separate MPostWriteBarrier node follows) stores inline; a value needing
      // a pre-write barrier falls through to bail (correct, rare).
      MStoreDynamicSlot* s = ins->toStoreDynamicSlot();
      uint32_t off = uint32_t(s->slot() * sizeof(JS::Value));
      if (s->needsBarrier()) {
        if (!EmitGuardedValuePreBarrier(e, be, [&]() {
              return GetOp(e, be, s->slots()) && e.writeOp(Op::I64Load) &&
                     e.writeVarU32(3) && e.writeVarU32(off);
            }))
          return EffectKind::Fail;
      }
      if (!GetOp(e, be, s->slots())) return EffectKind::Fail;
      if (!EmitSpillValue(e, be, s->value())) return EffectKind::Fail;
      if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(off))
        return EffectKind::Fail;
      return EffectKind::Emitted;
    }
    case MDefinition::Opcode::StoreElement: {
      if (getenv("GECKO_WJ_NONEWSTORE")) return EffectKind::Fail;
      // Dense store elements[index] = value. Object values (post-write barrier)
      // still bail. needsHoleCheck is handled inline by guarding the store is
      // in-bounds + not over a hole, deopting to PBL otherwise (grow/sparse) --
      // for pre-sized dense arrays (crypto bignums) the guards always pass, so
      // the hot store stays in JIT instead of bailing the whole function to PBL.
      MStoreElement* s = ins->toStoreElement();
      int32_t idx = be.local(s->index());
      if (idx < 0) return EffectKind::Fail;
      if (s->needsHoleCheck()) {
        // deopt unless (uint32)index < (uint32)initializedLength
        if (!GetOp(e, be, s->elements()) || !e.writeOp(Op::I32Const) ||
            !e.writeVarS32(js::ObjectElements::offsetOfInitializedLength()) ||
            !e.writeOp(Op::I32Add) || !e.writeOp(Op::I32Load) ||
            !e.writeVarU32(2) || !e.writeVarU32(0))
          return EffectKind::Fail;
        if (!GetLocal(e, uint32_t(idx)) || !e.writeOp(Op::I32LeU))
          return EffectKind::Fail;  // initLen <= index  -> out of bounds
        if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return EffectKind::Fail;
        if (!EmitDeopt(e, be)) return EffectKind::Fail;
        if (!e.writeOp(Op::End)) return EffectKind::Fail;
        // (No separate hole-value load: a dense in-bounds store just fills the
        // slot, matching Ion -- the bounds check above IS the needsHoleCheck.)
      }
      if (s->needsBarrier()) {
        // pre-write barrier on the OLD element value (incremental GC marking).
        if (!EmitGuardedValuePreBarrier(e, be, [&]() {
              return GetOp(e, be, s->elements()) && GetLocal(e, uint32_t(idx)) &&
                     e.writeOp(Op::I32Const) && e.writeVarS32(8) &&
                     e.writeOp(Op::I32Mul) && e.writeOp(Op::I32Add) &&
                     e.writeOp(Op::I64Load) && e.writeVarU32(3) && e.writeVarU32(0);
            }))
          return EffectKind::Fail;
      }
      if (!GetOp(e, be, s->elements()) || !GetLocal(e, uint32_t(idx)) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(8) ||
          !e.writeOp(Op::I32Mul) || !e.writeOp(Op::I32Add))
        return EffectKind::Fail;  // addr = elements + index*8
      if (!EmitSpillValue(e, be, s->value())) return EffectKind::Fail;
      if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
        return EffectKind::Fail;
      return EffectKind::Emitted;
    }
    case MDefinition::Opcode::SetInitializedLength: {
      // elements->initializedLength = index + 1 (Ion SetLengthFromIndex). Used to
      // finalize a dense array literal after its elements are stored (splay
      // GeneratePayloadTree's `[0,1,..,9]`).
      MDefinition* elems = ins->getOperand(0);
      MDefinition* idx = ins->getOperand(1);
      if (getenv("GECKO_WJ_SILDBG"))
        fprintf(stderr, "[wj-sil] idx const=%d val=%d\n", idx->isConstant(),
                idx->isConstant() ? idx->toConstant()->toInt32() : -1);
      if (getenv("GECKO_WJ_SILNOP")) return EffectKind::Emitted;  // isolate: no store
      int32_t off = js::ObjectElements::offsetOfInitializedLength();
      if (!GetOp(e, be, elems) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(off) || !e.writeOp(Op::I32Add))
        return EffectKind::Fail;  // addr = elements + offsetOfInitializedLength
      if (!GetOp(e, be, idx) || !e.writeOp(Op::I32Const) || !e.writeVarS32(1) ||
          !e.writeOp(Op::I32Add))
        return EffectKind::Fail;  // value = index + 1
      if (!e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
        return EffectKind::Fail;
      return EffectKind::Emitted;
    }
    case MDefinition::Opcode::MegamorphicStoreSlot: {
      if (getenv("GECKO_WJ_NONEWSTORE")) return EffectKind::Fail;
      MMegamorphicStoreSlot* m = ins->toMegamorphicStoreSlot();
      if (!EmitStageScratch(e, be, m->object(), 0)) return EffectKind::Fail;
      uint64_t nameBits = js::IdToValue(m->name()).asRawBits();
      if (!EmitStageConstBoxed(e, be, nameBits, 1)) return EffectKind::Fail;
      if (!EmitStageScratch(e, be, m->rhs(), 2)) return EffectKind::Fail;
      if (!EmitHelperCallResult(e, be, ins, WJH_SETPROP, m->strict() ? 1 : 0))
        return EffectKind::Fail;
      return e.writeOp(Op::Drop) ? EffectKind::Emitted : EffectKind::Fail;
    }
    case MDefinition::Opcode::SetPropertyCache: {
      if (getenv("GECKO_WJ_NONEWSTORE")) return EffectKind::Fail;
      MSetPropertyCache* s = ins->toSetPropertyCache();
      // Inline the per-site store IC for object receivers (writable own data
      // property -> direct slot store + barriers, no C++ hop); else generic set.
      static int noPropIC = getenv("GECKO_WJ_NOPROPIC") ? 1 : 0;
      if (!noPropIC && s->object()->type() == MIRType::Object) {
        uint32_t site = WJPropSite(be);
        if (site != 0) {
          return EmitPropStoreIC(e, be, ins, s->object(), s->idval(), s->value(),
                                 s->strict(), site)
                     ? EffectKind::Emitted
                     : EffectKind::Fail;
        }
      }
      if (!EmitStageScratch(e, be, s->object(), 0)) return EffectKind::Fail;
      if (!EmitStageScratch(e, be, s->idval(), 1)) return EffectKind::Fail;
      if (!EmitStageScratch(e, be, s->value(), 2)) return EffectKind::Fail;
      if (!EmitHelperCallResult(e, be, ins, WJH_SETPROP, s->strict() ? 1 : 0))
        return EffectKind::Fail;
      return e.writeOp(Op::Drop) ? EffectKind::Emitted : EffectKind::Fail;
    }
    case MDefinition::Opcode::PostWriteBarrier: {
      // Record the container in the GC store buffer after an object was stored
      // into one of its slots/elements. WITHOUT this, a tenured container's
      // edge to a nursery value is invisible to a minor GC, which frees the
      // value -> the slot reads back garbage (was the raytrace `dot` crash).
      int32_t ol = be.local(ins->getOperand(0));
      if (ol < 0) return EffectKind::Fail;
      uintptr_t objAddr = uintptr_t(static_cast<void*>(&gWJHelpObj));
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(objAddr)) ||
          !GetLocal(e, uint32_t(ol)) || !e.writeOp(Op::I32Store) ||
          !e.writeVarU32(2) || !e.writeVarU32(0))
        return EffectKind::Fail;
      if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(double(WJH_POSTBARRIER)) ||
          !e.writeOp(Op::F64Const) || !e.writeFixedF64(0.0) ||
          !e.writeOp(Op::Call) || !e.writeVarU32(0) || !e.writeOp(Op::Drop))
        return EffectKind::Fail;
      return EffectKind::Emitted;
    }
    case MDefinition::Opcode::KeepAliveObject:
      // Pure GC-liveness marker; our GC-root shadow stack already keeps call
      // operands alive, so this is a no-op for codegen.
      return EffectKind::Skip;
    default:
      // Pure node whose value is unused (DCE leftover, or a type we don't map to
      // a wasm local): safe to skip. Unknown EFFECTFUL node: bail the function.
      if (ins->isEffectful()) {
        WJTallyBail(ins->op());
        return EffectKind::Fail;
      }
      return EffectKind::Skip;
  }
}

// Store the (boxed) return value to gWJScratch[result] and return ok (0.0).
// Pop the persistent env-root shadow slot at a function exit (restore gWJRootSP
// to its entry value, envRootIdx). No-op unless the function rooted its env.
// Net-zero on the wasm value stack (pure memory store), so safe to interleave
// with an in-progress [flag,result] on the stack.
static bool EmitEnvRootPop(Encoder& e, WJBackend& be) {
  if (!be.useEnvRoot) return true;
  uintptr_t spAddr = uintptr_t(static_cast<void*>(&gWJRootSP));
  return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(spAddr)) &&
         GetLocal(e, be.envRootIdx) && e.writeOp(Op::I32Store) &&
         e.writeVarU32(2) && e.writeVarU32(0);
}

static bool EmitReturn(Encoder& e, WJBackend& be, MDefinition* val) {
  // Return [f64 flag=0.0, i64 boxed-result] in registers (no scratch store).
  if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(0.0)) return false;  // ok flag
  if (!GetOp(e, be, val)) return false;
  if (val->type() != MIRType::Value && val->type() != MIRType::Int64) {
    if (!EmitBoxFromStack(e, val->type())) return false;
  }
  if (!EmitEnvRootPop(e, be)) return false;
  return e.writeOp(Op::Return);  // -> [flag, result]
}

static bool EmitEdgeCopies(Encoder& e, WJBackend& be, MBasicBlock* from,
                           MBasicBlock* to) {
  uint32_t k = to->getPredecessorIndex(from);
  std::vector<int32_t> dsts;
  for (MPhiIterator p = to->phisBegin(); p != to->phisEnd(); p++) {
    MPhi* phi = *p;
    int32_t dst = be.local(phi);
    if (dst < 0) continue;
    if (!GetOp(e, be, phi->getOperand(k))) return false;  // push old value
    dsts.push_back(dst);
  }
  for (size_t i = dsts.size(); i-- > 0;) {
    if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(uint32_t(dsts[i]))) return false;
  }
  return true;
}

static bool GotoBlock(Encoder& e, uint32_t bidLocal, uint32_t loopDepth,
                      uint32_t targetIdx) {
  return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(targetIdx)) &&
         e.writeOp(Op::LocalSet) && e.writeVarU32(bidLocal) && e.writeOp(Op::Br) &&
         e.writeVarU32(loopDepth);
}

// Push i32 JS-truthiness (0/1) of the boxed Value in wasm local `v`. Covers
// int32/boolean/object/undefined/null/double; deopts to PBL for the rarer
// string/symbol/bigint (which need a length/zero check we don't emit yet).
static bool EmitValueTruthy(Encoder& e, WJBackend& be, uint32_t v) {
  const uint8_t kI32 = uint8_t(TypeCode::I32);
  auto tag = [&]() -> bool {
    return GetLocal(e, v) && e.writeOp(Op::I64Const) && e.writeVarS64(32) &&
           e.writeOp(Op::I64ShrU) && e.writeOp(Op::I32WrapI64);
  };
  auto tagEq = [&](JSValueType t) -> bool {
    return tag() && e.writeOp(Op::I32Const) &&
           e.writeVarS32(int32_t(TagWord(t))) && e.writeOp(Op::I32Eq);
  };
  if (!tagEq(JSVAL_TYPE_INT32)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kI32)) return false;
  if (!GetLocal(e, v) || !e.writeOp(Op::I32WrapI64) || !e.writeOp(Op::I32Const) ||
      !e.writeVarS32(0) || !e.writeOp(Op::I32Ne))
    return false;
  if (!e.writeOp(Op::Else)) return false;
  if (!tagEq(JSVAL_TYPE_BOOLEAN)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kI32)) return false;
  if (!GetLocal(e, v) || !e.writeOp(Op::I32WrapI64)) return false;
  if (!e.writeOp(Op::Else)) return false;
  if (!tagEq(JSVAL_TYPE_OBJECT)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kI32)) return false;
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(1)) return false;
  if (!e.writeOp(Op::Else)) return false;
  if (!tagEq(JSVAL_TYPE_UNDEFINED)) return false;
  if (!tagEq(JSVAL_TYPE_NULL)) return false;
  if (!e.writeOp(Op::I32Or)) return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kI32)) return false;
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(0)) return false;
  if (!e.writeOp(Op::Else)) return false;
  // double: tag u<= CLEAR => (d != 0) & (d == d)  [d==d is false for NaN]
  if (!tag() || !e.writeOp(Op::I32Const) ||
      !e.writeVarS32(int32_t(uint32_t(JSVAL_TAG_CLEAR))) || !e.writeOp(Op::I32LeU))
    return false;
  if (!e.writeOp(Op::If) || !e.writeFixedU8(kI32)) return false;
  if (!GetLocal(e, v) || !e.writeOp(Op::F64ReinterpretI64) ||
      !e.writeOp(Op::F64Const) || !e.writeFixedF64(0.0) || !e.writeOp(Op::F64Ne))
    return false;
  if (!GetLocal(e, v) || !e.writeOp(Op::F64ReinterpretI64) || !GetLocal(e, v) ||
      !e.writeOp(Op::F64ReinterpretI64) || !e.writeOp(Op::F64Eq))
    return false;
  if (!e.writeOp(Op::I32And)) return false;
  if (!e.writeOp(Op::Else)) return false;
  // string/symbol/bigint -> PBL. This deopt sits 5 `if(result i32)` levels deep
  // in the type chain above, so the OOL deopt br-depth needs +4 over its usual
  // single-`if` assumption (else it branches to a value block with an empty stack
  // -> invalid wasm; was the deltablue `if(this.myOutput.stay)` compile failure).
  be.deoptExtraNest = 4;
  bool dok = EmitDeopt(e, be);
  be.deoptExtraNest = 0;
  if (!dok) return false;
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(0)) return false;  // dead
  for (int i = 0; i < 5; i++)
    if (!e.writeOp(Op::End)) return false;
  return true;
}

// Emit one block's non-control instructions. Returns false on unsupported node.
static bool EmitBlockBody(Encoder& e, WJBackend& be, MBasicBlock* b) {
  // alwaysBails block: Warp couldn't type this (cold) path and marked it to bail
  // unconditionally. Emit a deopt-at-entry (resume to PBL from the block's entry
  // pc) and skip the body+terminator -- the caller checks alwaysBails() to elide
  // the terminator. Force INLINE resume (ends in `return`, no br-depth dependence
  // on the guard-if nesting the OOL path assumes). Keeps the function's hot blocks
  // compiled; this block only runs (deopts) if actually reached. (navier fix.)
  if (b->alwaysBails()) {
    gWJHadAlwaysBails = true;  // valve: a storming fn with this -> PBL, not recompile
    static int abdbg = getenv("GECKO_WJ_ABDBG") ? 1 : 0;
    be.curRp = b->entryResumePoint();
    if (!be.curRp) {
      if (abdbg) {
        JSScript* s = be.info ? be.info->script() : nullptr;
        fprintf(stderr, "[wj-abdbg] %s:%u alwaysBails block has NO entry RP\n",
                s ? s->filename() : "?", s ? unsigned(s->lineno()) : 0);
      }
      return false;
    }
    bool savedDispatch = be.inDispatchBody;
    be.inDispatchBody = false;
    bool ok = EmitDeopt(e, be);
    be.inDispatchBody = savedDispatch;
    if (!ok && abdbg) {
      JSScript* s = be.info ? be.info->script() : nullptr;
      fprintf(stderr, "[wj-abdbg] %s:%u alwaysBails EmitDeopt FAILED\n",
              s ? s->filename() : "?", s ? unsigned(s->lineno()) : 0);
    }
    return ok;
  }
  // Track the MOST-RECENT resume point in program order. A deopt at an op WITHOUT
  // its own resume point must resume from the last one SEEN (which reflects the
  // state AFTER the last committed side effect), NOT the block-entry RP -- else
  // PBL re-runs from the block start and RE-APPLIES already-committed stores
  // (randtest seed `+=`, navier loop stores -> wrong values). This is the sound
  // resume-point selection (mirrors Ion, where every bailing op has the correct
  // nearest dominating resume point).
  MResumePoint* lastRp = b->entryResumePoint();
  be.curRp = lastRp;  // valid even for an empty block (terminator uses it)
  // Shadow frame (hot path): spill the block-entry frame, then keep the shadow
  // current by spilling the delta whenever the resume point advances. A deopt
  // then reads MEMORY (gWJResumeVals) instead of the loop's wasm locals, so those
  // locals are not pinned live-out across the loop. spilledRp tracks what the
  // shadow currently reflects (null => nothing spilled yet => full spill).
  MResumePoint* spilledRp = nullptr;
  if (be.shadowDeopt && be.inDispatchBody) {
    if (!EmitFrameSpillDelta(e, be, lastRp, nullptr)) return false;
    spilledRp = lastRp;
  }
  for (MInstructionIterator it = b->begin(); it != b->end(); it++) {
    MInstruction* ins = *it;
    if (ins->isControlInstruction()) continue;
    MResumePoint* insRp = ins->resumePoint();
    if (getenv("GECKO_WJ_RPDBG") && be.info && be.info->script() &&
        be.info->script()->lineno() == uint32_t(atoi(getenv("GECKO_WJ_RPDBG")))) {
      fprintf(stderr, "[wj-rp] b%u op#%u(%s) rp=%s mode=%d\n", b->id(),
              unsigned(ins->op()), WJOpName(ins->op()), insRp ? "Y" : "n",
              insRp ? int(insRp->mode()) : -1);
    }
    // Deopt resume-point selection. Advance the deopt rp to this op's rp BEFORE
    // the op ONLY for resume-AT points: a mid-op guard then re-executes this op in
    // PBL from its own pc. For resume-AFTER points, KEEP the PRIOR rp during the op
    // -- the resume-after point's captured stack includes this op's RESULT, but a
    // mid-op guard deopts BEFORE the result is computed (before the local.set
    // below), so spilling the resume-after frame would spill the op's STALE/uninit
    // result local as the captured stack top -> PBL resumes with a garbage stack
    // value. The prior rp (the previous op's resume-after) captures all prior
    // effects and resumes just before this op, so PBL correctly re-executes ONLY
    // this op. (deltablue's LoadDynamicSlotAndUnbox fallible-unbox tag-guard hit
    // this: the bad spilled result fed the constraint solver -> "Cycle encountered"
    // / hang.) Advance to insRp AFTER the op completes (its result now in-local).
    bool spillAfter = insRp && IsResumeAfter(insRp->mode());
    if (insRp && !spillAfter) lastRp = insRp;
    be.curRp = lastRp;
    be.curOp = uint32_t(ins->op());
    bool doShadow = be.shadowDeopt && be.inDispatchBody;
    if (doShadow && !spillAfter && lastRp != spilledRp) {
      if (!EmitFrameSpillDelta(e, be, lastRp, spilledRp)) return false;
      spilledRp = lastRp;
    }
    int32_t l = be.local(ins);
    if (l < 0) {
      // No value local: either a no-op typed node or an effectful one.
      EffectKind k = EmitEffect(e, be, ins);
      if (k == EffectKind::Fail) {
        fprintf(stderr, "[wb-be] unsupported effect op#%u\n", unsigned(ins->op()));
        return false;
      }
    } else {
      if (!EmitValue(e, be, ins)) {
        fprintf(stderr, "[wb-be] unsupported value op#%u type%u\n",
                unsigned(ins->op()), unsigned(ins->type()));
        return false;
      }
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(uint32_t(l))) return false;
    }
    // Resume-AFTER op completed: its result is now in-local, so the resume-after
    // frame is consistent. Advance the deopt rp for subsequent ops, the terminator
    // (which keeps the body's last rp), + the shadow post-spill below.
    if (spillAfter) {
      lastRp = insRp;
      be.curRp = lastRp;
    }
    if (doShadow && spillAfter && lastRp != spilledRp) {
      if (!EmitFrameSpillDelta(e, be, lastRp, spilledRp)) return false;
      spilledRp = lastRp;
    }
  }
  return true;
}

// Emit a block terminator. loopDepth = depth of the dispatch loop $L from here.
static bool EmitTerminator(Encoder& e, WJBackend& be,
                           std::unordered_map<MBasicBlock*, uint32_t>& blockIdx,
                           uint32_t bidLocal, uint32_t loopDepth, MBasicBlock* b) {
  MControlInstruction* t = b->lastIns();
  if (t->resumePoint()) be.curRp = t->resumePoint();  // else keep body last RP
  if (t->isReturn()) {
    return EmitReturn(e, be, t->getOperand(0));
  }
  if (t->isUnreachable() || t->isThrow()) {
    // A reachable Throw/Unreachable means deopt to PBL to run real semantics.
    return EmitDeopt(e, be);
  }
  if (t->isGoto()) {
    MBasicBlock* s = t->toGoto()->target();
    if (!EmitEdgeCopies(e, be, b, s)) return false;
    return GotoBlock(e, bidLocal, loopDepth, blockIdx[s]);
  }
  if (t->isTest()) {
    MTest* test = t->toTest();
    MDefinition* cond = test->getOperand(0);
    MBasicBlock* tb = test->ifTrue();
    MBasicBlock* fb = test->ifFalse();
    if (WJValType(cond->type()) == uint8_t(TypeCode::I32)) {
      if (!GetOp(e, be, cond)) return false;
    } else if (cond->type() == MIRType::Value) {
      int32_t cl = be.local(cond);
      if (cl < 0) return false;
      if (!EmitValueTruthy(e, be, uint32_t(cl))) return false;
    } else {
      return false;
    }
    if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
    if (!EmitEdgeCopies(e, be, b, tb)) return false;
    if (!GotoBlock(e, bidLocal, loopDepth + 1, blockIdx[tb])) return false;
    if (!e.writeOp(Op::Else)) return false;
    if (!EmitEdgeCopies(e, be, b, fb)) return false;
    if (!GotoBlock(e, bidLocal, loopDepth + 1, blockIdx[fb])) return false;
    if (!e.writeOp(Op::End)) return false;
    return e.writeOp(Op::Unreachable);
  }
  return false;
}

// ---------------------------------------------------------------------------
// Structured-control-flow emitter ("relooper"). Reconstructs nested
// loop/block/if from the reducible MIR CFG so V8 sees real loops -- letting it
// register-allocate across iterations and keep the values OptimizeMIR's LICM
// already hoisted, which the flat br_table dispatch loop defeats. Each block
// gets a wasm `block` scope ending at it (so forward branches become `br`);
// each loop header also gets a `loop` scope (back-edges `br` to it). Begins are
// pulled back until all scopes nest. Returns false either before emitting
// (sets *started=false -> caller uses the dispatch loop) or mid-emit on an
// unsupported node (*started=true -> caller bails the whole compile).
static bool WJEmitStructured(Encoder& e, WJBackend& be,
                             std::vector<MBasicBlock*>& blocks,
                             std::unordered_map<MBasicBlock*, uint32_t>& blockIdx,
                             bool* started) {
  *started = false;
  uint32_t n = uint32_t(blocks.size());
  std::vector<int32_t> idomIdx(n, -1);
  std::vector<int32_t> loopEnd(n, -1);  // loopEnd[h] = end pos if h is a header
  for (uint32_t i = 0; i < n; i++) {
    MBasicBlock* d = blocks[i]->immediateDominator();
    if (d) {
      auto it = blockIdx.find(d);
      if (it != blockIdx.end()) idomIdx[i] = int32_t(it->second);
    }
    if (blocks[i]->isLoopHeader()) {
      MBasicBlock* bk = blocks[i]->backedge();
      if (!bk) return false;
      auto bit = blockIdx.find(bk);
      if (bit == blockIdx.end()) return false;
      loopEnd[i] = int32_t(bit->second) + 1;
    }
  }
  std::vector<uint32_t> bbegin(n, 0);  // block-scope begin position per block
  for (uint32_t t = 1; t < n; t++) {
    if (idomIdx[t] < 0) {
      js::wasm::gWJBailReason = "no-idom";
      return false;  // unreachable/no idom: bail to dispatch
    }
    bbegin[t] = uint32_t(idomIdx[t]);
  }
  // Nesting fix-up: two scopes [a,b) [c,d) improperly overlap iff a<c<b<d (or
  // symmetric). Pull a block scope's begin back to the other's begin until none
  // overlap. Loop begins are fixed (must sit at the header).
  auto improper = [](uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    return (a < c && c < b && b < d) || (c < a && a < d && d < b);
  };
  for (uint32_t iter = 0; iter <= n; iter++) {
    bool changed = false;
    for (uint32_t t = 1; t < n; t++) {
      for (uint32_t u = 1; u < n; u++) {
        if (u != t && improper(bbegin[u], u, bbegin[t], t)) {
          bbegin[t] = bbegin[u];
          changed = true;
        }
      }
      for (uint32_t h = 0; h < n; h++) {
        if (loopEnd[h] >= 0 &&
            improper(h, uint32_t(loopEnd[h]), bbegin[t], t)) {
          bbegin[t] = h;
          changed = true;
        }
      }
    }
    if (!changed) break;
  }
  struct Rec {
    uint32_t end;
    bool isLoop;
    uint32_t header;
  };
  std::vector<std::vector<Rec>> beginsAt(n + 1), endsAt(n + 1);
  for (uint32_t t = 1; t < n; t++) {
    Rec r{t, false, 0};
    beginsAt[bbegin[t]].push_back(r);
    endsAt[t].push_back(r);
  }
  for (uint32_t h = 0; h < n; h++) {
    if (loopEnd[h] < 0) continue;
    Rec r{uint32_t(loopEnd[h]), true, h};
    beginsAt[h].push_back(r);
    endsAt[uint32_t(loopEnd[h])].push_back(r);
  }
  // Open outermost (largest end) first; close innermost (smallest end) first.
  for (uint32_t p = 0; p <= n; p++) {
    std::sort(beginsAt[p].begin(), beginsAt[p].end(),
              [](const Rec& a, const Rec& b) { return a.end > b.end; });
    std::sort(endsAt[p].begin(), endsAt[p].end(),
              [](const Rec& a, const Rec& b) { return a.end < b.end; });
  }

  *started = true;
  std::vector<Rec> stk;
  auto depthOfBlock = [&](uint32_t tIdx) -> int32_t {
    for (int32_t k = int32_t(stk.size()) - 1; k >= 0; k--)
      if (!stk[k].isLoop && stk[k].end == tIdx) return int32_t(stk.size()) - 1 - k;
    return -1;
  };
  auto depthOfLoop = [&](uint32_t hIdx) -> int32_t {
    for (int32_t k = int32_t(stk.size()) - 1; k >= 0; k--)
      if (stk[k].isLoop && stk[k].header == hIdx) return int32_t(stk.size()) - 1 - k;
    return -1;
  };
  auto brTo = [&](MBasicBlock* s, uint32_t pos, uint32_t extra) -> int32_t {
    uint32_t sIdx = blockIdx[s];
    int32_t d = (sIdx > pos) ? depthOfBlock(sIdx) : depthOfLoop(sIdx);
    return d < 0 ? -1 : int32_t(uint32_t(d) + extra);
  };
  const uint8_t kVoid = 0x40;
  for (uint32_t pos = 0; pos <= n; pos++) {
    for (size_t i = 0; i < endsAt[pos].size(); i++) {
      if (stk.empty()) return false;
      stk.pop_back();
      if (!e.writeOp(Op::End)) return false;
    }
    for (Rec& r : beginsAt[pos]) {
      if (!e.writeOp(r.isLoop ? Op::Loop : Op::Block) || !e.writeFixedU8(kVoid))
        return false;
      stk.push_back(r);
    }
    if (pos == n) break;
    MBasicBlock* b = blocks[pos];
    if (!EmitBlockBody(e, be, b)) return false;
    if (b->alwaysBails()) continue;  // emitted its own deopt (returns); no terminator
    MControlInstruction* t = b->lastIns();
    if (t->resumePoint()) be.curRp = t->resumePoint();  // else keep body last RP
    if (t->isReturn()) {
      if (!EmitReturn(e, be, t->getOperand(0))) return false;
    } else if (t->isUnreachable() || t->isThrow()) {
      if (!EmitDeopt(e, be)) return false;
    } else if (t->isGoto()) {
      MBasicBlock* s = t->toGoto()->target();
      if (!EmitEdgeCopies(e, be, b, s)) return false;
      int32_t d = brTo(s, pos, 0);
      if (d < 0) { js::wasm::gWJBailReason = "reloop-goto-depth"; return false; }
      if (!e.writeOp(Op::Br) || !e.writeVarU32(uint32_t(d))) return false;
    } else if (t->isTest()) {
      MTest* test = t->toTest();
      MDefinition* cond = test->getOperand(0);
      if (WJValType(cond->type()) != uint8_t(TypeCode::I32)) {
        // Value-typed conditions need EmitValueTruthy; deferred -- the relooper has
        // a separate br-depth bug (invalid wasm) that surfaced when these compiled,
        // and it wasn't faster than dispatch anyway. Bail to dispatch.
        js::wasm::gWJBailReason = "reloop-test-cond";
        return false;
      }
      MBasicBlock* tb = test->ifTrue();
      MBasicBlock* fb = test->ifFalse();
      if (!GetOp(e, be, cond)) return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
      if (!EmitEdgeCopies(e, be, b, tb)) return false;
      int32_t dt = brTo(tb, pos, 1);
      if (dt < 0) { js::wasm::gWJBailReason = "reloop-test-depth-t"; return false; }
      if (!e.writeOp(Op::Br) || !e.writeVarU32(uint32_t(dt))) return false;
      if (!e.writeOp(Op::Else)) return false;
      if (!EmitEdgeCopies(e, be, b, fb)) return false;
      int32_t df = brTo(fb, pos, 1);
      if (df < 0) { js::wasm::gWJBailReason = "reloop-test-depth-f"; return false; }
      if (!e.writeOp(Op::Br) || !e.writeVarU32(uint32_t(df))) return false;
      if (!e.writeOp(Op::End)) return false;
    } else {
      js::wasm::gWJBailReason =
          t->isTableSwitch() ? "reloop-tableswitch" : "reloop-terminator";
      return false;
    }
  }
  if (!stk.empty()) return false;
  if (!e.writeOp(Op::Unreachable)) return false;
  return e.writeOp(Op::End);  // function body end
}

}  // namespace

bool js::wasm::WJEmitTrampoline(Encoder& e) {
  // locals: i32 sbI32 (1), f64 flag (2), i64 result (3). Param 0 = scratch-base.
  if (!e.writeVarU32(3) || !e.writeVarU32(1) ||
      !e.writeFixedU8(uint8_t(TypeCode::I32)) || !e.writeVarU32(1) ||
      !e.writeFixedU8(uint8_t(TypeCode::F64)) || !e.writeVarU32(1) ||
      !e.writeFixedU8(uint8_t(TypeCode::I64))) {
    return false;
  }
  // sbI32 (local 1) = (i32)param0
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(0)) return false;
  if (!e.writeOp(MiscOp::I32TruncSatF64S)) return false;
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(1)) return false;
  // Push main's args: sb (f64), this (i64), arg0..arg[kWJMaxArgs-1] (i64).
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(0)) return false;  // sb
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(1) || !e.writeOp(Op::I64Load) ||
      !e.writeVarU32(3) || !e.writeVarU32(kWJThisOff)) {
    return false;  // this
  }
  for (uint32_t i = 0; i < kWJMaxArgs; i++) {
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(1) ||
        !e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(i * 8)) {
      return false;  // arg i
    }
  }
  if (!e.writeOp(Op::Call) || !e.writeVarU32(1)) return false;  // main -> [flag,result]
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(3)) return false;  // result
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(2)) return false;  // flag
  // Store the boxed result to gWJScratch[kWJResultOff] for the host entry path.
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(1) || !e.writeOp(Op::LocalGet) ||
      !e.writeVarU32(3) || !e.writeOp(Op::I64Store) || !e.writeVarU32(3) ||
      !e.writeVarU32(kWJResultOff)) {
    return false;
  }
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(2)) return false;  // return flag
  return e.writeOp(Op::End);
}

bool js::wasm::WJEmitBody(MIRGenerator& mir, MIRGraph& graph, uint32_t nargs,
                          bool useThis, WarpSnapshot* snapshot, Encoder& e) {
  const uint8_t kVoid = 0x40;
  {
    JSScript* brs = mir.outerInfo().script();
    js::wasm::gWJBailLine = brs ? brs->lineno() : 0;
    js::wasm::gWJBailReason = "unknown";
  }
  // Bisection: GECKO_WJ_BAILLINE=<n[,n...]> bails (stays PBL) any function defined
  // at one of those source lines. Used to find which function's JIT codegen is
  // wrong (e.g. navier advect/project miscompute exposed by the number-unbox).
  if (const char* bl = getenv("GECKO_WJ_BAILLINE")) {
    JSScript* s = mir.outerInfo().script();
    if (s) {
      uint32_t ln = s->lineno();
      for (const char* p = bl; *p;) {
        uint32_t v = uint32_t(atoi(p));
        if (v == ln) return false;
        while (*p && *p != ',') p++;
        if (*p == ',') p++;
      }
    }
  }
  // Threshold bisection (for big files like earley): BAILGT=N bails fns at
  // lineno > N; BAILLT=N bails fns at lineno < N. Binary-search N to localize a
  // miscompiled function's region.
  {
    JSScript* s = mir.outerInfo().script();
    if (s) {
      uint32_t ln = s->lineno();
      if (const char* g = getenv("GECKO_WJ_BAILGT"))
        if (ln > uint32_t(atoi(g))) return false;
      if (const char* l = getenv("GECKO_WJ_BAILLT"))
        if (ln < uint32_t(atoi(l))) return false;
      // BAILRANGE=lo,hi bails fns with lo<=lineno<=hi (keeps the rest JIT'd, so
      // the run stays fast while bisecting a sub-region of a big file).
      if (const char* r = getenv("GECKO_WJ_BAILRANGE")) {
        uint32_t lo = uint32_t(atoi(r));
        const char* c = r;
        while (*c && *c != ',') c++;
        uint32_t hi = (*c == ',') ? uint32_t(atoi(c + 1)) : lo;
        if (ln >= lo && ln <= hi) return false;
      }
    }
  }
  // MIR dumper: GECKO_WJ_MIRDUMP=<lineno> prints every instruction (op + operand
  // ops + type) of the function defined at that line -- to inspect a specific
  // codegen pattern (e.g. closure-variable reads after FunctionEnvironment).
  if (const char* md = getenv("GECKO_WJ_MIRDUMP")) {
    JSScript* s = mir.outerInfo().script();
    fprintf(stderr, "[wb-mir-fn] compiling lineno=%u (want %s)\n",
            s ? s->lineno() : 0, md);
    if (s && s->lineno() == uint32_t(atoi(md))) {
      fprintf(stderr, "[wb-mir] === %s:%u ===\n", s->filename(), s->lineno());
      for (ReversePostorderIterator b = graph.rpoBegin(); b != graph.rpoEnd();
           b++) {
        fprintf(stderr, "[wb-mir] block%u (script %u)\n", b->id(),
                uint32_t(b->info().script()->lineno()));
        for (MInstructionIterator it = b->begin(); it != b->end(); it++) {
          char ops[256];
          int n = 0;
          for (size_t i = 0; i < it->numOperands() && n < 200; i++) {
            n += snprintf(ops + n, sizeof(ops) - n, "%s ",
                          WJOpName(it->getOperand(i)->op()));
          }
          fprintf(stderr, "[wb-mir]   %s <- %s\n", WJOpName(it->op()), ops);
        }
      }
    }
  }
  // Bisection knobs: bail (stay PBL) any function containing a disabled op kind.
  static int noCall = getenv("GECKO_WJ_NOCALL") ? 1 : 0;
  static int noStore = getenv("GECKO_WJ_NOSTORE") ? 1 : 0;
  static int noProp = getenv("GECKO_WJ_NOPROP") ? 1 : 0;
  static int noObjConst = getenv("GECKO_WJ_NOOBJCONST") ? 1 : 0;
  static int noGuardShape = getenv("GECKO_WJ_NOGUARDSHAPE") ? 1 : 0;
  if (noCall || noStore || noProp || noObjConst || noGuardShape) {
    for (ReversePostorderIterator b = graph.rpoBegin(); b != graph.rpoEnd(); b++) {
      for (MInstructionIterator it = b->begin(); it != b->end(); it++) {
        auto op = it->op();
        if (noCall && op == MDefinition::Opcode::Call) return false;
        if (noGuardShape && op == MDefinition::Opcode::GuardShape) return false;
        if (noStore && op == MDefinition::Opcode::StoreFixedSlot) return false;
        if (noObjConst && op == MDefinition::Opcode::Constant &&
            it->type() == MIRType::Object)
          return false;
        if (noProp && (op == MDefinition::Opcode::GuardShape ||
                       op == MDefinition::Opcode::Slots ||
                       op == MDefinition::Opcode::LoadFixedSlot ||
                       op == MDefinition::Opcode::LoadFixedSlotAndUnbox ||
                       op == MDefinition::Opcode::LoadDynamicSlotAndUnbox ||
                       op == MDefinition::Opcode::Elements))
          return false;
      }
    }
  }

  gWJHadAlwaysBails = false;  // set true if any block is emitted as alwaysBails deopt
  WJBackend be;
  be.info = &mir.outerInfo();
  be.snapshot = snapshot;
  // Params: 0 = scratch-base (f64), 1 = this (i64), 2..(1+kWJMaxArgs) = args (i64).
  be.paramCount = 2 + kWJMaxArgs;
  be.scratchBase = be.reserve(uint8_t(TypeCode::I32));  // first declared local
  be.unboxScratch = be.reserve(uint8_t(TypeCode::I64));  // unbox temp for fused loads
  be.callScratch = be.reserve(uint8_t(TypeCode::I32));   // call-site IC callee ptr
  be.callArgBase = uint32_t(be.reserve(uint8_t(TypeCode::I64)));  // boxed this
  for (uint32_t i = 0; i < kWJMaxArgs; i++) {
    be.reserve(uint8_t(TypeCode::I64));  // boxed arg i
  }
  be.callFlagLocal = uint32_t(be.reserve(uint8_t(TypeCode::F64)));
  be.callResultLocal = uint32_t(be.reserve(uint8_t(TypeCode::I64)));
  be.envLocal = uint32_t(be.reserve(uint8_t(TypeCode::I32)));  // saved env (see struct)
  be.envRootIdx = uint32_t(be.reserve(uint8_t(TypeCode::I32)));  // persistent env-root slot idx
  be.deoptSiteLocal = uint32_t(be.reserve(uint8_t(TypeCode::I32)));  // out-of-line deopt site id
  be.propObjLocal = uint32_t(be.reserve(uint8_t(TypeCode::I32)));   // prop-IC: receiver ptr
  be.propShapeLocal = uint32_t(be.reserve(uint8_t(TypeCode::I32)));  // prop-IC: receiver shape
  be.propTaggedLocal = uint32_t(be.reserve(uint8_t(TypeCode::I32)));  // prop-IC: tagged slot offset
  be.allocPosLocal = uint32_t(be.reserve(uint8_t(TypeCode::I32)));    // inline-alloc old position
  be.allocObjLocal = uint32_t(be.reserve(uint8_t(TypeCode::I32)));    // inline-alloc result obj
  be.rootBaseLocal = uint32_t(be.reserve(uint8_t(TypeCode::I32)));    // GC-root spill base addr

  std::vector<MBasicBlock*> blocks;
  std::unordered_map<MBasicBlock*, uint32_t> blockIdx;
  bool anyAlwaysBails = false;
  for (ReversePostorderIterator b = graph.rpoBegin(); b != graph.rpoEnd(); b++) {
    // A block that always bails (Warp gave up on a COLD path it couldn't type --
    // e.g. navier lin_solve's `if (a===0)` branch, never taken at runtime) is
    // emitted as a deopt-at-entry (resume to PBL) instead of bailing the WHOLE
    // function to PBL. This mirrors Ion, which keeps the hot path compiled and
    // only bails the cold block if it is ever reached. Bailing the whole function
    // here was the navier 1x ceiling: lin_solve/advect/project all have such a
    // cold branch, so none of them compiled. See EmitBlockBody (alwaysBails ->
    // EmitDeopt). The entry block bailing means the function always deopts on
    // entry, so there's no hot path to preserve -- bail it whole.
    if (b->alwaysBails()) anyAlwaysBails = true;
    blockIdx[*b] = uint32_t(blocks.size());
    blocks.push_back(*b);
  }
  const uint32_t n = uint32_t(blocks.size());
  if (n == 0 || blocks[0]->alwaysBails()) {
    js::wasm::gWJBailReason = (n == 0) ? "no-blocks" : "entry-alwaysBails";
    return false;
  }

  // Assign locals: phis first (edge copies target them), then value nodes.
  for (MBasicBlock* b : blocks) {
    for (MPhiIterator p = b->phisBegin(); p != b->phisEnd(); p++) {
      if (WJValType((*p)->type()) != 0) be.assign(*p);
    }
  }
  uint32_t maxLoopDepth = 0;
  bool hasCalls = false;
  for (MBasicBlock* b : blocks) {
    if (b->loopDepth() > maxLoopDepth) maxLoopDepth = b->loopDepth();
    for (MInstructionIterator it = b->begin(); it != b->end(); it++) {
      MInstruction* ins = *it;
      if (ins->op() == MDefinition::Opcode::FunctionEnvironment)
        be.useEnvRoot = true;  // env in envLocal must be rooted across calls
      switch (ins->op()) {
        case MDefinition::Opcode::Call:
        case MDefinition::Opcode::GetPropertyCache:
        case MDefinition::Opcode::SetPropertyCache:
          hasCalls = true;  // reloop's structural overhead is a loss when a
          break;            // C++/helper hop dominates the loop (richards)
        default:
          break;
      }
      if (ins->isControlInstruction()) continue;
      if (WJValType(ins->type()) == 0) continue;
      be.assign(ins);
    }
  }

  // alwaysBails-as-deopt: Warp marks a block alwaysBails when it ends in a cold
  // IC (never-profiled) whose transpile yields buildBailoutForColdIC. Ion handles
  // this by compiling the hot path and emitting a cheap bailout-to-Baseline for
  // the cold block; we mirror that by converting the alwaysBails block to a
  // deopt-at-block-entry (skip body+terminator) and compiling the rest of the fn.
  // This is the Ion-correct behavior -- a cold block should NOT bail the whole
  // function to the interpreter. The deopt-under-fast-call GC crash this used to
  // guard against is FIXED (always-push a fresh JitActivation in
  // WasmJitResumeViaPBL). GECKO_WJ_ABGATE=1 restores the old whole-function bail
  // (shallow-loop + alwaysBails -> PBL) for A/B debugging.
  if (anyAlwaysBails && maxLoopDepth < 2 && getenv("GECKO_WJ_ABGATE")) {
    js::wasm::gWJBailReason = "ab-gate";
    return false;
  }

  // Diagnostic: per-function MIR opcode histogram (GECKO_WJ_OPHIST). Used to see
  // whether hot property/array accesses fold to inline loads (LoadFixedSlot/
  // LoadElement) vs fall to C++ helper hops (GetPropertyCache/MegamorphicLoadSlot)
  // -- the codegen-quality ceiling for splay/navier/raytrace.
  static int opHist = getenv("GECKO_WJ_OPHIST") ? 1 : 0;
  if (opHist) {
    using Op_ = MDefinition::Opcode;
    auto cat = [](Op_ o) -> const char* {
      switch (o) {
        case Op_::GetPropertyCache: return "GetPropCache.HELPER";
        case Op_::SetPropertyCache: return "SetPropCache.HELPER";
        case Op_::MegamorphicLoadSlot: return "MegaLoad.HELPER";
        case Op_::MegamorphicSetElement: return "MegaSet.HELPER";
        case Op_::CallGetIntrinsicValue: return "other";
        case Op_::LoadFixedSlot: return "LoadFixedSlot.inline";
        case Op_::LoadFixedSlotAndUnbox: return "LoadFixedSlotUnbox.inline";
        case Op_::StoreFixedSlot: return "StoreFixedSlot.inline";
        case Op_::LoadDynamicSlot: return "LoadDynSlot.inline";
        case Op_::LoadDynamicSlotAndUnbox: return "LoadDynSlotUnbox.inline";
        case Op_::StoreDynamicSlot: return "StoreDynSlot.inline";
        case Op_::LoadElement: return "LoadElement.inline";
        case Op_::LoadElementAndUnbox: return "LoadElemUnbox.inline";
        case Op_::StoreElement: return "StoreElement.inline";
        case Op_::Call: return "Call";
        case Op_::GuardShape: return "GuardShape";
        case Op_::Unbox: return "Unbox";
        case Op_::Box: return "Box";
        default: return "other";
      }
    };
    std::map<std::string, uint32_t> hist;
    uint32_t total = 0;
    for (MBasicBlock* b : blocks) {
      for (MInstructionIterator it = b->begin(); it != b->end(); it++) {
        hist[cat((*it)->op())]++;
        total++;
      }
    }
    JSScript* s = mir.outerInfo().script();
    fprintf(stderr, "[wb-ophist] %s:%u total=%u", s ? s->filename() : "?",
            s ? s->lineno() : 0, total);
    for (auto& kv : hist) fprintf(stderr, " %s=%u", kv.first.c_str(), kv.second);
    fprintf(stderr, "\n");
  }

  const uint32_t bidLocal = be.paramCount + uint32_t(be.localTy.size());

  // Locals declaration: each declared local as its own group, plus $bid (i32).
  if (!e.writeVarU32(uint32_t(be.localTy.size()) + 1)) return false;
  for (uint8_t ty : be.localTy) {
    if (!e.writeVarU32(1) || !e.writeFixedU8(ty)) return false;
  }
  if (!e.writeVarU32(1) || !e.writeFixedU8(uint8_t(TypeCode::I32))) return false;

  // Prologue: scratchBase = (i32)param0.
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(0)) return false;
  if (!e.writeOp(MiscOp::I32TruncSatF64S)) return false;
  if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.scratchBase)) return false;

  // Prologue: envLocal = gWJCurrentEnv. The caller (entry trampoline, WJH_CALL,
  // or the fast inline call_indirect) sets gWJCurrentEnv to THIS invocation's
  // environment right before transferring control. Saving it to a local now --
  // before any nested call can overwrite the global -- makes MFunctionEnvironment
  // sound for closures (each invocation keeps its own env).
  {
    uintptr_t addr = uintptr_t(static_cast<void*>(&gWJCurrentEnv));
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(addr)) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
        !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.envLocal))
      return false;
  }
  // Persistently GC-root the env for closures: push boxed(envLocal) onto the
  // GC-root shadow stack at the current gWJRootSP, remember the slot index in
  // envRootIdx, and bump gWJRootSP. The slot is traced on EVERY GC (minor+major)
  // for the function's whole lifetime, so FunctionEnvironment can rematerialize a
  // GC-current env even across non-call GC points (allocations). Popped at exit.
  if (be.useEnvRoot) {
    uintptr_t spAddr = uintptr_t(static_cast<void*>(&gWJRootSP));
    uintptr_t rootsBase = uintptr_t(static_cast<void*>(&gWJCallRoots[0]));
    // envRootIdx = gWJRootSP
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(spAddr)) ||
        !e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(0) ||
        !e.writeOp(Op::LocalSet) || !e.writeVarU32(be.envRootIdx))
      return false;
    // gWJCallRoots[envRootIdx] = box(envLocal):  addr = rootsBase + envRootIdx*8
    if (!GetLocal(e, be.envRootIdx) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(8) || !e.writeOp(Op::I32Mul) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(int32_t(rootsBase)) || !e.writeOp(Op::I32Add))
      return false;
    if (!GetLocal(e, be.envLocal) || !EmitBoxFromStack(e, MIRType::Object) ||
        !e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
      return false;
    // gWJRootSP = envRootIdx + 1
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(spAddr)) ||
        !GetLocal(e, be.envRootIdx) || !e.writeOp(Op::I32Const) ||
        !e.writeVarS32(1) || !e.writeOp(Op::I32Add) || !e.writeOp(Op::I32Store) ||
        !e.writeVarU32(2) || !e.writeVarU32(0))
      return false;
  }

  // Initialize each Parameter's COPY local from its raw wasm param at the very
  // top of the body -- before any block-entry spill, call, or deopt. GetOp/spill
  // for a Parameter then read this copy local, which WJCollectRoots reloads across
  // every call/GC point, so a minor GC that moves an arg object (earley/nboyer
  // `new sc_Pair`) can't leave a stale raw param behind. Params are always boxed
  // i64 Values, so a plain LocalGet pidx -> LocalSet copy suffices.
  for (MInstructionIterator it = graph.entryBlock()->begin();
       it != graph.entryBlock()->end(); it++) {
    if (it->op() != MDefinition::Opcode::Parameter) continue;
    MParameter* p = it->toParameter();
    int32_t cl = be.local(p);
    if (cl < 0) continue;
    if (p->index() != MParameter::THIS_SLOT &&
        uint32_t(p->index()) >= kWJMaxArgs)
      continue;
    uint32_t pidx = (p->index() == MParameter::THIS_SLOT)
                        ? 1
                        : (2 + uint32_t(p->index()));
    if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(pidx)) return false;
    if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(uint32_t(cl))) return false;
  }

  static int cfInfo = getenv("GECKO_WJ_CFINFO") ? 1 : 0;
  size_t cfStartOff = e.currentOffset();
  uint32_t cfLineno = 0;
  if (cfInfo) {
    JSScript* s = mir.outerInfo().script();
    cfLineno = s ? s->lineno() : 0;
  }
  auto cfReport = [&](const char* path) {
    if (cfInfo) {
      fprintf(stderr, "[wb-cf] lineno=%u n=%u path=%s bodybytes=%zu\n", cfLineno,
              n, path, e.currentOffset() - cfStartOff);
    }
  };

  // Single-block fast path: straight-line, no dispatch loop.
  if (n == 1) {
    MBasicBlock* b = blocks[0];
    if (!EmitBlockBody(e, be, b)) return false;
    MControlInstruction* t = b->lastIns();
    if (t->resumePoint()) be.curRp = t->resumePoint();  // else keep body last RP
    if (t->isReturn()) {
      if (!EmitReturn(e, be, t->getOperand(0))) return false;
    } else if (t->isUnreachable() || t->isThrow()) {
      if (!EmitDeopt(e, be)) return false;
    } else {
      js::wasm::gWJBailReason = "single-bad-term";
      return false;
    }
    cfReport("single");
    return e.writeOp(Op::End);
  }

  // Structured control flow (relooper): real wasm loops/blocks/ifs, so the host
  // VM optimizes the loop natively instead of paying the br_table dispatch loop's
  // per-edge overhead (measured ~1.2x faster on splay's tree traversal, faster on
  // crypto). The relooper mis-handles NESTED loops on some CFGs (wrong control
  // flow -> e.g. navier advect wrong results), so restrict it to functions whose
  // max loop depth <= 1 (single, non-nested loops) -- correct there, and that's
  // where the tight-loop win is. Force on/off with GECKO_WJ_RELOOP / NORELOOP.
  // Relooper: natural wasm loops avoid the br_table dispatch loop's per-edge
  // overhead. This is a big win for TIGHT compute loops where the dispatch cost
  // is a large fraction of per-iteration work (crypto am3, splay tree ops: up to
  // 1.5x). But for call-heavy loops (richards) the call dominates, so reloop's
  // structural overhead is a net LOSS (regressed richards 1.67->0.91x). And the
  // relooper mis-handles NESTED loops (navier advect -> wrong results). So gate:
  // reloop only when maxLoopDepth<=1 AND the function makes no calls. Force/
  // disable with GECKO_WJ_RELOOP / GECKO_WJ_NORELOOP.
  // Auto-gate: enable the relooper for functions whose max loop depth <= 1 (no
  // nested loops -- the relooper mis-handles those AND is 20x slower there) AND
  // which make no calls (a call dominates the loop, so reloop's structure is a net
  // loss -- regressed richards). That's exactly where the natural-wasm-loop win
  // lands (crypto am3, splay tree ops). GECKO_WJ_RELOOP forces on; NORELOOP forces
  // off. If reloop bails mid-emit we fall through to the dispatch loop (started is
  // only set true once WJEmitStructured commits, in which case we must bail).
  // Megamorphic recompile flag MUST be set before the relooper branch: small
  // loopless call-less functions (deltablue output()/input()/isSatisfied) take the
  // reloop path, which returns before the dispatch-loop setup below -- if forceMega
  // were set only there, the relooper would emit the deopting GuardShape on a
  // polymorphic receiver and storm (deltablue 414 was here). Setting it up here
  // makes BOTH paths mega-convert.
  // Megamorphic codegen is now the DEFAULT for every compile (not just post-storm
  // recompiles): shape-guarded property reads/stores become self-guarding by-name
  // EmitPropIC (per-site shape cache + helper fallback, NO deopt on a polymorphic
  // receiver). This is the keystone that made deltablue correct AND fast (BROKEN
  // /timeout -> 109 = ~9x): the valve's PARTIAL mega left some reads as fixed-slot
  // under removed guards (wrong slot -> "Cycle"); full mega is consistent. Measured
  // no regression to the others (crypto 124->174, raytrace 40->52, richards/splay/
  // navier comparable). GECKO_WJ_NOMEGA reverts to fixed-slot+deopt codegen.
  be.forceMega = !getenv("GECKO_WJ_NOMEGA");
  static int forceReloop = getenv("GECKO_WJ_RELOOP") ? 1 : 0;
  static int noReloop = getenv("GECKO_WJ_NORELOOP") ? 1 : 0;
  bool useReloop = forceReloop ||
                   (!noReloop && maxLoopDepth <= 1 && !hasCalls);
  if (useReloop) {
    bool started = false;
    if (WJEmitStructured(e, be, blocks, blockIdx, &started)) {
      cfReport("reloop");
      return true;
    }
    if (started) {
      if (getenv("GECKO_WJ_CFINFO"))
        fprintf(stderr, "[wb-cf] lineno=%u reloop-bail reason=%s\n", cfLineno,
                js::wasm::gWJBailReason ? js::wasm::gWJBailReason : "?");
      cfReport("reloop-bail");
      js::wasm::gWJBailReason = "reloop-bail";
      return false;  // partially emitted -> bail this compile
    }
    cfReport("reloop-nostart");
  }

  // Out-of-line deopt: wrap the dispatch loop in `block $D`. A guard miss in a
  // hot block body sets deoptSiteLocal and `br $D`, jumping to the dispatcher
  // emitted AFTER the loop -- so the hot loop body stays tiny (no inline spill
  // bloat). Default on; GECKO_WJ_NOOOL falls back to inline deopt everywhere.
  static int noOol = getenv("GECKO_WJ_NOOOL") ? 1 : 0;
  be.oolDeopt = !noOol;
  // Shadow frame: spill single-frame resume operands to memory in the hot path so
  // deopts read memory, not pinned locals. Gated (GECKO_WJ_SHADOW) until validated.
  static int shadow = getenv("GECKO_WJ_SHADOW") ? 1 : 0;
  be.shadowDeopt = shadow && be.oolDeopt;
  if (be.shadowDeopt) {
    // ONE-TIME prologue spill of the function-level invariants (this + the
    // incoming args) to their fixed shadow slots [this@0, arg_i@1+i]. These never
    // change unless an arg is reassigned (then the per-iteration delta re-spills
    // that slot -- see EmitFrameSpillDelta, which skips slots still holding their
    // Parameter). Spilling them ONCE keeps them OUT of the hot loop, so they are
    // not re-read every iteration (which would re-pin them, defeating the shadow).
    uintptr_t valsBase = uintptr_t(static_cast<void*>(&gWJResumeVals[0]));
    uint32_t nargs = mir.outerInfo().nargs();
    for (uint32_t i = 0; i < 1 + nargs; i++) {  // slot 0 = this, 1.. = args
      uint32_t pidx = (i == 0) ? 1 : (1 + i);  // wasm param: this=1, arg j=2+j
      if (i > 0 && (i - 1) >= kWJMaxArgs) break;
      if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(valsBase + i * 8)) ||
          !GetLocal(e, pidx) || !e.writeOp(Op::I64Store) || !e.writeVarU32(3) ||
          !e.writeVarU32(0))
        return false;
    }
  }
  if (be.oolDeopt) {
    if (!e.writeOp(Op::Block) || !e.writeFixedU8(kVoid)) return false;  // $D
  }

  // Dispatch loop: loop $L { block $b0 {...{ block $b_{n-1} { br_table }}}}.
  if (!e.writeOp(Op::Loop) || !e.writeFixedU8(kVoid)) return false;
  for (uint32_t i = 0; i < n; i++) {
    if (!e.writeOp(Op::Block) || !e.writeFixedU8(kVoid)) return false;
  }
  if (!e.writeOp(Op::LocalGet) || !e.writeVarU32(bidLocal)) return false;
  if (!e.writeOp(Op::BrTable) || !e.writeVarU32(n)) return false;
  for (uint32_t i = 0; i < n; i++) {
    if (!e.writeVarU32(n - 1 - i)) return false;
  }
  if (!e.writeVarU32(0)) return false;  // default

  for (uint32_t ri = 0; ri < n; ri++) {
    uint32_t bi = n - 1 - ri;
    MBasicBlock* b = blocks[bi];
    if (!e.writeOp(Op::End)) return false;  // close block $b_{bi}
    uint32_t loopDepthHere = n - 1 - ri;
    be.bodyLoopIdx = loopDepthHere;  // br index of $L from this body base
    be.inDispatchBody = be.oolDeopt;  // in-body guards go out-of-line
    if (!EmitBlockBody(e, be, b)) return false;
    be.inDispatchBody = false;  // terminator deopts stay inline (at body base)
    // alwaysBails blocks emitted their own deopt (which exits); no terminator.
    if (b->alwaysBails()) continue;
    if (!EmitTerminator(e, be, blockIdx, bidLocal, loopDepthHere, b)) {
      js::wasm::gWJBailReason = "terminator";
      return false;
    }
  }

  if (!e.writeOp(Op::Unreachable)) return false;
  if (!e.writeOp(Op::End)) return false;  // end loop $L

  // Out-of-line deopt dispatcher: `br $D` from a guard lands here (just past the
  // loop). Switch on deoptSiteLocal and run the matching site's full resume.
  if (be.oolDeopt) {
    if (!e.writeOp(Op::End)) return false;  // end $D
    for (uint32_t s = 0; s < be.deoptSites.size(); s++) {
      if (!GetLocal(e, be.deoptSiteLocal) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(s)) || !e.writeOp(Op::I32Eq) ||
          !e.writeOp(Op::If) || !e.writeFixedU8(kVoid))
        return false;
      be.curRp = std::get<0>(be.deoptSites[s]);
      be.curOp = std::get<1>(be.deoptSites[s]);
      static int noSkip = getenv("GECKO_WJ_SHADOWNOSKIP") ? 1 : 0;
      bool isShadow = std::get<2>(be.deoptSites[s]) && !noSkip;
      // shadow sites: frame operands already in gWJResumeVals (hot path) -> skip
      // the cold re-read of locals (that read is what pins them live-out).
      if (!EmitDeoptResumeInline(e, be, isShadow)) {
        js::wasm::gWJBailReason = "deopt-resume";
        return false;  // ends in `return`
      }
      if (!e.writeOp(Op::End)) return false;
    }
  }

  if (!e.writeOp(Op::Unreachable)) return false;
  cfReport("dispatch");
  return e.writeOp(Op::End);  // end function body
}
