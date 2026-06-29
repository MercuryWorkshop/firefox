/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 *
 * Flavor-B wasm->wasm JIT: a host-wasm tier above the in-process interpreter.
 * A hot guest function is transcoded to a HOST wasm module that IMPORTS the
 * engine's own linear memory (the guest's memory lives inside it) and relocates
 * every guest memory access by +membase with an explicit bounds check, so the
 * interpreter and the host-JIT'd code SHARE the guest memory with no copy. The
 * host browser's real wasm JIT then runs the body at native speed; calls,
 * call_indirect, memory.grow and traps route back into the interpreter via a
 * single b_help dispatch import. Opt-in via GECKO_WASM_JIT; any unsupported
 * construct DECLINES and the interpreter runs the function unchanged.
 *
 * Included last from WasmInterp.h (needs Instance/Module + FuncInst/FuncIdx/
 * TableFromObject/GrowMemoryObject, all defined in the earlier -inl files).
 */

#ifndef wasm_WasmInterpJit_inl_h
#define wasm_WasmInterpJit_inl_h

#if defined(__EMSCRIPTEN__)

#  include "wasm/WasmBinary.h"  // Encoder, Bytes, MagicNumber, EncodingVersion

namespace js {
namespace wasm {
namespace interp {

// ---- host bridge (wasm-host-bridge.js) -------------------------------------
extern "C" {
int wasmhost_compile(const void* bytes, int len);
int wasmhost_instantiate(int handle, const int* callbackIds, int importCount);
double wasmhost_call(int handle, int index, const double* args, int argc);
int wasmhost_guest_mem_objid();
int wasmhost_guest_mem_shared();
}

// ---- per-thread scratch (arg/result channel + reloc header) ----------------
// The entry receives a pointer to this (as its f64 param), so it is thread-safe
// without baking any address. One buffer per thread, reused by nested calls:
// each entry's prologue copies membase/memsize/globals/params into locals before
// any nested call, so the buffer can be overwritten freely afterwards.
static constexpr uint32_t kBMaxSlots = 128;
static constexpr uint32_t kBSlotOff = 16;  // bytes; header is 16 bytes
struct BScratch {
  uint32_t membase;     // +0  guest memory base as an engine-memory offset
  uint32_t memsize;     // +4  guest memory current size in bytes (snapshot)
  uint32_t globals;     // +8  &Instance::globalCells[0] as an engine offset
  uint32_t memsizePtr;  // +12 &lm->size (engine offset) -- read LIVE for bounds
                        //     so another thread growing the shared memory mid
                        //     B-execution can't make us falsely trap.
  Cell slots[kBMaxSlots];  // +16
};
static thread_local BScratch gBScratch;
static thread_local Instance* gBInst = nullptr;

// b_help dispatch kinds.
enum {
  BH_TRAP = 0,          // a: JSMSG code
  BH_CALL = 1,          // a: func index (args/results in slots)
  BH_CALLINDIRECT = 2,  // a: typeIdx, b: tableIdx, c: elemIndex
  BH_MEMGROW = 3,       // a: delta pages -> returns old pages (or -1)
};

// Entry-function local layout. local 0 is the f64 scratch-pointer param.
enum {
  LB_PARAM = 0,
  LB_SP = 1,       // i32 scratch ptr
  LB_MEMBASE = 2,  // i32
  LB_MEMSIZE = 3,  // i32
  LB_GLOBALS = 4,  // i32
  LB_TMPA = 5,     // i32 (relocation address temp)
  LB_VALI32 = 6,
  LB_VALI64 = 7,
  LB_VALF32 = 8,
  LB_VALF64 = 9,
  LB_MEMSIZEPTR = 10,  // i32: &lm->size; bounds reads *LB_MEMSIZEPTR live
  LB_GBASE = 11,       // guest local g -> local (LB_GBASE + g)
  LB_NFIXED = 10,      // declared fixed locals (LB_SP..LB_MEMSIZEPTR)
};

static inline uint8_t VTByte(VT t) {
  switch (t) {
    case VT::I32: return uint8_t(TypeCode::I32);
    case VT::I64: return uint8_t(TypeCode::I64);
    case VT::F32: return uint8_t(TypeCode::F32);
    case VT::F64: return uint8_t(TypeCode::F64);
    case VT::FuncRef: return uint8_t(TypeCode::FuncRef);
    case VT::ExternRef: return uint8_t(TypeCode::ExternRef);
    case VT::V128: return uint8_t(TypeCode::V128);
    default: return uint8_t(TypeCode::I32);
  }
}

// Memory op classification (load/store width + store value type).
static bool BMemInfo(uint8_t op, uint32_t* size, bool* isStore, VT* valTy) {
  switch (Op(op)) {
    case Op::I32Load: *size = 4; *isStore = false; return true;
    case Op::I64Load: *size = 8; *isStore = false; return true;
    case Op::F32Load: *size = 4; *isStore = false; return true;
    case Op::F64Load: *size = 8; *isStore = false; return true;
    case Op::I32Load8S: case Op::I32Load8U: *size = 1; *isStore = false; return true;
    case Op::I32Load16S: case Op::I32Load16U: *size = 2; *isStore = false; return true;
    case Op::I64Load8S: case Op::I64Load8U: *size = 1; *isStore = false; return true;
    case Op::I64Load16S: case Op::I64Load16U: *size = 2; *isStore = false; return true;
    case Op::I64Load32S: case Op::I64Load32U: *size = 4; *isStore = false; return true;
    case Op::I32Store: *size = 4; *isStore = true; *valTy = VT::I32; return true;
    case Op::I64Store: *size = 8; *isStore = true; *valTy = VT::I64; return true;
    case Op::F32Store: *size = 4; *isStore = true; *valTy = VT::F32; return true;
    case Op::F64Store: *size = 8; *isStore = true; *valTy = VT::F64; return true;
    case Op::I32Store8: *size = 1; *isStore = true; *valTy = VT::I32; return true;
    case Op::I32Store16: *size = 2; *isStore = true; *valTy = VT::I32; return true;
    case Op::I64Store8: *size = 1; *isStore = true; *valTy = VT::I64; return true;
    case Op::I64Store16: *size = 2; *isStore = true; *valTy = VT::I64; return true;
    case Op::I64Store32: *size = 4; *isStore = true; *valTy = VT::I64; return true;
    default: return false;
  }
}

// True for value-stack ops that produce/consume a reference (decline: B has no
// GC rooting for refs held in host-JIT frames, and no ref scratch temp).
static bool BTypeHasRef(const FuncType& ft) {
  for (size_t i = 0; i < ft.params.length(); i++) {
    if (IsRefVT(ft.params[i]) || ft.params[i] == VT::V128) return true;
  }
  for (size_t i = 0; i < ft.results.length(); i++) {
    if (IsRefVT(ft.results[i]) || ft.results[i] == VT::V128) return true;
  }
  return false;
}

#  define WJB(expr) \
    do {            \
      if (!(expr)) return false; \
    } while (0)

static int BValTmpLocal(VT t) {
  switch (t) {
    case VT::I32: return LB_VALI32;
    case VT::I64: return LB_VALI64;
    case VT::F32: return LB_VALF32;
    case VT::F64: return LB_VALF64;
    default: return -1;
  }
}

// local.get $addrLocal ; <typed load at offset off> (absolute, NOT relocated).
static bool BEmitLoadAt(Encoder& e, uint32_t addrLocal, uint32_t off, VT t) {
  WJB(e.writeOp(Op::LocalGet) && e.writeVarU32(addrLocal));
  Op ld;
  uint32_t align;
  switch (t) {
    case VT::I32: ld = Op::I32Load; align = 2; break;
    case VT::I64: ld = Op::I64Load; align = 3; break;
    case VT::F32: ld = Op::F32Load; align = 2; break;
    case VT::F64: ld = Op::F64Load; align = 3; break;
    default: return false;
  }
  WJB(e.writeOp(ld) && e.writeVarU32(align) && e.writeVarU32(off));
  return true;
}

// local.get $addrLocal ; local.get $valLocal ; <typed store at offset off>.
static bool BEmitStoreAt(Encoder& e, uint32_t addrLocal, uint32_t valLocal,
                         uint32_t off, VT t) {
  WJB(e.writeOp(Op::LocalGet) && e.writeVarU32(addrLocal));
  WJB(e.writeOp(Op::LocalGet) && e.writeVarU32(valLocal));
  Op st;
  uint32_t align;
  switch (t) {
    case VT::I32: st = Op::I32Store; align = 2; break;
    case VT::I64: st = Op::I64Store; align = 3; break;
    case VT::F32: st = Op::F32Store; align = 2; break;
    case VT::F64: st = Op::F64Store; align = 3; break;
    default: return false;
  }
  WJB(e.writeOp(st) && e.writeVarU32(align) && e.writeVarU32(off));
  return true;
}

// Emit: call b_help(kind, a, b, c) ; the i32 result is left on the stack.
static bool BEmitHelpCall(Encoder& e, int kind, int a, int b, int c) {
  WJB(e.writeOp(Op::I32Const) && e.writeVarS32(kind));
  WJB(e.writeOp(Op::I32Const) && e.writeVarS32(a));
  WJB(e.writeOp(Op::I32Const) && e.writeVarS32(b));
  WJB(e.writeOp(Op::I32Const) && e.writeVarS32(c));
  WJB(e.writeOp(Op::Call) && e.writeVarU32(0));  // func 0 = m.help import
  return true;
}

// Emit a trapping return: b_help(TRAP, msg) ; drop ; i32.const 1 ; return.
static bool BEmitTrapReturn(Encoder& e, unsigned msg) {
  WJB(BEmitHelpCall(e, BH_TRAP, int(msg), 0, 0));
  WJB(e.writeOp(Op::Drop));
  WJB(e.writeOp(Op::I32Const) && e.writeVarS32(1));
  WJB(e.writeOp(Op::Return));
  return true;
}

// Bounds check on $tmpA: if ((u64)tmpA + endOff) > (u64)liveSize -> trap.
// liveSize is read fresh from *memsizePtr (the live lm->size cell), not the
// invoke-time snapshot, so a concurrent grow on another thread cannot make a
// valid access falsely trap. LB_MEMSIZE is refreshed to the live value here.
static bool BEmitBounds(Encoder& e, uint32_t endOff) {
  WJB(e.writeOp(Op::LocalGet) && e.writeVarU32(LB_MEMSIZEPTR));
  WJB(e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(0));
  WJB(e.writeOp(Op::LocalSet) && e.writeVarU32(LB_MEMSIZE));
  WJB(e.writeOp(Op::LocalGet) && e.writeVarU32(LB_TMPA));
  WJB(e.writeOp(Op::I64ExtendI32U));
  WJB(e.writeOp(Op::I64Const) && e.writeVarS64(int64_t(endOff)));
  WJB(e.writeOp(Op::I64Add));
  WJB(e.writeOp(Op::LocalGet) && e.writeVarU32(LB_MEMSIZE));
  WJB(e.writeOp(Op::I64ExtendI32U));
  WJB(e.writeOp(Op::I64GtU));
  WJB(e.writeOp(Op::If) && e.writeFixedU8(uint8_t(TypeCode::BlockVoid)));
  WJB(BEmitTrapReturn(e, JSMSG_WASM_OUT_OF_BOUNDS));
  WJB(e.writeOp(Op::End));
  return true;
}

static bool BEmitMarshalCall(Encoder& e, const FuncType& ft, int kind, int a,
                             int b, int c);
static bool BEmitMarshalCallIndirect(Encoder& e, const FuncType& ft,
                                     uint32_t typeIdx, uint32_t tableIdx);

// Transcode the guest body [fn.body, fn.bodyEnd) into the entry function body,
// already wrapped in `block $W <result>`. Returns false to DECLINE.
static bool BEmitGuestBody(JSContext* cx, Module* m, FuncDef& fn, Encoder& e) {
  Reader r(fn.body, fn.bodyEnd);
  uint32_t openCount = 0;
  while (r.p < r.end && r.ok) {
    const uint8_t* opStart = r.p;
    uint8_t op = r.u8();
    if (!r.ok) return false;
    switch (Op(op)) {
      // -- structured control flow: verbatim, track depth ---------------------
      case Op::Block:
      case Op::Loop:
      case Op::If: {
        if (!SkipBlockType(r)) return false;
        for (const uint8_t* q = opStart; q < r.p; q++) WJB(e.writeFixedU8(*q));
        openCount++;
        break;
      }
      case Op::Else:
        WJB(e.writeFixedU8(op));
        break;
      case Op::End:
        if (openCount == 0) return false;  // shouldn't happen (bodyEnd excludes)
        WJB(e.writeFixedU8(op));
        openCount--;
        break;
      case Op::Br:
      case Op::BrIf: {
        uint32_t d = r.u32leb();
        if (!r.ok) return false;
        WJB(e.writeFixedU8(op) && e.writeVarU32(d));
        break;
      }
      case Op::BrTable: {
        uint32_t n = r.u32leb();
        WJB(e.writeFixedU8(op) && e.writeVarU32(n));
        for (uint32_t i = 0; i < n + 1 && r.ok; i++) {
          uint32_t d = r.u32leb();
          WJB(e.writeVarU32(d));
        }
        if (!r.ok) return false;
        break;
      }
      case Op::Return:
        // Equivalent to a branch to the function-implicit block == $W (which is
        // at host depth openCount), carrying the result(s) already on the stack.
        WJB(e.writeOp(Op::Br) && e.writeVarU32(openCount));
        break;
      case Op::Unreachable:
        WJB(BEmitTrapReturn(e, JSMSG_WASM_UNREACHABLE));
        break;
      case Op::Nop:
        WJB(e.writeFixedU8(op));
        break;

      // -- locals: +LB_GBASE shift -------------------------------------------
      case Op::LocalGet:
      case Op::LocalSet:
      case Op::LocalTee: {
        uint32_t idx = r.u32leb();
        if (!r.ok) return false;
        WJB(e.writeFixedU8(op) && e.writeVarU32(LB_GBASE + idx));
        break;
      }

      // -- globals: direct via $globals base ---------------------------------
      case Op::GlobalGet:
      case Op::GlobalSet: {
        uint32_t idx = r.u32leb();
        if (!r.ok || idx >= m->globals.length()) return false;
        VT gt = m->globals[idx].type;
        if (IsRefVT(gt) || gt == VT::V128) return false;  // decline ref globals
        uint32_t off = idx * uint32_t(sizeof(Cell));
        if (Op(op) == Op::GlobalGet) {
          WJB(BEmitLoadAt(e, LB_GLOBALS, off, gt));
        } else {
          int vt = BValTmpLocal(gt);
          if (vt < 0) return false;
          WJB(e.writeOp(Op::LocalSet) && e.writeVarU32(uint32_t(vt)));  // pop val
          WJB(BEmitStoreAt(e, LB_GLOBALS, uint32_t(vt), off, gt));
        }
        break;
      }

      // -- memory load/store: relocate +membase, bounds vs memsize -----------
      case Op::I32Load: case Op::I64Load: case Op::F32Load: case Op::F64Load:
      case Op::I32Load8S: case Op::I32Load8U: case Op::I32Load16S:
      case Op::I32Load16U: case Op::I64Load8S: case Op::I64Load8U:
      case Op::I64Load16S: case Op::I64Load16U: case Op::I64Load32S:
      case Op::I64Load32U: case Op::I32Store: case Op::I64Store:
      case Op::F32Store: case Op::F64Store: case Op::I32Store8:
      case Op::I32Store16: case Op::I64Store8: case Op::I64Store16:
      case Op::I64Store32: {
        uint32_t align = r.u32leb();
        uint32_t off = r.u32leb();
        if (!r.ok) return false;
        uint32_t size;
        bool isStore;
        VT valTy = VT::I32;
        if (!BMemInfo(op, &size, &isStore, &valTy)) return false;
        if (isStore) {
          int vt = BValTmpLocal(valTy);
          if (vt < 0) return false;
          WJB(e.writeOp(Op::LocalSet) && e.writeVarU32(uint32_t(vt)));  // pop val
        }
        WJB(e.writeOp(Op::LocalSet) && e.writeVarU32(LB_TMPA));  // pop addr
        WJB(BEmitBounds(e, off + size));
        WJB(e.writeOp(Op::LocalGet) && e.writeVarU32(LB_MEMBASE));
        WJB(e.writeOp(Op::LocalGet) && e.writeVarU32(LB_TMPA));
        WJB(e.writeOp(Op::I32Add));
        if (isStore) {
          int vt = BValTmpLocal(valTy);
          WJB(e.writeOp(Op::LocalGet) && e.writeVarU32(uint32_t(vt)));
        }
        WJB(e.writeFixedU8(op) && e.writeVarU32(align) && e.writeVarU32(off));
        break;
      }
      case Op::MemorySize: {
        (void)r.u32leb();  // memidx
        if (!r.ok) return false;
        // live size in pages: (*memsizePtr) >> 16
        WJB(e.writeOp(Op::LocalGet) && e.writeVarU32(LB_MEMSIZEPTR));
        WJB(e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(0));
        WJB(e.writeOp(Op::I32Const) && e.writeVarS32(16));
        WJB(e.writeOp(Op::I32ShrU));
        break;
      }
      case Op::MemoryGrow: {
        (void)r.u32leb();  // memidx
        if (!r.ok) return false;
        WJB(e.writeOp(Op::LocalSet) && e.writeVarU32(LB_TMPA));  // delta
        WJB(e.writeOp(Op::I32Const) && e.writeVarS32(BH_MEMGROW));
        WJB(e.writeOp(Op::LocalGet) && e.writeVarU32(LB_TMPA));
        WJB(e.writeOp(Op::I32Const) && e.writeVarS32(0));
        WJB(e.writeOp(Op::I32Const) && e.writeVarS32(0));
        WJB(e.writeOp(Op::Call) && e.writeVarU32(0));  // -> old pages on stack
        // Reload membase/memsize (a non-shared grow may have moved the base).
        WJB(e.writeOp(Op::LocalGet) && e.writeVarU32(LB_SP));
        WJB(e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(0));
        WJB(e.writeOp(Op::LocalSet) && e.writeVarU32(LB_MEMBASE));
        WJB(e.writeOp(Op::LocalGet) && e.writeVarU32(LB_SP));
        WJB(e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(4));
        WJB(e.writeOp(Op::LocalSet) && e.writeVarU32(LB_MEMSIZE));
        break;
      }

      // -- calls: marshal via slots, dispatch through b_help -----------------
      case Op::Call: {
        uint32_t f = r.u32leb();
        if (!r.ok || f >= m->funcTypes.length()) return false;
        const FuncType& ft = m->funcType(f);
        if (BTypeHasRef(ft)) return false;  // ref args/results: no scratch temp
        if (ft.params.length() > kBMaxSlots || ft.results.length() > kBMaxSlots) {
          return false;
        }
        if (!BEmitMarshalCall(e, ft, BH_CALL, int(f), 0, 0)) return false;
        break;
      }
      case Op::CallIndirect: {
        uint32_t typeIdx = r.u32leb();
        uint32_t tableIdx = r.u32leb();
        if (!r.ok || typeIdx >= m->types.length()) return false;
        const FuncType& ft = m->types[typeIdx];
        if (BTypeHasRef(ft)) return false;
        if (ft.params.length() > kBMaxSlots || ft.results.length() > kBMaxSlots) {
          return false;
        }
        // elemIndex is on the stack -> pop into $tmpA, pass as b_help arg c.
        WJB(e.writeOp(Op::LocalSet) && e.writeVarU32(LB_TMPA));
        if (!BEmitMarshalCallIndirect(e, ft, typeIdx, tableIdx)) return false;
        break;
      }

      // -- parametric / numeric: verbatim ------------------------------------
      case Op::Drop:
      case Op::SelectNumeric:
      case Op::SelectTyped: {
        if (!SkipImmediates(r, op)) return false;
        for (const uint8_t* q = opStart; q < r.p; q++) WJB(e.writeFixedU8(*q));
        break;
      }

      // -- trapping ops: decline (interpreter handles them) ------------------
      case Op::I32DivS: case Op::I32DivU: case Op::I32RemS: case Op::I32RemU:
      case Op::I64DivS: case Op::I64DivU: case Op::I64RemS: case Op::I64RemU:
      case Op::I32TruncF32S: case Op::I32TruncF32U: case Op::I32TruncF64S:
      case Op::I32TruncF64U: case Op::I64TruncF32S: case Op::I64TruncF32U:
      case Op::I64TruncF64S: case Op::I64TruncF64U:
        return false;

      // -- references / tables: decline --------------------------------------
      case Op::RefNull: case Op::RefIsNull: case Op::RefFunc:
      case Op::TableGet: case Op::TableSet:
        return false;

      // -- prefixed: only non-trapping trunc_sat (MiscPrefix) is verbatim ----
      case Op::MiscPrefix: {
        uint32_t sub = r.u32leb();
        if (!r.ok) return false;
        if (sub <= uint32_t(MiscOp::I64TruncSatF64U)) {
          // trunc_sat family (0x00..0x07): no immediates, copy verbatim.
          for (const uint8_t* q = opStart; q < r.p; q++) WJB(e.writeFixedU8(*q));
          break;
        }
        return false;  // bulk-memory / table ops: decline
      }
      case Op::ThreadPrefix:
      case Op::SimdPrefix:
        return false;

      default:
        // Remaining Core ops (consts, arithmetic, comparisons, conversions,
        // sign-extension) have no memory/index semantics: copy verbatim.
        if (!SkipImmediates(r, op)) return false;
        for (const uint8_t* q = opStart; q < r.p; q++) WJB(e.writeFixedU8(*q));
        break;
    }
  }
  return r.ok && openCount == 0;
}

// Marshal a `call` (callee args on the stack) into slots and dispatch via
// b_help, then push the callee's results back onto the stack.
static bool BEmitMarshalCall(Encoder& e, const FuncType& ft, int kind, int a,
                             int b, int c) {
  uint32_t na = ft.params.length();
  for (int i = int(na) - 1; i >= 0; i--) {
    VT pt = ft.params[i];
    int vt = BValTmpLocal(pt);
    if (vt < 0) return false;
    WJB(e.writeOp(Op::LocalSet) && e.writeVarU32(uint32_t(vt)));  // pop arg i
    WJB(BEmitStoreAt(e, LB_SP, uint32_t(vt), kBSlotOff + uint32_t(i) * 16, pt));
  }
  WJB(BEmitHelpCall(e, kind, a, b, c));
  // status on stack: if non-zero, trap-return.
  WJB(e.writeOp(Op::If) && e.writeFixedU8(uint8_t(TypeCode::BlockVoid)));
  WJB(e.writeOp(Op::I32Const) && e.writeVarS32(1));
  WJB(e.writeOp(Op::Return));
  WJB(e.writeOp(Op::End));
  uint32_t nr = ft.results.length();
  for (uint32_t i = 0; i < nr; i++) {
    WJB(BEmitLoadAt(e, LB_SP, kBSlotOff + i * 16, ft.results[i]));
  }
  return true;
}

static bool BEmitMarshalCallIndirect(Encoder& e, const FuncType& ft,
                                     uint32_t typeIdx, uint32_t tableIdx) {
  // elemIndex already popped into $tmpA by the caller.
  uint32_t na = ft.params.length();
  for (int i = int(na) - 1; i >= 0; i--) {
    VT pt = ft.params[i];
    int vt = BValTmpLocal(pt);
    if (vt < 0) return false;
    WJB(e.writeOp(Op::LocalSet) && e.writeVarU32(uint32_t(vt)));
    WJB(BEmitStoreAt(e, LB_SP, uint32_t(vt), kBSlotOff + uint32_t(i) * 16, pt));
  }
  WJB(e.writeOp(Op::I32Const) && e.writeVarS32(BH_CALLINDIRECT));
  WJB(e.writeOp(Op::I32Const) && e.writeVarS32(int(typeIdx)));
  WJB(e.writeOp(Op::I32Const) && e.writeVarS32(int(tableIdx)));
  WJB(e.writeOp(Op::LocalGet) && e.writeVarU32(LB_TMPA));  // elemIndex
  WJB(e.writeOp(Op::Call) && e.writeVarU32(0));
  WJB(e.writeOp(Op::If) && e.writeFixedU8(uint8_t(TypeCode::BlockVoid)));
  WJB(e.writeOp(Op::I32Const) && e.writeVarS32(1));
  WJB(e.writeOp(Op::Return));
  WJB(e.writeOp(Op::End));
  uint32_t nr = ft.results.length();
  for (uint32_t i = 0; i < nr; i++) {
    WJB(BEmitLoadAt(e, LB_SP, kBSlotOff + i * 16, ft.results[i]));
  }
  return true;
}

// Emit the entry function's whole body (local decls + prologue + wrapped body +
// epilogue + final end). Returns false to DECLINE.
static bool BEmitEntry(JSContext* cx, Module* m, FuncDef& fn, Encoder& e) {
  const FuncType& ft = m->types[fn.typeIndex];
  uint32_t na = ft.params.length();
  uint32_t nr = ft.results.length();
  // MVP: single-result (the wrapper block type is one valtype) and no refs.
  if (nr > 1) return false;
  if (BTypeHasRef(ft)) return false;
  for (size_t i = 0; i < fn.localTypes.length(); i++) {
    if (IsRefVT(fn.localTypes[i]) || fn.localTypes[i] == VT::V128) return false;
  }
  if (na > kBMaxSlots) return false;

  uint32_t numGuestLocals = fn.localTypes.length();

  // ---- local declarations ----
  // [LB_NFIXED fixed] + [guest locals] + [nr result temps].
  Vector<uint8_t, 64, SystemAllocPolicy> locals;
  const uint8_t I32 = uint8_t(TypeCode::I32), I64 = uint8_t(TypeCode::I64),
                F32 = uint8_t(TypeCode::F32), F64 = uint8_t(TypeCode::F64);
  // sp, membase, memsize, globals, tmpA, valI32 (6 i32), valI64, valF32, valF64,
  // memsizePtr (i32).
  const uint8_t fixed[LB_NFIXED] = {I32, I32, I32, I32, I32,
                                    I32, I64, F32, F64, I32};
  for (uint8_t b : fixed) {
    if (!locals.append(b)) return false;
  }
  for (size_t i = 0; i < fn.localTypes.length(); i++) {
    if (!locals.append(VTByte(fn.localTypes[i]))) return false;
  }
  for (size_t i = 0; i < nr; i++) {
    if (!locals.append(VTByte(ft.results[i]))) return false;
  }
  // Emit local decls, run-length encoded (two passes, no storage).
  uint32_t ngroups = 0;
  for (size_t i = 0; i < locals.length();) {
    size_t j = i;
    while (j < locals.length() && locals[j] == locals[i]) j++;
    ngroups++;
    i = j;
  }
  WJB(e.writeVarU32(ngroups));
  for (size_t i = 0; i < locals.length();) {
    size_t j = i;
    while (j < locals.length() && locals[j] == locals[i]) j++;
    WJB(e.writeVarU32(uint32_t(j - i)) && e.writeFixedU8(locals[i]));
    i = j;
  }

  // ---- prologue ----
  // $sp = trunc_u(param) ; membase/memsize/globals = scratch header.
  WJB(e.writeOp(Op::LocalGet) && e.writeVarU32(LB_PARAM));
  WJB(e.writeOp(MiscOp::I32TruncSatF64U));
  WJB(e.writeOp(Op::LocalSet) && e.writeVarU32(LB_SP));
  struct {
    uint32_t off, local;
  } hdr[4] = {{0, LB_MEMBASE},
              {4, LB_MEMSIZE},
              {8, LB_GLOBALS},
              {12, LB_MEMSIZEPTR}};
  for (auto& h : hdr) {
    WJB(e.writeOp(Op::LocalGet) && e.writeVarU32(LB_SP));
    WJB(e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(h.off));
    WJB(e.writeOp(Op::LocalSet) && e.writeVarU32(h.local));
  }
  // Load params from slots into guest local slots.
  for (uint32_t i = 0; i < na; i++) {
    WJB(BEmitLoadAt(e, LB_SP, kBSlotOff + i * 16, ft.params[i]));
    WJB(e.writeOp(Op::LocalSet) && e.writeVarU32(LB_GBASE + i));
  }

  // ---- wrapped body ----
  // block $W (result = guest result type) { <body> }
  WJB(e.writeOp(Op::Block));
  if (nr == 0) {
    WJB(e.writeFixedU8(uint8_t(TypeCode::BlockVoid)));
  } else {
    WJB(e.writeFixedU8(VTByte(ft.results[0])));
  }
  if (!BEmitGuestBody(cx, m, fn, e)) return false;
  WJB(e.writeOp(Op::End));  // end $W

  // ---- epilogue ----
  uint32_t rtBase = LB_GBASE + numGuestLocals;
  for (int i = int(nr) - 1; i >= 0; i--) {
    WJB(e.writeOp(Op::LocalSet) && e.writeVarU32(rtBase + uint32_t(i)));
  }
  for (uint32_t i = 0; i < nr; i++) {
    WJB(BEmitStoreAt(e, LB_SP, rtBase + i, kBSlotOff + i * 16, ft.results[i]));
  }
  WJB(e.writeOp(Op::I32Const) && e.writeVarS32(0));  // status ok
  WJB(e.writeOp(Op::End));                           // end function
  return true;
}

// Assemble + host-compile + instantiate the per-function module. Returns a host
// handle (>=0) or -1 to DECLINE.
static int BCompileFunc(JSContext* cx, Module* m, FuncDef& fn) {
  if (!fn.body) return -1;
  int memId = wasmhost_guest_mem_objid();
  if (memId < 0) return -1;
  bool sharedMem = wasmhost_guest_mem_shared() != 0;

  Bytes out;
  Encoder e(out);
  const uint8_t I32 = uint8_t(TypeCode::I32), F64 = uint8_t(TypeCode::F64);
  if (!e.writeFixedU32(MagicNumber) || !e.writeFixedU32(EncodingVersionModule)) {
    return -1;
  }
  uint32_t nGuest = m->types.length();
  size_t s;
  // Type section: guest types (so blocktype/call_indirect indices stay valid) +
  // entry type (f64)->i32 + help type (i32,i32,i32,i32)->i32.
  if (!e.startSection(SectionId::Type, &s) || !e.writeVarU32(nGuest + 2)) return -1;
  for (uint32_t i = 0; i < nGuest; i++) {
    const FuncType& ft = m->types[i];
    if (!e.writeFixedU8(0x60) || !e.writeVarU32(ft.params.length())) return -1;
    for (size_t k = 0; k < ft.params.length(); k++) {
      if (!e.writeFixedU8(VTByte(ft.params[k]))) return -1;
    }
    if (!e.writeVarU32(ft.results.length())) return -1;
    for (size_t k = 0; k < ft.results.length(); k++) {
      if (!e.writeFixedU8(VTByte(ft.results[k]))) return -1;
    }
  }
  // entry type (index nGuest)
  if (!e.writeFixedU8(0x60) || !e.writeVarU32(1) || !e.writeFixedU8(F64) ||
      !e.writeVarU32(1) || !e.writeFixedU8(I32)) {
    return -1;
  }
  // help type (index nGuest+1)
  if (!e.writeFixedU8(0x60) || !e.writeVarU32(4) || !e.writeFixedU8(I32) ||
      !e.writeFixedU8(I32) || !e.writeFixedU8(I32) || !e.writeFixedU8(I32) ||
      !e.writeVarU32(1) || !e.writeFixedU8(I32)) {
    return -1;
  }
  e.finishSection(s);

  // Import section: m.help (func nGuest+1) then m.mem (memory).
  if (!e.startSection(SectionId::Import, &s) || !e.writeVarU32(2) ||
      !e.writeBytes("m", 1) || !e.writeBytes("help", 4) || !e.writeFixedU8(0x00) ||
      !e.writeVarU32(nGuest + 1)) {
    return -1;
  }
  if (!e.writeBytes("m", 1) || !e.writeBytes("mem", 3) || !e.writeFixedU8(0x02)) {
    return -1;
  }
  if (sharedMem) {
    if (!e.writeFixedU8(0x03) || !e.writeVarU32(1) || !e.writeVarU32(65536)) return -1;
  } else {
    if (!e.writeFixedU8(0x00) || !e.writeVarU32(0)) return -1;
  }
  e.finishSection(s);

  // Function section: 1 func, entry type (index nGuest). Func index 1.
  if (!e.startSection(SectionId::Function, &s) || !e.writeVarU32(1) ||
      !e.writeVarU32(nGuest)) {
    return -1;
  }
  e.finishSection(s);

  // Export section: "f" = entry (func idx 1).
  if (!e.startSection(SectionId::Export, &s) || !e.writeVarU32(1) ||
      !e.writeBytes("f", 1) || !e.writeFixedU8(0x00) || !e.writeVarU32(1)) {
    return -1;
  }
  e.finishSection(s);

  // Code section.
  if (!e.startSection(SectionId::Code, &s) || !e.writeVarU32(1)) return -1;
  size_t bodyOff;
  if (!e.writePatchableVarU32(&bodyOff)) return -1;
  size_t bodyStart = e.currentOffset();
  if (!BEmitEntry(cx, m, fn, e)) return -1;
  e.patchVarU32(bodyOff, uint32_t(e.currentOffset() - bodyStart));
  e.finishSection(s);

  if (getenv("GECKO_WASMJIT_DUMP")) {
    static int n = 0;
    char path[64];
    snprintf(path, sizeof(path), "/tmp/bjit_%d.wasm", n++);
    if (FILE* f = fopen(path, "wb")) {
      fwrite(out.begin(), 1, out.length(), f);
      fclose(f);
    }
  }

  int handle = wasmhost_compile(out.begin(), int(out.length()));
  if (handle < 0) return -1;
  const int importIds[2] = {-10, memId};  // help shim, guest memory
  if (wasmhost_instantiate(handle, importIds, 2) != 0) return -1;
  return handle;
}

// Run a compiled function: fill the scratch, call the host entry, read results.
static bool BInvoke(Instance* inst, const FuncType& ft, Cell* args,
                    Cell* results, int handle) {
  BScratch& sc = gBScratch;
  sc.membase = inst->lm ? uint32_t(uintptr_t(inst->lm->base)) : 0;
  sc.memsize = inst->lm ? inst->lm->size : 0;
  sc.memsizePtr = inst->lm ? uint32_t(uintptr_t(&inst->lm->size)) : 0;
  sc.globals = inst->globalCells.empty()
                   ? 0
                   : uint32_t(uintptr_t(inst->globalCells.begin()));
  uint32_t na = ft.params.length();
  for (uint32_t i = 0; i < na; i++) sc.slots[i] = args[i];
  Instance* saved = gBInst;
  gBInst = inst;
  double spF = double(uint32_t(uintptr_t(&sc)));
  double status = wasmhost_call(handle, 0, &spF, 1);
  gBInst = saved;
  if (status != 0) return false;  // trap: b_help set the pending exception
  uint32_t nrr = ft.results.length();
  for (uint32_t i = 0; i < nrr; i++) results[i] = sc.slots[i];
  return true;
}

// The single dispatch import for B-compiled code. Runs on the same thread that
// is executing the host module, so gBInst/gBScratch are valid.
extern "C" EMSCRIPTEN_KEEPALIVE int b_help(int kind, int a, int b, int c) {
  Instance* inst = gBInst;
  if (!inst) return 1;
  JSContext* cx = inst->cx;
  BScratch& sc = gBScratch;
  switch (kind) {
    case BH_TRAP:
      Trap(cx, unsigned(a));
      return 1;
    case BH_CALL: {
      uint32_t f = uint32_t(a);
      const FuncType& ft = inst->module->funcType(f);
      uint32_t na = ft.params.length(), nr = ft.results.length();
      Cell ab[64], rb[16];
      if (na > 64 || nr > 16) {
        JS_ReportErrorASCII(cx, "wasm B: call arity");
        return 1;
      }
      for (uint32_t i = 0; i < na; i++) ab[i] = sc.slots[i];
      Instance* saved = gBInst;
      bool ok = inst->invoke(f, ab, rb);
      gBInst = saved;
      if (!ok) return 1;
      for (uint32_t i = 0; i < nr; i++) sc.slots[i] = rb[i];
      return 0;
    }
    case BH_CALLINDIRECT: {
      uint32_t typeIdx = uint32_t(a), tableIdx = uint32_t(b), elem = uint32_t(c);
      if (tableIdx >= inst->tableObjs.length()) {
        Trap(cx, JSMSG_WASM_TABLE_OUT_OF_BOUNDS);
        return 1;
      }
      InstTable* t = TableFromObject(inst->tableObjs[tableIdx]);
      if (!t || elem >= t->funcs.length()) {
        Trap(cx, JSMSG_WASM_TABLE_OUT_OF_BOUNDS);
        return 1;
      }
      uint64_t fr = t->funcs[elem];
      if (FuncIsNull(fr)) {
        Trap(cx, JSMSG_WASM_IND_CALL_TO_NULL);
        return 1;
      }
      Instance* ti = FuncInst(fr);
      uint32_t tfi = FuncIdx(fr);
      if (!ti->module->funcType(tfi).equals(inst->module->types[typeIdx])) {
        Trap(cx, JSMSG_WASM_IND_CALL_BAD_SIG);
        return 1;
      }
      const FuncType& cft = inst->module->types[typeIdx];
      uint32_t na = cft.params.length(), nr = cft.results.length();
      Cell ab[64], rb[16];
      if (na > 64 || nr > 16) {
        JS_ReportErrorASCII(cx, "wasm B: call_indirect arity");
        return 1;
      }
      for (uint32_t i = 0; i < na; i++) ab[i] = sc.slots[i];
      Instance* saved = gBInst;
      bool ok = ti->invoke(tfi, ab, rb);
      gBInst = saved;
      if (!ok) return 1;
      for (uint32_t i = 0; i < nr; i++) sc.slots[i] = rb[i];
      return 0;
    }
    case BH_MEMGROW: {
      if (!inst->memObj || !inst->lm) return -1;
      JS::RootedObject memObj(cx, inst->memObj);
      int64_t old = GrowMemoryObject(cx, memObj, uint32_t(a));
      sc.membase = uint32_t(uintptr_t(inst->lm->base));
      sc.memsize = inst->lm->size;
      return int(old);
    }
    default:
      return 1;
  }
}

// ---- tier-up entry points (declared in WasmInterp.h) -----------------------
bool BEnabled() {
  static int v = -1;
  if (v < 0) v = getenv("GECKO_WASM_JIT") ? 1 : 0;
  return v != 0;
}

static uint32_t BThreshold() {
  static uint32_t v = 0;
  if (v == 0) {
    const char* s = getenv("GECKO_B_THRESHOLD");
    long n = s ? strtol(s, nullptr, 10) : 20;
    v = (n >= 1 && n < (1u << 30)) ? uint32_t(n) : 20;
  }
  return v;
}

struct BFunc {
  int handle = -1;
  int8_t state = 0;  // 0 cold, 1 compiled, 2 declined
};

static std::unordered_map<const FuncDef*, BFunc>& BCache() {
  static thread_local std::unordered_map<const FuncDef*, BFunc> cache;
  return cache;
}

// Returns 0 (handled ok), 1 (handled, trapped -> caller returns false), or -1
// (not handled -> caller runs the interpreter).
int BTryInvoke(Instance* inst, const FuncDef& fn, uint32_t fi, Cell* args,
               Cell* results) {
  Module* m = inst->module;
  if (!m->bCounts) return -1;
  uint32_t defIdx = fi - inst->numImportFuncs();
  uint32_t c = m->bCounts[defIdx].load(std::memory_order_relaxed);
  if (c < BThreshold()) {
    m->bCounts[defIdx].fetch_add(1, std::memory_order_relaxed);
    return -1;
  }
  BFunc& bf = BCache()[&fn];
  if (bf.state == 2) return -1;
  if (bf.state == 0) {
    int h = BCompileFunc(inst->cx, m, const_cast<FuncDef&>(fn));
    if (getenv("GECKO_WASMJIT_DEBUG")) {
      fprintf(stderr, "[b] func def#%u %s (handle=%d)\n", defIdx,
              h < 0 ? "DECLINED" : "COMPILED", h);
    }
    if (h < 0) {
      bf.state = 2;
      return -1;
    }
    bf.handle = h;
    bf.state = 1;
  }
  const FuncType& ft = m->funcType(fi);
  return BInvoke(inst, ft, args, results, bf.handle) ? 0 : 1;
}

#  undef WJB

}  // namespace interp
}  // namespace wasm
}  // namespace js

#endif  // __EMSCRIPTEN__
#endif  // wasm_WasmInterpJit_inl_h
