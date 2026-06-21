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
  const CompileInfo* info = nullptr;  // slot layout for resume points
  MResumePoint* curRp = nullptr;      // resume point for the instruction being emitted
  WarpSnapshot* snapshot = nullptr;   // for resolving NurseryObject references

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
  int32_t l = be.local(d);
  if (l < 0) return false;
  return e.writeOp(Op::LocalGet) && e.writeVarU32(uint32_t(l));
}

// Sound deopt: spill the current resume point's live state and continue in PBL
// (defined below). Returns false if it can't be emitted (caller bails the fn).
static bool EmitDeoptResume(Encoder& e, WJBackend& be);
static bool EmitValueTruthy(Encoder& e, WJBackend& be, uint32_t v);

// Emit a deopt at the current instruction's resume point. All guard misses route
// here; if there's no usable resume point the whole function bails at compile.
static bool EmitDeopt(Encoder& e, WJBackend& be) { return EmitDeoptResume(e, be); }

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
    case MIRType::Double:
      if (fallible && !EmitDoubleGuardLocal(e, be, localIdx)) return false;
      return GetLocal(e, localIdx) && e.writeOp(Op::F64ReinterpretI64);
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
  int32_t l = be.local(v);
  if (l < 0) return WJBAIL("spill: op#%u type %s has no local\n",
                           unsigned(v->op()), StringFromMIRType(v->type()));
  if (!GetLocal(e, uint32_t(l))) return false;
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
  int32_t l = be.local(v);
  if (l < 0) return false;
  if (v->type() == MIRType::Object) return GetLocal(e, uint32_t(l));
  if (v->type() == MIRType::Value) {
    return GetLocal(e, uint32_t(l)) && e.writeOp(Op::I32WrapI64);
  }
  return false;
}

// Sound deopt: box the resume point's live state [this,args,locals,stack] into
// gWJResumeVals, record pc + stack depth, and call wjhelp(WJH_RESUME), which
// rebuilds a PBL frame and finishes the function there.
static bool EmitDeoptResume(Encoder& e, WJBackend& be) {
  static int noResume = getenv("GECKO_WJ_NORESUME") ? 1 : 0;
  if (noResume) return false;  // bisect: bail any function needing a resume
  MResumePoint* rp = be.curRp;
  const CompileInfo* info = be.info;
  if (!rp || !info) return WJBAIL("resume: no rp/info\n");
  uintptr_t base = uintptr_t(static_cast<void*>(&gWJResumeVals[0]));
  uint32_t nargs = info->nargs();
  uint32_t nlocals = info->nlocals();
  uint32_t firstStack = info->firstStackSlot();
  if (rp->numOperands() < firstStack) return false;
  uint32_t stackDepth = uint32_t(rp->numOperands()) - firstStack;
  static int noStackResume = getenv("GECKO_WJ_NOSTACKRESUME") ? 1 : 0;
  if (noStackResume && stackDepth > 0) return false;

  uint32_t k = 0;
  auto spill = [&](MDefinition* v) -> bool {
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(base + k * 8)))
      return false;
    if (!EmitSpillValue(e, be, v)) return false;
    if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
      return false;
    k++;
    return true;
  };
  if (!spill(rp->getOperand(info->thisSlot()))) return false;
  for (uint32_t i = 0; i < nargs; i++) {
    if (!spill(rp->getOperand(info->argSlot(i)))) return false;
  }
  for (uint32_t i = 0; i < nlocals; i++) {
    if (!spill(rp->getOperand(info->localSlot(i)))) return false;
  }
  for (uint32_t i = 0; i < stackDepth; i++) {
    if (!spill(rp->getOperand(info->stackSlot(i)))) return false;
  }

  // Resume-after modes mean the op at rp->pc() already produced its result (on
  // the captured expr stack); baseline resumes at the NEXT bytecode rather than
  // re-executing it. Mirrors BaselineBailouts.cpp `if (IsResumeAfter) GetNextPc`.
  jsbytecode* resumePc = rp->pc();
  if (IsResumeAfter(rp->mode())) resumePc = GetNextPc(resumePc);
  uint32_t pcOff = uint32_t(resumePc - info->script()->code());
  uintptr_t pcAddr = uintptr_t(static_cast<void*>(&gWJResumePc));
  uintptr_t depthAddr = uintptr_t(static_cast<void*>(&gWJResumeStackDepth));
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(pcAddr)) ||
      !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(pcOff)) ||
      !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0)) {
    return false;
  }
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(depthAddr)) ||
      !e.writeOp(Op::I32Const) || !e.writeVarS32(int32_t(stackDepth)) ||
      !e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0)) {
    return false;
  }
  // Self-contained context: bake script ptr + nargs/nlocals, and spill the
  // resume point's environment-chain operand. This makes resume work regardless
  // of how the function was entered (RunCall / fast-call / wasm->wasm call).
  auto storeConstI32 = [&](void* addr, int32_t val) -> bool {
    return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(uintptr_t(addr))) &&
           e.writeOp(Op::I32Const) && e.writeVarS32(val) &&
           e.writeOp(Op::I32Store) && e.writeVarU32(2) && e.writeVarU32(0);
  };
  if (!storeConstI32(&gWJResumeScriptPtr, int32_t(uintptr_t(info->script()))))
    return false;
  if (!storeConstI32(&gWJResumeNArgs, int32_t(nargs))) return false;
  if (!storeConstI32(&gWJResumeNLocals, int32_t(nlocals))) return false;
  if (!e.writeOp(Op::I32Const) ||
      !e.writeVarS32(int32_t(uintptr_t(static_cast<void*>(&gWJResumeEnvPtr)))))
    return false;
  // Spill the resume point's env chain; if it isn't materializable here, store 0
  // and WJH_RESUME falls back to the function's environment (correct for the
  // common no-per-call-env-object case).
  if (!EmitObjPtr(e, be, rp->getOperand(info->environmentChainSlot()))) {
    if (!e.writeOp(Op::I32Const) || !e.writeVarS32(0)) return false;
  }
  if (!e.writeOp(Op::I32Store) || !e.writeVarU32(2) || !e.writeVarU32(0))
    return false;

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
  return e.writeOp(Op::Return);
}

// Emit a call to wjhelp(kind, site) for a VM op whose boxed operands the caller
// has already staged in gWJScratch[0..]. Roots the resume point's live
// Object/Value locals across the call (the helper may GC), propagates a thrown
// exception (return [1.0, 0]), reloads the roots, and leaves the boxed i64
// result (gWJScratch[kWJResultOff]) on the stack. `ins` is the node producing
// the result (excluded from rooting). Returns false => can't emit (bail fn).
static bool EmitHelperCallResult(Encoder& e, WJBackend& be, MInstruction* ins,
                                 int kind, uint32_t site) {
  std::vector<uint32_t> rootLocal;
  std::vector<uint8_t> rootIsObj;
  if (be.curRp) {
    for (uint32_t i = 0; i < be.curRp->numOperands(); i++) {
      MDefinition* op = be.curRp->getOperand(i);
      if (op == ins) continue;
      bool isObj = op->type() == MIRType::Object;
      bool isVal = op->type() == MIRType::Value;
      if (!isObj && !isVal) continue;
      int32_t l = be.local(op);
      if (l < 0) continue;
      rootLocal.push_back(uint32_t(l));
      rootIsObj.push_back(isObj ? 1 : 0);
    }
  }
  static int noRoot = getenv("GECKO_WJ_NOROOT") ? 1 : 0;
  if (noRoot) {
    rootLocal.clear();
    rootIsObj.clear();
  }
  uint32_t nRoots = uint32_t(rootLocal.size());
  uintptr_t rootsBase = uintptr_t(static_cast<void*>(&gWJCallRoots[0]));
  uintptr_t spAddr = uintptr_t(static_cast<void*>(&gWJRootSP));
  auto emitSlotAddr = [&](uint32_t k) -> bool {
    return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(spAddr)) &&
           e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(0) &&
           e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(k)) &&
           e.writeOp(Op::I32Add) && e.writeOp(Op::I32Const) && e.writeVarS32(8) &&
           e.writeOp(Op::I32Mul) && e.writeOp(Op::I32Const) &&
           e.writeVarS32(int32_t(rootsBase)) && e.writeOp(Op::I32Add);
  };
  auto emitSPAdjust = [&](int32_t delta) -> bool {
    return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(spAddr)) &&
           e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(spAddr)) &&
           e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(0) &&
           e.writeOp(Op::I32Const) && e.writeVarS32(delta) &&
           e.writeOp(Op::I32Add) && e.writeOp(Op::I32Store) && e.writeVarU32(2) &&
           e.writeVarU32(0);
  };
  for (uint32_t k = 0; k < nRoots; k++) {
    if (!emitSlotAddr(k)) return false;
    if (!GetLocal(e, rootLocal[k])) return false;
    if (rootIsObj[k] && !EmitBoxFromStack(e, MIRType::Object)) return false;
    if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
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
    if (!emitSlotAddr(k)) return false;
    if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0))
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

// Compute a node's value, leaving exactly one wasm value on the stack. (For
// effectful/void nodes EmitEffect is used instead.) Returns false => unsupported.
static bool EmitValue(Encoder& e, WJBackend& be, MInstruction* ins) {
  switch (ins->op()) {
    case MDefinition::Opcode::Parameter: {
      MParameter* p = ins->toParameter();
      // Register ABI: this = wasm param 1; arg i = wasm param 2+i.
      if (p->index() == MParameter::THIS_SLOT) {
        return e.writeOp(Op::LocalGet) && e.writeVarU32(1);
      }
      if (uint32_t(p->index()) >= kWJMaxArgs) return false;  // too many args: bail
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
      // A moving GC updates obj->shape_ in place but not our baked pointer, so a
      // moved shape simply fails the guard (deopt to PBL) -- safe for the pure
      // read-only functions we currently compile.
      MGuardShape* g = ins->toGuardShape();
      int32_t objLocal = be.local(g->object());
      if (objLocal < 0) return false;
      uint32_t shapeOff = uint32_t(offsetof(JS::shadow::Object, shape));
      if (!GetLocal(e, uint32_t(objLocal))) return false;  // obj
      if (!e.writeOp(Op::I32Load) || !e.writeVarU32(2) || !e.writeVarU32(shapeOff))
        return false;  // obj->shape_
      if (!e.writeOp(Op::I32Const) ||
          !e.writeVarS32(int32_t(uintptr_t(g->shape()))))
        return false;
      if (!e.writeOp(Op::I32Ne)) return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetLocal(e, uint32_t(objLocal));  // result = the object
    }
    case MDefinition::Opcode::GuardSpecificFunction: {
      // Passthrough operand 0 (the function); deopt if it != the expected
      // function (operand 1, an object pointer). Both are i32 object ptrs.
      MDefinition* obj = ins->getOperand(0);
      MDefinition* expected = ins->getOperand(1);
      int32_t ol = be.local(obj), xl = be.local(expected);
      if (ol < 0 || xl < 0) return false;
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
      int32_t el = be.local(ins->getOperand(0));  // Elements
      if (el < 0) return false;
      int32_t off = js::ObjectElements::offsetOfLength();  // negative (elem-relative)
      if (!GetLocal(e, uint32_t(el)) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(off) || !e.writeOp(Op::I32Add)) {
        return false;
      }
      return e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(0);
    }
    case MDefinition::Opcode::InitializedLength: {
      int32_t el = be.local(ins->getOperand(0));  // Elements
      if (el < 0) return false;
      int32_t off = js::ObjectElements::offsetOfInitializedLength();
      if (!GetLocal(e, uint32_t(el)) || !e.writeOp(Op::I32Const) ||
          !e.writeVarS32(off) || !e.writeOp(Op::I32Add)) {
        return false;
      }
      return e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(0);
    }
    case MDefinition::Opcode::BoundsCheck: {
      MBoundsCheck* bc = ins->toBoundsCheck();
      if (bc->minimum() != 0 || bc->maximum() != 0) return false;
      int32_t il = be.local(bc->index()), ll = be.local(bc->length());
      if (il < 0 || ll < 0) return false;
      // Deopt unless (uint32)index < (uint32)length (unsigned catches negatives).
      if (!GetLocal(e, uint32_t(il)) || !GetLocal(e, uint32_t(ll)) ||
          !e.writeOp(Op::I32GeU)) {
        return false;
      }
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
      int32_t el = be.local(elemsD), il = be.local(idxD);
      if (el < 0 || il < 0) return false;
      // addr = elements + index * sizeof(Value); load the boxed element.
      if (!GetLocal(e, uint32_t(el)) || !GetLocal(e, uint32_t(il)) ||
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
      int32_t objLocal = be.local(l->object());
      if (objLocal < 0) return false;
      uint32_t off = uint32_t(js::NativeObject::getFixedSlotOffset(l->slot()));
      if (!GetLocal(e, uint32_t(objLocal))) return false;
      return e.writeOp(Op::I64Load) && e.writeVarU32(3) && e.writeVarU32(off);
    }
    case MDefinition::Opcode::LoadFixedSlotAndUnbox: {
      MLoadFixedSlotAndUnbox* l = ins->toLoadFixedSlotAndUnbox();
      int32_t objLocal = be.local(l->object());
      if (objLocal < 0) return false;
      uint32_t off = uint32_t(js::NativeObject::getFixedSlotOffset(l->slot()));
      if (!GetLocal(e, uint32_t(objLocal))) return false;
      if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(off))
        return false;
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.unboxScratch))
        return false;
      return EmitUnboxLocal(e, be, be.unboxScratch, l->type(),
                            l->mode() == MUnbox::Fallible);
    }
    case MDefinition::Opcode::LoadDynamicSlotAndUnbox: {
      MLoadDynamicSlotAndUnbox* l = ins->toLoadDynamicSlotAndUnbox();
      int32_t slotsLocal = be.local(l->slots());
      if (slotsLocal < 0) return false;
      uint32_t off = uint32_t(l->slot() * sizeof(JS::Value));
      if (!GetLocal(e, uint32_t(slotsLocal))) return false;
      if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(off))
        return false;
      if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(be.unboxScratch))
        return false;
      return EmitUnboxLocal(e, be, be.unboxScratch, l->type(),
                            l->mode() == MUnbox::Fallible);
    }
    case MDefinition::Opcode::LoadDynamicSlot: {
      MLoadDynamicSlot* l = ins->toLoadDynamicSlot();
      int32_t slotsLocal = be.local(l->slots());
      if (slotsLocal < 0) return false;
      uint32_t off = uint32_t(l->slot() * sizeof(JS::Value));
      if (!GetLocal(e, uint32_t(slotsLocal))) return false;
      return e.writeOp(Op::I64Load) && e.writeVarU32(3) && e.writeVarU32(off);
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
      if (!GetOp(e, be, ins->getOperand(0)) || !GetOp(e, be, ins->getOperand(1))) {
        return false;
      }
      bool d = ins->type() == MIRType::Double;
      bool i = ins->type() == MIRType::Int32;
      if (ins->isAdd()) return e.writeOp(d ? Op::F64Add : i ? Op::I32Add : Op::Unreachable) && (d || i);
      if (ins->isSub()) return e.writeOp(d ? Op::F64Sub : i ? Op::I32Sub : Op::Unreachable) && (d || i);
      return e.writeOp(d ? Op::F64Mul : i ? Op::I32Mul : Op::Unreachable) && (d || i);
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
    case MDefinition::Opcode::Sqrt: {
      if (ins->type() != MIRType::Double) return false;
      if (!GetOp(e, be, ins->getOperand(0))) return false;
      return e.writeOp(Op::F64Sqrt);
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
    case MDefinition::Opcode::ToNumberInt32:
    case MDefinition::Opcode::TruncateToInt32: {
      // Result is Int32. Int32 input passes through; a Double input truncates
      // toward zero, deopting (to PBL's full ToInt32 wrap) if out of int32 range
      // or NaN -- so i32.trunc_f64_s never traps.
      MDefinition* in = ins->getOperand(0);
      if (in->type() == MIRType::Int32) return GetOp(e, be, in);
      if (in->type() != MIRType::Double) return false;
      if (!GetOp(e, be, in) || !e.writeOp(Op::F64Const) ||
          !e.writeFixedF64(-2147483648.0) || !e.writeOp(Op::F64Lt))
        return false;
      if (!GetOp(e, be, in) || !e.writeOp(Op::F64Const) ||
          !e.writeFixedF64(2147483647.0) || !e.writeOp(Op::F64Gt) ||
          !e.writeOp(Op::I32Or))
        return false;
      if (!GetOp(e, be, in) || !GetOp(e, be, in) || !e.writeOp(Op::F64Ne) ||
          !e.writeOp(Op::I32Or))
        return false;  // out-of-range OR NaN
      if (!e.writeOp(Op::If) || !e.writeFixedU8(0x40)) return false;
      if (!EmitDeopt(e, be)) return false;
      if (!e.writeOp(Op::End)) return false;
      return GetOp(e, be, in) && e.writeOp(Op::I32TruncF64S);
    }
    case MDefinition::Opcode::GetPropertyCache: {
      MGetPropertyCache* g = ins->toGetPropertyCache();
      if (!EmitStageScratch(e, be, g->value(), 0)) return false;
      if (!EmitStageScratch(e, be, g->idval(), 1)) return false;
      return EmitHelperCallResult(e, be, ins, WJH_GETPROP, 0);
    }
    case MDefinition::Opcode::MegamorphicLoadSlot: {
      // obj[name] via the generic get; name is a compile-time PropertyKey.
      MMegamorphicLoadSlot* m = ins->toMegamorphicLoadSlot();
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
      // Constructing-call support (InternalConstructWithProvidedThis via
      // WJH_CONSTRUCT) is implemented but currently DISABLED: it is correct in
      // isolation (verified: produces valid objects, correct newTarget =
      // getArg(numStackArgs()-1)), but in a function that ALSO contains a
      // property-cache load (GetPropertyCache/MegamorphicLoadSlot) it triggers a
      // wrong-result regression (octane raytrace `dot`). Disabling EITHER feature
      // fixes it; the GETPROP runtime is never even reached, so the fault is a
      // codegen interaction between two EmitHelperCallResult sites in one body
      // (likely a scratch-slot or GC-root-frame overlap). Re-enable once that is
      // root-caused. Until then constructing calls bail to PBL (correct).
      if (call->isConstructing()) return false;
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
      if (be.curRp) {
        for (uint32_t i = 0; i < be.curRp->numOperands(); i++) {
          MDefinition* op = be.curRp->getOperand(i);
          if (op == ins) continue;  // the call result: not computed yet
          bool isObj = op->type() == MIRType::Object;
          bool isVal = op->type() == MIRType::Value;
          if (!isObj && !isVal) continue;
          int32_t l = be.local(op);
          if (l < 0) continue;
          rootLocal.push_back(uint32_t(l));
          rootIsObj.push_back(isObj ? 1 : 0);
        }
      }
      static int noRoot = getenv("GECKO_WJ_NOROOT") ? 1 : 0;
      if (noRoot) {
        rootLocal.clear();
        rootIsObj.clear();
      }
      uint32_t nRoots = uint32_t(rootLocal.size());
      uintptr_t rootsBase = uintptr_t(static_cast<void*>(&gWJCallRoots[0]));
      uintptr_t spAddr = uintptr_t(static_cast<void*>(&gWJRootSP));
      // addr of slot k = gWJCallRoots + (gWJRootSP + k) * 8
      auto emitSlotAddr = [&](uint32_t k) -> bool {
        return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(spAddr)) &&
               e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(0) &&
               e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(k)) &&
               e.writeOp(Op::I32Add) && e.writeOp(Op::I32Const) &&
               e.writeVarS32(8) && e.writeOp(Op::I32Mul) &&
               e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(rootsBase)) &&
               e.writeOp(Op::I32Add);
      };
      auto emitSPAdjust = [&](int32_t delta) -> bool {
        return e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(spAddr)) &&
               e.writeOp(Op::I32Const) && e.writeVarS32(int32_t(spAddr)) &&
               e.writeOp(Op::I32Load) && e.writeVarU32(2) && e.writeVarU32(0) &&
               e.writeOp(Op::I32Const) && e.writeVarS32(delta) &&
               e.writeOp(Op::I32Add) && e.writeOp(Op::I32Store) &&
               e.writeVarU32(2) && e.writeVarU32(0);
      };
      for (uint32_t k = 0; k < nRoots; k++) {
        if (!emitSlotAddr(k)) return false;
        if (!GetLocal(e, rootLocal[k])) return false;
        if (rootIsObj[k] && !EmitBoxFromStack(e, MIRType::Object)) return false;
        if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
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
      for (uint32_t w = 0; w < kWJCallWays; w++) {
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
        if (!emitSlotAddr(k)) return false;
        if (!e.writeOp(Op::I64Load) || !e.writeVarU32(3) || !e.writeVarU32(0))
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
      return false;
  }
}

// True if the node has a side effect we must emit even though it produces no
// wasm value (or its value is unused). Returns false from EmitEffect => skip.
enum class EffectKind { Skip, Emitted, Fail };

static EffectKind EmitEffect(Encoder& e, WJBackend& be, MInstruction* ins) {
  switch (ins->op()) {
    case MDefinition::Opcode::Bail:
    case MDefinition::Opcode::Unreachable:
      return EmitDeopt(e, be) ? EffectKind::Emitted : EffectKind::Fail;
    case MDefinition::Opcode::Start:
    case MDefinition::Opcode::CheckOverRecursed:
    case MDefinition::Opcode::InterruptCheck:
      return EffectKind::Skip;  // prologue / ignored (host stack guards depth)
    case MDefinition::Opcode::StoreFixedSlot: {
      MStoreFixedSlot* s = ins->toStoreFixedSlot();
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
            !GetLocal(e, uint32_t(objLocal)) ||
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
      if (!GetLocal(e, uint32_t(objLocal))) return EffectKind::Fail;  // addr
      if (!EmitSpillValue(e, be, v)) return EffectKind::Fail;         // boxed value
      if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(off))
        return EffectKind::Fail;
      return EffectKind::Emitted;
    }
    case MDefinition::Opcode::StoreDynamicSlot: {
      // slots[slot] = value. The barrier-free case (primitive value, or a
      // separate MPostWriteBarrier node follows) stores inline; a value needing
      // a pre-write barrier falls through to bail (correct, rare).
      MStoreDynamicSlot* s = ins->toStoreDynamicSlot();
      if (s->needsBarrier()) return EffectKind::Fail;
      int32_t slotsLocal = be.local(s->slots());
      if (slotsLocal < 0) return EffectKind::Fail;
      uint32_t off = uint32_t(s->slot() * sizeof(JS::Value));
      if (!GetLocal(e, uint32_t(slotsLocal))) return EffectKind::Fail;
      if (!EmitSpillValue(e, be, s->value())) return EffectKind::Fail;
      if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(off))
        return EffectKind::Fail;
      return EffectKind::Emitted;
    }
    case MDefinition::Opcode::StoreElement: {
      // Dense in-bounds store elements[index] = value. Barrier-free / no-hole
      // cases store inline; object values (post-write barrier) or hole checks
      // fall through to bail (correct).
      MStoreElement* s = ins->toStoreElement();
      if (s->needsBarrier() || s->needsHoleCheck()) return EffectKind::Fail;
      int32_t el = be.local(s->elements());
      int32_t idx = be.local(s->index());
      if (el < 0 || idx < 0) return EffectKind::Fail;
      if (!GetLocal(e, uint32_t(el)) || !GetLocal(e, uint32_t(idx)) ||
          !e.writeOp(Op::I32Const) || !e.writeVarS32(8) ||
          !e.writeOp(Op::I32Mul) || !e.writeOp(Op::I32Add))
        return EffectKind::Fail;  // addr = elements + index*8
      if (!EmitSpillValue(e, be, s->value())) return EffectKind::Fail;
      if (!e.writeOp(Op::I64Store) || !e.writeVarU32(3) || !e.writeVarU32(0))
        return EffectKind::Fail;
      return EffectKind::Emitted;
    }
    case MDefinition::Opcode::MegamorphicStoreSlot: {
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
      MSetPropertyCache* s = ins->toSetPropertyCache();
      if (!EmitStageScratch(e, be, s->object(), 0)) return EffectKind::Fail;
      if (!EmitStageScratch(e, be, s->idval(), 1)) return EffectKind::Fail;
      if (!EmitStageScratch(e, be, s->value(), 2)) return EffectKind::Fail;
      if (!EmitHelperCallResult(e, be, ins, WJH_SETPROP, s->strict() ? 1 : 0))
        return EffectKind::Fail;
      return e.writeOp(Op::Drop) ? EffectKind::Emitted : EffectKind::Fail;
    }
    case MDefinition::Opcode::KeepAliveObject:
      // Pure GC-liveness marker; our GC-root shadow stack already keeps call
      // operands alive, so this is a no-op for codegen.
      return EffectKind::Skip;
    default:
      // Pure node whose value is unused (DCE leftover, or a type we don't map to
      // a wasm local): safe to skip. Unknown EFFECTFUL node: bail the function.
      return ins->isEffectful() ? EffectKind::Fail : EffectKind::Skip;
  }
}

// Store the (boxed) return value to gWJScratch[result] and return ok (0.0).
static bool EmitReturn(Encoder& e, WJBackend& be, MDefinition* val) {
  // Return [f64 flag=0.0, i64 boxed-result] in registers (no scratch store).
  if (!e.writeOp(Op::F64Const) || !e.writeFixedF64(0.0)) return false;  // ok flag
  if (!GetOp(e, be, val)) return false;
  if (val->type() != MIRType::Value && val->type() != MIRType::Int64) {
    if (!EmitBoxFromStack(e, val->type())) return false;
  }
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
  if (!EmitDeopt(e, be)) return false;  // string/symbol/bigint -> PBL
  if (!e.writeOp(Op::I32Const) || !e.writeVarS32(0)) return false;  // dead
  for (int i = 0; i < 5; i++)
    if (!e.writeOp(Op::End)) return false;
  return true;
}

// Emit one block's non-control instructions. Returns false on unsupported node.
static bool EmitBlockBody(Encoder& e, WJBackend& be, MBasicBlock* b) {
  for (MInstructionIterator it = b->begin(); it != b->end(); it++) {
    MInstruction* ins = *it;
    if (ins->isControlInstruction()) continue;
    be.curRp = ins->resumePoint() ? ins->resumePoint() : b->entryResumePoint();
    int32_t l = be.local(ins);
    if (l < 0) {
      // No value local: either a no-op typed node or an effectful one.
      EffectKind k = EmitEffect(e, be, ins);
      if (k == EffectKind::Fail) {
        fprintf(stderr, "[wb-be] unsupported effect op#%u\n", unsigned(ins->op()));
        return false;
      }
      continue;
    }
    if (!EmitValue(e, be, ins)) {
      fprintf(stderr, "[wb-be] unsupported value op#%u type%u\n", unsigned(ins->op()),
              unsigned(ins->type()));
      return false;
    }
    if (!e.writeOp(Op::LocalSet) || !e.writeVarU32(uint32_t(l))) return false;
  }
  return true;
}

// Emit a block terminator. loopDepth = depth of the dispatch loop $L from here.
static bool EmitTerminator(Encoder& e, WJBackend& be,
                           std::unordered_map<MBasicBlock*, uint32_t>& blockIdx,
                           uint32_t bidLocal, uint32_t loopDepth, MBasicBlock* b) {
  MControlInstruction* t = b->lastIns();
  be.curRp = t->resumePoint() ? t->resumePoint() : b->entryResumePoint();
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
    if (idomIdx[t] < 0) return false;  // unreachable/no idom: bail to dispatch
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
    MControlInstruction* t = b->lastIns();
    be.curRp = t->resumePoint() ? t->resumePoint() : b->entryResumePoint();
    if (t->isReturn()) {
      if (!EmitReturn(e, be, t->getOperand(0))) return false;
    } else if (t->isUnreachable() || t->isThrow()) {
      if (!EmitDeopt(e, be)) return false;
    } else if (t->isGoto()) {
      MBasicBlock* s = t->toGoto()->target();
      if (!EmitEdgeCopies(e, be, b, s)) return false;
      int32_t d = brTo(s, pos, 0);
      if (d < 0) return false;
      if (!e.writeOp(Op::Br) || !e.writeVarU32(uint32_t(d))) return false;
    } else if (t->isTest()) {
      MTest* test = t->toTest();
      MDefinition* cond = test->getOperand(0);
      if (WJValType(cond->type()) != uint8_t(TypeCode::I32)) return false;
      MBasicBlock* tb = test->ifTrue();
      MBasicBlock* fb = test->ifFalse();
      if (!GetOp(e, be, cond)) return false;
      if (!e.writeOp(Op::If) || !e.writeFixedU8(kVoid)) return false;
      if (!EmitEdgeCopies(e, be, b, tb)) return false;
      int32_t dt = brTo(tb, pos, 1);
      if (dt < 0) return false;
      if (!e.writeOp(Op::Br) || !e.writeVarU32(uint32_t(dt))) return false;
      if (!e.writeOp(Op::Else)) return false;
      if (!EmitEdgeCopies(e, be, b, fb)) return false;
      int32_t df = brTo(fb, pos, 1);
      if (df < 0) return false;
      if (!e.writeOp(Op::Br) || !e.writeVarU32(uint32_t(df))) return false;
      if (!e.writeOp(Op::End)) return false;
    } else {
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

  std::vector<MBasicBlock*> blocks;
  std::unordered_map<MBasicBlock*, uint32_t> blockIdx;
  for (ReversePostorderIterator b = graph.rpoBegin(); b != graph.rpoEnd(); b++) {
    // A block that always bails (Warp gave up on an op, e.g. an un-inlined
    // constructor) gains nothing from compilation -- it would just resume on
    // every execution. Bail the whole function to PBL instead.
    if (b->alwaysBails()) return false;
    blockIdx[*b] = uint32_t(blocks.size());
    blocks.push_back(*b);
  }
  const uint32_t n = uint32_t(blocks.size());

  // Assign locals: phis first (edge copies target them), then value nodes.
  for (MBasicBlock* b : blocks) {
    for (MPhiIterator p = b->phisBegin(); p != b->phisEnd(); p++) {
      if (WJValType((*p)->type()) != 0) be.assign(*p);
    }
  }
  for (MBasicBlock* b : blocks) {
    for (MInstructionIterator it = b->begin(); it != b->end(); it++) {
      MInstruction* ins = *it;
      if (ins->isControlInstruction()) continue;
      if (WJValType(ins->type()) == 0) continue;
      be.assign(ins);
    }
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

  // Single-block fast path: straight-line, no dispatch loop.
  if (n == 1) {
    MBasicBlock* b = blocks[0];
    if (!EmitBlockBody(e, be, b)) return false;
    MControlInstruction* t = b->lastIns();
    be.curRp = t->resumePoint() ? t->resumePoint() : b->entryResumePoint();
    if (t->isReturn()) {
      if (!EmitReturn(e, be, t->getOperand(0))) return false;
    } else if (t->isUnreachable() || t->isThrow()) {
      if (!EmitDeopt(e, be)) return false;
    } else {
      return false;
    }
    return e.writeOp(Op::End);
  }

  // Structured control flow (relooper): real loops/blocks/ifs so V8 can
  // optimize loops. Measured perf-neutral vs the dispatch loop AND has a
  // correctness bug on some complex CFGs (wrong control flow -> heap
  // corruption), so it is OFF by default; opt in with GECKO_WJ_RELOOP=1.
  static int useReloop = getenv("GECKO_WJ_RELOOP") ? 1 : 0;
  if (useReloop) {
    bool started = false;
    if (WJEmitStructured(e, be, blocks, blockIdx, &started)) return true;
    if (started) return false;  // partially emitted -> bail this compile
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
    if (!EmitBlockBody(e, be, b)) return false;
    uint32_t loopDepthHere = n - 1 - ri;
    if (!EmitTerminator(e, be, blockIdx, bidLocal, loopDepthHere, b)) return false;
  }

  if (!e.writeOp(Op::Unreachable)) return false;
  if (!e.writeOp(Op::End)) return false;  // end loop $L
  if (!e.writeOp(Op::Unreachable)) return false;
  return e.writeOp(Op::End);  // end function body
}
