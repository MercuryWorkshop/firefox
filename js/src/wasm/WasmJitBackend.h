/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef wasm_WasmJitBackend_h
#define wasm_WasmJitBackend_h

#include <stdint.h>

namespace js {
namespace jit {
class MIRGenerator;
class MIRGraph;
class WarpSnapshot;
}  // namespace jit
namespace wasm {

class Encoder;

// MIR->wasm back-end (Phase 1+). Emits the wasm function BODY (the locals
// declaration + instruction stream that goes inside the single Code-section
// entry) for an OptimizeMIR'd Warp graph, using the scratch-buffer ABI:
//   param0 (f64) = base address of the runtime's gWJScratch buffer
//   args are boxed JS::Value bits at gWJScratch[i] (i64), `this` at kWJThisOff
//   result (boxed) is stored to gWJScratch[kWJResultOff]
//   the function returns an f64 deopt flag: 0.0 = ok, 1.0 = bail back to PBL
// Returns false (caller keeps the function in PBL) if the graph uses a node the
// back-end can't lower yet.
bool WJEmitBody(jit::MIRGenerator& mir, jit::MIRGraph& graph, uint32_t nargs,
                bool useThis, jit::WarpSnapshot* snapshot, Encoder& e);

// Emit the per-module entry trampoline body (wasm type (f64 scratchbase)->f64):
// loads `this` + args from gWJScratch and calls the module's main (func idx 1)
// with the register-argument ABI. Used for the host-entry path. Returns false on
// encoder failure.
bool WJEmitTrampoline(Encoder& e);

// GC-constant pool (WASMJIT_REARCH_PLAN.md §4.2). Interns a boxed JS::Value (a
// GC pointer baked into emitted wasm cannot be relocated by a moving GC) and
// returns the *byte address* of its pool slot, which the emitted code loads at
// runtime. WJTraceRoots traces + updates the pool. Defined in WasmJitRuntime.cpp.
// Returns 0 if the pool is full (caller bails the compile).
uintptr_t WJInternConstant(uint64_t valueBits);

// wjhelp() helper kinds (the f64 first argument to the imported "m"."help").
enum WJHelpKind : int {
  WJH_RESUME = 1,   // deopt: rebuild a PBL frame from gWJResume* and run from pc
  WJH_CALL = 2,     // invoke gWJCallCallee with argc args staged in gWJScratch
  WJH_SETSLOT = 3,  // gWJHelpObj->setSlot(gWJHelpSlot, gWJHelpVal) (barriered)
  // Generic VM-op helpers. Operands are staged boxed in gWJScratch[0..]; the
  // (boxed) result is left in gWJScratch[kWJResultSlot]; returns 0.0 ok / 1.0
  // threw. May GC -- caller roots live values around the call.
  WJH_GETPROP = 4,   // scratch[0]=obj, scratch[1]=id  -> GetElementOperation
  WJH_SETPROP = 5,   // scratch[0]=obj, scratch[1]=id, scratch[2]=val, strict flag=site
  WJH_GETELEM = 6,   // (alias of GETPROP path; kept for clarity at the call site)
  WJH_INSTANCEOF = 7,  // scratch[0]=obj, scratch[1]=ctor -> InstanceofOperation
  WJH_ARRAYPUSH = 8,   // scratch[0]=array(obj), scratch[1]=val -> push, result=newlen
  WJH_ARRAYPOPSHIFT = 9,  // scratch[0]=array(obj), site: 0=pop 1=shift -> result=elem
  WJH_CREATETHIS = 10,    // scratch[0]=callee, scratch[1]=newTarget -> result=this
  WJH_CONSTRUCT = 11,     // gWJCallCallee + scratch[0..argc-1] args + this@kWJThisSlot
                          // + gWJConstructNewTarget -> InternalConstructWithProvidedThis
};

// Resume state buffer (WASMJIT_REARCH_PLAN.md §4.1). On a guard miss the emitted
// code boxes the MResumePoint's live values [this, args, locals, stack] into
// gWJResumeVals, sets gWJResumePc / gWJResumeStackDepth, and calls wjhelp(WJH_RESUME),
// which reconstructs a PBL frame and continues. Traced by WJTraceRoots.
extern uint64_t gWJResumeVals[];
extern uint32_t gWJResumePc;
extern uint32_t gWJResumeStackDepth;
// Self-contained resume context (set by the emitted deopt code, so a function
// resumes correctly regardless of how it was entered -- RunCall, fast-call, or a
// pure wasm->wasm call_indirect). Script ptr is baked; env is the resume point's
// environment-chain operand; nargs/nlocals are compile-time constants.
extern uint32_t gWJResumeScriptPtr;  // JSScript* as i32 (wasm32)
extern uint32_t gWJResumeEnvPtr;     // JSObject* env chain as i32
extern uint32_t gWJResumeNArgs;
extern uint32_t gWJResumeNLocals;
// Call boundary: callee + argc for wjhelp(WJH_CALL).
extern uint64_t gWJCallCallee;
extern uint32_t gWJCallArgc;
// Constructing call: newTarget (boxed) for wjhelp(WJH_CREATETHIS isn't this --
// this is for the construct CALL itself). `this` is staged at gWJScratch[kWJThisSlot].
extern uint64_t gWJConstructNewTarget;
// Barriered slot store: wjhelp(WJH_SETSLOT) does gWJHelpObj->setSlot(slot, val).
extern uint32_t gWJHelpObj;
extern uint32_t gWJHelpSlot;
extern uint64_t gWJHelpVal;

// Monomorphic call-site IC for direct wasm->wasm calls. Per call site: the cached
// callee object pointer and its shared-table index. A JIT'd caller compares the
// actual callee to gWJCallFn[site]; on a hit it `call_indirect`s the callee's
// wasm via the shared table (no C++/JS hop); on a miss it calls wjhelp(WJH_CALL),
// which runs the callee and fills the IC (only for compiled callees).
static constexpr uint32_t kWJCallSites = 16384;
// Polymorphic IC: up to kWJCallWays cached (callee, table-slot) pairs per site,
// so a polymorphic dispatch (e.g. richards' task.run over 4-5 task types) does a
// guarded wasm->wasm call_indirect per way instead of the slow wjhelp hop. Way 0
// is checked first, so monomorphic calls stay one compare. Arrays are
// [kWJCallSites * kWJCallWays], indexed [site * kWJCallWays + way].
static constexpr uint32_t kWJCallWays = 8;
extern uint32_t gWJCallFn[];      // cached callee obj ptr (0 = empty)
extern int32_t gWJCallTblIdx[];   // callee's shared-table index
// Next unused call-site id (process-global; assigned at compile).
uint32_t WJAllocCallSite();

// GC-root shadow stack. The value-per-local backend holds object pointers in
// wasm locals, invisible to the GC. Before a call (which can trigger a moving
// minor GC) the emitted code spills its live object/Value locals here, bumps
// gWJRootSP, makes the call, restores gWJRootSP, then reloads the (GC-updated)
// values. WJTraceRoots traces [0, gWJRootSP). A stack discipline handles nested
// JIT->JIT calls. Defined in WasmJitRuntime.cpp.
static constexpr uint32_t kWJCallRootsSize = 262144;
extern uint64_t gWJCallRoots[];
extern uint32_t gWJRootSP;

// Scratch-buffer layout (shared with WasmJitRuntime.cpp's gWJScratch + the host
// call marshalling). Sized for nargs<=64 with slack.
// Register-argument call ABI. JIT'd "main" functions take wasm params
// (f64 scratch-base, i64 this, i64 arg0..arg[kWJMaxArgs-1]) so args pass in
// registers instead of round-tripping through the gWJScratch buffer. A per-
// module trampoline (f64 scratch-base)->f64 reads scratch and calls main, for
// the host-entry path (the JS<->wasm boundary can't carry i64 safely). Functions
// with more than kWJMaxArgs actual args fall back (stay PBL).
static constexpr uint32_t kWJMaxArgs = 8;

static constexpr uint32_t kWJResultSlot = 64;
static constexpr uint32_t kWJThisSlot = 65;
static constexpr uint32_t kWJScratchSlots = 72;
static constexpr uint32_t kWJResultOff = kWJResultSlot * 8;
static constexpr uint32_t kWJThisOff = kWJThisSlot * 8;

}  // namespace wasm
}  // namespace js

#endif  // wasm_WasmJitBackend_h
