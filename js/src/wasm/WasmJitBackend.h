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

// Intern a Shape* for a GuardShape: returns the byte address of a GC-traced pool
// slot holding the (relocatable) shape pointer. The emitted guard loads the slot
// at runtime so a GC-moved shape doesn't cause a permanent deopt. 0 if pool full.
uintptr_t WJInternShape(uintptr_t shapeBits);

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
  WJH_POSTBARRIER = 12,   // gWJHelpObj = container -> jit::PostWriteBarrier (store buffer)
  WJH_PREBARRIER = 13,    // gWJHelpVal = old value -> gc::ValuePreWriteBarrier (incremental mark)
  WJH_PROPIC = 14,        // prop-IC miss: scratch[0]=obj, site arg -> get gWJPropKey[site],
                          // fill gWJPropShape/Off[site], result -> scratch[kWJResultSlot]
  WJH_SETPROPIC = 15,     // set-prop-IC miss: scratch[0]=obj, scratch[1]=value, site arg
                          // -> fill gWJPropShape/Off[site] (writable own data prop) + store
  WJH_NEWPLAIN = 16,      // gWJNewShapeSlot(pool idx)/gWJNewAux(allocKind)/gWJNewHeap
                          // -> NewPlainObjectOptimizedFallback; result -> kWJResultSlot
  WJH_NEWARROBJ = 17,     // gWJNewShapeSlot/gWJNewAux(length)/gWJNewHeap
                          // -> NewArrayObjectOptimizedFallback
  WJH_NEWARR = 18,        // gWJNewAux(length) -> NewArrayOperation (NewArray/DynamicLength)
};

// Allocation-helper staging (non-GC ints; the shape is in the traced shape pool).
extern uint32_t gWJNewShapeSlot;  // gWJShapePool index of the template shape
extern uint32_t gWJNewAux;        // allocKind (NewPlainObject) or array length
extern uint32_t gWJNewHeap;       // gc::Heap (0=Default nursery)

// Compile-time-baked address of the current zone's needs-marking-barrier flag
// (zone->addressOfNeedsMarkingBarrier()). The emitted pre-write-barrier fast path
// loads + tests this; it's 0 except during an incremental GC slice. Set by
// WJWarpCompile. Single-zone shell, so one address suffices.
extern uintptr_t gWJMarkBarrierAddr;

// Inline nursery bump-allocation params (baked in WJWarpCompile; read by the
// backend at emit time, emitted as immediates). gWJNurseryPosAddr = address of
// the zone's nursery position_ (the emitted code loads/bumps/stores it);
// gWJObjHeaderWord = NurseryCellHeader::MakeValue(catchAllAllocSite, Object) for
// the cell header. The rest (header size, currentEnd offset, thingSize,
// emptyObjectSlots/Elements) are compile-time constants the backend computes.
extern uintptr_t gWJNurseryPosAddr;
extern uintptr_t gWJObjHeaderWord;

// Boxed ObjectValue bits of the realm's global lexical environment, refreshed
// each WJWarpCompile. MFunctionEnvironment bakes this for top-level-scoped
// functions (whose canonical JSFunction has no compile-time environment).
extern uint64_t gWJGlobalLexEnvVal;

// Raw pointer to the currently-executing JIT function's environment object. Set
// (to the runtime closure's environment) right before EVERY JIT invocation --
// the entry path (WasmJitRunCall's envChain) and the fast inter-JIT call_indirect
// (callee->environment()). MFunctionEnvironment loads this at function entry,
// giving the CORRECT runtime environment even for closures (unlike a baked one).
extern uint32_t gWJCurrentEnv;

// Set by WJEmitBody (reset at entry) to true iff the compiled function contains an
// alwaysBails block emitted as a deopt-at-entry (a cold/untyped Warp branch). Read
// by the deopt-storm valve: a storming function WITH an alwaysBails block deopts
// from that (cold-IC) block, which recompiling reproduces identically -- so it goes
// straight to PBL (Failed) instead of churning recompiles (deltablue wrong-value/
// crash). A storming function WITHOUT one (a stale monomorphic guard) still
// recompiles to specialize (crypto). Single-threaded synchronous compile.
extern bool gWJHadAlwaysBails;

// Compile bail-reason tracking (GECKO_WJ_CDBG): the most recent reason a function
// stayed in PBL plus its source line, printed by WJWarpCompile when WJEmitBody
// returns false. gWJBailReason points to a string literal (no ownership).
extern const char* gWJBailReason;
extern uint32_t gWJBailLine;

// Resume state buffer (WASMJIT_REARCH_PLAN.md §4.1). On a guard miss the emitted
// code boxes the resume point's live values [this, args, locals, stack] into
// gWJResumeVals and calls wjhelp(WJH_RESUME), which reconstructs PBL frame(s) and
// continues. Traced by WJTraceRoots.
//
// MULTI-FRAME (inline bailout): a deopt inside an INLINED callee must rebuild the
// whole inline frame chain. gWJResumeNFrames is the chain length; index 0 is the
// INNERMOST (deopt) frame, the last is the OUTERMOST (compiled function). Each
// frame's per-frame state is in the arrays below at index i; its boxed values
// start at gWJResumeVals[gWJResumeValsOff[i]] laid out [this, args.., locals..,
// stack..]. WJH_RESUME runs them innermost->outermost, threading each frame's
// return into the next outer frame's call-result stack slot. nFrames==1 is the
// ordinary (non-inlined) single-frame resume.
static constexpr uint32_t kWJMaxResumeFrames = 16;
extern uint64_t gWJResumeVals[];
extern uint32_t gWJResumeNFrames;
extern uint32_t gWJResumePc[];          // resume bytecode offset, per frame
extern uint32_t gWJResumeStackDepth[];  // expr-stack depth, per frame
extern uint32_t gWJResumeScriptPtr[];   // JSScript* (i32), per frame
extern uint32_t gWJResumeEnvPtr[];      // JSObject* env chain (i32), per frame
// The function's ENCLOSING (captured) environment for the resumed frame -- i.e.
// the closure's environment() = gWJCurrentEnv at entry (snapshotted to envLocal).
// WJH_RESUME passes it so the PBL prologue sets up the frame env from the CORRECT
// runtime enclosing scope, not the CANONICAL script->function()->environment()
// (wrong/null for a re-instantiated closure -> navier/earley crash). Per frame; 0
// = unknown (PBL falls back to func->environment()). Only the outermost frame
// (this compiled fn) has it; inlined frames keep 0.
extern uint32_t gWJResumeEnclosingEnv[];
// Debug (GECKO_WJ_DEOPTHIST): per-MIR-op deopt counter. The emitted deopt path
// increments gWJDeoptByOp[curOp]; WJH_RESUME periodically prints the histogram so
// we can see WHICH guard kind deopts most (MIRDUMP is unreliable for nondeterm-
// inistically-compiled fns like navier's). Indexed by MDefinition::Opcode.
static constexpr uint32_t kWJNumOps = 700;
extern uint32_t gWJDeoptByOp[];
extern uint32_t gWJResumeNArgs[];       // per frame
extern uint32_t gWJResumeNLocals[];     // per frame
extern uint32_t gWJResumeValsOff[];     // start index into gWJResumeVals, per frame
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

// Inline property-load IC (for MegamorphicLoadSlot / named GetPropertyCache of
// own data properties). Mirrors the call IC: per-site cached (receiver shape,
// TaggedSlotOffset). The JIT'd code loads obj->shape(), compares to the cached
// shape; on a hit it loads the slot inline (no C++ hop), on a miss it calls
// wjhelp(WJH_PROPIC) which does a pure lookup, fills the IC, and returns the
// value. A Shape is immutable and uniquely determines an own data property's
// slot, so a shape-identity match is sufficient for correctness; the only hazard
// is shape-pointer REUSE after a compacting GC, handled by clearing this IC on
// major-GC marking in WJTraceRoots. Indexed [site * kWJPropWays + way].
static constexpr uint32_t kWJPropSites = 16384;
static constexpr uint32_t kWJPropWays = 4;
extern uint32_t gWJPropShape[];   // cached receiver Shape* (0 = empty)
extern uint32_t gWJPropOff[];     // cached TaggedSlotOffset bits ((off<<1)|isFixed)
uint32_t WJAllocPropSite();
// Per-site baked PropertyKey for the WJH_PROPIC fill helper (the property name).
extern uint64_t gWJPropKey[];     // jsid raw bits per site
extern uint8_t gWJPropStrict[];   // per-site strict flag (set-prop fallback)
// Staging for the prop-IC miss helper: obj (boxed) -> gWJScratch[0]; result in
// gWJScratch[kWJResultSlot]. site passed as wjhelp's 2nd arg.
void WJClearPropIC();

// GC-root shadow stack. The value-per-local backend holds object pointers in
// wasm locals, invisible to the GC. Before a call (which can trigger a moving
// minor GC) the emitted code spills its live object/Value locals here, bumps
// gWJRootSP, makes the call, restores gWJRootSP, then reloads the (GC-updated)
// values. WJTraceRoots traces [0, gWJRootSP). A stack discipline handles nested
// JIT->JIT calls. Defined in WasmJitRuntime.cpp.
static constexpr uint32_t kWJCallRootsSize = 1048576;
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
