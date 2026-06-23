/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// WasmJitRuntime.cpp -- runtime integration for the re-architected JS->wasm JIT:
// the compile trigger (delays until ICs warm, then drives WasmJitWarp), call
// routing + arg/result marshalling through gWJScratch, the wjhelp/wasmjit_invoke
// trampolines, and GC root tracing of the scratch buffer.

#include "wasm/WasmJit.h"
#include "wasm/WasmJitBackend.h"  // kWJResultSlot / kWJThisSlot / kWJScratchSlots

#include <stdint.h>
#include <stdlib.h>
#include <unordered_map>

#include "js/CallAndConstruct.h"  // JS::Call
#include "js/GCAPI.h"
#include "js/RootingAPI.h"
#include "js/Value.h"
#include "vm/JSContext.h"
#include "vm/JSFunction.h"
#include "jit/JitScript.h"
#include "jit/VMFunctions.h"  // CreateThisFromIon
#include "vm/JSScript.h"
#include "vm/BytecodeUtil.h"  // js::CodeName
#include "vm/NativeObject.h"
#include "vm/Interpreter.h"  // SetObjectElement, InstanceofOperator, LessThan, ...
#include "vm/EqualityOperations.h"  // LooselyEqual, StrictlyEqual
#include "vm/Stack.h"  // ConstructArgs
#include "vm/ProxyObject.h"
#include "gc/Barrier.h"  // gc::ValuePreWriteBarrier
#include "gc/Tracer.h"   // js::TraceRoot
#include "vm/Shape.h"    // js::Shape
#include "vm/ArrayObject.h"   // js::ArrayObject
#include "builtin/Array.h"    // js::SetLengthProperty
#include "builtin/String.h"   // js::StringFromCharCode
#include "vm/StringType.h"    // js::ToString

#include "vm/NativeObject-inl.h"
#include "vm/Interpreter-inl.h"  // GetElementOperation
#include "vm/JSObject-inl.h"     // GuessArrayGCKind

#if defined(__EMSCRIPTEN__)
#  include <emscripten.h>
#else
#  define EMSCRIPTEN_KEEPALIVE
#endif

using namespace js;
using namespace js::wasm;

extern "C" {
double wasmhost_call(int handle, int index, const double* args, int argc);
}

namespace js {
namespace wasm {
// Front-end + compile orchestrator (WasmJitWarp.cpp).
int WJWarpCompile(JSContext* cx, JSScript* script, uint32_t* nargsOut,
                  uint32_t* nlocalsOut, int* tblSlotOut);
}  // namespace wasm
namespace pbl {
// Resume a partially-executed JIT'd function in the Portable Baseline Interpreter
// at `pcOff`, seeding fixed locals from osrLocals[] and the expr stack from osrStack[].
extern bool WasmJitResumeViaPBL(JSContext* cx, JSScript* script, uint64_t thisBits,
                                const JS::Value* args, uint32_t argc,
                                JSObject* envChain, const uint64_t* osrLocals,
                                uint32_t nLocals, uint32_t pcOff, uint64_t* retBits,
                                const uint64_t* osrStack, uint32_t osrStackDepth,
                                JSObject* enclosingEnv = nullptr,
                                bool keepFrameEnv = false);
}  // namespace pbl
}  // namespace js

// --- Resume state (set partly by RunCall, partly by the emitted spill code) ---
// These five are declared in WasmJitBackend.h (js::wasm) and read by the emitted
// code's addresses, so their definitions must live in js::wasm too.
namespace js {
namespace wasm {
uint64_t gWJResumeVals[1024];   // all frames' [this, args.., locals.., stack..]
uint32_t gWJResumeNFrames = 0;             // inline chain length (1 = no inlining)
uint32_t gWJResumePc[kWJMaxResumeFrames] = {0};
uint32_t gWJResumeStackDepth[kWJMaxResumeFrames] = {0};
uint32_t gWJResumeScriptPtr[kWJMaxResumeFrames] = {0};
uint32_t gWJResumeEnvPtr[kWJMaxResumeFrames] = {0};
uint32_t gWJResumeEnclosingEnv[kWJMaxResumeFrames] = {0};
uint32_t gWJDeoptByOp[js::wasm::kWJNumOps] = {0};
uint32_t gWJResumeNArgs[kWJMaxResumeFrames] = {0};
uint32_t gWJResumeNLocals[kWJMaxResumeFrames] = {0};
uint32_t gWJResumeValsOff[kWJMaxResumeFrames] = {0};
uint64_t gWJCallCallee = 0;     // boxed callee Value (set by emitted code)
uint32_t gWJCallArgc = 0;
uint64_t gWJConstructNewTarget = 0;  // boxed newTarget for constructing calls
uintptr_t gWJMarkBarrierAddr = 0;  // baked zone needs-marking-barrier flag address
uint64_t gWJGlobalLexEnvVal = 0;   // boxed global lexical env (for FunctionEnvironment)
uint32_t gWJCurrentEnv = 0;        // current fn's runtime environment (raw ptr)
int gWJExecDepth = 0;              // >0 while emitted wasm-JIT code is on the stack
                                   // (drives inWasmJit()/inJit()/inIon() in the shell)
bool gWJHadAlwaysBails = false;    // last compile emitted an alwaysBails deopt block
bool gWJForceMega = false;         // megamorphic recompile (set per-compile by the valve)
bool gWJForceNumberArith = false;  // de-speculate Int32 arith/elem (set per-compile by the valve)
const char* gWJBailReason = "unknown";  // why the last WJEmitBody returned false
uint32_t gWJBailLine = 0;          // source line of the last bailed function
uint32_t gWJNewShapeSlot = 0;      // alloc helper: shape pool index
uint32_t gWJNewAux = 0;            // alloc helper: allocKind or array length
uint32_t gWJNewHeap = 0;           // alloc helper: gc::Heap
uint32_t gWJNewObjScript = 0;      // WJH_NEWOBJECT: JSScript* (raw)
uint32_t gWJNewObjPcOff = 0;       // WJH_NEWOBJECT: bytecode offset
uintptr_t gWJNurseryPosAddr = 0;   // address of zone nursery position_ (inline alloc)
uintptr_t gWJObjHeaderWord = 0;    // NurseryCellHeader value for Object cells
uint32_t gWJHelpObj = 0;        // object ptr for WJH_SETSLOT
uint32_t gWJHelpSlot = 0;
uint64_t gWJHelpVal = 0;        // boxed value for WJH_SETSLOT
uint32_t gWJCallFn[kWJCallSites * kWJCallWays];     // polymorphic call IC: callee ptrs
int32_t gWJCallTblIdx[kWJCallSites * kWJCallWays];  // polymorphic call IC: table slots
static uint32_t gWJNextCallSite = 0;
uint32_t WJAllocCallSite() {
  if (gWJNextCallSite >= kWJCallSites) return 0;  // site 0 is a safe sentinel
  return gWJNextCallSite++;
}

// Inline property-load IC (see WasmJitBackend.h).
uint32_t gWJPropShape[kWJPropSites * kWJPropWays];
uint32_t gWJPropOff[kWJPropSites * kWJPropWays];
uint64_t gWJPropKey[kWJPropSites];
uint8_t gWJPropStrict[kWJPropSites];
static uint32_t gWJNextPropSite = 1;  // site 0 is a never-filled sentinel
uint32_t WJAllocPropSite() {
  if (gWJNextPropSite >= kWJPropSites) return 0;
  return gWJNextPropSite++;
}
// Keyed allocation: the SAME read (identified by a stable (script, read-index) key)
// REUSES its site across recompiles instead of consuming a fresh one each time.
// Without this, the deopt-storm valve's repeated mega-recompiles leak sites until
// the pool exhausts -- then a late recompile's read can't get a site, can't convert
// to a by-name EmitPropIC, and (with its shape guard already removed) reads the
// WRONG fixed slot for a polymorphic receiver (deltablue 741 "Cycle"). Reuse bounds
// the pool to the number of distinct read sites in the program. Bonus: the reused
// site keeps its warm shape cache across recompiles.
static std::unordered_map<uint64_t, uint32_t>* gWJPropSiteMap = nullptr;
uint32_t WJAllocPropSiteKeyed(uint64_t key) {
  if (!gWJPropSiteMap) gWJPropSiteMap = new std::unordered_map<uint64_t, uint32_t>();
  auto it = gWJPropSiteMap->find(key);
  if (it != gWJPropSiteMap->end()) return it->second;
  if (gWJNextPropSite >= kWJPropSites) return 0;
  uint32_t s = gWJNextPropSite++;
  (*gWJPropSiteMap)[key] = s;
  return s;
}

// GetName IC site state (see WasmJitBackend.h).
uintptr_t gWJNameHolder[kWJNameSites];
uint32_t gWJNameShape[kWJNameSites];
uint32_t gWJNameOff[kWJNameSites];
uint64_t gWJNameKey[kWJNameSites];
static uint32_t gWJNextNameSite = 1;  // site 0 is a never-allocated sentinel
uint32_t WJAllocNameSite() {
  if (gWJNextNameSite >= kWJNameSites) return 0;
  return gWJNextNameSite++;
}
void WJClearPropIC() {
  memset(gWJPropShape, 0, sizeof(gWJPropShape));
}
}  // namespace wasm
}  // namespace js

// Argument/result transfer buffer in the guest heap (a fixed global => a stable
// address the emitted wasm can compute from). Args at [0..nargs), `this` at
// [kWJThisSlot], result at [kWJResultSlot]. GC-traced by WJTraceRoots.
alignas(8) uint64_t gWJScratch[js::wasm::kWJScratchSlots];

uint64_t gWJWasmRuns = 0;
uint64_t gWJFastCalls = 0;
uint64_t gWJSlowCalls = 0;
uint64_t gWJWasmDeopts = 0;
// Set by WJH_RESUME; checked by WasmJitRunCall to tell a deopted entry from one
// that ran fully in JIT (both return flag 0), driving the deopt-storm safety valve.
static bool gWJDidResume = false;

// GC-constant pool: boxed JS::Values (object/string constants) baked into JIT'd
// modules. Traced + relocated in place by WJTraceRoots so the emitted code,
// which loads pool[i] at runtime, always sees the live pointer.
static constexpr uint32_t kWJConstPoolSize = 4096;
alignas(8) uint64_t gWJConstPool[kWJConstPoolSize];
static uint32_t gWJConstPoolCount = 0;

uintptr_t js::wasm::WJInternConstant(uint64_t valueBits) {
  // Dedupe (pools are shared across all compiled functions for the process).
  for (uint32_t i = 0; i < gWJConstPoolCount; i++) {
    if (gWJConstPool[i] == valueBits) {
      return uintptr_t(static_cast<void*>(&gWJConstPool[i]));
    }
  }
  if (gWJConstPoolCount >= kWJConstPoolSize) return 0;
  uint32_t i = gWJConstPoolCount++;
  gWJConstPool[i] = valueBits;
  return uintptr_t(static_cast<void*>(&gWJConstPool[i]));
}

// Shape pool: GuardShape bakes the address of a pool slot and loads the CURRENT
// shape pointer from it at runtime. WJTraceRoots traces + relocates each slot, so
// when a compacting GC moves a shape the guard still compares against the live
// pointer instead of a stale one (which would deopt-storm -- crypto's am3 was
// deopting ~475k times because its `this.array` shape moved post-compile).
static constexpr uint32_t kWJShapePoolSize = 4096;
uintptr_t gWJShapePool[kWJShapePoolSize];
static uint32_t gWJShapePoolCount = 0;

uintptr_t js::wasm::WJInternShape(uintptr_t shapeBits) {
  for (uint32_t i = 0; i < gWJShapePoolCount; i++) {
    if (gWJShapePool[i] == shapeBits) {
      return uintptr_t(static_cast<void*>(&gWJShapePool[i]));
    }
  }
  if (gWJShapePoolCount >= kWJShapePoolSize) return 0;
  uint32_t i = gWJShapePoolCount++;
  gWJShapePool[i] = shapeBits;
  return uintptr_t(static_cast<void*>(&gWJShapePool[i]));
}

namespace {

struct WJEntry {
  enum class State : uint8_t { Cold, Compiled, Failed };
  State state = State::Cold;
  int handle = -1;
  int tblSlot = -1;  // dense shared-table slot (-1 = not call_indirect-able)
  int directIdx = -1;  // slot in MAIN indirect table for direct PBL->JIT entry
  uint32_t jitRuns = 0;  // entries that ran fully in JIT (no deopt)
  uint32_t deopts = 0;   // entries that deopted to PBL (resume)
  uint32_t recompiles = 0;  // deopt-storm-triggered recompiles so far
  bool hasAlwaysBails = false;  // compiled fn has a cold-IC alwaysBails deopt block
  uint32_t nargs = 0;
  uint32_t nlocals = 0;
  uint32_t observes = 0;
  uint32_t nextTry = 0;  // observe count at which to (re)attempt compilation
  uint32_t fails = 0;    // failed attempts so far
  // After a GuardShape deopt-storm, recompiling immediately reproduces the same
  // MONOMORPHIC guard (the fn compiled before PBL saw the receiver's other
  // shapes). Force it to run in PBL for this many more observes first, so the
  // property ICs accumulate the polymorphic shapes -> the recompile reads a
  // polymorphic IC and Warp emits GuardShapeList (handled, no deopt) instead.
  uint32_t recompileFloor = 0;
  // Set after a (non-alwaysBails) deopt storm: the next recompile uses megamorphic
  // property codegen (multi-shape EmitPropIC, no deopt) for GuardShape-guarded
  // reads -- fixes a monomorphic GuardShape storming on a polymorphic receiver.
  bool forceMega = false;
};

// Per-script compile state. Keyed by JSScript*; entries persist for the process.
static std::unordered_map<JSScript*, WJEntry>* gEntries = nullptr;

static WJEntry& EntryFor(JSScript* script) {
  if (!gEntries) gEntries = new std::unordered_map<JSScript*, WJEntry>();
  static JSScript* sLast = nullptr;
  static WJEntry* sLastE = nullptr;
  if (script == sLast && sLastE) return *sLastE;  // skip the map find on repeats
  WJEntry& e = (*gEntries)[script];
  sLast = script;
  sLastE = &e;
  return e;
}

// Warm-up delay: WarpOracle needs PBL-attached CacheIR stubs, which only appear
// after the body has run many times. Compile after this many observations.
static uint32_t WarmupDelay() {
  static uint32_t n = 0;
  if (!n) {
    const char* s = getenv("GECKO_WJWARP_DELAY");
    n = s ? uint32_t(atoi(s)) : 200;
    if (!n) n = 200;
  }
  return n;
}

}  // namespace

bool js::wasm::WasmJitInWasm() { return js::wasm::gWJExecDepth > 0; }

bool js::wasm::WasmJitObserveCall(JSScript* script) {
  static int sEnabled = -1;
  if (sEnabled < 0) sEnabled = getenv("GECKO_NOWASMJIT") ? 0 : 1;
  if (!sEnabled) return false;
  if (!script->function() || script->isModule() || script->length() > 4096) {
    return false;
  }

  WJEntry& e = EntryFor(script);
  if (e.state == WJEntry::State::Compiled) return true;
  if (e.state == WJEntry::State::Failed) return false;

  if (e.nextTry == 0) e.nextTry = WarmupDelay();
  // Trigger on EITHER call count (our observes) OR the script's loop-aware
  // warmUpCount (PBL bumps it on every LoopHead). The latter catches the hot
  // driver functions that are entered rarely but loop thousands of times
  // internally (e.g. richards' schedule) -- by their 2nd call the accumulated
  // loop warmup is huge, so we compile them and let Warp inline their dispatch.
  ++e.observes;
  // Post-storm: stay in PBL until the IC warms polymorphic (overrides the warm-
  // path bypass below, which would otherwise recompile early + monomorphic).
  if (e.recompileFloor && e.observes < e.recompileFloor) return false;
  uint32_t warm =
      script->hasJitScript() ? script->jitScript()->warmUpCount() : 0;
  static uint32_t kLoopWarm = 0;
  if (!kLoopWarm) {
    const char* s = getenv("GECKO_WJWARP_LOOPWARM");
    kLoopWarm = s ? uint32_t(atoi(s)) : 2000;
    if (!kLoopWarm) kLoopWarm = 2000;
  }
  if (e.observes < e.nextTry && warm < kLoopWarm) return false;

  JSContext* cx = js::TlsContext.get();
  if (!cx) return false;
  uint32_t nargs = 0, nlocals = 0;
  // Preferred shared-table slot: reuse the prior slot on a deopt-storm recompile
  // (callers' call ICs cache funPtr->slot; funPtr is unchanged, so the new module
  // must take over the same slot). -1 (first compile) allocates a fresh one.
  int tblSlot = e.tblSlot;
  js::wasm::gWJForceMega = e.forceMega;  // megamorphic recompile (post-storm)
  int handle = js::wasm::WJWarpCompile(cx, script, &nargs, &nlocals, &tblSlot);
  js::wasm::gWJForceMega = false;
  if (e.forceMega && getenv("GECKO_WJ_MEGADBG")) {
    fprintf(stderr, "[wj-mega-compile] %s:%u handle=%d\n",
            script->filename() ? script->filename() : "?",
            unsigned(script->lineno()), handle);
  }
  if (handle < 0) {
    // Recompile-when-warm: a bail is often a cold IC (Warp emits an unconditional
    // bailout for an op whose baseline IC hasn't specialized yet). Retry later --
    // as the bench runs, callee/op ICs warm up and the bail disappears. Cap retries
    // so a genuinely-unsupported function eventually gives up (stays in PBL).
    const char* reason =
        js::wasm::gWJBailReason ? js::wasm::gWJBailReason : "?";
    // LOG-BAIL (non-fatal coverage audit): print each compile bail so a bench that
    // silently degrades to ~1x (hot fns in PBL) can be diagnosed. Post-process with
    // sort|uniq -c (retries repeat the same line). Off by default.
    if (getenv("GECKO_WJ_LOGBAIL")) {
      fprintf(stderr, "[WJ-BAIL] %s:%u reason=%s\n",
              script->filename() ? script->filename() : "?",
              unsigned(script->lineno()), reason);
    }
    // FAIL-ON-BAIL: a compile bail means the function runs in PBL forever -- the
    // exact "running in PBL not JIT" signal we must never miss. With FAILONBAIL set,
    // print it loudly and ABORT so any bench with an unJIT-able function insta-fails
    // instead of silently degrading to ~1x. (Off by default so normal runs tolerate
    // the rare residual bail; turn on to audit coverage.)
    if (getenv("GECKO_WJ_FAILONBAIL")) {
      fprintf(stderr,
              "\n[WJ-COMPILE-FAIL] %s:%u reason=%s -- function will run in PBL, "
              "not JIT. Aborting (GECKO_WJ_FAILONBAIL).\n",
              script->filename() ? script->filename() : "?",
              unsigned(script->lineno()), reason);
      fflush(stderr);
      MOZ_CRASH("WJ compile bail with GECKO_WJ_FAILONBAIL");
    }
    if (++e.fails >= 8) {
      e.state = WJEntry::State::Failed;
    } else {
      e.nextTry = e.observes + (WarmupDelay() << e.fails);
    }
    return false;
  }
  e.state = WJEntry::State::Compiled;
  e.handle = handle;
  e.tblSlot = tblSlot;
  e.hasAlwaysBails = js::wasm::gWJHadAlwaysBails;
  if (getenv("GECKO_WJ_COMPILECNT")) {
    static uint64_t cc = 0;
    fprintf(stderr, "[wj-compile] #%llu %s:%u hasAB=%d recomp=%u\n",
            (unsigned long long)(++cc),
            script->filename() ? script->filename() : "?",
            unsigned(script->lineno()), e.hasAlwaysBails, e.recompiles);
  }
  e.nargs = nargs;
  e.nlocals = nlocals;
  // Register the trampoline in the MAIN indirect table for direct (no-JS-hop)
  // PBL->JIT entry via a C function pointer. -1 if registration failed.
  if (!getenv("GECKO_WJ_NODIRECT")) {
    e.directIdx = int(wasmhost_call(handle, -1, nullptr, 0));
  }
  return true;
}

// Set during the differential verifier's interpreter re-run so nested calls
// decline JIT entry (run in PBL) -- otherwise the "interp" re-run would re-JIT
// and we'd compare JIT-vs-JIT (never catching a JIT bug).
static bool gWJVerifyReentry = false;

int js::wasm::WasmJitRunCall(JSScript* script, uint64_t thisBits,
                             const JS::Value* argv, uint32_t argc,
                             JSObject* envChain, uint64_t* retBits) {
  if (gWJVerifyReentry) return 0;  // verifier re-run: force PBL
  if (!gEntries) return 0;
  // Last-lookup cache: hot functions are entered repeatedly with the same script,
  // so skip the unordered_map find (per-PBL->JIT-entry cost on entry-heavy benches
  // like splay). Map element addresses are stable across rehashes, so caching the
  // WJEntry* is safe; state is re-read fresh each call.
  static JSScript* sLastScript = nullptr;
  static WJEntry* sLastEntry = nullptr;
  WJEntry* ep;
  if (script == sLastScript) {
    ep = sLastEntry;
  } else {
    auto it = gEntries->find(script);
    if (it == gEntries->end()) return 0;
    ep = &it->second;
    sLastScript = script;
    sLastEntry = ep;
  }
  WJEntry& e = *ep;
  if (e.state != WJEntry::State::Compiled) return 0;
  if (argc < e.nargs) return 0;  // underflow: let the interpreter pad

  for (uint32_t i = 0; i < e.nargs; i++) {
    gWJScratch[i] = argv[i].asRawBits();
  }
  gWJScratch[js::wasm::kWJThisSlot] = thisBits;

  // Mutation verifier (GECKO_WJ_VERIFYMUT): snapshot this/arg object slots BEFORE
  // the wasm runs, so we can later compare the wasm's mutations against a PBL
  // re-run (catches wrong SIDE EFFECTS the return-value verifier can't see).
  static int verifyMut = getenv("GECKO_WJ_VERIFYMUT") ? 1 : 0;
  struct MutSnap {
    uint32_t rootIdx;  // index into mutRoots (the GC-rooted object handle)
    uint32_t n;
    uint64_t before[16];
    uint64_t wasmAfter[16];
  };
  std::vector<MutSnap> msnaps;
  JSContext* mutCx = verifyMut ? js::TlsContext.get() : nullptr;
  // Persisted, GC-rooted handles to the snapshot objects so they survive the
  // wasm call's GC (raw JSObject* would go stale -> false garbage mutations).
  JS::RootedValueVector mutRoots(mutCx ? mutCx : js::TlsContext.get());
  if (verifyMut && mutCx) {
    auto snap = [&](uint64_t bits) {
      JS::Value v = JS::Value::fromRawBits(bits);
      if (v.isObject() && v.toObject().is<js::NativeObject>()) {
        js::NativeObject& o = v.toObject().as<js::NativeObject>();
        MutSnap sp;
        sp.rootIdx = uint32_t(mutRoots.length());
        (void)mutRoots.append(v);
        sp.n = std::min<uint32_t>(o.slotSpan(), 16);
        for (uint32_t s = 0; s < sp.n; s++) sp.before[s] = o.getSlot(s).asRawBits();
        msnaps.push_back(sp);
      }
    };
    snap(thisBits);
    for (uint32_t i = 0; i < argc; i++) snap(argv[i].asRawBits());
    // One level deep: also snapshot objects reachable through this/args' slots
    // (e.g. this.position, this.color) so nested-object mutations are caught.
    uint32_t topLevel = uint32_t(msnaps.size());
    for (uint32_t m = 0; m < topLevel; m++) {
      for (uint32_t s = 0; s < msnaps[m].n; s++) snap(msnaps[m].before[s]);
    }
  }

  // Resume context is now self-contained: the emitted deopt code sets
  // gWJResumeScriptPtr/EnvPtr/NArgs/NLocals itself, so no setup is needed here.
  // The function's runtime environment (for MFunctionEnvironment): stash it so the
  // JIT'd code reads the correct closure env at entry (no GC before it reads it).
  gWJCurrentEnv = uint32_t(uintptr_t(static_cast<void*>(envChain)));
  if (getenv("GECKO_WJ_ENVDBG")) {
    fprintf(stderr, "[wb-envdbg] ENTRY env=%u %s:%u\n", gWJCurrentEnv,
            script ? script->filename() : "?", script ? script->lineno() : 0);
  }
  double ptr = double(uintptr_t(static_cast<void*>(gWJScratch)));
  // Direct entry: call the trampoline via a C function pointer (call_indirect
  // through the MAIN indirect table) -- no wasm->JS->wasm hop. Falls back to the
  // JS shim if registration failed.
  double flag;
  gWJDidResume = false;
  gWJExecDepth++;
  if (e.directIdx >= 0) {
    typedef double (*WJTrampFn)(double);
    WJTrampFn fp = reinterpret_cast<WJTrampFn>(uintptr_t(e.directIdx));
    flag = fp(ptr);
  } else {
    flag = wasmhost_call(e.handle, 0, &ptr, 1);
  }
  gWJExecDepth--;
  // Convention: 0.0 = result ready in gWJScratch (normal completion OR sound
  // resume); 1.0 = an exception is pending (a call/resume threw) -> propagate.
  if (flag != 0.0) {
    gWJWasmDeopts++;
    return 2;  // propagate pending exception
  }
  // Deopt-storm handling: a function deopting on (almost) every entry compiled
  // with stale/monomorphic IC info. RECOMPILE it (Ion-style) -- by now PBL has
  // populated its ICs with the shapes that caused the deopts, so Warp should emit
  // a polymorphic/megamorphic load instead of the failing monomorphic GuardShape,
  // keeping it in JIT (the goal: almost never engage PBL). Cap recompiles; only
  // after repeated failure fall back to PBL (last resort).
  if (gWJDidResume) {
    e.deopts++;
    static int valveN = getenv("GECKO_WJ_VALVEN") ? atoi(getenv("GECKO_WJ_VALVEN")) : 300;
    if (e.deopts >= uint32_t(valveN) && e.deopts > (e.jitRuns + 1) &&
        !getenv("GECKO_WJ_NODEOPTVALVE")) {
      // A storming fn with an alwaysBails block deopts from that COLD-IC block;
      // recompiling reproduces it identically (verified: deltablue 741's recompiled
      // MIR is byte-identical), so recompiling is pointless AND the recompile churn
      // corrupts state (deltablue wrong-value/GC-crash). Go straight to PBL. A
      // storming fn WITHOUT one deopts from a stale monomorphic guard -> recompile
      // to specialize it polymorphic (crypto 3.1x). (Cold branches that NEVER fire,
      // e.g. navier lin_solve's a===0, never reach here: deopts stays 0.)
      if (getenv("GECKO_WJ_VALVEDBG")) {
        fprintf(stderr, "[wj-valve] %s:%u hasAB=%d deopts=%u jitRuns=%u recomp=%u len=%u -> %s\n",
                script->filename() ? script->filename() : "?",
                unsigned(script->lineno()), e.hasAlwaysBails, e.deopts, e.jitRuns,
                e.recompiles, unsigned(script->length()),
                e.hasAlwaysBails ? "FAIL" : "recompile");
      }
      if (e.hasAlwaysBails) {
        e.state = WJEntry::State::Failed;
        return 0;
      }
      // A TINY function that storms via polymorphic guards (hasAB=0): mega-recompile
      // makes it deopt-free but adds per-access IC overhead, and for a tiny body the
      // JIT call/GC-root/boxing overhead exceeds the body's work -> PBL is faster
      // (deltablue's constraint accessors: output/input/isSatisfied, ~30-80 bytecodes,
      // called polymorphically from the planner). Larger storming bodies (crypto
      // bignum loops) still recompile-mega (the loop work dominates the overhead, so
      // staying in JIT wins). Tunable via GECKO_WJ_TINYBAIL (default 120; 0 disables).
      {
        static int tinyBail = getenv("GECKO_WJ_TINYBAIL")
                                  ? atoi(getenv("GECKO_WJ_TINYBAIL"))
                                  : 0;
        if (tinyBail > 0 && e.recompiles == 0 &&
            script->length() < uint32_t(tinyBail)) {
          e.state = WJEntry::State::Failed;
          return 0;
        }
      }
      // Mega-recompile budget. The FIRST storm triggers a megamorphic recompile
      // (GuardShape-guarded reads -> multi-shape EmitPropIC, no deopt on polymorphic
      // receivers). crypto's storming bignum loops become deopt-free this way and
      // stay fast in JIT. But if a function STILL reaches the storm threshold AFTER
      // being mega-recompiled, its deopts are from guards mega CAN'T convert
      // (BoxNonStrictThis / fallible Unbox / call-guards on a genuinely polymorphic
      // receiver -- deltablue's constraint accessors). More mega passes are
      // identical and pointless; the function is deopt-bound and SLOWER than PBL
      // (deltablue regressed to a timeout here -- ~30% of millions of calls deopt,
      // below the 50% re-trigger but enough to crawl). Fall back to PBL (the valve's
      // documented last resort). GECKO_WJ_MEGARETRIES tunes the budget (default 1).
      static int megaRetries = getenv("GECKO_WJ_MEGARETRIES")
                                   ? atoi(getenv("GECKO_WJ_MEGARETRIES"))
                                   : 1;
      if (int(e.recompiles) < megaRetries) {
        e.recompiles++;
        e.forceMega = true;
        e.state = WJEntry::State::Cold;  // re-observe + recompile with fresh ICs
        e.handle = -1;
        e.directIdx = -1;
        // KEEP e.tblSlot: the recompile reuses it so callers' cached call ICs
        // (funPtr->slot, funPtr unchanged) keep resolving to the new module.
        // Resetting it to -1 allocated a NEW slot, leaving cached ICs pointing at
        // the stale old module -> deltablue wrong results.
        e.deopts = 0;
        e.jitRuns = 0;
        e.observes = 0;
        e.nextTry = 0;
        return 0;
      }
      // Still storming after the mega recompile(s): deopt-bound, slower than PBL.
      e.state = WJEntry::State::Failed;
      return 0;
    }
  } else {
    e.jitRuns++;
  }
  gWJWasmRuns++;
  if (((gWJFastCalls+gWJSlowCalls) % 20000)==0 && (gWJFastCalls+gWJSlowCalls)>0 && (getenv("GECKO_WJWARP_DUMP")||getenv("GECKO_DEBUG_JIT"))) fprintf(stderr, "[wb-calls] fast=%llu slow=%llu\n", (unsigned long long)gWJFastCalls,(unsigned long long)gWJSlowCalls);
  if (((gWJWasmRuns + gWJWasmDeopts) % 5000) == 0 &&
      (getenv("GECKO_WJWARP_DUMP") || getenv("GECKO_DEBUG_JIT"))) {
    fprintf(stderr, "[wb-stats] wasm runs=%llu deopts=%llu\n",
            (unsigned long long)gWJWasmRuns, (unsigned long long)gWJWasmDeopts);
  }
  *retBits = gWJScratch[js::wasm::kWJResultSlot];

  if (verifyMut && !msnaps.empty() && mutCx) {
    JSContext* cx = mutCx;
    if (!cx->isExceptionPending()) {
      // Save the wasm's mutations (via the GC-rooted, updated object handles),
      // then restore originals so the PBL re-run sees the same starting state.
      for (auto& sp : msnaps) {
        js::NativeObject& o = mutRoots[sp.rootIdx].toObject().as<js::NativeObject>();
        for (uint32_t s = 0; s < sp.n; s++) sp.wasmAfter[s] = o.getSlot(s).asRawBits();
        for (uint32_t s = 0; s < sp.n; s++)
          o.setSlot(s, JS::Value::fromRawBits(sp.before[s]));
      }
      RootedValue fval(cx, JS::ObjectValue(*script->function()));
      RootedValue thisv(cx, JS::Value::fromRawBits(thisBits));
      JS::RootedValueVector av(cx);
      bool okv = av.reserve(argc);
      for (uint32_t i = 0; okv && i < argc; i++) av.infallibleAppend(argv[i]);
      RootedValue rv(cx);
      gWJVerifyReentry = true;
      bool okcall = okv && JS::Call(cx, thisv, fval, JS::HandleValueArray(av), &rv);
      gWJVerifyReentry = false;
      if (okcall) {
        for (auto& sp : msnaps) {
          js::NativeObject& o =
              mutRoots[sp.rootIdx].toObject().as<js::NativeObject>();
          for (uint32_t s = 0; s < sp.n; s++) {
            if (o.getSlot(s).asRawBits() != sp.wasmAfter[s]) {
              static int mn = 0;
              if (mn++ < 40) {
                JS::Value wv = JS::Value::fromRawBits(sp.wasmAfter[s]);
                JS::Value pv = o.getSlot(s);
                fprintf(stderr,
                        "[wb-MUT] %s:%u slot%u wasm=%g/%s pbl=%g/%s\n",
                        script->filename(), uint32_t(script->lineno()), s,
                        wv.isNumber() ? wv.toNumber() : -1,
                        wv.isObject() ? "obj" : wv.isUndefined() ? "undef" : "prim",
                        pv.isNumber() ? pv.toNumber() : -1,
                        pv.isObject() ? "obj" : pv.isUndefined() ? "undef" : "prim");
              }
            }
          }
        }
      }
    }
  }

  // Differential verifier (GECKO_WJ_VERIFY): re-run the call in the interpreter
  // and compare. Only sound for pure functions (re-runs side effects), so a
  // debugging aid, not a correctness mechanism.
  static int verify = getenv("GECKO_WJ_VERIFY") ? 1 : 0;
  if (verify) {
    uint64_t wasmRes = *retBits;
    JSContext* cx = js::TlsContext.get();
    if (cx && !cx->isExceptionPending()) {
      RootedValue fval(cx, JS::ObjectValue(*script->function()));
      RootedValue thisv(cx, JS::Value::fromRawBits(thisBits));
      JS::RootedValueVector av(cx);
      bool okv = av.reserve(argc);
      for (uint32_t i = 0; okv && i < argc; i++) av.infallibleAppend(argv[i]);
      RootedValue rv(cx);
      // ROOT the wasm result across the re-run: JS::Call GCs, which would move a
      // GC-thing result and leave `wasmRes` (raw bits) a stale pointer -> false
      // "garbage class" mismatch. The Rooted value is updated by the GC.
      RootedValue wasmRooted(cx, JS::Value::fromRawBits(wasmRes));
      gWJVerifyReentry = true;  // force the re-run to use PBL (not re-JIT)
      bool okcall = okv && JS::Call(cx, thisv, fval, JS::HandleValueArray(av), &rv);
      gWJVerifyReentry = false;
      if (okcall) {
        JS::Value wasmV = wasmRooted;
        bool gcResult = wasmV.isGCThing() || rv.isGCThing();
        // Strong wrongness signal: JIT returned a primitive/undefined where the
        // interpreter produced an object (or different class), OR both are native
        // objects with differing fixed-slot contents. Catches "returns undefined
        // / wrong-field Vector" -- the raytrace failure mode.
        bool kindMismatch =
            (wasmV.isObject() != rv.isObject()) ||
            (wasmV.isObject() && rv.isObject() &&
             wasmV.toObject().getClass() != rv.toObject().getClass());
        static int slotdbg = getenv("GECKO_WJ_SLOTDBG") ? 1 : 0;
        if (!kindMismatch && wasmV.isObject() && rv.isObject() &&
            wasmV.toObject().is<js::NativeObject>() &&
            rv.toObject().is<js::NativeObject>()) {
          js::NativeObject& wo = wasmV.toObject().as<js::NativeObject>();
          js::NativeObject& io = rv.toObject().as<js::NativeObject>();
          uint32_t n = std::min(wo.slotSpan(), io.slotSpan());
          for (uint32_t s = 0; s < n && s < 8; s++) {
            if (wo.getSlot(s).asRawBits() != io.getSlot(s).asRawBits()) {
              kindMismatch = true;
              if (slotdbg) {
                JS::Value wv = wo.getSlot(s), iv = io.getSlot(s);
                auto ty = [](JS::Value v) {
                  return v.isInt32() ? "i32" : v.isDouble() ? "dbl"
                         : v.isObject() ? "obj" : v.isUndefined() ? "undef"
                         : v.isBoolean() ? "bool" : "other";
                };
                fprintf(stderr,
                        "[wb-SLOT] %s:%u slot%u wasm=%g/%s interp=%g/%s\n",
                        script->filename(), uint32_t(script->lineno()), s,
                        wv.isNumber() ? wv.toNumber() : -999, ty(wv),
                        iv.isNumber() ? iv.toNumber() : -999, ty(iv));
              }
              break;
            }
          }
        }
        if ((!gcResult && rv.asRawBits() != wasmRes) || kindMismatch) {
          static int vn = 0;
          if (vn++ < 30)
            fprintf(stderr,
                    "[wb-VERIFY] MISMATCH %s:%u argc=%u wasm=%s interp=%s\n",
                    script->filename() ? script->filename() : "?",
                    uint32_t(script->lineno()), argc,
                    wasmV.isObject() ? wasmV.toObject().getClass()->name
                                     : (wasmV.isUndefined() ? "undef"
                                        : wasmV.isNumber() ? "num" : "prim"),
                    rv.isObject() ? rv.toObject().getClass()->name
                                  : (rv.isUndefined() ? "undef"
                                     : rv.isNumber() ? "num" : "prim"));
        }
      } else if (cx->isExceptionPending()) {
        cx->clearPendingException();
      }
    }
  }
  return 1;
}

// wasm -> C++ trampoline imported by JIT'd modules ("m"."help"). Returns 0.0 on
// success (result, if any, in gWJScratch[kWJResultSlot]) or 1.0 if it threw.
extern "C" EMSCRIPTEN_KEEPALIVE double wjhelp(double kindF, double siteF) {
  JSContext* cx = js::TlsContext.get();
  if (!cx) return 1.0;
  int kind = int(kindF);

  static int helpHist = getenv("GECKO_WJ_HELPHIST") ? 1 : 0;
  if (helpHist) {
    static uint64_t hc[32] = {0};
    static uint64_t tot = 0;
    if (kind >= 0 && kind < 32) hc[kind]++;
    if ((++tot % 200000) == 0) {
      static const char* nm[16] = {"?",        "RESUME",  "CALL",
                                   "SETSLOT",  "GETPROP", "SETPROP",
                                   "GETELEM",  "INSTOF",  "ARRPUSH",
                                   "ARRPOP",   "CRTHIS",  "CONSTRUCT",
                                   "POSTBAR",  "PREBAR",  "?14",  "?15"};
      fprintf(stderr, "[wb-helphist] %llu calls:", (unsigned long long)tot);
      for (int k = 0; k < 16; k++)
        if (hc[k]) fprintf(stderr, " %s=%llu", nm[k], (unsigned long long)hc[k]);
      fprintf(stderr, "\n");
    }
  }

  if (kind == js::wasm::WJH_RESUME) {
    gWJDidResume = true;  // this JIT entry deopted (safety-valve accounting)
    if(getenv("GECKO_DEBUG_JIT")){static uint64_t c=0; if((++c%5000)==0) fprintf(stderr,"[wb-resume-count] %llu\n",(unsigned long long)c);}
    if (getenv("GECKO_WJ_DEOPTHIST")) {
      static uint64_t dc = 0;
      uint64_t dmod = getenv("GECKO_WJ_DEOPTHISTN") ? 10 : 200;
      if ((++dc % dmod) == 0) {
        fprintf(stderr, "[wb-deopthist] after %llu resumes:\n",
                (unsigned long long)dc);
        for (uint32_t o = 0; o < js::wasm::kWJNumOps; o++)
          if (gWJDeoptByOp[o])
            fprintf(stderr, "[wb-deopthist]   op#%u = %u\n", o, gWJDeoptByOp[o]);
      }
    }
    if (getenv("GECKO_WJ_STORMLINE")) {
      static uint64_t sc = 0;
      ++sc;
      uint64_t mod = getenv("GECKO_WJ_STORMFIRST") ? 1 : 2000;
      uint64_t cap = getenv("GECKO_WJ_STORMFIRST") ? 40 : ~0ull;
      if (sc <= cap && (sc % mod) == 0 && gWJResumeNFrames) {
        uint32_t lf = gWJResumeNFrames - 1;
        JSScript* ds =
            reinterpret_cast<JSScript*>(uintptr_t(gWJResumeScriptPtr[lf]));
        if (ds) {
          uint32_t pcoff = gWJResumePc[lf];
          const char* opname = "?";
          if (pcoff < ds->length()) {
            JSOp jop = JSOp(*(ds->code() + pcoff));
            opname = js::CodeName(jop);
          }
          fprintf(stderr, "[wj-storm] %s:%u pc=%u op=%s (resumes=%llu)\n",
                  ds->filename() ? ds->filename() : "?", unsigned(ds->lineno()),
                  pcoff, opname, (unsigned long long)sc);
        }
      }
    }
    // Multi-frame inline bailout: run frames innermost (0) -> outermost. Each
    // frame's return is threaded into the next outer frame's call-result stack
    // slot (the top of its expr stack at the resume-after-call point).
    uint32_t nframes = gWJResumeNFrames;
    if (nframes == 0 || nframes > js::wasm::kWJMaxResumeFrames) return 1.0;
    static int rdbg =
        (getenv("GECKO_WJWARP_DUMP") || getenv("GECKO_WJ_RESUMEDBG")) ? 1 : 0;
    static int rn = 0;
    uint64_t rbits = JS::UndefinedValue().asRawBits();
    bool haveInner = false;
    for (uint32_t f = 0; f < nframes; f++) {
      RootedScript script(
          cx, reinterpret_cast<JSScript*>(uintptr_t(gWJResumeScriptPtr[f])));
      if (!script || !script->hasJitScript()) return 1.0;
      JSObject* envObj = reinterpret_cast<JSObject*>(uintptr_t(gWJResumeEnvPtr[f]));
      if (!envObj && script->function()) envObj = script->function()->environment();
      RootedObject env(cx, envObj);
      // Correct enclosing env for the frame's PBL prologue (see header). 0 ->
      // PBL falls back to the canonical func->environment().
      RootedObject enclosingEnv(
          cx, reinterpret_cast<JSObject*>(uintptr_t(gWJResumeEnclosingEnv[f])));
      // NOTE: keepFrameEnv (reuse the captured gWJResumeEnvPtr as the frame env,
      // skipping InitFunctionEnvironmentObjects) was tried to fix navier's mid-
      // function-deopt checksum bug, but gWJResumeEnvPtr is unreliable (wrong for
      // earley -> regressed it, no help for navier). Disabled; use the enclosing-
      // env + Init path (earley correct). navier's resume bug needs gWJResumeEnvPtr
      // to be made reliable first (EmitObjPtr of the RP env operand). See memory.
      bool keepFrameEnv = false;
      uint32_t nargs = gWJResumeNArgs[f];
      uint32_t nlocals = gWJResumeNLocals[f];
      uint32_t depth = gWJResumeStackDepth[f];
      uint32_t off = gWJResumeValsOff[f];
      uint32_t total = 1 + nargs + nlocals + depth;
      // Caller (inlined-call) frames resume AT the call with the call inputs on
      // the expr stack (ResumeAt), so PBL RE-EXECUTES the call -- re-running the
      // callee normally. We therefore do NOT thread the inner frame's return;
      // running the inner frame first is only used for the innermost (deopt)
      // frame, and outer frames simply re-execute. (haveInner kept for clarity.)
      (void)haveInner;
      JS::RootedValueVector vals(cx);
      if (!vals.reserve(total)) return 1.0;
      static int rchk = getenv("GECKO_WJ_RESUMECHK") ? 1 : 0;
      for (uint32_t i = 0; i < total; i++) {
        uint64_t bits = gWJResumeVals[off + i];
        if (rchk) {
          JS::Value v = JS::Value::fromRawBits(bits);
          bool ok = v.isObject() || v.isString() || v.isSymbol() ||
                    v.isBigInt() || v.isInt32() || v.isDouble() || v.isBoolean() ||
                    v.isNull() || v.isUndefined() || v.isMagic();
          if (!ok || (v.isGCThing() && (uintptr_t(v.toGCThing()) < 0x1000 ||
                                        (uintptr_t(v.toGCThing()) & 7)))) {
            const char* kind = i == 0 ? "this"
                               : i < 1 + nargs ? "arg"
                               : i < 1 + nargs + nlocals ? "local" : "stack";
            fprintf(stderr,
                    "[wj-badval] %s:%u f=%u slot=%u(%s) bits=%016llx gcthing=%d\n",
                    script->filename() ? script->filename() : "?",
                    unsigned(script->lineno()), f, i, kind,
                    (unsigned long long)bits, v.isGCThing());
          }
        }
        vals.infallibleAppend(JS::Value::fromRawBits(bits));
      }
      uint64_t thisBits = vals[0].asRawBits();
      const JS::Value* args = vals.begin() + 1;
      const uint64_t* locals =
          reinterpret_cast<const uint64_t*>(vals.begin() + 1 + nargs);
      const uint64_t* stack =
          reinterpret_cast<const uint64_t*>(vals.begin() + 1 + nargs + nlocals);
      if (rdbg && rn < 4000) {
        rn++;
        fprintf(stderr,
                "[wb-resume] frame %u/%u %s:%u pc=%u nargs=%u nlocals=%u depth=%u this=%s",
                f, nframes, script->filename() ? script->filename() : "?",
                uint32_t(script->lineno()), gWJResumePc[f], nargs, nlocals, depth,
                vals[0].isObject() ? (vals[0].toObject().is<JSFunction>() ? "fn" : "obj")
                                   : (vals[0].isUndefined() ? "undef" : "prim"));
        for (uint32_t i = 0; i < nargs && i < 6; i++) {
          JS::Value av = vals[1 + i];
          if (av.isInt32()) fprintf(stderr, " a%u=i%d", i, av.toInt32());
          else if (av.isDouble()) fprintf(stderr, " a%u=d%g", i, av.toDouble());
          else fprintf(stderr, " a%u=%s", i, av.isObject() ? "obj" : "prim");
        }
        for (uint32_t i = 0; i < nlocals && i < 6; i++) {
          JS::Value lv = vals[1 + nargs + i];
          if (lv.isInt32()) fprintf(stderr, " L%u=i%d", i, lv.toInt32());
          else if (lv.isDouble()) fprintf(stderr, " L%u=d%g", i, lv.toDouble());
          else fprintf(stderr, " L%u=%s", i,
                       lv.isUndefined() ? "undef" : lv.isObject() ? "obj" : "prim");
        }
        for (uint32_t i = 0; i < depth && i < 6; i++) {
          JS::Value sv = vals[1 + nargs + nlocals + i];
          if (sv.isInt32()) fprintf(stderr, " S%u=i%d", i, sv.toInt32());
          else if (sv.isDouble()) fprintf(stderr, " S%u=d%g", i, sv.toDouble());
          else if (sv.isBoolean()) fprintf(stderr, " S%u=b%d", i, sv.toBoolean());
          else fprintf(stderr, " S%u=%s", i,
                       sv.isUndefined() ? "undef" : sv.isObject()
                       ? (sv.toObject().is<JSFunction>() ? "fn" : "obj") : "prim");
        }
        fprintf(stderr, "\n");
      }
      if (!js::pbl::WasmJitResumeViaPBL(cx, script, thisBits, args, nargs, env,
                                        locals, nlocals, gWJResumePc[f], &rbits,
                                        stack, depth, enclosingEnv, keepFrameEnv)) {
        return 1.0;  // resumed execution threw -> propagate
      }
      haveInner = true;
    }
    gWJScratch[js::wasm::kWJResultSlot] = rbits;
    return 0.0;
  }

  if (kind == js::wasm::WJH_CALL) {
    RootedValue callee(cx, JS::Value::fromRawBits(gWJCallCallee));
    uint32_t argc = gWJCallArgc;

    // Fast path: if the callee is itself a compiled WJ function, run its wasm
    // directly (its args + `this` are already staged in gWJScratch by the
    // caller's marshalling), skipping JS::Call's generic dispatch machinery.
    // GECKO_WJ_FORCESLOW (debug) skips this so callees re-enter via JS::Call ->
    // WasmJitRunCall, where the differential verifier can check them.
    static int forceSlow = getenv("GECKO_WJ_FORCESLOW") ? 1 : 0;
    if (!forceSlow && gEntries && callee.isObject() &&
        callee.toObject().is<JSFunction>()) {
      JSFunction* fun = &callee.toObject().as<JSFunction>();
      if (fun->isInterpreted() && fun->hasBytecode()) {
        JSScript* cs = fun->nonLazyScript();
        // Drive compilation of JIT-only callees: a function called ONLY from
        // already-compiled code never reaches the interpreter/PBL observe hook,
        // so without this it stays COLD forever and every call falls to the slow
        // JS::Call path (richards: 160k slow calls). Observing here warms it to
        // the threshold, compiles it, and lets subsequent calls take the direct
        // wasm path + fill the inline call IC.
        js::wasm::WasmJitObserveCall(cs);
        auto it = gEntries->find(cs);
        if (it != gEntries->end() &&
            it->second.state == WJEntry::State::Compiled &&
            argc >= it->second.nargs) {
          WJEntry& ce = it->second;
          // Fill the caller's call-site IC so the next call goes direct
          // (call_indirect) with no helper hop. site is wjhelp's 2nd arg.
          uint32_t site = uint32_t(siteF);
          // Fill a free way of the polymorphic IC (or refresh the matching one).
          // Only if the callee has a real table slot; else stays on the slow path.
          static int noICFill = getenv("GECKO_WJ_NOICFILL") ? 1 : 0;
          if (noICFill) { /* force all calls through this C++ path (debug) */ }
          else if (site < kWJCallSites && ce.tblSlot >= 0) {
            uint32_t funPtr = uint32_t(uintptr_t(static_cast<void*>(fun)));
            uint32_t base = site * kWJCallWays;
            uint32_t w = 0;
            for (; w < kWJCallWays; w++) {
              if (gWJCallFn[base + w] == 0 || gWJCallFn[base + w] == funPtr) break;
            }
            if (w == kWJCallWays) w = 0;  // all ways full: evict way 0
            gWJCallFn[base + w] = funPtr;
            gWJCallTblIdx[base + w] = ce.tblSlot;
          }
          // Resume context is self-contained (emitted code sets it), so just run.
          gWJCurrentEnv = uint32_t(uintptr_t(static_cast<void*>(fun->environment())));
          if (getenv("GECKO_WJ_ENVDBG")) {
            js::BaseScript* dsc = fun->hasBaseScript() ? fun->baseScript() : nullptr;
            fprintf(stderr, "[wb-envdbg] WJH_CALL fun=%p env=%u %s:%u\n",
                    (void*)fun, gWJCurrentEnv, dsc ? dsc->filename() : "?",
                    dsc ? dsc->lineno() : 0);
          }
          double ptr = double(uintptr_t(static_cast<void*>(gWJScratch)));
          double flag = wasmhost_call(ce.handle, 0, &ptr, 1);
          gWJFastCalls++;
          return flag;  // 0 = result in gWJScratch[result]; 1 = threw
        }
      }
    }

    RootedValue thisv(cx, JS::Value::fromRawBits(gWJScratch[js::wasm::kWJThisSlot]));
    JS::RootedValueVector argv(cx);
    if (!argv.reserve(argc)) return 1.0;
    for (uint32_t i = 0; i < argc; i++) {
      argv.infallibleAppend(JS::Value::fromRawBits(gWJScratch[i]));
    }
    RootedValue rval(cx);
    gWJSlowCalls++;
    if (!JS::Call(cx, thisv, callee, JS::HandleValueArray(argv), &rval)) {
      return 1.0;  // callee threw -> propagate
    }
    gWJScratch[js::wasm::kWJResultSlot] = rval.asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_POSTBARRIER) {
    JSObject* obj = reinterpret_cast<JSObject*>(uintptr_t(gWJHelpObj));
    // PostWriteBarrier (putWholeCellDontCheckLast) assumes a TENURED container;
    // buffering a nursery cell corrupts the store buffer. Ion inlines this guard.
    if (obj && obj->isTenured()) {
      js::jit::PostWriteBarrier(cx->runtime(), obj);
    }
    return 0.0;
  }

  if (kind == js::wasm::WJH_PREBARRIER) {
    // Incremental-GC pre-write barrier on the OLD value being overwritten. Only
    // reached when the zone's marking-barrier flag is set (the emitted fast path
    // skips otherwise), so this is rare. ValuePreWriteBarrier no-ops non-GC values.
    js::gc::ValuePreWriteBarrier(JS::Value::fromRawBits(gWJHelpVal));
    return 0.0;
  }

  if (kind == js::wasm::WJH_SETSLOT) {
    JSObject* obj = reinterpret_cast<JSObject*>(uintptr_t(gWJHelpObj));
    if (getenv("GECKO_WJ_SLOTDBG2")) {
      js::NativeObject& no = obj->as<js::NativeObject>();
      fprintf(stderr, "[wb-setslot] obj=%p class=%s nfixed=%u slotSpan=%u slot=%u\n",
              (void*)obj, obj->getClass()->name, no.numFixedSlots(),
              uint32_t(no.slotSpan()), gWJHelpSlot);
    }
    obj->as<js::NativeObject>().setSlot(gWJHelpSlot,
                                        JS::Value::fromRawBits(gWJHelpVal));
    return 0.0;
  }

  // Generic VM-op helpers. gWJScratch is GC-traced (WJTraceRoots), so the staged
  // operands and result survive any GC the operation triggers.
  if (kind == js::wasm::WJH_GETPROP || kind == js::wasm::WJH_GETELEM) {
    RootedValue lref(cx, JS::Value::fromRawBits(gWJScratch[0]));
    RootedValue rref(cx, JS::Value::fromRawBits(gWJScratch[1]));
    RootedValue res(cx);
    if (!js::GetElementOperation(cx, lref, rref, &res)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = res.asRawBits();
    return 0.0;
  }

  // Generic binary arithmetic (MBinaryCache): a cold/unspecialized arith IC. The
  // operation is identified by the bytecode JSOp passed as `site`. Mirrors the
  // bytecode interpreter / DoBinaryArithFallback -- correct for any operand types
  // (int/double/bigint/string-concat). Operands boxed in scratch[0]/[1].
  if (kind == js::wasm::WJH_BINARYARITH) {
    JSOp op = JSOp(int(siteF));
    if (getenv("GECKO_WJ_ARITHDBG"))
      fprintf(stderr, "[wj-arith] BIN op=%d lhs=%016llx rhs=%016llx\n", int(op),
              (unsigned long long)gWJScratch[0], (unsigned long long)gWJScratch[1]);
    RootedValue lhs(cx, JS::Value::fromRawBits(gWJScratch[0]));
    RootedValue rhs(cx, JS::Value::fromRawBits(gWJScratch[1]));
    RootedValue res(cx);
    bool ok;
    switch (op) {
      case JSOp::Add: ok = js::AddValues(cx, &lhs, &rhs, &res); break;
      case JSOp::Sub: ok = js::SubValues(cx, &lhs, &rhs, &res); break;
      case JSOp::Mul: ok = js::MulValues(cx, &lhs, &rhs, &res); break;
      case JSOp::Div: ok = js::DivValues(cx, &lhs, &rhs, &res); break;
      case JSOp::Mod: ok = js::ModValues(cx, &lhs, &rhs, &res); break;
      case JSOp::Pow: ok = js::PowValues(cx, &lhs, &rhs, &res); break;
      case JSOp::BitOr: ok = js::BitOr(cx, &lhs, &rhs, &res); break;
      case JSOp::BitAnd: ok = js::BitAnd(cx, &lhs, &rhs, &res); break;
      case JSOp::BitXor: ok = js::BitXor(cx, &lhs, &rhs, &res); break;
      case JSOp::Lsh: ok = js::BitLsh(cx, &lhs, &rhs, &res); break;
      case JSOp::Rsh: ok = js::BitRsh(cx, &lhs, &rhs, &res); break;
      case JSOp::Ursh: ok = js::UrshValues(cx, &lhs, &rhs, &res); break;
      default: return 1.0;  // unsupported op (shouldn't be emitted)
    }
    if (!ok) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = res.asRawBits();
    return 0.0;
  }

  // Generic unary arithmetic (MUnaryCache). Operand boxed in scratch[0]; JSOp in
  // `site`. Matches jit::DoUnaryArithFallback exactly (Inc/Dec assume an already-
  // numeric operand -- the bytecode applies ToNumeric before them).
  if (kind == js::wasm::WJH_UNARYARITH) {
    JSOp op = JSOp(int(siteF));
    if (getenv("GECKO_WJ_ARITHDBG"))
      fprintf(stderr, "[wj-arith] UN op=%d val=%016llx\n", int(op),
              (unsigned long long)gWJScratch[0]);
    RootedValue val(cx, JS::Value::fromRawBits(gWJScratch[0]));
    RootedValue res(cx, val);
    bool ok;
    switch (op) {
      case JSOp::BitNot: ok = js::BitNot(cx, &res, &res); break;
      case JSOp::Pos: ok = js::ToNumber(cx, &res); break;
      case JSOp::Neg: ok = js::NegOperation(cx, &res, &res); break;
      case JSOp::Inc: ok = js::IncOperation(cx, val, &res); break;
      case JSOp::Dec: ok = js::DecOperation(cx, val, &res); break;
      case JSOp::ToNumeric: ok = js::ToNumeric(cx, &res); break;
      default: return 1.0;
    }
    if (!ok) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = res.asRawBits();
    return 0.0;
  }

  // Generic comparison (Boolean MBinaryCache): a cold/untyped ==,!=,<,<=,>,>=,===,
  // !== that our inline Compare codegen can't type. scratch[0]=lhs, scratch[1]=rhs,
  // site=JSOp. Mirrors jit::DoCompareFallback -> the VM comparison ops. Result is a
  // boxed Boolean. (Without this, such a function bails WHOLE to PBL -- deltablue's
  // constraint comparisons -> slow.)
  if (kind == js::wasm::WJH_COMPARE) {
    JSOp op = JSOp(int(siteF));
    RootedValue lhs(cx, JS::Value::fromRawBits(gWJScratch[0]));
    RootedValue rhs(cx, JS::Value::fromRawBits(gWJScratch[1]));
    bool b = false, ok = true;
    switch (op) {
      case JSOp::Eq: ok = js::LooselyEqual(cx, lhs, rhs, &b); break;
      case JSOp::Ne: ok = js::LooselyEqual(cx, lhs, rhs, &b); b = !b; break;
      case JSOp::StrictEq: ok = js::StrictlyEqual(cx, lhs, rhs, &b); break;
      case JSOp::StrictNe: ok = js::StrictlyEqual(cx, lhs, rhs, &b); b = !b; break;
      case JSOp::Lt: ok = js::LessThan(cx, &lhs, &rhs, &b); break;
      case JSOp::Le: ok = js::LessThanOrEqual(cx, &lhs, &rhs, &b); break;
      case JSOp::Gt: ok = js::GreaterThan(cx, &lhs, &rhs, &b); break;
      case JSOp::Ge: ok = js::GreaterThanOrEqual(cx, &lhs, &rhs, &b); break;
      default: return 1.0;
    }
    if (!ok) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::BooleanValue(b).asRawBits();
    return 0.0;
  }

  // Object-literal allocation (MNewObject, ObjectLiteral mode -- no compile-time
  // template). The shape is in the bytecode; NewObjectOperation(script, pc) builds
  // it. splay GeneratePayloadTree's `{array:..,string:..}` / `{left:..,right:..}`.
  if (kind == js::wasm::WJH_NEWOBJECT) {
    RootedScript script(cx, reinterpret_cast<JSScript*>(uintptr_t(gWJNewObjScript)));
    if (!script) return 1.0;
    jsbytecode* pc = script->code() + gWJNewObjPcOff;
    JSObject* o = js::NewObjectOperation(cx, script, pc);
    if (!o) return 1.0;
    if (getenv("GECKO_WJ_NEWOBJDBG")) {
      static uint64_t c = 0;
      if ((++c % 50000) == 1) {
        js::NativeObject* no = &o->as<js::NativeObject>();
        fprintf(stderr,
                "[wj-newobj-rt] op=%s nfixed=%u slotSpan=%u class=%s\n",
                js::CodeName(JSOp(*pc)), no->numFixedSlots(),
                unsigned(no->slotSpan()), o->getClass()->name);
      }
    }
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*o).asRawBits();
    return 0.0;
  }

  // BindName (MBindNameCache): resolve the env object holding `name`'s binding for
  // a following SetName. scratch[0]=env chain, scratch[1]=name(StringValue).
  // gbemu deopt-stormed here before this case existed.
  if (kind == js::wasm::WJH_BINDNAME) {
    RootedObject env(cx, &JS::Value::fromRawBits(gWJScratch[0]).toObject());
    JSString* s = JS::Value::fromRawBits(gWJScratch[1]).toString();
    Rooted<js::PropertyName*> name(cx, s->asAtom().asPropertyName());
    JSObject* holder = js::LookupNameUnqualified(cx, name, env);
    if (!holder) return 1.0;  // error -> deopt/throw path
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*holder).asRawBits();
    return 0.0;
  }

  // AllocateAndStoreSlot's slot growth (MAllocateAndStoreSlot). scratch[0]=object,
  // gWJNewAux=new dynamic-slot capacity. growSlotsPure mallocs the slots buffer
  // (no GC of JS objects, so obj doesn't move) and returns false on OOM without a
  // pending exception -> report it so the [1.0] flag path throws cleanly.
  if (kind == js::wasm::WJH_GROWSLOTS) {
    js::NativeObject* obj =
        &JS::Value::fromRawBits(gWJScratch[0]).toObject().as<js::NativeObject>();
    if (!js::NativeObject::growSlotsPure(cx, obj, uint32_t(gWJNewAux))) {
      js::ReportOutOfMemory(cx);
      return 1.0;
    }
    return 0.0;
  }

  // Global/lexical name lookup (MGetNameCache) + IC fill. scratch[0]=env chain
  // object, scratch[1]=name (StringValue atom, baked). site=siteF.
  if (kind == js::wasm::WJH_GETNAME) {
    uint32_t site = uint32_t(siteF);
    if (getenv("GECKO_WJ_NAMEDBG")) {
      static uint64_t calls = 0, fills = 0;
      if ((++calls % 200000) == 0)
        fprintf(stderr, "[wj-name] calls=%llu (helper hit/miss path)\n",
                (unsigned long long)calls);
      (void)fills;
    }
    RootedObject env(cx, &JS::Value::fromRawBits(gWJScratch[0]).toObject());
    JSString* s = JS::Value::fromRawBits(gWJScratch[1]).toString();
    Rooted<js::PropertyName*> name(cx, s->asAtom().asPropertyName());
    // Resolve holder+property without GC (fast); on success fill the per-site IC
    // ONLY when the holder is a realm singleton (global object / global lexical),
    // whose data-property SLOT is stable for the run (value may still mutate).
    js::PropertyResult prop;
    js::NativeObject* pobj = nullptr;
    if (js::LookupNameNoGC(cx, name, env, &pobj, &prop)) {
      JS::Value v;
      if (js::FetchNameNoGC(pobj, prop, &v)) {
        gWJScratch[js::wasm::kWJResultSlot] = v.asRawBits();
        if (site && prop.isNativeProperty() &&
            prop.propertyInfo().isDataProperty() && !v.isMagic() &&
            (pobj->is<js::GlobalObject>() ||
             pobj->is<js::GlobalLexicalEnvironmentObject>())) {
          js::TaggedSlotOffset t =
              pobj->getTaggedSlotOffset(prop.propertyInfo().slot());
          gWJNameHolder[site] = uintptr_t(static_cast<void*>(pobj));
          gWJNameShape[site] = uint32_t(uintptr_t(static_cast<void*>(pobj->shape())));
          gWJNameOff[site] = (t.offset() << js::TaggedSlotOffset::OffsetShift) |
                             (t.isFixedSlot() ? js::TaggedSlotOffset::IsFixedSlotFlag : 0);
          if (getenv("GECKO_WJ_NAMEDBG")) {
            // Self-check: decode the cached offset exactly as the JIT fast path
            // does and confirm it loads `v`. A mismatch = bad offset/holder.
            uint32_t tag = uint32_t(gWJNameOff[site]);
            char* obj = reinterpret_cast<char*>(pobj);
            char* base = (tag & 1) ? obj
                         : *reinterpret_cast<char**>(
                               obj + js::NativeObject::offsetOfSlots());
            uint64_t loaded = *reinterpret_cast<uint64_t*>(base + (tag >> 1));
            if (loaded != v.asRawBits()) {
              static int nm = 0;
              if (nm++ < 20)
                fprintf(stderr,
                        "[wj-name-BAD] site=%u fixed=%d off=%u loaded=%llx v=%llx\n",
                        site, int(tag & 1), tag >> 1,
                        (unsigned long long)loaded, (unsigned long long)v.asRawBits());
            }
          }
        }
        return 0.0;
      }
    }
    // Slow path (proxies / accessors / not found): no caching.
    RootedValue res(cx);
    if (!js::GetEnvironmentName<js::GetNameMode::Normal>(cx, env, name, &res))
      return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = res.asRawBits();
    return 0.0;
  }

  // ToPropertyKey (MToPropertyKeyCache): scratch[0]=input value -> property key
  // as a Value (string/symbol/int).
  if (kind == js::wasm::WJH_TOPROPKEY) {
    RootedValue in(cx, JS::Value::fromRawBits(gWJScratch[0]));
    JS::RootedId id(cx);
    if (!js::ToPropertyKey(cx, in, &id)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = js::IdToValue(id).asRawBits();
    return 0.0;
  }

  // String.charCodeAt (MCharCodeAt): scratch[0]=string, scratch[1]=index(int32).
  if (kind == js::wasm::WJH_CHARCODEAT) {
    RootedString str(cx, JS::Value::fromRawBits(gWJScratch[0]).toString());
    int32_t index = JS::Value::fromRawBits(gWJScratch[1]).toInt32();
    uint32_t code = 0;
    if (!js::jit::CharCodeAt(cx, str, index, &code)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::Int32Value(int32_t(code)).asRawBits();
    return 0.0;
  }

  // String.fromCharCode (MFromCharCode): scratch[0]=code(int32) -> 1-char string.
  if (kind == js::wasm::WJH_FROMCHARCODE) {
    int32_t code = JS::Value::fromRawBits(gWJScratch[0]).toInt32();
    JSString* s = js::StringFromCharCode(cx, code);
    if (!s) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::StringValue(s).asRawBits();
    return 0.0;
  }

  // ToString (MToString): scratch[0]=input value -> string.
  if (kind == js::wasm::WJH_TOSTRING) {
    RootedValue in(cx, JS::Value::fromRawBits(gWJScratch[0]));
    JSString* s = js::ToString<js::CanGC>(cx, in);
    if (!s) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::StringValue(s).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_PROPIC) {
    // Inline property-load IC miss: the JIT'd fast path didn't match a cached
    // shape. Get the value AND, for an OWN data property of a native receiver,
    // fill the per-site IC (shape -> TaggedSlotOffset) so the next access loads
    // the slot inline. Property name is baked per-site in gWJPropKey.
    uint32_t site = uint32_t(siteF);
    RootedValue objv(cx, JS::Value::fromRawBits(gWJScratch[0]));
    JS::RootedId id(cx, JS::PropertyKey::fromRawBits(uintptr_t(gWJPropKey[site])));
    if (objv.isObject() && objv.toObject().is<js::NativeObject>()) {
      js::NativeObject* nobj = &objv.toObject().as<js::NativeObject>();
      mozilla::Maybe<js::PropertyInfo> prop = nobj->lookupPure(id);
      if (prop.isSome() && prop->isDataProperty()) {
        uint32_t base = site * js::wasm::kWJPropWays;
        uint32_t shapeBits =
            uint32_t(uintptr_t(static_cast<void*>(nobj->shape())));
        js::TaggedSlotOffset t = nobj->getTaggedSlotOffset(prop->slot());
        uint32_t offBits =
            (t.offset() << js::TaggedSlotOffset::OffsetShift) |
            (t.isFixedSlot() ? js::TaggedSlotOffset::IsFixedSlotFlag : 0);
        uint32_t w = 0;
        for (; w < js::wasm::kWJPropWays; w++) {
          if (gWJPropShape[base + w] == 0 || gWJPropShape[base + w] == shapeBits)
            break;
        }
        if (w == js::wasm::kWJPropWays) w = 0;  // evict way 0
        static int noFill = -1;
        if (noFill < 0) noFill = getenv("GECKO_WJ_PROPNOFILL") ? 1 : 0;
        if (!noFill) {
          gWJPropShape[base + w] = shapeBits;
          gWJPropOff[base + w] = offBits;
        }
        gWJScratch[js::wasm::kWJResultSlot] = nobj->getSlot(prop->slot()).asRawBits();
        return 0.0;
      }
    }
    // Fallback: proto/accessor/non-native/missing property -> generic get, no
    // caching (the site keeps missing for this shape, always correct). Use the
    // same generic element get as WJH_GETPROP, with the baked key as a Value.
    RootedValue keyv(cx, js::IdToValue(id));
    RootedValue res(cx);
    if (!js::GetElementOperation(cx, objv, keyv, &res)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = res.asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_SETPROPIC) {
    // Set-prop IC miss: store value into obj.name. For a WRITABLE own data
    // property of a native receiver, fill the per-site IC (shape ->
    // TaggedSlotOffset) so the next store writes the slot inline, and store via
    // setSlot (which runs the proper pre/post barriers). Otherwise fall back to
    // the generic set (no caching).
    uint32_t site = uint32_t(siteF);
    RootedValue objv(cx, JS::Value::fromRawBits(gWJScratch[0]));
    RootedValue idv(cx, JS::Value::fromRawBits(gWJScratch[1]));
    RootedValue val(cx, JS::Value::fromRawBits(gWJScratch[2]));
    JS::RootedId id(cx);
    if (!js::ToPropertyKey(cx, idv, &id)) return 1.0;
    if (objv.isObject() && objv.toObject().is<js::NativeObject>()) {
      js::NativeObject* nobj = &objv.toObject().as<js::NativeObject>();
      mozilla::Maybe<js::PropertyInfo> prop = nobj->lookupPure(id);
      if (prop.isSome() && prop->isDataProperty() && prop->writable()) {
        uint32_t base = site * js::wasm::kWJPropWays;
        uint32_t shapeBits =
            uint32_t(uintptr_t(static_cast<void*>(nobj->shape())));
        js::TaggedSlotOffset t = nobj->getTaggedSlotOffset(prop->slot());
        uint32_t offBits =
            (t.offset() << js::TaggedSlotOffset::OffsetShift) |
            (t.isFixedSlot() ? js::TaggedSlotOffset::IsFixedSlotFlag : 0);
        uint32_t w = 0;
        for (; w < js::wasm::kWJPropWays; w++) {
          if (gWJPropShape[base + w] == 0 || gWJPropShape[base + w] == shapeBits)
            break;
        }
        if (w == js::wasm::kWJPropWays) w = 0;
        static int noFillS = -1;
        if (noFillS < 0) noFillS = getenv("GECKO_WJ_PROPNOFILL") ? 1 : 0;
        if (!noFillS) {
          gWJPropShape[base + w] = shapeBits;
          gWJPropOff[base + w] = offBits;
        }
        nobj->setSlot(prop->slot(), val);  // pre+post write barriers included
        return 0.0;
      }
    }
    // Fallback: setter / non-writable / proto / add / non-native -> generic set.
    if (!objv.isObject()) return 1.0;
    RootedObject obj(cx, &objv.toObject());
    RootedValue keyv(cx, js::IdToValue(id));
    bool strict = gWJPropStrict[site] != 0;
    if (!js::SetObjectElement(cx, obj, keyv, val, strict)) return 1.0;
    return 0.0;
  }

  if (kind == js::wasm::WJH_SETPROP) {
    RootedValue objv(cx, JS::Value::fromRawBits(gWJScratch[0]));
    if (!objv.isObject()) return 1.0;
    RootedObject obj(cx, &objv.toObject());
    RootedValue idx(cx, JS::Value::fromRawBits(gWJScratch[1]));
    RootedValue val(cx, JS::Value::fromRawBits(gWJScratch[2]));
    bool strict = (int(siteF) != 0);
    if (!js::SetObjectElement(cx, obj, idx, val, strict)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = val.asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_ARRAYPUSH) {
    // arr.push(val): set arr[length] = val (auto-grows dense + bumps length),
    // result = new length (Int32). MArrayPush guarantees an Array operand.
    RootedValue objv(cx, JS::Value::fromRawBits(gWJScratch[0]));
    if (!objv.isObject() || !objv.toObject().is<js::ArrayObject>()) return 1.0;
    RootedObject arr(cx, &objv.toObject());
    RootedValue val(cx, JS::Value::fromRawBits(gWJScratch[1]));
    uint32_t len = arr->as<js::ArrayObject>().length();
    RootedValue idx(cx, JS::NumberValue(double(len)));
    if (!js::SetObjectElement(cx, arr, idx, val, false)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] =
        JS::Int32Value(int32_t(len + 1)).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_ARRAYPOPSHIFT) {
    // site 0 = pop (remove last), site 1 = shift (remove first). Result = the
    // removed element (or undefined if empty). Routed through the native ops.
    RootedValue objv(cx, JS::Value::fromRawBits(gWJScratch[0]));
    if (!objv.isObject() || !objv.toObject().is<js::ArrayObject>()) return 1.0;
    RootedObject arr(cx, &objv.toObject());
    uint32_t len = arr->as<js::ArrayObject>().length();
    RootedValue res(cx, JS::UndefinedValue());
    if (len != 0) {
      uint32_t removeIdx = (int(siteF) == 0) ? (len - 1) : 0;
      RootedValue idxv(cx, JS::NumberValue(double(removeIdx)));
      if (!js::GetObjectElementOperation(cx, JSOp::GetElem, arr, objv, idxv,
                                         &res)) {
        return 1.0;
      }
      // pop: just truncate length. shift: delegate to the generic path below by
      // bailing (rare in the hot path) -- pop is the common dense case.
      if (int(siteF) == 0) {
        if (!js::SetLengthProperty(cx, arr, len - 1)) return 1.0;
      } else {
        return 1.0;  // shift: element-move not inlined; deopt to PBL
      }
    }
    gWJScratch[js::wasm::kWJResultSlot] = res.asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_INSTANCEOF) {
    RootedValue v(cx, JS::Value::fromRawBits(gWJScratch[0]));
    RootedValue ctorv(cx, JS::Value::fromRawBits(gWJScratch[1]));
    if (!ctorv.isObject()) return 1.0;
    RootedObject ctor(cx, &ctorv.toObject());
    bool res = false;
    if (!js::InstanceofOperator(cx, ctor, v, &res)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::BooleanValue(res).asRawBits();
    return 0.0;
  }

  if (kind == js::wasm::WJH_CREATETHIS) {
    if (getenv("GECKO_WJ_CTCOUNT")) {
      static uint64_t ctc = 0;
      if ((++ctc % 100000) == 0)
        fprintf(stderr, "[wj-createthis] %llu\n", (unsigned long long)ctc);
    }
    RootedValue calleev(cx, JS::Value::fromRawBits(gWJScratch[0]));
    RootedValue ntv(cx, JS::Value::fromRawBits(gWJScratch[1]));
    if (!calleev.isObject() || !ntv.isObject()) return 1.0;
    RootedObject callee(cx, &calleev.toObject());
    RootedObject newTarget(cx, &ntv.toObject());
    RootedValue rval(cx);
    if (!js::jit::CreateThisFromIon(cx, callee, newTarget, &rval)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = rval.asRawBits();
    gWJScratch[0] = 0;
    gWJScratch[1] = 0;
    return 0.0;
  }

  if (kind == js::wasm::WJH_CONSTRUCT) {
    RootedValue fval(cx, JS::Value::fromRawBits(gWJCallCallee));
    RootedValue thisv(cx,
                      JS::Value::fromRawBits(gWJScratch[js::wasm::kWJThisSlot]));
    RootedValue newTarget(cx, JS::Value::fromRawBits(gWJConstructNewTarget));
    uint32_t argc = gWJCallArgc;
    js::ConstructArgs cargs(cx);
    if (!cargs.init(cx, argc)) return 1.0;
    for (uint32_t i = 0; i < argc; i++) {
      cargs[i].set(JS::Value::fromRawBits(gWJScratch[i]));
    }
    RootedValue rval(cx);
    if (!js::InternalConstructWithProvidedThis(cx, fval, thisv, cargs, newTarget,
                                               &rval)) {
      return 1.0;
    }
    gWJScratch[js::wasm::kWJResultSlot] = rval.asRawBits();
    // Clear staged operands: WJTraceRoots scans gWJScratch[0..kWJThisSlot] as GC
    // roots on EVERY GC, but these slots are dead after the helper returns. A
    // later GC tracing a stale (freed/moved) nursery pointer left here corrupts
    // the heap -- the raytrace `ray`-goes-garbage bug. (Regular calls pass args
    // in locals, not scratch, which is why only construct/helper paths hit this.)
    for (uint32_t i = 0; i < argc; i++) gWJScratch[i] = 0;
    gWJScratch[js::wasm::kWJThisSlot] = 0;
    return 0.0;
  }

  if (kind == js::wasm::WJH_NEWPLAIN) {
    // gWJNewShapeSlot = address of the traced shape-pool slot (relocated by
    // WJTraceRoots); load the GC-current Shape* from it.
    js::Shape* sh =
        *reinterpret_cast<js::Shape**>(uintptr_t(gWJNewShapeSlot));
    if (!sh) return 1.0;
    JS::Rooted<js::SharedShape*> shape(cx, &sh->asShared());
    JSObject* obj = js::NewPlainObjectOptimizedFallback(
        cx, shape, js::gc::AllocKind(gWJNewAux), js::gc::Heap(gWJNewHeap));
    if (!obj) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*obj).asRawBits();
    return 0.0;
  }
  if (kind == js::wasm::WJH_NEWARROBJ) {
    js::gc::AllocKind ak = GuessArrayGCKind(gWJNewAux);
    js::NewObjectKind nk = gWJNewHeap == uint32_t(js::gc::Heap::Tenured)
                               ? js::TenuredObject
                               : js::GenericObject;
    js::ArrayObject* arr =
        js::NewArrayObjectOptimizedFallback(cx, gWJNewAux, ak, nk);
    if (!arr) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*arr).asRawBits();
    return 0.0;
  }
  if (kind == js::wasm::WJH_NEWARR) {
    js::ArrayObject* arr = js::NewArrayOperation(cx, gWJNewAux);
    if (!arr) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = JS::ObjectValue(*arr).asRawBits();
    return 0.0;
  }

  return 1.0;
}

// Compiled-callee invoke trampoline ("m"."call" import). Unused until the call
// path is rebuilt (Phase 5).
extern "C" EMSCRIPTEN_KEEPALIVE double wasmjit_invoke(int site, int argc) {
  return 1.0;
}

// Registered as an extra GC roots tracer (gc/RootMarking.cpp). The scratch buffer
// holds live boxed JS::Values (args + receiver + result) across a JIT'd call.
// GC-root shadow stack (see WasmJitBackend.h). Active region is [0, gWJRootSP).
namespace js {
namespace wasm {
alignas(8) uint64_t gWJCallRoots[kWJCallRootsSize];
uint32_t gWJRootSP = 0;
}  // namespace wasm
}  // namespace js

extern "C" EMSCRIPTEN_KEEPALIVE void WJTraceRoots(JSTracer* trc, void*) {
  for (uint32_t i = 0; i <= js::wasm::kWJThisSlot; i++) {
    JS::TraceRoot(trc, reinterpret_cast<JS::Value*>(&gWJScratch[i]), "wjscratch");
  }
  for (uint32_t i = 0; i < gWJConstPoolCount; i++) {
    JS::TraceRoot(trc, reinterpret_cast<JS::Value*>(&gWJConstPool[i]), "wjconst");
  }
  for (uint32_t i = 0; i < gWJShapePoolCount; i++) {
    if (gWJShapePool[i]) {
      js::TraceRoot(trc, reinterpret_cast<js::Shape**>(&gWJShapePool[i]), "wjshape");
    }
  }
  // Prop-IC cached shapes: trace+relocate (wasm32, so the uint32 IS the Shape*).
  // Keeps cached shapes live and pointer-current, so a shape match is always
  // correct (no stale/reused-pointer hazard). The paired offset stays valid
  // because a Shape's property layout is immutable.
  {
    uint32_t n = js::wasm::gWJNextPropSite * js::wasm::kWJPropWays;
    for (uint32_t i = 0; i < n; i++) {
      if (gWJPropShape[i]) {
        js::TraceRoot(trc, reinterpret_cast<js::Shape**>(&gWJPropShape[i]),
                      "wjpropic");
      }
    }
  }
  // GetName IC: trace+relocate the cached holder object AND its shape so both
  // stay pointer-current across a compacting GC (else the holder load reads a
  // moved object / the shape guard compares a stale pointer -> wrong value).
  {
    uint32_t n = js::wasm::gWJNextNameSite;
    for (uint32_t i = 0; i < n; i++) {
      if (gWJNameHolder[i]) {
        js::TraceRoot(trc, reinterpret_cast<JSObject**>(&gWJNameHolder[i]),
                      "wjnameholder");
        js::TraceRoot(trc, reinterpret_cast<js::Shape**>(&gWJNameShape[i]),
                      "wjnameshape");
      }
    }
  }
  // Polymorphic CALL IC: trace+relocate the cached callee function pointers so the
  // IC's `gWJCallFn[w] == currentCallee` dispatch compare stays pointer-current
  // across a GC that MOVES a cached function (nursery promotion on a minor GC, or
  // compaction). Without this, a moved function leaves a stale cached pointer; if a
  // DIFFERENT function is later allocated at that stale address the compare FALSE-
  // MATCHES and dispatches via the stale table slot -> calls the WRONG function ->
  // wrong result (raytrace "Scene rendered incorrectly"; NO_NURSERY masked it by
  // never moving objects). gWJCallTblIdx is a plain table index (not a pointer) and
  // needs no tracing. Functions kept alive this way (a strong root) is a bounded
  // acceptable leak, matching gWJShapePool/gWJPropShape.
  {
    uint32_t n = js::wasm::gWJNextCallSite * js::wasm::kWJCallWays;
    for (uint32_t i = 0; i < n; i++) {
      if (gWJCallFn[i]) {
        js::TraceRoot(trc, reinterpret_cast<JSObject**>(&gWJCallFn[i]), "wjcallfn");
      }
    }
  }
  uint32_t sp = gWJRootSP;
  if (sp > js::wasm::kWJCallRootsSize) sp = js::wasm::kWJCallRootsSize;
  for (uint32_t i = 0; i < sp; i++) {
    JS::TraceRoot(trc, reinterpret_cast<JS::Value*>(&gWJCallRoots[i]), "wjcallroot");
  }
}
