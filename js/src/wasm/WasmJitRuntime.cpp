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
#include "vm/NativeObject.h"
#include "vm/Interpreter.h"  // SetObjectElement, InstanceofOperator
#include "vm/Stack.h"  // ConstructArgs

#include "vm/NativeObject-inl.h"
#include "vm/Interpreter-inl.h"  // GetElementOperation

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
                                const uint64_t* osrStack, uint32_t osrStackDepth);
}  // namespace pbl
}  // namespace js

// --- Resume state (set partly by RunCall, partly by the emitted spill code) ---
// These five are declared in WasmJitBackend.h (js::wasm) and read by the emitted
// code's addresses, so their definitions must live in js::wasm too.
namespace js {
namespace wasm {
uint64_t gWJResumeVals[1024];   // [this, args.., locals.., stack..] boxed Values
uint32_t gWJResumePc = 0;       // resume bytecode offset (set by emitted code)
uint32_t gWJResumeStackDepth = 0;  // expr-stack depth at resume (set by emitted code)
uint32_t gWJResumeScriptPtr = 0;   // JSScript* (set by emitted code, baked)
uint32_t gWJResumeEnvPtr = 0;      // JSObject* env chain (set by emitted code)
uint32_t gWJResumeNArgs = 0;       // set by emitted code (baked constant)
uint32_t gWJResumeNLocals = 0;     // set by emitted code (baked constant)
uint64_t gWJCallCallee = 0;     // boxed callee Value (set by emitted code)
uint32_t gWJCallArgc = 0;
uint64_t gWJConstructNewTarget = 0;  // boxed newTarget for constructing calls
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

namespace {

struct WJEntry {
  enum class State : uint8_t { Cold, Compiled, Failed };
  State state = State::Cold;
  int handle = -1;
  int tblSlot = -1;  // dense shared-table slot (-1 = not call_indirect-able)
  uint32_t nargs = 0;
  uint32_t nlocals = 0;
  uint32_t observes = 0;
  uint32_t nextTry = 0;  // observe count at which to (re)attempt compilation
  uint32_t fails = 0;    // failed attempts so far
};

// Per-script compile state. Keyed by JSScript*; entries persist for the process.
static std::unordered_map<JSScript*, WJEntry>* gEntries = nullptr;

static WJEntry& EntryFor(JSScript* script) {
  if (!gEntries) gEntries = new std::unordered_map<JSScript*, WJEntry>();
  return (*gEntries)[script];
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
  int tblSlot = -1;
  int handle = js::wasm::WJWarpCompile(cx, script, &nargs, &nlocals, &tblSlot);
  if (handle < 0) {
    // Recompile-when-warm: a bail is often a cold IC (Warp emits an unconditional
    // bailout for an op whose baseline IC hasn't specialized yet). Retry later --
    // as the bench runs, callee/op ICs warm up and the bail disappears. Cap retries
    // so a genuinely-unsupported function eventually gives up (stays in PBL).
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
  e.nargs = nargs;
  e.nlocals = nlocals;
  return true;
}

int js::wasm::WasmJitRunCall(JSScript* script, uint64_t thisBits,
                             const JS::Value* argv, uint32_t argc,
                             JSObject* envChain, uint64_t* retBits) {
  if (!gEntries) return 0;
  auto it = gEntries->find(script);
  if (it == gEntries->end() || it->second.state != WJEntry::State::Compiled) {
    return 0;
  }
  WJEntry& e = it->second;
  if (argc < e.nargs) return 0;  // underflow: let the interpreter pad

  for (uint32_t i = 0; i < e.nargs; i++) {
    gWJScratch[i] = argv[i].asRawBits();
  }
  gWJScratch[js::wasm::kWJThisSlot] = thisBits;

  // Resume context is now self-contained: the emitted deopt code sets
  // gWJResumeScriptPtr/EnvPtr/NArgs/NLocals itself, so no setup is needed here.
  (void)envChain;
  double ptr = double(uintptr_t(static_cast<void*>(gWJScratch)));
  double flag = wasmhost_call(e.handle, 0, &ptr, 1);
  // Convention: 0.0 = result ready in gWJScratch (normal completion OR sound
  // resume); 1.0 = an exception is pending (a call/resume threw) -> propagate.
  if (flag != 0.0) {
    gWJWasmDeopts++;
    return 2;  // propagate pending exception
  }
  gWJWasmRuns++;
  if (((gWJFastCalls+gWJSlowCalls) % 20000)==0 && (gWJFastCalls+gWJSlowCalls)>0 && (getenv("GECKO_WJWARP_DUMP")||getenv("GECKO_DEBUG_JIT"))) fprintf(stderr, "[wb-calls] fast=%llu slow=%llu\n", (unsigned long long)gWJFastCalls,(unsigned long long)gWJSlowCalls);
  if (((gWJWasmRuns + gWJWasmDeopts) % 5000) == 0 &&
      (getenv("GECKO_WJWARP_DUMP") || getenv("GECKO_DEBUG_JIT"))) {
    fprintf(stderr, "[wb-stats] wasm runs=%llu deopts=%llu\n",
            (unsigned long long)gWJWasmRuns, (unsigned long long)gWJWasmDeopts);
  }
  *retBits = gWJScratch[js::wasm::kWJResultSlot];

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
      if (okv && JS::Call(cx, thisv, fval, JS::HandleValueArray(av), &rv)) {
        // Skip GC-thing results: a re-run allocates fresh objects, so pointer
        // identity legitimately differs. Only value types must be bit-identical.
        bool gcResult = JS::Value::fromRawBits(wasmRes).isGCThing() ||
                        rv.isGCThing();
        if (!gcResult && rv.asRawBits() != wasmRes) {
          fprintf(stderr,
                  "[wb-VERIFY] MISMATCH %s:%u argc=%u wasm=%#llx interp=%#llx\n",
                  script->filename() ? script->filename() : "?",
                  uint32_t(script->lineno()), argc, (unsigned long long)wasmRes,
                  (unsigned long long)rv.asRawBits());
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

  if (kind == js::wasm::WJH_RESUME) {
    RootedScript script(cx, reinterpret_cast<JSScript*>(uintptr_t(gWJResumeScriptPtr)));
    if (!script) return 1.0;
    JSObject* envObj = reinterpret_cast<JSObject*>(uintptr_t(gWJResumeEnvPtr));
    if (!envObj && script->function()) envObj = script->function()->environment();
    RootedObject env(cx, envObj);
    uint32_t nargs = gWJResumeNArgs;
    uint32_t nlocals = gWJResumeNLocals;
    uint32_t depth = gWJResumeStackDepth;
    uint32_t total = 1 + nargs + nlocals + depth;
    // Root the spilled live values (the spill->here window is GC-free, but the
    // resume below allocates/GCs). Reinterpret the Value array as uint64 bits for
    // the osrLocals/osrStack params.
    JS::RootedValueVector vals(cx);
    if (!vals.reserve(total)) return 1.0;
    for (uint32_t i = 0; i < total; i++) {
      vals.infallibleAppend(JS::Value::fromRawBits(gWJResumeVals[i]));
    }
    uint64_t thisBits = vals[0].asRawBits();
    const JS::Value* args = vals.begin() + 1;
    const uint64_t* locals = reinterpret_cast<const uint64_t*>(vals.begin() + 1 + nargs);
    const uint64_t* stack =
        reinterpret_cast<const uint64_t*>(vals.begin() + 1 + nargs + nlocals);
    uint64_t rbits = JS::UndefinedValue().asRawBits();
    static int rdbg = getenv("GECKO_WJWARP_DUMP") ? 1 : 0;
    static int rn = 0;
    if (rdbg && rn < 15) {
      rn++;
      fprintf(stderr, "[wb-resume] %s:%u pc=%u nargs=%u nlocals=%u depth=%u\n",
              script->filename() ? script->filename() : "?",
              uint32_t(script->lineno()), gWJResumePc, nargs, nlocals, depth);
    }
    if (!js::pbl::WasmJitResumeViaPBL(cx, script, thisBits, args, nargs, env,
                                      locals, nlocals, gWJResumePc, &rbits, stack,
                                      depth)) {
      return 1.0;  // resumed execution threw -> propagate
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
    if (gEntries && callee.isObject() && callee.toObject().is<JSFunction>()) {
      JSFunction* fun = &callee.toObject().as<JSFunction>();
      if (fun->isInterpreted() && fun->hasBytecode()) {
        JSScript* cs = fun->nonLazyScript();
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
          if (site < kWJCallSites && ce.tblSlot >= 0) {
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

  if (kind == js::wasm::WJH_SETSLOT) {
    JSObject* obj = reinterpret_cast<JSObject*>(uintptr_t(gWJHelpObj));
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
    RootedValue calleev(cx, JS::Value::fromRawBits(gWJScratch[0]));
    RootedValue ntv(cx, JS::Value::fromRawBits(gWJScratch[1]));
    if (!calleev.isObject() || !ntv.isObject()) return 1.0;
    RootedObject callee(cx, &calleev.toObject());
    RootedObject newTarget(cx, &ntv.toObject());
    RootedValue rval(cx);
    if (!js::jit::CreateThisFromIon(cx, callee, newTarget, &rval)) return 1.0;
    gWJScratch[js::wasm::kWJResultSlot] = rval.asRawBits();
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
  uint32_t sp = gWJRootSP;
  if (sp > js::wasm::kWJCallRootsSize) sp = js::wasm::kWJCallRootsSize;
  for (uint32_t i = 0; i < sp; i++) {
    JS::TraceRoot(trc, reinterpret_cast<JS::Value*>(&gWJCallRoots[i]), "wjcallroot");
  }
}
