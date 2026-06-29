/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * The interpreter Instance + bytecode execution loop. Included by WasmInterp.h.
 */

#ifndef wasm_WasmInterpRun_inl_h
#define wasm_WasmInterpRun_inl_h

namespace js {
namespace wasm {
namespace interp {

// ---- funcref packing (wasm32: Instance* fits in the high 32 bits) ----------

static inline uint64_t PackFunc(Instance* inst, uint32_t idx) {
  return inst ? ((uint64_t(uintptr_t(inst)) << 32) | idx) : 0;
}
static inline bool FuncIsNull(uint64_t p) { return (p >> 32) == 0; }
static inline Instance* FuncInst(uint64_t p) {
  return reinterpret_cast<Instance*>(uintptr_t(p >> 32));
}
static inline uint32_t FuncIdx(uint64_t p) { return uint32_t(p & 0xffffffffu); }

static inline bool Trap(JSContext* cx, unsigned msg) {
  ReportTrapError(cx, msg);
  return false;
}

// ---- shared-memory futex (memory.atomic.wait / .notify) --------------------
// Cross-thread futex routed through SpiderMonkey's Atomics on the inner memory's
// SharedArrayRawBuffer. CRITICAL: this hits the SAME waiter list that the inner
// content's JS Atomics.wait/waitAsync/notify use -- emscripten's pthread mailbox
// awaits notifications with Atomics.waitAsync, and proxied work (FS, pthread
// spawn, PROXY_TO_PTHREAD calls) is delivered by notifying that mailbox. The host
// wasm-engine futex (__builtin_wasm_memory_atomic_*) operates on a DIFFERENT
// table -- the OUTER engine's own WebAssembly.Memory -- so a wasm-side notify
// there never wakes the inner's JS-side waitAsync, deadlocking at the first
// proxied op. atomics_*_impl on the inner SAB makes wasm<->JS notifies
// interoperate. timeoutNs < 0 = infinite. Returns 0 = woken(ok), 1 = not-equal,
// 2 = timed-out, -1 = error (exception pending, caller must propagate).
static int32_t FutexWait(JSContext* cx, void* rawbuf, size_t byteOffset,
                         uint64_t expected, bool expected64, int64_t timeoutNs) {
  if (!rawbuf) {
    // Non-shared memory: no other thread can wake us. Match wasm semantics as
    // best we can without blocking forever (single-threaded shell path).
    return 2;  // timed-out
  }
  if (!cx->fx.canWait()) {
    // memory.atomic.wait is only valid on an agent that can block (a worker, not
    // the main browser thread). Propagate as an exception rather than tripping
    // the canWait() assert inside FutexThread::wait.
    JS_ReportErrorASCII(cx, "wasm interp: atomic.wait not allowed on this thread");
    return -1;
  }
  auto* sarb = reinterpret_cast<js::SharedArrayRawBuffer*>(rawbuf);
  mozilla::Maybe<mozilla::TimeDuration> timeout;
  if (timeoutNs >= 0) {
    timeout = mozilla::Some(
        mozilla::TimeDuration::FromMicroseconds(double(timeoutNs) / 1000.0));
  }
  static const bool dbg = getenv("GECKO_INTERP_FUTEX") != nullptr;
  if (dbg) {
    fprintf(stderr, "[interp] futex WAIT off=%zu exp=%llu to=%lld\n", byteOffset,
            (unsigned long long)expected, (long long)timeoutNs);
  }
  js::FutexThread::WaitResult r;
  if (expected64) {
    r = js::atomics_wait_impl(cx, sarb, byteOffset, (int64_t)expected, timeout);
  } else {
    r = js::atomics_wait_impl(cx, sarb, byteOffset, (int32_t)expected, timeout);
  }
  if (dbg) fprintf(stderr, "[interp] futex WOKE off=%zu r=%d\n", byteOffset, int(r));
  switch (r) {
    case js::FutexThread::WaitResult::OK:       return 0;
    case js::FutexThread::WaitResult::NotEqual: return 1;
    case js::FutexThread::WaitResult::TimedOut: return 2;
    case js::FutexThread::WaitResult::Error:    return -1;
  }
  return 0;
}
static int32_t FutexNotify(JSContext* cx, void* rawbuf, size_t byteOffset,
                           uint32_t count) {
  if (!rawbuf) return 0;
  auto* sarb = reinterpret_cast<js::SharedArrayRawBuffer*>(rawbuf);
  int64_t woken = 0;
  if (!js::atomics_notify_impl(cx, sarb, byteOffset, (int64_t)count, &woken)) {
    return -1;
  }
  static const bool dbg = getenv("GECKO_INTERP_FUTEX") != nullptr;
  if (dbg && (count == 0 || woken > 0)) {
    fprintf(stderr, "[interp] futex NOTIFY off=%zu cnt=%u woke=%lld\n", byteOffset,
            count, (long long)woken);
  }
  return (int32_t)woken;
}

// ---- saturating / trapping float->int conversions --------------------------

template <typename I, typename F>
static inline bool TruncTrap(JSContext* cx, F f, F lo, F hi, I* out) {
  if (std::isnan(f)) return Trap(cx, JSMSG_WASM_INVALID_CONVERSION);
  if (!(f >= lo && f < hi)) return Trap(cx, JSMSG_WASM_INTEGER_OVERFLOW);
  *out = I(f);
  return true;
}
template <typename I, typename F>
static inline I TruncSat(F f, F lo, F hi, I imin, I imax) {
  if (std::isnan(f)) return 0;
  if (f < lo) return imin;
  if (f >= hi) return imax;
  return I(f);
}

template <typename F>
static inline F WasmMin(F a, F b) {
  if (std::isnan(a) || std::isnan(b)) return a + b;  // -> NaN
  if (a == 0 && b == 0) return std::signbit(a) ? a : b;
  return a < b ? a : b;
}
template <typename F>
static inline F WasmMax(F a, F b) {
  if (std::isnan(a) || std::isnan(b)) return a + b;  // -> NaN
  if (a == 0 && b == 0) return std::signbit(a) ? b : a;
  return a > b ? a : b;
}

// A runtime control-stack entry (one per active block/loop/if).
struct Ctrl {
  bool isLoop;
  uint32_t targetPc;  // body-relative: loop start, or (end op + 1) for block/if
  uint32_t arity;     // values preserved across a branch to this label
  uint32_t spBase;    // operand-stack height the label resets to
  uint32_t endOpPc;   // body-relative pc of the matching `end` opcode
};

struct Frame {
  const FuncDef* fn;
  Cell* locals;
  Cell* stack;
  uint8_t* tags;  // per stack slot: 1 = externref (GC-traced)
  uint32_t sp;
  Frame* prev;
};

class Instance {
 public:
  JSContext* cx;
  Module* module = nullptr;
  JSObject* instanceObj = nullptr;  // owner (not rooted by us)
  JSObject* moduleObj = nullptr;    // keep module alive (rooted)
  JSObject* memObj = nullptr;       // MemoryObject (rooted), or null
  LinearMemory* lm = nullptr;       // cached from memObj

  Vector<JS::Value, 0, SystemAllocPolicy> importFuncs;  // index == func index
  Vector<Cell, 0, SystemAllocPolicy> globalCells;       // by global index
  Vector<JSObject*, 0, SystemAllocPolicy> tableObjs;    // by table index
  Vector<JSObject*, 0, SystemAllocPolicy> funcWrappers; // lazy, by func index
  Vector<bool, 0, SystemAllocPolicy> dataDropped;
  Vector<bool, 0, SystemAllocPolicy> elemDropped;

  Frame* topFrame = nullptr;

  Instance* nextLive = nullptr;  // intrusive live list for the GC tracer

  ~Instance() = default;

  uint32_t numImportFuncs() const { return module->numImportFuncs; }

  // -- GC tracing (called by the extra-roots tracer) --
  void traceValueBits(JSTracer* trc, uint64_t* bits, const char* name) {
    JS::Value v = JS::Value::fromRawBits(*bits);
    if (v.isGCThing()) {
      JS::TraceRoot(trc, &v, name);
      *bits = v.asRawBits();
    }
  }
  void trace(JSTracer* trc) {
    if (moduleObj) JS::TraceRoot(trc, &moduleObj, "interp-module");
    if (memObj) JS::TraceRoot(trc, &memObj, "interp-mem");
    for (size_t i = 0; i < importFuncs.length(); i++) {
      JS::TraceRoot(trc, &importFuncs[i], "interp-importfn");
    }
    for (size_t i = 0; i < tableObjs.length(); i++) {
      if (tableObjs[i]) JS::TraceRoot(trc, &tableObjs[i], "interp-table");
    }
    for (size_t i = 0; i < funcWrappers.length(); i++) {
      if (funcWrappers[i]) JS::TraceRoot(trc, &funcWrappers[i], "interp-wrap");
    }
    for (size_t i = 0; i < globalCells.length(); i++) {
      if (module->globals[i].type == VT::ExternRef) {
        traceValueBits(trc, &globalCells[i].u64, "interp-global");
      } else if (module->globals[i].type == VT::FuncRef) {
        traceFuncrefOwner(trc, globalCells[i].u64);
      }
    }
    for (Frame* f = topFrame; f; f = f->prev) {
      for (size_t i = 0; i < f->fn->localTypes.length(); i++) {
        if (f->fn->localTypes[i] == VT::ExternRef) {
          traceValueBits(trc, &f->locals[i].u64, "interp-local");
        } else if (f->fn->localTypes[i] == VT::FuncRef) {
          traceFuncrefOwner(trc, f->locals[i].u64);
        }
      }
      for (uint32_t i = 0; i < f->sp; i++) {
        if (f->tags[i]) traceValueBits(trc, &f->stack[i].u64, "interp-stk");
      }
    }
  }
  // Keep the InstanceObject that owns a funcref alive (the packed Instance* is
  // a stable C++ pointer, but its instanceObj is GC).
  void traceFuncrefOwner(JSTracer* trc, uint64_t packed) {
    if ((packed >> 32) == 0) return;
    Instance* fi = reinterpret_cast<Instance*>(uintptr_t(packed >> 32));
    if (fi && fi->instanceObj) {
      JS::TraceRoot(trc, &fi->instanceObj, "interp-funcref-owner");
    }
  }

  // -- value marshaling at the JS boundary --
  bool valueToCell(JS::HandleValue v, VT t, Cell* out) {
    switch (t) {
      case VT::I32: {
        int32_t i;
        if (!JS::ToInt32(cx, v, &i)) return false;
        out->i32 = i;
        return true;
      }
      case VT::I64: {
        JS::BigInt* bi = js::ToBigInt(cx, v);
        if (!bi) return false;
        out->i64 = JS::BigInt::toInt64(bi);
        return true;
      }
      case VT::F32: {
        double d;
        if (!JS::ToNumber(cx, v, &d)) return false;
        out->f32 = float(d);
        return true;
      }
      case VT::F64: {
        double d;
        if (!JS::ToNumber(cx, v, &d)) return false;
        out->f64 = d;
        return true;
      }
      case VT::ExternRef:
        out->u64 = v.asRawBits();
        return true;
      case VT::FuncRef: {
        if (v.isNull()) {
          out->u64 = 0;
          return true;
        }
        if (v.isObject()) {
          uint64_t packed;
          if (GetFuncWrapperPacked(&v.toObject(), &packed)) {
            out->u64 = packed;
            return true;
          }
        }
        // TypeError (not a generic Error) so emscripten addFunction's
        // try/catch(TypeError) fallback fires.
        JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr,
                                  JSMSG_NOT_FUNCTION, "argument");
        return false;
      }
      default:
        return false;
    }
  }
  bool cellToValue(const Cell* c, VT t, JS::MutableHandleValue out) {
    switch (t) {
      case VT::I32:
        out.setInt32(c->i32);
        return true;
      case VT::I64: {
        JS::BigInt* bi = JS::BigInt::createFromInt64(cx, c->i64);
        if (!bi) return false;
        out.setBigInt(bi);
        return true;
      }
      case VT::F32:
        out.setDouble(double(c->f32));
        return true;
      case VT::F64:
        out.setDouble(c->f64);
        return true;
      case VT::ExternRef:
        out.set(JS::Value::fromRawBits(c->u64));
        return true;
      case VT::FuncRef: {
        if (FuncIsNull(c->u64)) {
          out.setNull();
          return true;
        }
        JSObject* w =
            GetOrCreateFuncWrapper(cx, FuncInst(c->u64), FuncIdx(c->u64));
        if (!w) return false;
        out.setObject(*w);
        return true;
      }
      default:
        return false;
    }
  }

  // -- call an imported (host) function --
  bool callImport(uint32_t fi, Cell* args, Cell* results) {
    const FuncType& ft = module->funcType(fi);
    JS::RootedValue fnval(cx, importFuncs[fi]);
    if (!fnval.isObject()) {
      JS_ReportErrorASCII(cx, "wasm interp: import is not callable");
      return false;
    }
    JS::RootedObject fnObj(cx, &fnval.toObject());
    if (getenv("GECKO_INTERP_DEBUG")) {
      bool cMis = fnObj->compartment() != cx->compartment();
      bool zMis = fnObj->zone() != cx->zone();
      if (cMis || zMis) {
        fprintf(stderr,
                "[interp] callImport fi=%u callable=%d compMismatch=%d "
                "(fnComp=%p cxComp=%p) zoneMismatch=%d (fnZone=%p cxZone=%p)\n",
                fi, IsCallable(fnval), cMis, (void*)fnObj->compartment(),
                (void*)cx->compartment(), zMis, (void*)fnObj->zone(),
                (void*)cx->zone());
      }
    }
    JS::RootedValue rval(cx);
    bool callOk;
    {
      // Call the import in ITS realm. Building argv here (after entering the
      // realm) puts the args in the callee's compartment, which JS::Call's
      // same-compartment checks require; the result is wrapped back below. This
      // matters when content spans realms (e.g. libcurl + emscripten glue).
      JSAutoRealm ar(cx, fnObj);
      JS::RootedValueVector argv(cx);
      for (size_t i = 0; i < ft.params.length(); i++) {
        JS::RootedValue v(cx);
        if (!cellToValue(&args[i], ft.params[i], &v)) return false;
        if (!JS_WrapValue(cx, &v)) return false;  // into the callee's realm
        if (!argv.append(v)) {
          ReportOutOfMemory(cx);
          return false;
        }
      }
      callOk = JS::Call(cx, JS::UndefinedHandleValue, fnval,
                        JS::HandleValueArray(argv), &rval);
    }
    if (!callOk) {
      if (getenv("GECKO_INTERP_DEBUG")) {
        JS::RootedValue exc(cx);
        if (cx->isExceptionPending() && JS_GetPendingException(cx, &exc)) {
          JS::AutoSaveExceptionState saved(cx);
          JS::RootedString s(cx, JS::ToString(cx, exc));
          JS::UniqueChars c = s ? JS_EncodeStringToUTF8(cx, s) : nullptr;
          fprintf(stderr, "[interp] callImport fi=%u threw: %s\n", fi,
                  c ? c.get() : "(unstringifiable)");
          saved.restore();
        } else {
          fprintf(stderr, "[interp] callImport fi=%u failed (no pending exc)\n",
                  fi);
        }
      }
      return false;  // propagate the exception
    }
    if (!JS_WrapValue(cx, &rval)) return false;  // back to the caller's realm
    size_t nr = ft.results.length();
    if (nr == 1) {
      return valueToCell(rval, ft.results[0], &results[0]);
    }
    if (nr > 1) {
      JS::RootedObject arr(cx, rval.isObject() ? &rval.toObject() : nullptr);
      if (!arr) {
        JS_ReportErrorASCII(cx, "wasm interp: multi-result import not array");
        return false;
      }
      for (size_t i = 0; i < nr; i++) {
        JS::RootedValue e(cx);
        if (!JS_GetElement(cx, arr, uint32_t(i), &e)) return false;
        if (!valueToCell(e, ft.results[i], &results[i])) return false;
      }
    }
    return true;
  }

  bool invoke(uint32_t fi, Cell* args, Cell* results);
  bool runFunc(const FuncDef& fn, uint32_t fi, Cell* args, Cell* results);
};

bool Instance::invoke(uint32_t fi, Cell* args, Cell* results) {
  if (fi < numImportFuncs()) {
    return callImport(fi, args, results);
  }
  uint32_t defIdx = fi - numImportFuncs();
  FuncDef& fn = module->defined[defIdx];
  // Lazy decode: build this function's locals + control-flow side-table on first
  // call (deferred from module decode). Thread-safe for a shared (cross-thread)
  // module: acquire-load the prepared flag (fast path); on the first call,
  // prepare under prepareLock and release-store the flag so another thread that
  // sees it set also sees the FuncDef writes.
  if (module->preparedFlags &&
      module->preparedFlags[defIdx].load(std::memory_order_acquire) == 0) {
    std::lock_guard<std::mutex> guard(module->prepareLock);
    if (module->preparedFlags[defIdx].load(std::memory_order_relaxed) == 0) {
      if (!PrepareFunc(cx, module, fn, fn.codeStart, fn.codeEnd)) {
        if (!cx->isExceptionPending()) {
          JS_ReportErrorASCII(cx, "wasm interp: function prepare failed");
        }
        return false;
      }
      module->preparedFlags[defIdx].store(1, std::memory_order_release);
    }
  }
  return runFunc(fn, fi, args, results);
}

// The execution loop.
bool Instance::runFunc(const FuncDef& fn, uint32_t fi, Cell* args,
                       Cell* results) {
  // The interpreter recurses in C++ for every wasm call, so deep wasm call
  // chains (e.g. a TLS handshake) can overflow the native stack. Trap cleanly
  // ("too much recursion") via the engine's stack-limit check instead.
  AutoCheckRecursionLimit recursion(cx);
  if (!recursion.check(cx)) {
    return false;
  }
  static thread_local uint32_t gDepth = 0, gMaxDepth = 0;
  struct DepthGuard {
    DepthGuard() { gDepth++; if (gDepth > gMaxDepth) { gMaxDepth = gDepth; } }
    ~DepthGuard() { gDepth--; }
  } depthGuard;
  if (getenv("GECKO_INTERP_DEBUG") && gDepth == gMaxDepth && (gDepth % 200) == 0 && gDepth) {
    fprintf(stderr, "[interp] call depth %u\n", gDepth);
  }

  const FuncType& ft = module->funcType(fi);
  uint32_t nParams = ft.params.length();
  uint32_t nResults = ft.results.length();
  uint32_t nLocals = fn.localTypes.length();
  // The real operand-stack depth never exceeds the instruction count; use that
  // as a hard ceiling but start small and grow on demand, so a large function
  // body doesn't force a huge per-call allocation.
  uint32_t stkMax = uint32_t(fn.bodyEnd - fn.body) + 8;
  uint32_t stkCap = stkMax < 1024 ? stkMax : 1024;

  Cell* locals = js_pod_calloc<Cell>(nLocals ? nLocals : 1);
  Cell* stk = js_pod_calloc<Cell>(stkCap ? stkCap : 1);
  uint8_t* tags = js_pod_calloc<uint8_t>(stkCap ? stkCap : 1);
  if (!locals || !stk || !tags) {
    js_free(locals); js_free(stk); js_free(tags);
    if (getenv("GECKO_INTERP_DEBUG"))
      fprintf(stderr, "[interp] OOM initial alloc fi=%u nLocals=%u stkCap=%u\n",
              fi, nLocals, stkCap);
    ReportOutOfMemory(cx);
    return false;
  }
  for (uint32_t i = 0; i < nParams; i++) locals[i] = args[i];

  Frame frame{&fn, locals, stk, tags, 0, topFrame};
  topFrame = &frame;

  // Small inline capacity: this lives on the C++ stack and the interpreter
  // recurses per guest call, so a large inline control stack would multiply
  // native-stack use and cap guest recursion depth (deep blocks spill to heap).
  Vector<Ctrl, 8, SystemAllocPolicy> cs;
  bool ok = true;
  bool returning = false;

  const uint8_t* pc = fn.body;
  const uint8_t* end = fn.bodyEnd;
  uint32_t sp = 0;

  auto rdU32 = [&]() -> uint32_t {
    uint32_t r = 0; unsigned s = 0; uint8_t b;
    do { b = *pc++; r |= uint32_t(b & 0x7f) << s; s += 7; } while (b & 0x80);
    return r;
  };
  auto rdS32 = [&]() -> int32_t {
    int32_t r = 0; unsigned s = 0; uint8_t b;
    do { b = *pc++; r |= int32_t(b & 0x7f) << s; s += 7; } while (b & 0x80);
    if (s < 32 && (b & 0x40)) r |= int32_t(-1) << s;
    return r;
  };
  auto rdS64 = [&]() -> int64_t {
    int64_t r = 0; unsigned s = 0; uint8_t b;
    do { b = *pc++; r |= int64_t(b & 0x7f) << s; s += 7; } while (b & 0x80);
    if (s < 64 && (b & 0x40)) r |= int64_t(-1) << s;
    return r;
  };
  auto branchTo = [&](uint32_t depth) {
    if (depth >= cs.length()) {
      for (uint32_t i = 0; i < nResults; i++) results[i] = stk[sp - nResults + i];
      returning = true;
      return;
    }
    Ctrl& t = cs[cs.length() - 1 - depth];
    uint32_t arity = t.arity;
    for (uint32_t i = 0; i < arity; i++) {
      stk[t.spBase + i] = stk[sp - arity + i];
      tags[t.spBase + i] = tags[sp - arity + i];
    }
    sp = t.spBase + arity;
    pc = fn.body + t.targetPc;
    cs.shrinkBy(t.isLoop ? depth : depth + 1);
  };
  // Grow the operand stack to hold at least `need` more slots. Returns false on
  // OOM. Reallocation moves stk/tags; the by-ref lambdas + frame are updated.
  auto ensureStack = [&](uint32_t need) -> bool {
    if (sp + need <= stkCap) return true;
    uint32_t nc = stkCap * 2;
    if (nc < sp + need) nc = sp + need;
    if (nc > stkMax) nc = stkMax;
    if (getenv("GECKO_INTERP_DEBUG"))
      fprintf(stderr, "[interp] grow stack %u->%u (sp=%u need=%u max=%u)\n",
              stkCap, nc, sp, need, stkMax);
    Cell* ns = js_pod_realloc<Cell>(stk, stkCap, nc);
    if (!ns) { ReportOutOfMemory(cx); return false; }
    stk = ns; frame.stack = stk;
    uint8_t* nt = js_pod_realloc<uint8_t>(tags, stkCap, nc);
    if (!nt) { ReportOutOfMemory(cx); return false; }
    tags = nt; frame.tags = tags;
    stkCap = nc;
    return true;
  };

#define TRAP(msg) do { \
  if (getenv("GECKO_INTERP_DEBUG")) fprintf(stderr, "[interp] TRAP msg=%u fi=%u pc=%zu\n", unsigned(msg), fi, size_t(pc - fn.body)); \
  ReportTrapError(cx, msg); ok = false; goto done; } while (0)
#define FAIL() do { \
  if (getenv("GECKO_INTERP_DEBUG")) fprintf(stderr, "[interp] FAIL fi=%u pc=%zu\n", fi, size_t(pc - fn.body)); \
  ok = false; goto done; } while (0)
#define PUSH64(v) do { stk[sp].u64 = (v); tags[sp] = 0; sp++; } while (0)
#define PUSHEXT(v) do { stk[sp].u64 = (v); tags[sp] = 1; sp++; } while (0)

  while (pc < end) {
    if (sp + 8 > stkCap && !ensureStack(8)) FAIL();
    // Diagnostic heartbeat (outer-engine stderr is reliably forwarded, unlike
    // the inner content console): shows whether a thread is spinning (repeating
    // fi) or blocked (no heartbeat = parked in a futex wait), and where. Cached
    // env check -> ~zero cost when GECKO_INTERP_DEBUG is unset.
    {
      static const bool hb = getenv("GECKO_INTERP_DEBUG") != nullptr;
      if (hb) {
        static thread_local uint64_t gHB = 0;
        if ((++gHB & 0x3ffffff) == 0) {
          fprintf(stderr, "[interp] hb fi=%u depth=%u pc=%zu\n", fi, gDepth,
                  size_t(pc - fn.body));
        }
      }
    }
    // Publish the live operand-stack height so the GC tracer (which reads
    // frame.sp) traces this frame's operand stack. A GC can fire inside any op
    // that allocates or re-enters JS (calls, host imports, BigInt/wrapper
    // creation, memory.grow); capturing sp before the op runs is safe (it never
    // under-traces: every slot in [0, sp) holds a value that was valid here).
    frame.sp = sp;
    uint32_t opPc = uint32_t(pc - fn.body);
    uint8_t op = *pc++;
    switch (Op(op)) {
      case Op::Unreachable: TRAP(JSMSG_WASM_UNREACHABLE);
      case Op::Nop: break;
      case Op::Block:
      case Op::Loop:
      case Op::If: {
        Reader r(pc, end);
        uint32_t nP, nR;
        if (!DecodeBlockType(r, *module, &nP, &nR)) FAIL();
        pc = r.p;
        auto it = fn.ctrl.find(opPc);
        if (it == fn.ctrl.end()) FAIL();
        const CtrlMeta& meta = it->second;
        if (Op(op) == Op::Loop) {
          Ctrl c{true, uint32_t(pc - fn.body), nP, sp - nP, meta.endOpPc};
          if (!cs.append(c)) { ReportOutOfMemory(cx); FAIL(); }
        } else {
          uint32_t cond = 1;
          if (Op(op) == Op::If) cond = stk[--sp].u32;
          Ctrl c{false, meta.endOpPc + 1, nR, sp - nP, meta.endOpPc};
          if (!cs.append(c)) { ReportOutOfMemory(cx); FAIL(); }
          if (Op(op) == Op::If && cond == 0) {
            pc = fn.body + (meta.elsePc ? meta.elsePc : meta.endOpPc);
          }
        }
        break;
      }
      case Op::Else:
        pc = fn.body + cs.back().endOpPc;
        break;
      case Op::End:
        cs.popBack();
        break;
      case Op::Br: {
        uint32_t depth = rdU32();
        branchTo(depth);
        if (returning) goto done;
        break;
      }
      case Op::BrIf: {
        uint32_t depth = rdU32();
        uint32_t c = stk[--sp].u32;
        if (c) {
          branchTo(depth);
          if (returning) goto done;
        }
        break;
      }
      case Op::BrTable: {
        uint32_t n = rdU32();
        uint32_t idx = stk[--sp].u32;
        uint32_t depth = 0;
        for (uint32_t i = 0; i <= n; i++) {
          uint32_t d = rdU32();
          if ((idx < n && i == idx) || (idx >= n && i == n)) depth = d;
        }
        branchTo(depth);
        if (returning) goto done;
        break;
      }
      case Op::Return:
        for (uint32_t i = 0; i < nResults; i++) {
          results[i] = stk[sp - nResults + i];
        }
        goto done;
      case Op::Call: {
        uint32_t f = rdU32();
        const FuncType& cft = module->funcType(f);
        uint32_t na = cft.params.length();
        uint32_t nr = cft.results.length();
        Cell ab[64];
        Cell rb[16];
        if (na > 64 || nr > 16) { JS_ReportErrorASCII(cx, "wasm interp: arity"); FAIL(); }
        for (uint32_t i = 0; i < na; i++) ab[i] = stk[sp - na + i];
        sp -= na;
        if (!invoke(f, ab, rb)) FAIL();
        if (!ensureStack(nr)) FAIL();
        for (uint32_t i = 0; i < nr; i++) {
          stk[sp].u64 = rb[i].u64;
          tags[sp] = (cft.results[i] == VT::ExternRef) ? 1 : 0;
          sp++;
        }
        break;
      }
      case Op::CallIndirect: {
        uint32_t typeIdx = rdU32();
        uint32_t tableIdx = rdU32();
        uint32_t i = stk[--sp].u32;
        InstTable* t = TableFromObject(tableObjs[tableIdx]);
        if (i >= t->funcs.length()) TRAP(JSMSG_WASM_TABLE_OUT_OF_BOUNDS);
        uint64_t fr = t->funcs[i];
        if (FuncIsNull(fr)) TRAP(JSMSG_WASM_IND_CALL_TO_NULL);
        Instance* ti = FuncInst(fr);
        uint32_t tfi = FuncIdx(fr);
        if (!ti->module->funcType(tfi).equals(module->types[typeIdx])) {
          TRAP(JSMSG_WASM_IND_CALL_BAD_SIG);
        }
        const FuncType& cft = module->types[typeIdx];
        uint32_t na = cft.params.length();
        uint32_t nr = cft.results.length();
        Cell ab[64];
        Cell rb[16];
        if (na > 64 || nr > 16) { JS_ReportErrorASCII(cx, "wasm interp: arity"); FAIL(); }
        for (uint32_t k = 0; k < na; k++) ab[k] = stk[sp - na + k];
        sp -= na;
        if (!ti->invoke(tfi, ab, rb)) FAIL();
        if (!ensureStack(nr)) FAIL();
        for (uint32_t k = 0; k < nr; k++) {
          stk[sp].u64 = rb[k].u64;
          tags[sp] = (cft.results[k] == VT::ExternRef) ? 1 : 0;
          sp++;
        }
        break;
      }
      case Op::Drop:
        sp--;
        break;
      case Op::SelectNumeric:
      case Op::SelectTyped: {
        bool isExt = false;
        if (Op(op) == Op::SelectTyped) {
          uint32_t n = rdU32();
          for (uint32_t i = 0; i < n; i++) {
            uint8_t tb = *pc++;
            VT t;
            if (DecodeVT(tb, &t) && t == VT::ExternRef) isExt = true;
          }
        }
        uint32_t c = stk[--sp].u32;
        Cell b = stk[--sp];
        uint8_t bt = tags[sp];
        Cell a = stk[--sp];
        uint8_t at = tags[sp];
        stk[sp] = c ? a : b;
        tags[sp] = isExt ? 1 : (c ? at : bt);
        sp++;
        break;
      }
      case Op::LocalGet: {
        uint32_t i = rdU32();
        stk[sp].u64 = locals[i].u64;
        tags[sp] = (fn.localTypes[i] == VT::ExternRef) ? 1 : 0;
        sp++;
        break;
      }
      case Op::LocalSet: {
        uint32_t i = rdU32();
        locals[i].u64 = stk[--sp].u64;
        break;
      }
      case Op::LocalTee: {
        uint32_t i = rdU32();
        locals[i].u64 = stk[sp - 1].u64;
        break;
      }
      case Op::GlobalGet: {
        uint32_t i = rdU32();
        stk[sp].u64 = globalCells[i].u64;
        tags[sp] = (module->globals[i].type == VT::ExternRef) ? 1 : 0;
        sp++;
        break;
      }
      case Op::GlobalSet: {
        uint32_t i = rdU32();
        globalCells[i].u64 = stk[--sp].u64;
        break;
      }
      case Op::TableGet: {
        uint32_t ti = rdU32();
        InstTable* t = TableFromObject(tableObjs[ti]);
        uint32_t i = stk[--sp].u32;
        if (i >= t->length()) TRAP(JSMSG_WASM_TABLE_OUT_OF_BOUNDS);
        if (t->elem == VT::FuncRef) {
          PUSH64(t->funcs[i]);
        } else {
          PUSHEXT(t->refs[i].asRawBits());
        }
        break;
      }
      case Op::TableSet: {
        uint32_t ti = rdU32();
        InstTable* t = TableFromObject(tableObjs[ti]);
        uint64_t val = stk[--sp].u64;
        uint32_t i = stk[--sp].u32;
        if (i >= t->length()) TRAP(JSMSG_WASM_TABLE_OUT_OF_BOUNDS);
        if (t->elem == VT::FuncRef) {
          t->funcs[i] = val;
        } else {
          t->refs[i] = JS::Value::fromRawBits(val);
        }
        break;
      }

      // ---- memory load/store ----
#define LOADMEM(TYPE, FIELD, CAST)                                  \
  {                                                                 \
    rdU32();                                                        \
    uint64_t off = rdU32();                                         \
    uint64_t a = stk[--sp].u32;                                     \
    uint64_t ea = a + off;                                          \
    if (ea + sizeof(TYPE) > lm->size) TRAP(JSMSG_WASM_OUT_OF_BOUNDS); \
    TYPE tv;                                                        \
    memcpy(&tv, lm->base + ea, sizeof(TYPE));                       \
    stk[sp].FIELD = CAST(tv);                                       \
    tags[sp] = 0; sp++;                                             \
    break;                                                          \
  }
      case Op::I32Load: LOADMEM(uint32_t, i32, int32_t);
      case Op::I64Load: LOADMEM(uint64_t, i64, int64_t);
      case Op::F32Load: LOADMEM(float, f32, float);
      case Op::F64Load: LOADMEM(double, f64, double);
      case Op::I32Load8S: LOADMEM(int8_t, i32, int32_t);
      case Op::I32Load8U: LOADMEM(uint8_t, i32, uint32_t);
      case Op::I32Load16S: LOADMEM(int16_t, i32, int32_t);
      case Op::I32Load16U: LOADMEM(uint16_t, i32, uint32_t);
      case Op::I64Load8S: LOADMEM(int8_t, i64, int64_t);
      case Op::I64Load8U: LOADMEM(uint8_t, i64, uint64_t);
      case Op::I64Load16S: LOADMEM(int16_t, i64, int64_t);
      case Op::I64Load16U: LOADMEM(uint16_t, i64, uint64_t);
      case Op::I64Load32S: LOADMEM(int32_t, i64, int64_t);
      case Op::I64Load32U: LOADMEM(uint32_t, i64, uint64_t);
#undef LOADMEM
#define STOREMEM(TYPE, FIELD)                                       \
  {                                                                 \
    rdU32();                                                        \
    uint64_t off = rdU32();                                         \
    TYPE v = (TYPE)stk[--sp].FIELD;                                 \
    uint64_t a = stk[--sp].u32;                                     \
    uint64_t ea = a + off;                                          \
    if (ea + sizeof(TYPE) > lm->size) TRAP(JSMSG_WASM_OUT_OF_BOUNDS); \
    memcpy(lm->base + ea, &v, sizeof(TYPE));                        \
    break;                                                          \
  }
      case Op::I32Store: STOREMEM(uint32_t, u32);
      case Op::I64Store: STOREMEM(uint64_t, u64);
      case Op::F32Store: STOREMEM(float, f32);
      case Op::F64Store: STOREMEM(double, f64);
      case Op::I32Store8: STOREMEM(uint8_t, u32);
      case Op::I32Store16: STOREMEM(uint16_t, u32);
      case Op::I64Store8: STOREMEM(uint8_t, u64);
      case Op::I64Store16: STOREMEM(uint16_t, u64);
      case Op::I64Store32: STOREMEM(uint32_t, u64);
#undef STOREMEM
      case Op::MemorySize: {
        rdU32();
        PUSH64(uint64_t(uint32_t(lm->size / 65536)));
        stk[sp - 1].i32 = int32_t(lm->size / 65536);
        break;
      }
      case Op::MemoryGrow: {
        rdU32();
        uint32_t delta = stk[--sp].u32;
        JS::RootedObject mo(cx, memObj);
        int64_t oldPages = GrowMemoryObject(cx, mo, delta);
        if (cx->isExceptionPending()) FAIL();
        stk[sp].i32 = int32_t(oldPages);
        tags[sp] = 0; sp++;
        break;
      }

      // ---- constants ----
      case Op::I32Const: { int32_t v = rdS32(); PUSH64(uint64_t(uint32_t(v))); stk[sp-1].i32 = v; break; }
      case Op::I64Const: { int64_t v = rdS64(); PUSH64(uint64_t(v)); break; }
      case Op::F32Const: { float v; memcpy(&v, pc, 4); pc += 4; stk[sp].f32 = v; tags[sp]=0; sp++; break; }
      case Op::F64Const: { double v; memcpy(&v, pc, 8); pc += 8; stk[sp].f64 = v; tags[sp]=0; sp++; break; }

      // ---- i32 comparisons ----
      case Op::I32Eqz: { uint32_t a = stk[--sp].u32; PUSH64(0); stk[sp-1].i32 = (a==0); break; }
      case Op::I32Eq: { uint32_t b=stk[--sp].u32,a=stk[--sp].u32; stk[sp].i32=(a==b); tags[sp]=0; sp++; break; }
      case Op::I32Ne: { uint32_t b=stk[--sp].u32,a=stk[--sp].u32; stk[sp].i32=(a!=b); tags[sp]=0; sp++; break; }
      case Op::I32LtS: { int32_t b=stk[--sp].i32,a=stk[--sp].i32; stk[sp].i32=(a<b); tags[sp]=0; sp++; break; }
      case Op::I32LtU: { uint32_t b=stk[--sp].u32,a=stk[--sp].u32; stk[sp].i32=(a<b); tags[sp]=0; sp++; break; }
      case Op::I32GtS: { int32_t b=stk[--sp].i32,a=stk[--sp].i32; stk[sp].i32=(a>b); tags[sp]=0; sp++; break; }
      case Op::I32GtU: { uint32_t b=stk[--sp].u32,a=stk[--sp].u32; stk[sp].i32=(a>b); tags[sp]=0; sp++; break; }
      case Op::I32LeS: { int32_t b=stk[--sp].i32,a=stk[--sp].i32; stk[sp].i32=(a<=b); tags[sp]=0; sp++; break; }
      case Op::I32LeU: { uint32_t b=stk[--sp].u32,a=stk[--sp].u32; stk[sp].i32=(a<=b); tags[sp]=0; sp++; break; }
      case Op::I32GeS: { int32_t b=stk[--sp].i32,a=stk[--sp].i32; stk[sp].i32=(a>=b); tags[sp]=0; sp++; break; }
      case Op::I32GeU: { uint32_t b=stk[--sp].u32,a=stk[--sp].u32; stk[sp].i32=(a>=b); tags[sp]=0; sp++; break; }

      // ---- i64 comparisons ----
      case Op::I64Eqz: { uint64_t a=stk[--sp].u64; stk[sp].i32=(a==0); tags[sp]=0; sp++; break; }
      case Op::I64Eq: { uint64_t b=stk[--sp].u64,a=stk[--sp].u64; stk[sp].i32=(a==b); tags[sp]=0; sp++; break; }
      case Op::I64Ne: { uint64_t b=stk[--sp].u64,a=stk[--sp].u64; stk[sp].i32=(a!=b); tags[sp]=0; sp++; break; }
      case Op::I64LtS: { int64_t b=stk[--sp].i64,a=stk[--sp].i64; stk[sp].i32=(a<b); tags[sp]=0; sp++; break; }
      case Op::I64LtU: { uint64_t b=stk[--sp].u64,a=stk[--sp].u64; stk[sp].i32=(a<b); tags[sp]=0; sp++; break; }
      case Op::I64GtS: { int64_t b=stk[--sp].i64,a=stk[--sp].i64; stk[sp].i32=(a>b); tags[sp]=0; sp++; break; }
      case Op::I64GtU: { uint64_t b=stk[--sp].u64,a=stk[--sp].u64; stk[sp].i32=(a>b); tags[sp]=0; sp++; break; }
      case Op::I64LeS: { int64_t b=stk[--sp].i64,a=stk[--sp].i64; stk[sp].i32=(a<=b); tags[sp]=0; sp++; break; }
      case Op::I64LeU: { uint64_t b=stk[--sp].u64,a=stk[--sp].u64; stk[sp].i32=(a<=b); tags[sp]=0; sp++; break; }
      case Op::I64GeS: { int64_t b=stk[--sp].i64,a=stk[--sp].i64; stk[sp].i32=(a>=b); tags[sp]=0; sp++; break; }
      case Op::I64GeU: { uint64_t b=stk[--sp].u64,a=stk[--sp].u64; stk[sp].i32=(a>=b); tags[sp]=0; sp++; break; }

      // ---- f32 comparisons ----
      case Op::F32Eq: { float b=stk[--sp].f32,a=stk[--sp].f32; stk[sp].i32=(a==b); tags[sp]=0; sp++; break; }
      case Op::F32Ne: { float b=stk[--sp].f32,a=stk[--sp].f32; stk[sp].i32=(a!=b); tags[sp]=0; sp++; break; }
      case Op::F32Lt: { float b=stk[--sp].f32,a=stk[--sp].f32; stk[sp].i32=(a<b); tags[sp]=0; sp++; break; }
      case Op::F32Gt: { float b=stk[--sp].f32,a=stk[--sp].f32; stk[sp].i32=(a>b); tags[sp]=0; sp++; break; }
      case Op::F32Le: { float b=stk[--sp].f32,a=stk[--sp].f32; stk[sp].i32=(a<=b); tags[sp]=0; sp++; break; }
      case Op::F32Ge: { float b=stk[--sp].f32,a=stk[--sp].f32; stk[sp].i32=(a>=b); tags[sp]=0; sp++; break; }

      // ---- f64 comparisons ----
      case Op::F64Eq: { double b=stk[--sp].f64,a=stk[--sp].f64; stk[sp].i32=(a==b); tags[sp]=0; sp++; break; }
      case Op::F64Ne: { double b=stk[--sp].f64,a=stk[--sp].f64; stk[sp].i32=(a!=b); tags[sp]=0; sp++; break; }
      case Op::F64Lt: { double b=stk[--sp].f64,a=stk[--sp].f64; stk[sp].i32=(a<b); tags[sp]=0; sp++; break; }
      case Op::F64Gt: { double b=stk[--sp].f64,a=stk[--sp].f64; stk[sp].i32=(a>b); tags[sp]=0; sp++; break; }
      case Op::F64Le: { double b=stk[--sp].f64,a=stk[--sp].f64; stk[sp].i32=(a<=b); tags[sp]=0; sp++; break; }
      case Op::F64Ge: { double b=stk[--sp].f64,a=stk[--sp].f64; stk[sp].i32=(a>=b); tags[sp]=0; sp++; break; }

      // ---- i32 arithmetic ----
      case Op::I32Clz: { uint32_t a=stk[sp-1].u32; stk[sp-1].i32 = a?__builtin_clz(a):32; break; }
      case Op::I32Ctz: { uint32_t a=stk[sp-1].u32; stk[sp-1].i32 = a?__builtin_ctz(a):32; break; }
      case Op::I32Popcnt: { uint32_t a=stk[sp-1].u32; stk[sp-1].i32 = __builtin_popcount(a); break; }
      case Op::I32Add: { uint32_t b=stk[--sp].u32,a=stk[--sp].u32; stk[sp].u32=a+b; tags[sp]=0; sp++; break; }
      case Op::I32Sub: { uint32_t b=stk[--sp].u32,a=stk[--sp].u32; stk[sp].u32=a-b; tags[sp]=0; sp++; break; }
      case Op::I32Mul: { uint32_t b=stk[--sp].u32,a=stk[--sp].u32; stk[sp].u32=a*b; tags[sp]=0; sp++; break; }
      case Op::I32DivS: { int32_t b=stk[--sp].i32,a=stk[--sp].i32; if(b==0) TRAP(JSMSG_WASM_INT_DIVIDE_BY_ZERO); if(a==INT32_MIN&&b==-1) TRAP(JSMSG_WASM_INTEGER_OVERFLOW); stk[sp].i32=a/b; tags[sp]=0; sp++; break; }
      case Op::I32DivU: { uint32_t b=stk[--sp].u32,a=stk[--sp].u32; if(b==0) TRAP(JSMSG_WASM_INT_DIVIDE_BY_ZERO); stk[sp].u32=a/b; tags[sp]=0; sp++; break; }
      case Op::I32RemS: { int32_t b=stk[--sp].i32,a=stk[--sp].i32; if(b==0) TRAP(JSMSG_WASM_INT_DIVIDE_BY_ZERO); stk[sp].i32=(a==INT32_MIN&&b==-1)?0:a%b; tags[sp]=0; sp++; break; }
      case Op::I32RemU: { uint32_t b=stk[--sp].u32,a=stk[--sp].u32; if(b==0) TRAP(JSMSG_WASM_INT_DIVIDE_BY_ZERO); stk[sp].u32=a%b; tags[sp]=0; sp++; break; }
      case Op::I32And: { uint32_t b=stk[--sp].u32,a=stk[--sp].u32; stk[sp].u32=a&b; tags[sp]=0; sp++; break; }
      case Op::I32Or: { uint32_t b=stk[--sp].u32,a=stk[--sp].u32; stk[sp].u32=a|b; tags[sp]=0; sp++; break; }
      case Op::I32Xor: { uint32_t b=stk[--sp].u32,a=stk[--sp].u32; stk[sp].u32=a^b; tags[sp]=0; sp++; break; }
      case Op::I32Shl: { uint32_t b=stk[--sp].u32,a=stk[--sp].u32; stk[sp].u32=a<<(b&31); tags[sp]=0; sp++; break; }
      case Op::I32ShrS: { uint32_t b=stk[--sp].u32; int32_t a=stk[--sp].i32; stk[sp].i32=a>>(b&31); tags[sp]=0; sp++; break; }
      case Op::I32ShrU: { uint32_t b=stk[--sp].u32,a=stk[--sp].u32; stk[sp].u32=a>>(b&31); tags[sp]=0; sp++; break; }
      case Op::I32Rotl: { uint32_t b=stk[--sp].u32&31,a=stk[--sp].u32; stk[sp].u32=b?((a<<b)|(a>>(32-b))):a; tags[sp]=0; sp++; break; }
      case Op::I32Rotr: { uint32_t b=stk[--sp].u32&31,a=stk[--sp].u32; stk[sp].u32=b?((a>>b)|(a<<(32-b))):a; tags[sp]=0; sp++; break; }

      // ---- i64 arithmetic ----
      case Op::I64Clz: { uint64_t a=stk[sp-1].u64; stk[sp-1].i64 = a?__builtin_clzll(a):64; break; }
      case Op::I64Ctz: { uint64_t a=stk[sp-1].u64; stk[sp-1].i64 = a?__builtin_ctzll(a):64; break; }
      case Op::I64Popcnt: { uint64_t a=stk[sp-1].u64; stk[sp-1].i64 = __builtin_popcountll(a); break; }
      case Op::I64Add: { uint64_t b=stk[--sp].u64,a=stk[--sp].u64; stk[sp].u64=a+b; tags[sp]=0; sp++; break; }
      case Op::I64Sub: { uint64_t b=stk[--sp].u64,a=stk[--sp].u64; stk[sp].u64=a-b; tags[sp]=0; sp++; break; }
      case Op::I64Mul: { uint64_t b=stk[--sp].u64,a=stk[--sp].u64; stk[sp].u64=a*b; tags[sp]=0; sp++; break; }
      case Op::I64DivS: { int64_t b=stk[--sp].i64,a=stk[--sp].i64; if(b==0) TRAP(JSMSG_WASM_INT_DIVIDE_BY_ZERO); if(a==INT64_MIN&&b==-1) TRAP(JSMSG_WASM_INTEGER_OVERFLOW); stk[sp].i64=a/b; tags[sp]=0; sp++; break; }
      case Op::I64DivU: { uint64_t b=stk[--sp].u64,a=stk[--sp].u64; if(b==0) TRAP(JSMSG_WASM_INT_DIVIDE_BY_ZERO); stk[sp].u64=a/b; tags[sp]=0; sp++; break; }
      case Op::I64RemS: { int64_t b=stk[--sp].i64,a=stk[--sp].i64; if(b==0) TRAP(JSMSG_WASM_INT_DIVIDE_BY_ZERO); stk[sp].i64=(a==INT64_MIN&&b==-1)?0:a%b; tags[sp]=0; sp++; break; }
      case Op::I64RemU: { uint64_t b=stk[--sp].u64,a=stk[--sp].u64; if(b==0) TRAP(JSMSG_WASM_INT_DIVIDE_BY_ZERO); stk[sp].u64=a%b; tags[sp]=0; sp++; break; }
      case Op::I64And: { uint64_t b=stk[--sp].u64,a=stk[--sp].u64; stk[sp].u64=a&b; tags[sp]=0; sp++; break; }
      case Op::I64Or: { uint64_t b=stk[--sp].u64,a=stk[--sp].u64; stk[sp].u64=a|b; tags[sp]=0; sp++; break; }
      case Op::I64Xor: { uint64_t b=stk[--sp].u64,a=stk[--sp].u64; stk[sp].u64=a^b; tags[sp]=0; sp++; break; }
      case Op::I64Shl: { uint64_t b=stk[--sp].u64,a=stk[--sp].u64; stk[sp].u64=a<<(b&63); tags[sp]=0; sp++; break; }
      case Op::I64ShrS: { uint64_t b=stk[--sp].u64; int64_t a=stk[--sp].i64; stk[sp].i64=a>>(b&63); tags[sp]=0; sp++; break; }
      case Op::I64ShrU: { uint64_t b=stk[--sp].u64,a=stk[--sp].u64; stk[sp].u64=a>>(b&63); tags[sp]=0; sp++; break; }
      case Op::I64Rotl: { uint64_t b=stk[--sp].u64&63,a=stk[--sp].u64; stk[sp].u64=b?((a<<b)|(a>>(64-b))):a; tags[sp]=0; sp++; break; }
      case Op::I64Rotr: { uint64_t b=stk[--sp].u64&63,a=stk[--sp].u64; stk[sp].u64=b?((a>>b)|(a<<(64-b))):a; tags[sp]=0; sp++; break; }

      // ---- f32 arithmetic ----
      case Op::F32Abs: { stk[sp-1].f32=std::fabs(stk[sp-1].f32); break; }
      case Op::F32Neg: { stk[sp-1].f32=-stk[sp-1].f32; break; }
      case Op::F32Ceil: { stk[sp-1].f32=std::ceil(stk[sp-1].f32); break; }
      case Op::F32Floor: { stk[sp-1].f32=std::floor(stk[sp-1].f32); break; }
      case Op::F32Trunc: { stk[sp-1].f32=std::trunc(stk[sp-1].f32); break; }
      case Op::F32Nearest: { stk[sp-1].f32=std::nearbyint(stk[sp-1].f32); break; }
      case Op::F32Sqrt: { stk[sp-1].f32=std::sqrt(stk[sp-1].f32); break; }
      case Op::F32Add: { float b=stk[--sp].f32,a=stk[--sp].f32; stk[sp].f32=a+b; tags[sp]=0; sp++; break; }
      case Op::F32Sub: { float b=stk[--sp].f32,a=stk[--sp].f32; stk[sp].f32=a-b; tags[sp]=0; sp++; break; }
      case Op::F32Mul: { float b=stk[--sp].f32,a=stk[--sp].f32; stk[sp].f32=a*b; tags[sp]=0; sp++; break; }
      case Op::F32Div: { float b=stk[--sp].f32,a=stk[--sp].f32; stk[sp].f32=a/b; tags[sp]=0; sp++; break; }
      case Op::F32Min: { float b=stk[--sp].f32,a=stk[--sp].f32; stk[sp].f32=WasmMin(a,b); tags[sp]=0; sp++; break; }
      case Op::F32Max: { float b=stk[--sp].f32,a=stk[--sp].f32; stk[sp].f32=WasmMax(a,b); tags[sp]=0; sp++; break; }
      case Op::F32CopySign: { float b=stk[--sp].f32,a=stk[--sp].f32; stk[sp].f32=std::copysign(a,b); tags[sp]=0; sp++; break; }

      // ---- f64 arithmetic ----
      case Op::F64Abs: { stk[sp-1].f64=std::fabs(stk[sp-1].f64); break; }
      case Op::F64Neg: { stk[sp-1].f64=-stk[sp-1].f64; break; }
      case Op::F64Ceil: { stk[sp-1].f64=std::ceil(stk[sp-1].f64); break; }
      case Op::F64Floor: { stk[sp-1].f64=std::floor(stk[sp-1].f64); break; }
      case Op::F64Trunc: { stk[sp-1].f64=std::trunc(stk[sp-1].f64); break; }
      case Op::F64Nearest: { stk[sp-1].f64=std::nearbyint(stk[sp-1].f64); break; }
      case Op::F64Sqrt: { stk[sp-1].f64=std::sqrt(stk[sp-1].f64); break; }
      case Op::F64Add: { double b=stk[--sp].f64,a=stk[--sp].f64; stk[sp].f64=a+b; tags[sp]=0; sp++; break; }
      case Op::F64Sub: { double b=stk[--sp].f64,a=stk[--sp].f64; stk[sp].f64=a-b; tags[sp]=0; sp++; break; }
      case Op::F64Mul: { double b=stk[--sp].f64,a=stk[--sp].f64; stk[sp].f64=a*b; tags[sp]=0; sp++; break; }
      case Op::F64Div: { double b=stk[--sp].f64,a=stk[--sp].f64; stk[sp].f64=a/b; tags[sp]=0; sp++; break; }
      case Op::F64Min: { double b=stk[--sp].f64,a=stk[--sp].f64; stk[sp].f64=WasmMin(a,b); tags[sp]=0; sp++; break; }
      case Op::F64Max: { double b=stk[--sp].f64,a=stk[--sp].f64; stk[sp].f64=WasmMax(a,b); tags[sp]=0; sp++; break; }
      case Op::F64CopySign: { double b=stk[--sp].f64,a=stk[--sp].f64; stk[sp].f64=std::copysign(a,b); tags[sp]=0; sp++; break; }

      // ---- conversions ----
      case Op::I32WrapI64: { stk[sp-1].i32 = int32_t(stk[sp-1].u64); break; }
      case Op::I32TruncF32S: { int32_t o; if(!TruncTrap<int32_t,float>(cx,stk[sp-1].f32,-2147483648.0f,2147483648.0f,&o)) FAIL(); stk[sp-1].i32=o; break; }
      case Op::I32TruncF32U: { uint32_t o; float f=stk[sp-1].f32; if(std::isnan(f)) TRAP(JSMSG_WASM_INVALID_CONVERSION); if(!(f>-1.0f&&f<4294967296.0f)) TRAP(JSMSG_WASM_INTEGER_OVERFLOW); o=uint32_t(f); stk[sp-1].u32=o; break; }
      case Op::I32TruncF64S: { int32_t o; if(!TruncTrap<int32_t,double>(cx,stk[sp-1].f64,-2147483648.0,2147483648.0,&o)) FAIL(); stk[sp-1].i32=o; break; }
      case Op::I32TruncF64U: { double f=stk[sp-1].f64; if(std::isnan(f)) TRAP(JSMSG_WASM_INVALID_CONVERSION); if(!(f>-1.0&&f<4294967296.0)) TRAP(JSMSG_WASM_INTEGER_OVERFLOW); stk[sp-1].u32=uint32_t(f); break; }
      case Op::I64ExtendI32S: { stk[sp-1].i64 = int64_t(stk[sp-1].i32); break; }
      case Op::I64ExtendI32U: { stk[sp-1].u64 = uint64_t(stk[sp-1].u32); break; }
      case Op::I64TruncF32S: { int64_t o; if(!TruncTrap<int64_t,float>(cx,stk[sp-1].f32,-9223372036854775808.0f,9223372036854775808.0f,&o)) FAIL(); stk[sp-1].i64=o; break; }
      case Op::I64TruncF32U: { float f=stk[sp-1].f32; if(std::isnan(f)) TRAP(JSMSG_WASM_INVALID_CONVERSION); if(!(f>-1.0f&&f<18446744073709551616.0f)) TRAP(JSMSG_WASM_INTEGER_OVERFLOW); stk[sp-1].u64=uint64_t(f); break; }
      case Op::I64TruncF64S: { int64_t o; if(!TruncTrap<int64_t,double>(cx,stk[sp-1].f64,-9223372036854775808.0,9223372036854775808.0,&o)) FAIL(); stk[sp-1].i64=o; break; }
      case Op::I64TruncF64U: { double f=stk[sp-1].f64; if(std::isnan(f)) TRAP(JSMSG_WASM_INVALID_CONVERSION); if(!(f>-1.0&&f<18446744073709551616.0)) TRAP(JSMSG_WASM_INTEGER_OVERFLOW); stk[sp-1].u64=uint64_t(f); break; }
      case Op::F32ConvertI32S: { stk[sp-1].f32 = float(stk[sp-1].i32); break; }
      case Op::F32ConvertI32U: { stk[sp-1].f32 = float(stk[sp-1].u32); break; }
      case Op::F32ConvertI64S: { stk[sp-1].f32 = float(stk[sp-1].i64); break; }
      case Op::F32ConvertI64U: { stk[sp-1].f32 = float(stk[sp-1].u64); break; }
      case Op::F32DemoteF64: { stk[sp-1].f32 = float(stk[sp-1].f64); break; }
      case Op::F64ConvertI32S: { stk[sp-1].f64 = double(stk[sp-1].i32); break; }
      case Op::F64ConvertI32U: { stk[sp-1].f64 = double(stk[sp-1].u32); break; }
      case Op::F64ConvertI64S: { stk[sp-1].f64 = double(stk[sp-1].i64); break; }
      case Op::F64ConvertI64U: { stk[sp-1].f64 = double(stk[sp-1].u64); break; }
      case Op::F64PromoteF32: { stk[sp-1].f64 = double(stk[sp-1].f32); break; }
      case Op::I32ReinterpretF32: break;  // bit-preserving (same cell)
      case Op::I64ReinterpretF64: break;
      case Op::F32ReinterpretI32: break;
      case Op::F64ReinterpretI64: break;

      // ---- sign extension ----
      case Op::I32Extend8S: { stk[sp-1].i32 = int32_t(int8_t(stk[sp-1].u32)); break; }
      case Op::I32Extend16S: { stk[sp-1].i32 = int32_t(int16_t(stk[sp-1].u32)); break; }
      case Op::I64Extend8S: { stk[sp-1].i64 = int64_t(int8_t(stk[sp-1].u64)); break; }
      case Op::I64Extend16S: { stk[sp-1].i64 = int64_t(int16_t(stk[sp-1].u64)); break; }
      case Op::I64Extend32S: { stk[sp-1].i64 = int64_t(int32_t(stk[sp-1].u64)); break; }

      // ---- reference types ----
      case Op::RefNull: { (void)*pc++; PUSH64(0); break; }
      case Op::RefIsNull: { uint64_t r=stk[--sp].u64; uint8_t wasExt=tags[sp]; bool isn = wasExt ? JS::Value::fromRawBits(r).isNull() : FuncIsNull(r); stk[sp].i32=isn?1:0; tags[sp]=0; sp++; break; }
      case Op::RefFunc: { uint32_t f=rdU32(); PUSH64(PackFunc(this, f)); break; }

      case Op::MiscPrefix: {
        uint32_t sub = rdU32();
        switch (MiscOp(sub)) {
          case MiscOp::I32TruncSatF32S: { stk[sp-1].i32 = TruncSat<int32_t,float>(stk[sp-1].f32,-2147483648.0f,2147483648.0f,INT32_MIN,INT32_MAX); break; }
          case MiscOp::I32TruncSatF32U: { float f=stk[sp-1].f32; stk[sp-1].u32 = std::isnan(f)?0:(f<0?0:(f>=4294967296.0f?UINT32_MAX:uint32_t(f))); break; }
          case MiscOp::I32TruncSatF64S: { stk[sp-1].i32 = TruncSat<int32_t,double>(stk[sp-1].f64,-2147483648.0,2147483648.0,INT32_MIN,INT32_MAX); break; }
          case MiscOp::I32TruncSatF64U: { double f=stk[sp-1].f64; stk[sp-1].u32 = std::isnan(f)?0:(f<0?0:(f>=4294967296.0?UINT32_MAX:uint32_t(f))); break; }
          case MiscOp::I64TruncSatF32S: { stk[sp-1].i64 = TruncSat<int64_t,float>(stk[sp-1].f32,-9223372036854775808.0f,9223372036854775808.0f,INT64_MIN,INT64_MAX); break; }
          case MiscOp::I64TruncSatF32U: { float f=stk[sp-1].f32; stk[sp-1].u64 = std::isnan(f)?0:(f<0?0:(f>=18446744073709551616.0f?UINT64_MAX:uint64_t(f))); break; }
          case MiscOp::I64TruncSatF64S: { stk[sp-1].i64 = TruncSat<int64_t,double>(stk[sp-1].f64,-9223372036854775808.0,9223372036854775808.0,INT64_MIN,INT64_MAX); break; }
          case MiscOp::I64TruncSatF64U: { double f=stk[sp-1].f64; stk[sp-1].u64 = std::isnan(f)?0:(f<0?0:(f>=18446744073709551616.0?UINT64_MAX:uint64_t(f))); break; }
          case MiscOp::MemoryCopy: {
            rdU32(); rdU32();
            uint32_t n=stk[--sp].u32, s=stk[--sp].u32, d=stk[--sp].u32;
            if (uint64_t(s)+n > lm->size || uint64_t(d)+n > lm->size) TRAP(JSMSG_WASM_OUT_OF_BOUNDS);
            memmove(lm->base + d, lm->base + s, n);
            break;
          }
          case MiscOp::MemoryFill: {
            rdU32();
            uint32_t n=stk[--sp].u32, val=stk[--sp].u32, d=stk[--sp].u32;
            if (uint64_t(d)+n > lm->size) TRAP(JSMSG_WASM_OUT_OF_BOUNDS);
            memset(lm->base + d, int(val & 0xff), n);
            break;
          }
          case MiscOp::MemoryInit: {
            uint32_t seg=rdU32(); rdU32();
            uint32_t n=stk[--sp].u32, s=stk[--sp].u32, d=stk[--sp].u32;
            DataSeg& ds = module->datas[seg];
            size_t avail = dataDropped[seg] ? 0 : ds.bytes.length();
            if (uint64_t(s)+n > avail || uint64_t(d)+n > lm->size) TRAP(JSMSG_WASM_OUT_OF_BOUNDS);
            if (n) memcpy(lm->base + d, ds.bytes.begin() + s, n);
            break;
          }
          case MiscOp::DataDrop: { uint32_t seg=rdU32(); dataDropped[seg]=true; break; }
          case MiscOp::TableCopy: {
            uint32_t dti=rdU32(), sti=rdU32();
            InstTable* dt=TableFromObject(tableObjs[dti]);
            InstTable* st=TableFromObject(tableObjs[sti]);
            uint32_t n=stk[--sp].u32, s=stk[--sp].u32, d=stk[--sp].u32;
            if (uint64_t(s)+n > st->length() || uint64_t(d)+n > dt->length()) TRAP(JSMSG_WASM_TABLE_OUT_OF_BOUNDS);
            if (dt->elem==VT::FuncRef) { if(n) memmove(dt->funcs.begin()+d, st->funcs.begin()+s, n*sizeof(uint64_t)); }
            else { for(uint32_t k=0;k<n;k++) dt->refs[d+k]=st->refs[s+k]; }
            break;
          }
          case MiscOp::TableInit: {
            uint32_t seg=rdU32(); uint32_t ti=rdU32();
            InstTable* t=TableFromObject(tableObjs[ti]);
            ElemSeg& es = module->elems[seg];
            size_t avail = elemDropped[seg] ? 0 : es.funcIndices.length();
            uint32_t n=stk[--sp].u32, s=stk[--sp].u32, d=stk[--sp].u32;
            if (uint64_t(s)+n > avail || uint64_t(d)+n > t->length()) TRAP(JSMSG_WASM_TABLE_OUT_OF_BOUNDS);
            for (uint32_t k=0;k<n;k++) {
              uint32_t fi2 = es.funcIndices[s+k];
              if (t->elem==VT::FuncRef) t->funcs[d+k] = (fi2==UINT32_MAX)?0:PackFunc(this, fi2);
              else t->refs[d+k] = JS::NullValue();
            }
            break;
          }
          case MiscOp::ElemDrop: { uint32_t seg=rdU32(); elemDropped[seg]=true; break; }
          case MiscOp::TableGrow: {
            uint32_t ti=rdU32();
            uint32_t delta=stk[--sp].u32;
            uint64_t initFunc = stk[sp-1].u64; uint8_t initTag = tags[sp-1]; sp--;
            JS::RootedValue initRef(cx, initTag ? JS::Value::fromRawBits(initFunc) : JS::UndefinedValue());
            JS::RootedObject to(cx, tableObjs[ti]);
            int64_t oldLen = GrowTableObject(cx, to, delta, initFunc, initRef);
            if (cx->isExceptionPending()) FAIL();
            stk[sp].i32 = int32_t(oldLen); tags[sp]=0; sp++;
            break;
          }
          case MiscOp::TableSize: {
            uint32_t ti=rdU32();
            InstTable* t=TableFromObject(tableObjs[ti]);
            stk[sp].i32 = int32_t(t->length()); tags[sp]=0; sp++;
            break;
          }
          case MiscOp::TableFill: {
            uint32_t ti=rdU32();
            InstTable* t=TableFromObject(tableObjs[ti]);
            uint32_t n=stk[--sp].u32; uint64_t val=stk[sp-1].u64; uint8_t vt=tags[sp-1]; sp--; uint32_t d=stk[--sp].u32;
            if (uint64_t(d)+n > t->length()) TRAP(JSMSG_WASM_TABLE_OUT_OF_BOUNDS);
            for (uint32_t k=0;k<n;k++) { if(t->elem==VT::FuncRef) t->funcs[d+k]=val; else t->refs[d+k]=JS::Value::fromRawBits(val); }
            (void)vt;
            break;
          }
          default:
            JS_ReportErrorASCII(cx, "wasm interp: unsupported misc op");
            FAIL();
        }
        break;
      }

      case Op::ThreadPrefix: {
        uint32_t sub = rdU32();
        // Address + memarg + bounds/alignment for an atomic of SZ bytes. POPVALS
        // pops the operand(s) above the address (value / expected+replacement /
        // timeout+expected) -- the address is always the deepest operand. The
        // (shared) memory lives in the outer engine's shared heap, so __atomic_*
        // builtins give real cross-thread atomicity.
#define ATOMIC_ADDR(SZ, POPVALS)                            \
  rdU32(); /* align (ignored) */                            \
  uint64_t off = rdU32();                                   \
  POPVALS                                                   \
  uint64_t addr = stk[--sp].u32;                            \
  uint64_t ea = addr + off;                                 \
  if (ea & ((SZ) - 1)) TRAP(JSMSG_WASM_UNALIGNED_ACCESS);   \
  if (ea + (SZ) > lm->size) TRAP(JSMSG_WASM_OUT_OF_BOUNDS); \
  uint8_t* ptr = lm->base + ea;
        switch (ThreadOp(sub)) {
          case ThreadOp::Fence:
            (void)*pc++;  // reserved 0x00
            __atomic_thread_fence(__ATOMIC_SEQ_CST);
            break;
#define ALOAD(T, SZ, FIELD)                          \
  {                                                  \
    ATOMIC_ADDR(SZ, )                                \
    T v = __atomic_load_n((T*)ptr, __ATOMIC_SEQ_CST); \
    stk[sp].FIELD = v;                               \
    tags[sp] = 0;                                    \
    sp++;                                            \
    break;                                           \
  }
          case ThreadOp::I32AtomicLoad:    ALOAD(uint32_t, 4, u32)
          case ThreadOp::I64AtomicLoad:    ALOAD(uint64_t, 8, u64)
          case ThreadOp::I32AtomicLoad8U:  ALOAD(uint8_t, 1, u32)
          case ThreadOp::I32AtomicLoad16U: ALOAD(uint16_t, 2, u32)
          case ThreadOp::I64AtomicLoad8U:  ALOAD(uint8_t, 1, u64)
          case ThreadOp::I64AtomicLoad16U: ALOAD(uint16_t, 2, u64)
          case ThreadOp::I64AtomicLoad32U: ALOAD(uint32_t, 4, u64)
#undef ALOAD
#define ASTORE(T, SZ, FIELD)                              \
  {                                                       \
    ATOMIC_ADDR(SZ, T sv = (T)stk[--sp].FIELD;)           \
    __atomic_store_n((T*)ptr, sv, __ATOMIC_SEQ_CST);      \
    break;                                                \
  }
          case ThreadOp::I32AtomicStore:    ASTORE(uint32_t, 4, u32)
          case ThreadOp::I64AtomicStore:    ASTORE(uint64_t, 8, u64)
          case ThreadOp::I32AtomicStore8U:  ASTORE(uint8_t, 1, u32)
          case ThreadOp::I32AtomicStore16U: ASTORE(uint16_t, 2, u32)
          case ThreadOp::I64AtomicStore8U:  ASTORE(uint8_t, 1, u64)
          case ThreadOp::I64AtomicStore16U: ASTORE(uint16_t, 2, u64)
          case ThreadOp::I64AtomicStore32U: ASTORE(uint32_t, 4, u64)
#undef ASTORE
#define ARMW(T, SZ, FIELD, FN)                            \
  {                                                       \
    ATOMIC_ADDR(SZ, T rv = (T)stk[--sp].FIELD;)           \
    T old = FN((T*)ptr, rv, __ATOMIC_SEQ_CST);            \
    stk[sp].FIELD = old;                                  \
    tags[sp] = 0;                                         \
    sp++;                                                 \
    break;                                                \
  }
          case ThreadOp::I32AtomicAdd:     ARMW(uint32_t, 4, u32, __atomic_fetch_add)
          case ThreadOp::I64AtomicAdd:     ARMW(uint64_t, 8, u64, __atomic_fetch_add)
          case ThreadOp::I32AtomicAdd8U:   ARMW(uint8_t, 1, u32, __atomic_fetch_add)
          case ThreadOp::I32AtomicAdd16U:  ARMW(uint16_t, 2, u32, __atomic_fetch_add)
          case ThreadOp::I64AtomicAdd8U:   ARMW(uint8_t, 1, u64, __atomic_fetch_add)
          case ThreadOp::I64AtomicAdd16U:  ARMW(uint16_t, 2, u64, __atomic_fetch_add)
          case ThreadOp::I64AtomicAdd32U:  ARMW(uint32_t, 4, u64, __atomic_fetch_add)
          case ThreadOp::I32AtomicSub:     ARMW(uint32_t, 4, u32, __atomic_fetch_sub)
          case ThreadOp::I64AtomicSub:     ARMW(uint64_t, 8, u64, __atomic_fetch_sub)
          case ThreadOp::I32AtomicSub8U:   ARMW(uint8_t, 1, u32, __atomic_fetch_sub)
          case ThreadOp::I32AtomicSub16U:  ARMW(uint16_t, 2, u32, __atomic_fetch_sub)
          case ThreadOp::I64AtomicSub8U:   ARMW(uint8_t, 1, u64, __atomic_fetch_sub)
          case ThreadOp::I64AtomicSub16U:  ARMW(uint16_t, 2, u64, __atomic_fetch_sub)
          case ThreadOp::I64AtomicSub32U:  ARMW(uint32_t, 4, u64, __atomic_fetch_sub)
          case ThreadOp::I32AtomicAnd:     ARMW(uint32_t, 4, u32, __atomic_fetch_and)
          case ThreadOp::I64AtomicAnd:     ARMW(uint64_t, 8, u64, __atomic_fetch_and)
          case ThreadOp::I32AtomicAnd8U:   ARMW(uint8_t, 1, u32, __atomic_fetch_and)
          case ThreadOp::I32AtomicAnd16U:  ARMW(uint16_t, 2, u32, __atomic_fetch_and)
          case ThreadOp::I64AtomicAnd8U:   ARMW(uint8_t, 1, u64, __atomic_fetch_and)
          case ThreadOp::I64AtomicAnd16U:  ARMW(uint16_t, 2, u64, __atomic_fetch_and)
          case ThreadOp::I64AtomicAnd32U:  ARMW(uint32_t, 4, u64, __atomic_fetch_and)
          case ThreadOp::I32AtomicOr:      ARMW(uint32_t, 4, u32, __atomic_fetch_or)
          case ThreadOp::I64AtomicOr:      ARMW(uint64_t, 8, u64, __atomic_fetch_or)
          case ThreadOp::I32AtomicOr8U:    ARMW(uint8_t, 1, u32, __atomic_fetch_or)
          case ThreadOp::I32AtomicOr16U:   ARMW(uint16_t, 2, u32, __atomic_fetch_or)
          case ThreadOp::I64AtomicOr8U:    ARMW(uint8_t, 1, u64, __atomic_fetch_or)
          case ThreadOp::I64AtomicOr16U:   ARMW(uint16_t, 2, u64, __atomic_fetch_or)
          case ThreadOp::I64AtomicOr32U:   ARMW(uint32_t, 4, u64, __atomic_fetch_or)
          case ThreadOp::I32AtomicXor:     ARMW(uint32_t, 4, u32, __atomic_fetch_xor)
          case ThreadOp::I64AtomicXor:     ARMW(uint64_t, 8, u64, __atomic_fetch_xor)
          case ThreadOp::I32AtomicXor8U:   ARMW(uint8_t, 1, u32, __atomic_fetch_xor)
          case ThreadOp::I32AtomicXor16U:  ARMW(uint16_t, 2, u32, __atomic_fetch_xor)
          case ThreadOp::I64AtomicXor8U:   ARMW(uint8_t, 1, u64, __atomic_fetch_xor)
          case ThreadOp::I64AtomicXor16U:  ARMW(uint16_t, 2, u64, __atomic_fetch_xor)
          case ThreadOp::I64AtomicXor32U:  ARMW(uint32_t, 4, u64, __atomic_fetch_xor)
#undef ARMW
#define AXCHG(T, SZ, FIELD)                                  \
  {                                                          \
    ATOMIC_ADDR(SZ, T rv = (T)stk[--sp].FIELD;)              \
    T old = __atomic_exchange_n((T*)ptr, rv, __ATOMIC_SEQ_CST); \
    stk[sp].FIELD = old;                                     \
    tags[sp] = 0;                                            \
    sp++;                                                    \
    break;                                                   \
  }
          case ThreadOp::I32AtomicXchg:    AXCHG(uint32_t, 4, u32)
          case ThreadOp::I64AtomicXchg:    AXCHG(uint64_t, 8, u64)
          case ThreadOp::I32AtomicXchg8U:  AXCHG(uint8_t, 1, u32)
          case ThreadOp::I32AtomicXchg16U: AXCHG(uint16_t, 2, u32)
          case ThreadOp::I64AtomicXchg8U:  AXCHG(uint8_t, 1, u64)
          case ThreadOp::I64AtomicXchg16U: AXCHG(uint16_t, 2, u64)
          case ThreadOp::I64AtomicXchg32U: AXCHG(uint32_t, 4, u64)
#undef AXCHG
#define ACMPX(T, SZ, FIELD)                                          \
  {                                                                  \
    ATOMIC_ADDR(SZ, T rep = (T)stk[--sp].FIELD; T exp = (T)stk[--sp].FIELD;) \
    __atomic_compare_exchange_n((T*)ptr, &exp, rep, false,           \
                                __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST); \
    stk[sp].FIELD = exp;                                             \
    tags[sp] = 0;                                                    \
    sp++;                                                            \
    break;                                                          \
  }
          case ThreadOp::I32AtomicCmpXchg:    ACMPX(uint32_t, 4, u32)
          case ThreadOp::I64AtomicCmpXchg:    ACMPX(uint64_t, 8, u64)
          case ThreadOp::I32AtomicCmpXchg8U:  ACMPX(uint8_t, 1, u32)
          case ThreadOp::I32AtomicCmpXchg16U: ACMPX(uint16_t, 2, u32)
          case ThreadOp::I64AtomicCmpXchg8U:  ACMPX(uint8_t, 1, u64)
          case ThreadOp::I64AtomicCmpXchg16U: ACMPX(uint16_t, 2, u64)
          case ThreadOp::I64AtomicCmpXchg32U: ACMPX(uint32_t, 4, u64)
#undef ACMPX
          case ThreadOp::Notify: {
            ATOMIC_ADDR(4, uint32_t cnt = stk[--sp].u32;)
            (void)ptr;
            int32_t w = FutexNotify(cx, lm->rawbuf, size_t(ea), cnt);
            if (w < 0) FAIL();
            stk[sp].i32 = w; tags[sp] = 0; sp++;
            break;
          }
          case ThreadOp::I32Wait: {
            ATOMIC_ADDR(4, int64_t to = stk[--sp].i64; uint32_t expv = stk[--sp].u32;)
            (void)ptr;
            int32_t r = FutexWait(cx, lm->rawbuf, size_t(ea), expv,
                                  /*expected64=*/false, to);
            if (r < 0) FAIL();
            stk[sp].i32 = r; tags[sp] = 0; sp++;
            break;
          }
          case ThreadOp::I64Wait: {
            ATOMIC_ADDR(8, int64_t to = stk[--sp].i64; uint64_t expv = stk[--sp].u64;)
            (void)ptr;
            int32_t r = FutexWait(cx, lm->rawbuf, size_t(ea), expv,
                                  /*expected64=*/true, to);
            if (r < 0) FAIL();
            stk[sp].i32 = r; tags[sp] = 0; sp++;
            break;
          }
          default:
            JS_ReportErrorASCII(cx, "wasm interp: unsupported atomic op 0x%x",
                                unsigned(sub));
            FAIL();
        }
#undef ATOMIC_ADDR
        break;
      }

      // ---- SIMD (v128, 0xFD prefix) ------------------------------------------
      // Pass-through: each guest SIMD op maps to the host's native wasm SIMD via
      // wasm_simd128.h intrinsics (the engine is built -msimd128), so the guest's
      // vector op executes as ONE host SIMD instruction. Ops with runtime
      // immediates the intrinsics can't take as constants (extract/replace lane,
      // shuffle, load/store lane) are done by indexing the 16 v128 bytes.
      case Op::SimdPrefix: {
#if defined(__wasm_simd128__)
        uint32_t sub = rdU32();
#define SIMD_BINOP(FN)                                       \
  {                                                          \
    v128_t b_ = wasm_v128_load(stk[sp - 1].v128);            \
    v128_t a_ = wasm_v128_load(stk[sp - 2].v128);            \
    sp--;                                                    \
    wasm_v128_store(stk[sp - 1].v128, (FN)(a_, b_));         \
    tags[sp - 1] = 0;                                        \
    break;                                                   \
  }
#define SIMD_UNOP(FN)                                        \
  {                                                          \
    v128_t a_ = wasm_v128_load(stk[sp - 1].v128);            \
    wasm_v128_store(stk[sp - 1].v128, (FN)(a_));             \
    tags[sp - 1] = 0;                                        \
    break;                                                   \
  }
#define SIMD_SPLAT(FN, FIELD)                                \
  {                                                          \
    auto s_ = stk[sp - 1].FIELD;                             \
    wasm_v128_store(stk[sp - 1].v128, (FN)(s_));             \
    tags[sp - 1] = 0;                                        \
    break;                                                   \
  }
#define SIMD_SHIFT(FN)                                       \
  {                                                          \
    uint32_t c_ = stk[sp - 1].u32;                           \
    v128_t a_ = wasm_v128_load(stk[sp - 2].v128);            \
    sp--;                                                    \
    wasm_v128_store(stk[sp - 1].v128, (FN)(a_, c_));         \
    tags[sp - 1] = 0;                                        \
    break;                                                   \
  }
#define SIMD_LOAD(SZ, FN)                                          \
  {                                                                \
    rdU32();                                                       \
    uint64_t off_ = rdU32();                                       \
    uint64_t a_ = stk[sp - 1].u32;                                 \
    uint64_t ea_ = a_ + off_;                                      \
    if (ea_ + (SZ) > lm->size) TRAP(JSMSG_WASM_OUT_OF_BOUNDS);     \
    wasm_v128_store(stk[sp - 1].v128, (FN)(lm->base + ea_));       \
    tags[sp - 1] = 0;                                              \
    break;                                                         \
  }
#define SIMD_EXTRACT(CTYPE, DST, SEXT)                       \
  {                                                          \
    uint8_t lane_ = *pc++;                                   \
    CTYPE* p_ = (CTYPE*)stk[sp - 1].v128;                    \
    stk[sp - 1].DST = (SEXT)p_[lane_];                       \
    tags[sp - 1] = 0;                                        \
    break;                                                   \
  }
#define SIMD_REPLACE(CTYPE, SRC)                             \
  {                                                          \
    uint8_t lane_ = *pc++;                                   \
    CTYPE v_ = (CTYPE)stk[sp - 1].SRC;                       \
    sp--;                                                    \
    ((CTYPE*)stk[sp - 1].v128)[lane_] = v_;                  \
    tags[sp - 1] = 0;                                        \
    break;                                                   \
  }
#define SIMD_TEST(EXPR)                                      \
  {                                                          \
    v128_t a_ = wasm_v128_load(stk[sp - 1].v128);            \
    stk[sp - 1].i32 = (int32_t)(EXPR);                       \
    tags[sp - 1] = 0;                                        \
    break;                                                   \
  }
        switch (SimdOp(sub)) {
          // -- loads / stores --
          case SimdOp::V128Load: SIMD_LOAD(16, wasm_v128_load)
          case SimdOp::V128Load8Splat: SIMD_LOAD(1, wasm_v128_load8_splat)
          case SimdOp::V128Load16Splat: SIMD_LOAD(2, wasm_v128_load16_splat)
          case SimdOp::V128Load32Splat: SIMD_LOAD(4, wasm_v128_load32_splat)
          case SimdOp::V128Load64Splat: SIMD_LOAD(8, wasm_v128_load64_splat)
          case SimdOp::V128Load8x8S: SIMD_LOAD(8, wasm_i16x8_load8x8)
          case SimdOp::V128Load8x8U: SIMD_LOAD(8, wasm_u16x8_load8x8)
          case SimdOp::V128Load16x4S: SIMD_LOAD(8, wasm_i32x4_load16x4)
          case SimdOp::V128Load16x4U: SIMD_LOAD(8, wasm_u32x4_load16x4)
          case SimdOp::V128Load32x2S: SIMD_LOAD(8, wasm_i64x2_load32x2)
          case SimdOp::V128Load32x2U: SIMD_LOAD(8, wasm_u64x2_load32x2)
          case SimdOp::V128Load32Zero: SIMD_LOAD(4, wasm_v128_load32_zero)
          case SimdOp::V128Load64Zero: SIMD_LOAD(8, wasm_v128_load64_zero)
          case SimdOp::V128Store: {
            rdU32();
            uint64_t off_ = rdU32();
            v128_t v_ = wasm_v128_load(stk[sp - 1].v128);
            uint64_t a_ = stk[sp - 2].u32;
            uint64_t ea_ = a_ + off_;
            if (ea_ + 16 > lm->size) TRAP(JSMSG_WASM_OUT_OF_BOUNDS);
            wasm_v128_store(lm->base + ea_, v_);
            sp -= 2;
            break;
          }
#define SIMD_LOAD_LANE(SZ, CTYPE)                                  \
  {                                                                \
    rdU32();                                                       \
    uint64_t off_ = rdU32();                                       \
    uint8_t lane_ = *pc++;                                         \
    v128_t v_ = wasm_v128_load(stk[sp - 1].v128);                  \
    uint64_t a_ = stk[sp - 2].u32;                                 \
    uint64_t ea_ = a_ + off_;                                      \
    if (ea_ + (SZ) > lm->size) TRAP(JSMSG_WASM_OUT_OF_BOUNDS);     \
    CTYPE e_;                                                      \
    memcpy(&e_, lm->base + ea_, SZ);                              \
    ((CTYPE*)&v_)[lane_] = e_;                                     \
    sp--;                                                          \
    wasm_v128_store(stk[sp - 1].v128, v_);                         \
    tags[sp - 1] = 0;                                              \
    break;                                                         \
  }
          case SimdOp::V128Load8Lane: SIMD_LOAD_LANE(1, uint8_t)
          case SimdOp::V128Load16Lane: SIMD_LOAD_LANE(2, uint16_t)
          case SimdOp::V128Load32Lane: SIMD_LOAD_LANE(4, uint32_t)
          case SimdOp::V128Load64Lane: SIMD_LOAD_LANE(8, uint64_t)
#undef SIMD_LOAD_LANE
#define SIMD_STORE_LANE(SZ, CTYPE)                                 \
  {                                                                \
    rdU32();                                                       \
    uint64_t off_ = rdU32();                                       \
    uint8_t lane_ = *pc++;                                         \
    v128_t v_ = wasm_v128_load(stk[sp - 1].v128);                  \
    uint64_t a_ = stk[sp - 2].u32;                                 \
    uint64_t ea_ = a_ + off_;                                      \
    if (ea_ + (SZ) > lm->size) TRAP(JSMSG_WASM_OUT_OF_BOUNDS);     \
    CTYPE e_ = ((CTYPE*)&v_)[lane_];                               \
    memcpy(lm->base + ea_, &e_, SZ);                              \
    sp -= 2;                                                       \
    break;                                                         \
  }
          case SimdOp::V128Store8Lane: SIMD_STORE_LANE(1, uint8_t)
          case SimdOp::V128Store16Lane: SIMD_STORE_LANE(2, uint16_t)
          case SimdOp::V128Store32Lane: SIMD_STORE_LANE(4, uint32_t)
          case SimdOp::V128Store64Lane: SIMD_STORE_LANE(8, uint64_t)
#undef SIMD_STORE_LANE

          // -- const / shuffle / swizzle --
          case SimdOp::V128Const: {
            memcpy(stk[sp].v128, pc, 16);
            pc += 16;
            tags[sp] = 0;
            sp++;
            break;
          }
          case SimdOp::I8x16Shuffle: {
            const uint8_t* idx_ = pc;
            pc += 16;
            uint8_t a_[16], b_[16], out_[16];
            memcpy(a_, stk[sp - 2].v128, 16);
            memcpy(b_, stk[sp - 1].v128, 16);
            for (int i = 0; i < 16; i++) {
              uint8_t k = idx_[i];
              out_[i] = k < 16 ? a_[k] : b_[k - 16];
            }
            sp--;
            memcpy(stk[sp - 1].v128, out_, 16);
            tags[sp - 1] = 0;
            break;
          }
          case SimdOp::I8x16Swizzle: SIMD_BINOP(wasm_i8x16_swizzle)

          // -- splat --
          case SimdOp::I8x16Splat: SIMD_SPLAT(wasm_i8x16_splat, i32)
          case SimdOp::I16x8Splat: SIMD_SPLAT(wasm_i16x8_splat, i32)
          case SimdOp::I32x4Splat: SIMD_SPLAT(wasm_i32x4_splat, i32)
          case SimdOp::I64x2Splat: SIMD_SPLAT(wasm_i64x2_splat, i64)
          case SimdOp::F32x4Splat: SIMD_SPLAT(wasm_f32x4_splat, f32)
          case SimdOp::F64x2Splat: SIMD_SPLAT(wasm_f64x2_splat, f64)

          // -- extract / replace lane (runtime lane index) --
          case SimdOp::I8x16ExtractLaneS: SIMD_EXTRACT(int8_t, i32, int32_t)
          case SimdOp::I8x16ExtractLaneU: SIMD_EXTRACT(uint8_t, i32, uint32_t)
          case SimdOp::I16x8ExtractLaneS: SIMD_EXTRACT(int16_t, i32, int32_t)
          case SimdOp::I16x8ExtractLaneU: SIMD_EXTRACT(uint16_t, i32, uint32_t)
          case SimdOp::I32x4ExtractLane: SIMD_EXTRACT(int32_t, i32, int32_t)
          case SimdOp::I64x2ExtractLane: SIMD_EXTRACT(int64_t, i64, int64_t)
          case SimdOp::F32x4ExtractLane: SIMD_EXTRACT(float, f32, float)
          case SimdOp::F64x2ExtractLane: SIMD_EXTRACT(double, f64, double)
          case SimdOp::I8x16ReplaceLane: SIMD_REPLACE(int8_t, i32)
          case SimdOp::I16x8ReplaceLane: SIMD_REPLACE(int16_t, i32)
          case SimdOp::I32x4ReplaceLane: SIMD_REPLACE(int32_t, i32)
          case SimdOp::I64x2ReplaceLane: SIMD_REPLACE(int64_t, i64)
          case SimdOp::F32x4ReplaceLane: SIMD_REPLACE(float, f32)
          case SimdOp::F64x2ReplaceLane: SIMD_REPLACE(double, f64)

          // -- comparisons --
          case SimdOp::I8x16Eq: SIMD_BINOP(wasm_i8x16_eq)
          case SimdOp::I8x16Ne: SIMD_BINOP(wasm_i8x16_ne)
          case SimdOp::I8x16LtS: SIMD_BINOP(wasm_i8x16_lt)
          case SimdOp::I8x16LtU: SIMD_BINOP(wasm_u8x16_lt)
          case SimdOp::I8x16GtS: SIMD_BINOP(wasm_i8x16_gt)
          case SimdOp::I8x16GtU: SIMD_BINOP(wasm_u8x16_gt)
          case SimdOp::I8x16LeS: SIMD_BINOP(wasm_i8x16_le)
          case SimdOp::I8x16LeU: SIMD_BINOP(wasm_u8x16_le)
          case SimdOp::I8x16GeS: SIMD_BINOP(wasm_i8x16_ge)
          case SimdOp::I8x16GeU: SIMD_BINOP(wasm_u8x16_ge)
          case SimdOp::I16x8Eq: SIMD_BINOP(wasm_i16x8_eq)
          case SimdOp::I16x8Ne: SIMD_BINOP(wasm_i16x8_ne)
          case SimdOp::I16x8LtS: SIMD_BINOP(wasm_i16x8_lt)
          case SimdOp::I16x8LtU: SIMD_BINOP(wasm_u16x8_lt)
          case SimdOp::I16x8GtS: SIMD_BINOP(wasm_i16x8_gt)
          case SimdOp::I16x8GtU: SIMD_BINOP(wasm_u16x8_gt)
          case SimdOp::I16x8LeS: SIMD_BINOP(wasm_i16x8_le)
          case SimdOp::I16x8LeU: SIMD_BINOP(wasm_u16x8_le)
          case SimdOp::I16x8GeS: SIMD_BINOP(wasm_i16x8_ge)
          case SimdOp::I16x8GeU: SIMD_BINOP(wasm_u16x8_ge)
          case SimdOp::I32x4Eq: SIMD_BINOP(wasm_i32x4_eq)
          case SimdOp::I32x4Ne: SIMD_BINOP(wasm_i32x4_ne)
          case SimdOp::I32x4LtS: SIMD_BINOP(wasm_i32x4_lt)
          case SimdOp::I32x4LtU: SIMD_BINOP(wasm_u32x4_lt)
          case SimdOp::I32x4GtS: SIMD_BINOP(wasm_i32x4_gt)
          case SimdOp::I32x4GtU: SIMD_BINOP(wasm_u32x4_gt)
          case SimdOp::I32x4LeS: SIMD_BINOP(wasm_i32x4_le)
          case SimdOp::I32x4LeU: SIMD_BINOP(wasm_u32x4_le)
          case SimdOp::I32x4GeS: SIMD_BINOP(wasm_i32x4_ge)
          case SimdOp::I32x4GeU: SIMD_BINOP(wasm_u32x4_ge)
          case SimdOp::I64x2Eq: SIMD_BINOP(wasm_i64x2_eq)
          case SimdOp::I64x2Ne: SIMD_BINOP(wasm_i64x2_ne)
          case SimdOp::I64x2LtS: SIMD_BINOP(wasm_i64x2_lt)
          case SimdOp::I64x2GtS: SIMD_BINOP(wasm_i64x2_gt)
          case SimdOp::I64x2LeS: SIMD_BINOP(wasm_i64x2_le)
          case SimdOp::I64x2GeS: SIMD_BINOP(wasm_i64x2_ge)
          case SimdOp::F32x4Eq: SIMD_BINOP(wasm_f32x4_eq)
          case SimdOp::F32x4Ne: SIMD_BINOP(wasm_f32x4_ne)
          case SimdOp::F32x4Lt: SIMD_BINOP(wasm_f32x4_lt)
          case SimdOp::F32x4Gt: SIMD_BINOP(wasm_f32x4_gt)
          case SimdOp::F32x4Le: SIMD_BINOP(wasm_f32x4_le)
          case SimdOp::F32x4Ge: SIMD_BINOP(wasm_f32x4_ge)
          case SimdOp::F64x2Eq: SIMD_BINOP(wasm_f64x2_eq)
          case SimdOp::F64x2Ne: SIMD_BINOP(wasm_f64x2_ne)
          case SimdOp::F64x2Lt: SIMD_BINOP(wasm_f64x2_lt)
          case SimdOp::F64x2Gt: SIMD_BINOP(wasm_f64x2_gt)
          case SimdOp::F64x2Le: SIMD_BINOP(wasm_f64x2_le)
          case SimdOp::F64x2Ge: SIMD_BINOP(wasm_f64x2_ge)

          // -- bitwise / boolean --
          case SimdOp::V128Not: SIMD_UNOP(wasm_v128_not)
          case SimdOp::V128And: SIMD_BINOP(wasm_v128_and)
          case SimdOp::V128AndNot: SIMD_BINOP(wasm_v128_andnot)
          case SimdOp::V128Or: SIMD_BINOP(wasm_v128_or)
          case SimdOp::V128Xor: SIMD_BINOP(wasm_v128_xor)
          case SimdOp::V128AnyTrue: SIMD_TEST(wasm_v128_any_true(a_) ? 1 : 0)
          case SimdOp::I8x16AllTrue: SIMD_TEST(wasm_i8x16_all_true(a_) ? 1 : 0)
          case SimdOp::I16x8AllTrue: SIMD_TEST(wasm_i16x8_all_true(a_) ? 1 : 0)
          case SimdOp::I32x4AllTrue: SIMD_TEST(wasm_i32x4_all_true(a_) ? 1 : 0)
          case SimdOp::I64x2AllTrue: SIMD_TEST(wasm_i64x2_all_true(a_) ? 1 : 0)
          case SimdOp::I8x16Bitmask: SIMD_TEST(wasm_i8x16_bitmask(a_))
          case SimdOp::I16x8Bitmask: SIMD_TEST(wasm_i16x8_bitmask(a_))
          case SimdOp::I32x4Bitmask: SIMD_TEST(wasm_i32x4_bitmask(a_))
          case SimdOp::I64x2Bitmask: SIMD_TEST(wasm_i64x2_bitmask(a_))
          case SimdOp::V128Bitselect: {
            v128_t m_ = wasm_v128_load(stk[sp - 1].v128);
            v128_t b_ = wasm_v128_load(stk[sp - 2].v128);
            v128_t a_ = wasm_v128_load(stk[sp - 3].v128);
            sp -= 2;
            wasm_v128_store(stk[sp - 1].v128, wasm_v128_bitselect(a_, b_, m_));
            tags[sp - 1] = 0;
            break;
          }

          // -- integer unary --
          case SimdOp::I8x16Abs: SIMD_UNOP(wasm_i8x16_abs)
          case SimdOp::I8x16Neg: SIMD_UNOP(wasm_i8x16_neg)
          case SimdOp::I8x16Popcnt: SIMD_UNOP(wasm_i8x16_popcnt)
          case SimdOp::I16x8Abs: SIMD_UNOP(wasm_i16x8_abs)
          case SimdOp::I16x8Neg: SIMD_UNOP(wasm_i16x8_neg)
          case SimdOp::I32x4Abs: SIMD_UNOP(wasm_i32x4_abs)
          case SimdOp::I32x4Neg: SIMD_UNOP(wasm_i32x4_neg)
          case SimdOp::I64x2Abs: SIMD_UNOP(wasm_i64x2_abs)
          case SimdOp::I64x2Neg: SIMD_UNOP(wasm_i64x2_neg)

          // -- integer add/sub/mul/min/max/sat/avgr --
          case SimdOp::I8x16Add: SIMD_BINOP(wasm_i8x16_add)
          case SimdOp::I8x16AddSatS: SIMD_BINOP(wasm_i8x16_add_sat)
          case SimdOp::I8x16AddSatU: SIMD_BINOP(wasm_u8x16_add_sat)
          case SimdOp::I8x16Sub: SIMD_BINOP(wasm_i8x16_sub)
          case SimdOp::I8x16SubSatS: SIMD_BINOP(wasm_i8x16_sub_sat)
          case SimdOp::I8x16SubSatU: SIMD_BINOP(wasm_u8x16_sub_sat)
          case SimdOp::I8x16MinS: SIMD_BINOP(wasm_i8x16_min)
          case SimdOp::I8x16MinU: SIMD_BINOP(wasm_u8x16_min)
          case SimdOp::I8x16MaxS: SIMD_BINOP(wasm_i8x16_max)
          case SimdOp::I8x16MaxU: SIMD_BINOP(wasm_u8x16_max)
          case SimdOp::I8x16AvgrU: SIMD_BINOP(wasm_u8x16_avgr)
          case SimdOp::I16x8Add: SIMD_BINOP(wasm_i16x8_add)
          case SimdOp::I16x8AddSatS: SIMD_BINOP(wasm_i16x8_add_sat)
          case SimdOp::I16x8AddSatU: SIMD_BINOP(wasm_u16x8_add_sat)
          case SimdOp::I16x8Sub: SIMD_BINOP(wasm_i16x8_sub)
          case SimdOp::I16x8SubSatS: SIMD_BINOP(wasm_i16x8_sub_sat)
          case SimdOp::I16x8SubSatU: SIMD_BINOP(wasm_u16x8_sub_sat)
          case SimdOp::I16x8Mul: SIMD_BINOP(wasm_i16x8_mul)
          case SimdOp::I16x8MinS: SIMD_BINOP(wasm_i16x8_min)
          case SimdOp::I16x8MinU: SIMD_BINOP(wasm_u16x8_min)
          case SimdOp::I16x8MaxS: SIMD_BINOP(wasm_i16x8_max)
          case SimdOp::I16x8MaxU: SIMD_BINOP(wasm_u16x8_max)
          case SimdOp::I16x8AvgrU: SIMD_BINOP(wasm_u16x8_avgr)
          case SimdOp::I16x8Q15MulrSatS: SIMD_BINOP(wasm_i16x8_q15mulr_sat)
          case SimdOp::I32x4Add: SIMD_BINOP(wasm_i32x4_add)
          case SimdOp::I32x4Sub: SIMD_BINOP(wasm_i32x4_sub)
          case SimdOp::I32x4Mul: SIMD_BINOP(wasm_i32x4_mul)
          case SimdOp::I32x4MinS: SIMD_BINOP(wasm_i32x4_min)
          case SimdOp::I32x4MinU: SIMD_BINOP(wasm_u32x4_min)
          case SimdOp::I32x4MaxS: SIMD_BINOP(wasm_i32x4_max)
          case SimdOp::I32x4MaxU: SIMD_BINOP(wasm_u32x4_max)
          case SimdOp::I32x4DotI16x8S: SIMD_BINOP(wasm_i32x4_dot_i16x8)
          case SimdOp::I64x2Add: SIMD_BINOP(wasm_i64x2_add)
          case SimdOp::I64x2Sub: SIMD_BINOP(wasm_i64x2_sub)
          case SimdOp::I64x2Mul: SIMD_BINOP(wasm_i64x2_mul)

          // -- shifts --
          case SimdOp::I8x16Shl: SIMD_SHIFT(wasm_i8x16_shl)
          case SimdOp::I8x16ShrS: SIMD_SHIFT(wasm_i8x16_shr)
          case SimdOp::I8x16ShrU: SIMD_SHIFT(wasm_u8x16_shr)
          case SimdOp::I16x8Shl: SIMD_SHIFT(wasm_i16x8_shl)
          case SimdOp::I16x8ShrS: SIMD_SHIFT(wasm_i16x8_shr)
          case SimdOp::I16x8ShrU: SIMD_SHIFT(wasm_u16x8_shr)
          case SimdOp::I32x4Shl: SIMD_SHIFT(wasm_i32x4_shl)
          case SimdOp::I32x4ShrS: SIMD_SHIFT(wasm_i32x4_shr)
          case SimdOp::I32x4ShrU: SIMD_SHIFT(wasm_u32x4_shr)
          case SimdOp::I64x2Shl: SIMD_SHIFT(wasm_i64x2_shl)
          case SimdOp::I64x2ShrS: SIMD_SHIFT(wasm_i64x2_shr)
          case SimdOp::I64x2ShrU: SIMD_SHIFT(wasm_u64x2_shr)

          // -- narrow / extend / extadd / extmul --
          case SimdOp::I8x16NarrowI16x8S: SIMD_BINOP(wasm_i8x16_narrow_i16x8)
          case SimdOp::I8x16NarrowI16x8U: SIMD_BINOP(wasm_u8x16_narrow_i16x8)
          case SimdOp::I16x8NarrowI32x4S: SIMD_BINOP(wasm_i16x8_narrow_i32x4)
          case SimdOp::I16x8NarrowI32x4U: SIMD_BINOP(wasm_u16x8_narrow_i32x4)
          case SimdOp::I16x8ExtendLowI8x16S: SIMD_UNOP(wasm_i16x8_extend_low_i8x16)
          case SimdOp::I16x8ExtendHighI8x16S: SIMD_UNOP(wasm_i16x8_extend_high_i8x16)
          case SimdOp::I16x8ExtendLowI8x16U: SIMD_UNOP(wasm_u16x8_extend_low_u8x16)
          case SimdOp::I16x8ExtendHighI8x16U: SIMD_UNOP(wasm_u16x8_extend_high_u8x16)
          case SimdOp::I32x4ExtendLowI16x8S: SIMD_UNOP(wasm_i32x4_extend_low_i16x8)
          case SimdOp::I32x4ExtendHighI16x8S: SIMD_UNOP(wasm_i32x4_extend_high_i16x8)
          case SimdOp::I32x4ExtendLowI16x8U: SIMD_UNOP(wasm_u32x4_extend_low_u16x8)
          case SimdOp::I32x4ExtendHighI16x8U: SIMD_UNOP(wasm_u32x4_extend_high_u16x8)
          case SimdOp::I64x2ExtendLowI32x4S: SIMD_UNOP(wasm_i64x2_extend_low_i32x4)
          case SimdOp::I64x2ExtendHighI32x4S: SIMD_UNOP(wasm_i64x2_extend_high_i32x4)
          case SimdOp::I64x2ExtendLowI32x4U: SIMD_UNOP(wasm_u64x2_extend_low_u32x4)
          case SimdOp::I64x2ExtendHighI32x4U: SIMD_UNOP(wasm_u64x2_extend_high_u32x4)
          case SimdOp::I16x8ExtaddPairwiseI8x16S: SIMD_UNOP(wasm_i16x8_extadd_pairwise_i8x16)
          case SimdOp::I16x8ExtaddPairwiseI8x16U: SIMD_UNOP(wasm_u16x8_extadd_pairwise_u8x16)
          case SimdOp::I32x4ExtaddPairwiseI16x8S: SIMD_UNOP(wasm_i32x4_extadd_pairwise_i16x8)
          case SimdOp::I32x4ExtaddPairwiseI16x8U: SIMD_UNOP(wasm_u32x4_extadd_pairwise_u16x8)
          case SimdOp::I16x8ExtmulLowI8x16S: SIMD_BINOP(wasm_i16x8_extmul_low_i8x16)
          case SimdOp::I16x8ExtmulHighI8x16S: SIMD_BINOP(wasm_i16x8_extmul_high_i8x16)
          case SimdOp::I16x8ExtmulLowI8x16U: SIMD_BINOP(wasm_u16x8_extmul_low_u8x16)
          case SimdOp::I16x8ExtmulHighI8x16U: SIMD_BINOP(wasm_u16x8_extmul_high_u8x16)
          case SimdOp::I32x4ExtmulLowI16x8S: SIMD_BINOP(wasm_i32x4_extmul_low_i16x8)
          case SimdOp::I32x4ExtmulHighI16x8S: SIMD_BINOP(wasm_i32x4_extmul_high_i16x8)
          case SimdOp::I32x4ExtmulLowI16x8U: SIMD_BINOP(wasm_u32x4_extmul_low_u16x8)
          case SimdOp::I32x4ExtmulHighI16x8U: SIMD_BINOP(wasm_u32x4_extmul_high_u16x8)
          case SimdOp::I64x2ExtmulLowI32x4S: SIMD_BINOP(wasm_i64x2_extmul_low_i32x4)
          case SimdOp::I64x2ExtmulHighI32x4S: SIMD_BINOP(wasm_i64x2_extmul_high_i32x4)
          case SimdOp::I64x2ExtmulLowI32x4U: SIMD_BINOP(wasm_u64x2_extmul_low_u32x4)
          case SimdOp::I64x2ExtmulHighI32x4U: SIMD_BINOP(wasm_u64x2_extmul_high_u32x4)

          // -- float unary --
          case SimdOp::F32x4Abs: SIMD_UNOP(wasm_f32x4_abs)
          case SimdOp::F32x4Neg: SIMD_UNOP(wasm_f32x4_neg)
          case SimdOp::F32x4Sqrt: SIMD_UNOP(wasm_f32x4_sqrt)
          case SimdOp::F32x4Ceil: SIMD_UNOP(wasm_f32x4_ceil)
          case SimdOp::F32x4Floor: SIMD_UNOP(wasm_f32x4_floor)
          case SimdOp::F32x4Trunc: SIMD_UNOP(wasm_f32x4_trunc)
          case SimdOp::F32x4Nearest: SIMD_UNOP(wasm_f32x4_nearest)
          case SimdOp::F64x2Abs: SIMD_UNOP(wasm_f64x2_abs)
          case SimdOp::F64x2Neg: SIMD_UNOP(wasm_f64x2_neg)
          case SimdOp::F64x2Sqrt: SIMD_UNOP(wasm_f64x2_sqrt)
          case SimdOp::F64x2Ceil: SIMD_UNOP(wasm_f64x2_ceil)
          case SimdOp::F64x2Floor: SIMD_UNOP(wasm_f64x2_floor)
          case SimdOp::F64x2Trunc: SIMD_UNOP(wasm_f64x2_trunc)
          case SimdOp::F64x2Nearest: SIMD_UNOP(wasm_f64x2_nearest)

          // -- float binary --
          case SimdOp::F32x4Add: SIMD_BINOP(wasm_f32x4_add)
          case SimdOp::F32x4Sub: SIMD_BINOP(wasm_f32x4_sub)
          case SimdOp::F32x4Mul: SIMD_BINOP(wasm_f32x4_mul)
          case SimdOp::F32x4Div: SIMD_BINOP(wasm_f32x4_div)
          case SimdOp::F32x4Min: SIMD_BINOP(wasm_f32x4_min)
          case SimdOp::F32x4Max: SIMD_BINOP(wasm_f32x4_max)
          case SimdOp::F32x4PMin: SIMD_BINOP(wasm_f32x4_pmin)
          case SimdOp::F32x4PMax: SIMD_BINOP(wasm_f32x4_pmax)
          case SimdOp::F64x2Add: SIMD_BINOP(wasm_f64x2_add)
          case SimdOp::F64x2Sub: SIMD_BINOP(wasm_f64x2_sub)
          case SimdOp::F64x2Mul: SIMD_BINOP(wasm_f64x2_mul)
          case SimdOp::F64x2Div: SIMD_BINOP(wasm_f64x2_div)
          case SimdOp::F64x2Min: SIMD_BINOP(wasm_f64x2_min)
          case SimdOp::F64x2Max: SIMD_BINOP(wasm_f64x2_max)
          case SimdOp::F64x2PMin: SIMD_BINOP(wasm_f64x2_pmin)
          case SimdOp::F64x2PMax: SIMD_BINOP(wasm_f64x2_pmax)

          // -- conversions --
          case SimdOp::I32x4TruncSatF32x4S: SIMD_UNOP(wasm_i32x4_trunc_sat_f32x4)
          case SimdOp::I32x4TruncSatF32x4U: SIMD_UNOP(wasm_u32x4_trunc_sat_f32x4)
          case SimdOp::I32x4TruncSatF64x2SZero: SIMD_UNOP(wasm_i32x4_trunc_sat_f64x2_zero)
          case SimdOp::I32x4TruncSatF64x2UZero: SIMD_UNOP(wasm_u32x4_trunc_sat_f64x2_zero)
          case SimdOp::F32x4ConvertI32x4S: SIMD_UNOP(wasm_f32x4_convert_i32x4)
          case SimdOp::F32x4ConvertI32x4U: SIMD_UNOP(wasm_f32x4_convert_u32x4)
          case SimdOp::F64x2ConvertLowI32x4S: SIMD_UNOP(wasm_f64x2_convert_low_i32x4)
          case SimdOp::F64x2ConvertLowI32x4U: SIMD_UNOP(wasm_f64x2_convert_low_u32x4)
          case SimdOp::F32x4DemoteF64x2Zero: SIMD_UNOP(wasm_f32x4_demote_f64x2_zero)
          case SimdOp::F64x2PromoteLowF32x4: SIMD_UNOP(wasm_f64x2_promote_low_f32x4)

          default:
            JS_ReportErrorASCII(cx, "wasm interp: unsupported SIMD op 0x%x",
                                unsigned(sub));
            FAIL();
        }
#undef SIMD_BINOP
#undef SIMD_UNOP
#undef SIMD_SPLAT
#undef SIMD_SHIFT
#undef SIMD_LOAD
#undef SIMD_EXTRACT
#undef SIMD_REPLACE
#undef SIMD_TEST
        break;
#else
        JS_ReportErrorASCII(cx, "wasm interp: SIMD unsupported in this build");
        FAIL();
#endif
      }

      default:
        JS_ReportErrorASCII(cx, "wasm interp: unsupported opcode 0x%x", op);
        FAIL();
    }
  }

  // Fell off the end of the body == implicit return.
  for (uint32_t i = 0; i < nResults; i++) {
    results[i] = stk[sp - nResults + i];
  }

done:
  topFrame = frame.prev;
  js_free(locals);
  js_free(stk);
  js_free(tags);
  return ok;

#undef TRAP
#undef FAIL
#undef PUSH64
#undef PUSHEXT
}

}  // namespace interp
}  // namespace wasm
}  // namespace js

#endif  // wasm_WasmInterpRun_inl_h
