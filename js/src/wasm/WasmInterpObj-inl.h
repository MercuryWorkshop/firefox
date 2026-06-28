/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * JSObject integration for the in-process wasm interpreter: the Module /
 * Instance / Memory / Table / Global wrapper objects, export-function natives,
 * the GC roots tracer, and the public CompileBytes / InstantiateModuleObject
 * entry points. Included by WasmInterp.h.
 */

#ifndef wasm_WasmInterpObj_inl_h
#define wasm_WasmInterpObj_inl_h

namespace js {
namespace wasm {
namespace interp {

// ---- GC: a process-global list of live instances, traced as roots ----------

// THREAD-LOCAL: the full Gecko runs wasm on multiple threads, each with its own
// JSRuntime/GC. An interp Instance is created, used, GC'd, and finalized all on
// one thread (no shared memory), so each thread keeps its own live lists and
// registers its own roots tracer. A process-global list would let one thread's
// GC trace another runtime's GC things -> cross-runtime onObjectEdge OOB.
static thread_local Instance* gLiveInstances = nullptr;
static thread_local InstTable* gExternTables = nullptr;  // externref tables (GC)
static thread_local InstTable* gFuncTables = nullptr;    // funcref tables (GC)
static thread_local bool gTracerRegistered = false;

// A funcref cell packs (Instance*<<32 | funcIndex); 0 = null. The Instance* is a
// stable js_new'd pointer (not GC-moved), but the InstanceObject it owns is GC
// and must be kept alive while any table/global/stack funcref references it.
static inline void TraceFuncrefOwner(JSTracer* trc, uint64_t packed) {
  if ((packed >> 32) == 0) return;  // null
  Instance* inst = reinterpret_cast<Instance*>(uintptr_t(packed >> 32));
  if (inst && inst->instanceObj) {
    JS::TraceRoot(trc, &inst->instanceObj, "interp-funcref-owner");
  }
}

// Exported (not file-static) so emscripten's PIC linker keeps a real function
// for the JS_AddExtraGCRootsTracer callback rather than turning its address
// into an unresolved env import (same fix the JS->wasm JIT needed).
extern "C" EMSCRIPTEN_KEEPALIVE void InterpTraceRoots(JSTracer* trc,
                                                      void* data) {
  for (Instance* i = gLiveInstances; i; i = i->nextLive) {
    i->trace(trc);
  }
  for (InstTable* t = gExternTables; t; t = t->nextExtern) {
    for (size_t i = 0; i < t->refs.length(); i++) {
      JS::TraceRoot(trc, &t->refs[i], "interp-extern-table");
    }
  }
  for (InstTable* t = gFuncTables; t; t = t->nextFuncTable) {
    for (size_t i = 0; i < t->funcs.length(); i++) {
      TraceFuncrefOwner(trc, t->funcs[i]);
    }
  }
}

static bool EnsureTracer(JSContext* cx) {
  if (gTracerRegistered) return true;
  if (!JS_AddExtraGCRootsTracer(cx, (JSTraceDataOp)InterpTraceRoots, nullptr)) {
    return false;
  }
  gTracerRegistered = true;
  return true;
}

// ---- Module wrapper --------------------------------------------------------

static void ModuleObj_finalize(JS::GCContext* gcx, JSObject* obj) {
  Module* m = JS::GetMaybePtrFromReservedSlot<Module>(obj, 0);
  // Pinned == shared cross-thread (another thread's wrapper still references it).
  // Leak rather than free, since we don't cross-thread refcount.
  if (m && !m->pinned) js_delete(m);
}
static const JSClassOps kModuleClassOps = {.finalize = ModuleObj_finalize};
static const JSClass kModuleClass = {
    "WasmInterpModule",
    JSCLASS_HAS_RESERVED_SLOTS(1) | JSCLASS_FOREGROUND_FINALIZE,
    &kModuleClassOps};

bool IsModuleObject(JSObject* obj) { return obj->hasClass(&kModuleClass); }
static Module* ModuleFromObject(JSObject* obj) {
  return JS::GetMaybePtrFromReservedSlot<Module>(obj, 0);
}

// ---- Instance wrapper ------------------------------------------------------

static void InstanceObj_finalize(JS::GCContext* gcx, JSObject* obj) {
  Instance* inst = JS::GetMaybePtrFromReservedSlot<Instance>(obj, 0);
  if (!inst) return;
  // Unlink from the live list.
  Instance** pp = &gLiveInstances;
  while (*pp) {
    if (*pp == inst) {
      *pp = inst->nextLive;
      break;
    }
    pp = &(*pp)->nextLive;
  }
  js_delete(inst);
}
static const JSClassOps kInstanceClassOps = {.finalize = InstanceObj_finalize};
static const JSClass kInstanceClass = {
    "WasmInterpInstance",
    JSCLASS_HAS_RESERVED_SLOTS(1) | JSCLASS_FOREGROUND_FINALIZE,
    &kInstanceClassOps};
bool IsInstanceObject(JSObject* obj) { return obj->hasClass(&kInstanceClass); }

// ---- Memory wrapper --------------------------------------------------------

static void MemoryObj_finalize(JS::GCContext* gcx, JSObject* obj) {
  LinearMemory* lm = JS::GetMaybePtrFromReservedSlot<LinearMemory>(obj, 0);
  if (!lm) return;
  // Pinned == shared cross-thread (another thread's wrapper references this lm
  // and the same backing). Leak rather than free.
  if (lm->pinned) return;
  // Non-shared memory owns a js_pod_calloc'd base; shared memory's base belongs
  // to the SharedArrayBuffer (GC-owned), so we must not free it here.
  if (!lm->rawbuf) {
    js_free(lm->base);
  }
  js_delete(lm);
}
static const JSClassOps kMemoryClassOps = {.finalize = MemoryObj_finalize};
static const JSClass kMemoryClass = {
    "WasmInterpMemory",
    JSCLASS_HAS_RESERVED_SLOTS(1) | JSCLASS_FOREGROUND_FINALIZE,
    &kMemoryClassOps};

LinearMemory* MemoryFromObject(JSObject* obj) {
  return JS::GetMaybePtrFromReservedSlot<LinearMemory>(obj, 0);
}

// Create an interp wrapper object whose prototype is the matching
// WebAssembly.X.prototype, so content's `obj instanceof WebAssembly.Memory`
// (etc.) succeeds -- emscripten's pthread loader asserts this. Our own
// "buffer"/"grow"/... own-properties still shadow the real prototype's methods
// (which would not understand our custom object).
static JSObject* InterpNewObject(JSContext* cx, const JSClass* clasp,
                                 JSProtoKey key) {
  JS::RootedObject proto(cx, GlobalObject::getOrCreatePrototype(cx, key));
  if (!proto) return nullptr;
  return JS_NewObjectWithGivenProto(cx, clasp, proto);
}

static JSObject* MakeMemoryAB(JSContext* cx, LinearMemory* lm) {
  return JS::NewArrayBufferWithUserOwnedContents(cx, lm->size, lm->base);
}

static bool Memory_grow(JSContext* cx, unsigned argc, JS::Value* vp);

// Build a kMemoryClass wrapper around an existing LinearMemory + buffer object.
// Used both for freshly-created memories and for cross-thread clones (where lm is
// a shared C++ pointer and `ab` is a SAB reconstructed over the shared backing).
static JSObject* WrapMemoryObject(JSContext* cx, LinearMemory* lm,
                                  JS::HandleObject ab) {
  JS::RootedObject obj(cx, InterpNewObject(cx, &kMemoryClass, JSProto_WasmMemory));
  if (!obj) return nullptr;
  JS::SetReservedSlot(obj, 0, JS::PrivateValue(lm));
  JS::RootedValue abv(cx, JS::ObjectValue(*ab));
  if (!JS_DefineProperty(cx, obj, "buffer", abv, JSPROP_ENUMERATE)) {
    return nullptr;
  }
  if (!JS_DefineFunction(cx, obj, "grow", Memory_grow, 1, JSPROP_ENUMERATE)) {
    return nullptr;
  }
  return obj;
}

static JSObject* CreateMemoryObject(JSContext* cx, uint32_t initialPages,
                                    uint32_t maxPages, bool shared) {
  uint32_t maxP = (maxPages == UINT32_MAX) ? 65535 : (maxPages > 65535 ? 65535 : maxPages);
  size_t bytes = size_t(initialPages) * 65536;

  LinearMemory* lm = js_new<LinearMemory>();
  if (!lm) {
    ReportOutOfMemory(cx);
    return nullptr;
  }
  lm->maxBytes = maxP * 65536;
  lm->shared = shared;

  JS::RootedObject ab(cx);
  // Use a real SharedArrayBuffer for shared memory when the realm allows it
  // (cross-origin-isolated content): cross-thread clone-shareable AND emscripten's
  // JS futex (Atomics.wait/notify on HEAP) needs a real SAB. Otherwise (non-shared,
  // or a realm without SAB) use a USER_OWNED ArrayBuffer over a js_pod_calloc base.
  bool useSAB =
      shared && cx->realm()->creationOptions().getSharedMemoryAndAtomicsEnabled();
  if (getenv("GECKO_INTERP_DEBUG")) {
    fprintf(stderr, "[interp] create-memory shared=%d useSAB=%d bytes=%zu\n",
            shared, useSAB, bytes);
  }
  if (useSAB) {
    JS::Rooted<FixedLengthSharedArrayBufferObject*> sab(
        cx, SharedArrayBufferObject::New(cx, bytes));
    if (!sab) {
      js_delete(lm);
      return nullptr;
    }
    bool isShared = false;
    {
      JS::AutoCheckCannotGC nogc;
      lm->base = JS::GetSharedArrayBufferData(sab, &isShared, nogc);
    }
    lm->size = uint32_t(bytes);
    lm->rawbuf = sab->rawBufferObject();
    ab = sab;
  } else {
    uint8_t* base = js_pod_calloc<uint8_t>(bytes ? bytes : 1);
    if (!base) {
      js_delete(lm);
      ReportOutOfMemory(cx);
      return nullptr;
    }
    lm->base = base;
    lm->size = uint32_t(bytes);
    ab = MakeMemoryAB(cx, lm);
    if (!ab) {
      js_free(base);
      js_delete(lm);
      return nullptr;
    }
  }
  return WrapMemoryObject(cx, lm, ab);
}

JSObject* NewMemoryObjectJS(JSContext* cx, uint32_t initialPages,
                            uint32_t maxPages, bool shared) {
  return CreateMemoryObject(cx, initialPages, maxPages, shared);
}

int64_t GrowMemoryObject(JSContext* cx, JS::HandleObject memObj,
                         uint32_t deltaPages) {
  LinearMemory* lm = MemoryFromObject(memObj);
  uint32_t oldPages = lm->size / 65536;
  uint64_t newBytes = uint64_t(oldPages + deltaPages) * 65536;
  if (getenv("GECKO_INTERP_DEBUG")) {
    fprintf(stderr, "[interp] grow delta=%u old=%uB new=%lluB max=%uB\n",
            deltaPages, lm->size, (unsigned long long)newBytes, lm->maxBytes);
  }
  if (newBytes > lm->maxBytes) return -1;
  // Detach the old buffer (USER_OWNED: detach frees nothing of ours).
  JS::RootedValue bufv(cx);
  if (!JS_GetProperty(cx, memObj, "buffer", &bufv)) return -1;
  if (bufv.isObject()) {
    JS::RootedObject old(cx, &bufv.toObject());
    JS::DetachArrayBuffer(cx, old);
  }
  uint8_t* nb = js_pod_realloc<uint8_t>(lm->base, lm->size ? lm->size : 1,
                                        newBytes ? newBytes : 1);
  if (!nb) {
    ReportOutOfMemory(cx);
    return -1;
  }
  if (newBytes > lm->size) memset(nb + lm->size, 0, newBytes - lm->size);
  lm->base = nb;
  lm->size = uint32_t(newBytes);
  JS::RootedObject ab(cx, MakeMemoryAB(cx, lm));
  if (!ab) return -1;
  JS::RootedValue abv(cx, JS::ObjectValue(*ab));
  if (!JS_DefineProperty(cx, memObj, "buffer", abv, JSPROP_ENUMERATE)) {
    return -1;
  }
  return int64_t(oldPages);
}

static bool Memory_grow(JSContext* cx, unsigned argc, JS::Value* vp) {
  JS::CallArgs args = JS::CallArgsFromVp(argc, vp);
  if (!args.thisv().isObject() ||
      !args.thisv().toObject().hasClass(&kMemoryClass)) {
    JS_ReportErrorASCII(cx, "Memory.grow: bad this");
    return false;
  }
  JS::RootedObject self(cx, &args.thisv().toObject());
  uint32_t delta = 0;
  if (args.length() > 0 && !JS::ToUint32(cx, args[0], &delta)) return false;
  int64_t old = GrowMemoryObject(cx, self, delta);
  if (cx->isExceptionPending()) return false;
  if (old < 0) {
    JS_ReportErrorASCII(cx, "Memory.grow failed");
    return false;
  }
  args.rval().setNumber(double(old));
  return true;
}

// ---- Table wrapper ---------------------------------------------------------

static void TableObj_finalize(JS::GCContext* gcx, JSObject* obj) {
  InstTable* t = JS::GetMaybePtrFromReservedSlot<InstTable>(obj, 0);
  if (!t) return;
  if (t->elem == VT::ExternRef) {
    InstTable** pp = &gExternTables;
    while (*pp) {
      if (*pp == t) { *pp = t->nextExtern; break; }
      pp = &(*pp)->nextExtern;
    }
  } else {
    InstTable** pp = &gFuncTables;
    while (*pp) {
      if (*pp == t) { *pp = t->nextFuncTable; break; }
      pp = &(*pp)->nextFuncTable;
    }
  }
  js_delete(t);
}
static const JSClassOps kTableClassOps = {.finalize = TableObj_finalize};
static const JSClass kTableClass = {
    "WasmInterpTable",
    JSCLASS_HAS_RESERVED_SLOTS(1) | JSCLASS_FOREGROUND_FINALIZE,
    &kTableClassOps};

InstTable* TableFromObject(JSObject* obj) {
  return JS::GetMaybePtrFromReservedSlot<InstTable>(obj, 0);
}

static bool Table_get(JSContext* cx, unsigned argc, JS::Value* vp);
static bool Table_set(JSContext* cx, unsigned argc, JS::Value* vp);
static bool Table_grow(JSContext* cx, unsigned argc, JS::Value* vp);
static bool Table_lengthGetter(JSContext* cx, unsigned argc, JS::Value* vp);

static JSObject* CreateTableObject(JSContext* cx, VT elem, uint32_t initial,
                                   uint32_t maximum) {
  InstTable* t = js_new<InstTable>();
  if (!t) {
    ReportOutOfMemory(cx);
    return nullptr;
  }
  t->elem = elem;
  t->maxLen = maximum;
  bool ok = true;
  if (elem == VT::FuncRef) {
    ok = t->funcs.resize(initial);
    for (uint32_t i = 0; i < initial && ok; i++) t->funcs[i] = 0;  // null
  } else {
    ok = t->refs.resize(initial);
    for (uint32_t i = 0; i < initial && ok; i++) t->refs[i] = JS::NullValue();
  }
  if (!ok) {
    js_delete(t);
    ReportOutOfMemory(cx);
    return nullptr;
  }
  JS::RootedObject obj(cx, InterpNewObject(cx, &kTableClass, JSProto_WasmTable));
  if (!obj) {
    js_delete(t);
    return nullptr;
  }
  JS::SetReservedSlot(obj, 0, JS::PrivateValue(t));
  if (elem == VT::ExternRef) {
    t->nextExtern = gExternTables;
    gExternTables = t;
  } else {
    t->nextFuncTable = gFuncTables;
    gFuncTables = t;
  }
  if (!JS_DefineFunction(cx, obj, "get", Table_get, 1, JSPROP_ENUMERATE) ||
      !JS_DefineFunction(cx, obj, "set", Table_set, 2, JSPROP_ENUMERATE) ||
      !JS_DefineFunction(cx, obj, "grow", Table_grow, 1, JSPROP_ENUMERATE)) {
    return nullptr;
  }
  if (!JS_DefineProperty(cx, obj, "length", Table_lengthGetter, nullptr,
                         JSPROP_ENUMERATE)) {
    return nullptr;
  }
  return obj;
}

JSObject* NewTableObjectJS(JSContext* cx, VT elem, uint32_t initial,
                           uint32_t maximum) {
  return CreateTableObject(cx, elem, initial, maximum);
}

// Convert a JS value -> table cell bits (funcref packed, or externref Value).
static bool TableValToBits(JSContext* cx, InstTable* t, JS::HandleValue v,
                           uint64_t* out) {
  if (t->elem == VT::FuncRef) {
    if (v.isNull()) {
      *out = 0;
      return true;
    }
    if (v.isObject() && GetFuncWrapperPacked(&v.toObject(), out)) return true;
    // A plain JS function: throw a TypeError so emscripten's addFunction
    // catches it and retries via convertJsFunctionToWasm (whose export IS one
    // of our wrappers). A generic Error would not trigger that fallback.
    JS_ReportErrorNumberASCII(cx, GetErrorMessage, nullptr, JSMSG_NOT_FUNCTION,
                              "argument");
    return false;
  }
  *out = v.asRawBits();
  return true;
}

static bool Table_get(JSContext* cx, unsigned argc, JS::Value* vp) {
  JS::CallArgs args = JS::CallArgsFromVp(argc, vp);
  JS::RootedObject self(cx, &args.thisv().toObject());
  InstTable* t = TableFromObject(self);
  uint32_t i = 0;
  if (!JS::ToUint32(cx, args.get(0), &i)) return false;
  if (i >= t->length()) {
    JS_ReportErrorASCII(cx, "table.get: out of bounds");
    return false;
  }
  if (t->elem == VT::FuncRef) {
    uint64_t fr = t->funcs[i];
    if (FuncIsNull(fr)) {
      args.rval().setNull();
      return true;
    }
    JSObject* w = GetOrCreateFuncWrapper(cx, FuncInst(fr), FuncIdx(fr));
    if (!w) return false;
    args.rval().setObject(*w);
  } else {
    args.rval().set(t->refs[i]);
  }
  return true;
}

static bool Table_set(JSContext* cx, unsigned argc, JS::Value* vp) {
  JS::CallArgs args = JS::CallArgsFromVp(argc, vp);
  JS::RootedObject self(cx, &args.thisv().toObject());
  InstTable* t = TableFromObject(self);
  uint32_t i = 0;
  if (!JS::ToUint32(cx, args.get(0), &i)) return false;
  if (i >= t->length()) {
    JS_ReportErrorASCII(cx, "table.set: out of bounds");
    return false;
  }
  uint64_t bits;
  if (!TableValToBits(cx, t, args.get(1), &bits)) return false;
  if (t->elem == VT::FuncRef) {
    t->funcs[i] = bits;
  } else {
    t->refs[i] = JS::Value::fromRawBits(bits);
  }
  args.rval().setUndefined();
  return true;
}

int64_t GrowTableObject(JSContext* cx, JS::HandleObject tableObj,
                        uint32_t delta, uint64_t initFunc,
                        JS::HandleValue initRef) {
  InstTable* t = TableFromObject(tableObj);
  uint32_t oldLen = t->length();
  uint64_t newLen = uint64_t(oldLen) + delta;
  if (newLen > t->maxLen) return -1;
  if (t->elem == VT::FuncRef) {
    if (!t->funcs.resize(newLen)) {
      ReportOutOfMemory(cx);
      return -1;
    }
    for (uint32_t i = oldLen; i < newLen; i++) t->funcs[i] = initFunc;
  } else {
    if (!t->refs.resize(newLen)) {
      ReportOutOfMemory(cx);
      return -1;
    }
    for (uint32_t i = oldLen; i < newLen; i++) t->refs[i] = initRef;
  }
  return int64_t(oldLen);
}

static bool Table_grow(JSContext* cx, unsigned argc, JS::Value* vp) {
  JS::CallArgs args = JS::CallArgsFromVp(argc, vp);
  JS::RootedObject self(cx, &args.thisv().toObject());
  InstTable* t = TableFromObject(self);
  uint32_t delta = 0;
  if (!JS::ToUint32(cx, args.get(0), &delta)) return false;
  uint64_t initFunc = 0;
  JS::RootedValue initRef(cx, args.get(1));
  if (t->elem == VT::FuncRef && !args.get(1).isNullOrUndefined()) {
    if (!TableValToBits(cx, t, args.get(1), &initFunc)) return false;
  }
  int64_t old = GrowTableObject(cx, self, delta, initFunc, initRef);
  if (cx->isExceptionPending()) return false;
  if (old < 0) {
    JS_ReportErrorASCII(cx, "table.grow failed");
    return false;
  }
  args.rval().setNumber(double(old));
  return true;
}

static bool Table_lengthGetter(JSContext* cx, unsigned argc, JS::Value* vp) {
  JS::CallArgs args = JS::CallArgsFromVp(argc, vp);
  if (!args.thisv().isObject()) {
    args.rval().setInt32(0);
    return true;
  }
  InstTable* t = TableFromObject(&args.thisv().toObject());
  args.rval().setInt32(int32_t(t ? t->length() : 0));
  return true;
}

// ---- Global wrapper (minimal: numeric + exported module globals) ------------

static void GlobalObj_finalize(JS::GCContext* gcx, JSObject* obj) {
  // slot1 == -1 means standalone (we own a Cell at slot0).
  JS::Value m = JS::GetReservedSlot(obj, 1);
  if (m.isInt32() && m.toInt32() == -1) {
    Cell* c = JS::GetMaybePtrFromReservedSlot<Cell>(obj, 0);
    if (c) js_free(c);
  }
}
static const JSClassOps kGlobalClassOps = {.finalize = GlobalObj_finalize};
static const JSClass kGlobalClass = {
    "WasmInterpGlobal",
    JSCLASS_HAS_RESERVED_SLOTS(3) | JSCLASS_FOREGROUND_FINALIZE,
    &kGlobalClassOps};

static bool Global_valueGetter(JSContext* cx, unsigned argc, JS::Value* vp);

// Build a wrapper for an exported module global at (inst, globalIndex).
static JSObject* MakeGlobalExport(JSContext* cx, Instance* inst,
                                  uint32_t index) {
  JS::RootedObject obj(cx, InterpNewObject(cx, &kGlobalClass, JSProto_WasmGlobal));
  if (!obj) return nullptr;
  JS::SetReservedSlot(obj, 0, JS::PrivateValue(inst));
  JS::SetReservedSlot(obj, 1, JS::Int32Value(int32_t(index)));
  JS::SetReservedSlot(obj, 2,
                      JS::Int32Value(int32_t(inst->module->globals[index].type)));
  if (!JS_DefineProperty(cx, obj, "value", Global_valueGetter, nullptr,
                         JSPROP_ENUMERATE)) {
    return nullptr;
  }
  return obj;
}

static bool Global_valueGetter(JSContext* cx, unsigned argc, JS::Value* vp) {
  JS::CallArgs args = JS::CallArgsFromVp(argc, vp);
  JS::RootedObject self(cx, &args.thisv().toObject());
  JS::Value idxv = JS::GetReservedSlot(self, 1);
  VT t = VT(JS::GetReservedSlot(self, 2).toInt32());
  Cell c;
  Instance* inst = nullptr;
  if (idxv.toInt32() == -1) {
    c = *JS::GetMaybePtrFromReservedSlot<Cell>(self, 0);
  } else {
    inst = JS::GetMaybePtrFromReservedSlot<Instance>(self, 0);
    c = inst->globalCells[idxv.toInt32()];
  }
  // Reuse the marshaler via a temporary instance pointer when available.
  switch (t) {
    case VT::I32: args.rval().setInt32(c.i32); return true;
    case VT::I64: {
      JS::BigInt* bi = JS::BigInt::createFromInt64(cx, c.i64);
      if (!bi) return false;
      args.rval().setBigInt(bi);
      return true;
    }
    case VT::F32: args.rval().setDouble(double(c.f32)); return true;
    case VT::F64: args.rval().setDouble(c.f64); return true;
    case VT::ExternRef: args.rval().set(JS::Value::fromRawBits(c.u64)); return true;
    case VT::FuncRef: {
      if (FuncIsNull(c.u64)) { args.rval().setNull(); return true; }
      JSObject* w = GetOrCreateFuncWrapper(cx, FuncInst(c.u64), FuncIdx(c.u64));
      if (!w) return false;
      args.rval().setObject(*w);
      return true;
    }
    default: args.rval().setUndefined(); return true;
  }
}

JSObject* NewGlobalObjectJS(JSContext* cx, VT type, bool isMutable,
                            const Cell* initVal, JS::HandleValue initRef) {
  Cell* c = js_pod_malloc<Cell>(1);
  if (!c) { ReportOutOfMemory(cx); return nullptr; }
  *c = *initVal;
  if (IsRefVT(type)) c->u64 = initRef.isUndefined() ? 0 : initRef.asRawBits();
  JS::RootedObject obj(cx, InterpNewObject(cx, &kGlobalClass, JSProto_WasmGlobal));
  if (!obj) { js_free(c); return nullptr; }
  JS::SetReservedSlot(obj, 0, JS::PrivateValue(c));
  JS::SetReservedSlot(obj, 1, JS::Int32Value(-1));
  JS::SetReservedSlot(obj, 2, JS::Int32Value(int32_t(type)));
  (void)isMutable;
  if (!JS_DefineProperty(cx, obj, "value", Global_valueGetter, nullptr,
                         JSPROP_ENUMERATE)) {
    return nullptr;
  }
  return obj;
}

// ---- export-function natives + funcref wrappers ----------------------------

static bool InterpExportCall(JSContext* cx, unsigned argc, JS::Value* vp) {
  JS::CallArgs args = JS::CallArgsFromVp(argc, vp);
  JSFunction& callee = args.callee().as<JSFunction>();
  JSObject* io = &callee.getExtendedSlot(0).toObject();
  Instance* inst = JS::GetMaybePtrFromReservedSlot<Instance>(io, 0);
  uint32_t fi = uint32_t(callee.getExtendedSlot(1).toInt32());

  const FuncType& ft = inst->module->funcType(fi);
  uint32_t nP = ft.params.length();
  uint32_t nR = ft.results.length();
  Cell ab[64];
  Cell rb[16];
  if (nP > 64 || nR > 16) {
    JS_ReportErrorASCII(cx, "wasm interp: arity too large");
    return false;
  }
  for (uint32_t i = 0; i < nP; i++) {
    JS::RootedValue v(cx, args.get(i));
    if (!inst->valueToCell(v, ft.params[i], &ab[i])) return false;
  }
  if (!inst->invoke(fi, ab, rb)) {
    if (getenv("GECKO_INTERP_DEBUG")) {
      fprintf(stderr, "[interp] export-call fi=%u FAILED (pinned=%d)\n", fi,
              inst->module->pinned);
    }
    return false;
  }
  if (nR == 0) {
    args.rval().setUndefined();
  } else if (nR == 1) {
    JS::RootedValue r(cx);
    if (!inst->cellToValue(&rb[0], ft.results[0], &r)) return false;
    args.rval().set(r);
  } else {
    JS::RootedValueVector vals(cx);
    for (uint32_t i = 0; i < nR; i++) {
      JS::RootedValue r(cx);
      if (!inst->cellToValue(&rb[i], ft.results[i], &r)) return false;
      if (!vals.append(r)) { ReportOutOfMemory(cx); return false; }
    }
    JSObject* arr = JS::NewArrayObject(cx, vals);
    if (!arr) return false;
    args.rval().setObject(*arr);
  }
  return true;
}

bool GetFuncWrapperPacked(JSObject* obj, uint64_t* packedOut) {
  if (!obj->is<JSFunction>()) return false;
  JSFunction& fn = obj->as<JSFunction>();
  if (fn.maybeNative() != InterpExportCall) return false;
  JSObject* io = &fn.getExtendedSlot(0).toObject();
  Instance* inst = JS::GetMaybePtrFromReservedSlot<Instance>(io, 0);
  uint32_t fi = uint32_t(fn.getExtendedSlot(1).toInt32());
  *packedOut = PackFunc(inst, fi);
  return true;
}

JSObject* GetOrCreateFuncWrapper(JSContext* cx, Instance* inst,
                                 uint32_t funcIndex) {
  if (funcIndex < inst->funcWrappers.length() &&
      inst->funcWrappers[funcIndex]) {
    return inst->funcWrappers[funcIndex];
  }
  JS::Rooted<JSFunction*> fn(
      cx, NewNativeFunction(cx, InterpExportCall, 0, nullptr,
                            gc::AllocKind::FUNCTION_EXTENDED));
  if (!fn) return nullptr;
  fn->setExtendedSlot(0, JS::ObjectValue(*inst->instanceObj));
  fn->setExtendedSlot(1, JS::Int32Value(int32_t(funcIndex)));
  if (funcIndex < inst->funcWrappers.length()) {
    inst->funcWrappers[funcIndex] = fn;
  }
  return fn;
}

// ---- compile ---------------------------------------------------------------

JSObject* CompileBytes(JSContext* cx, const uint8_t* bytes, size_t len) {
  // Validate with Core features only so SIMD/threads/EH/GC modules are rejected
  // cleanly instead of trapping mid-execution.
  FeatureOptions options;
  UniqueChars error;
  BytecodeSource source(bytes, len);
  if (!Validate(cx, source, options, &error)) {
    if (error) {
      JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                               JSMSG_WASM_COMPILE_ERROR, error.get());
    } else if (!cx->isExceptionPending()) {
      JS_ReportErrorASCII(cx, "wasm interp: validation failed");
    }
    return nullptr;
  }
  if (getenv("GECKO_INTERP_DEBUG")) {
    fprintf(stderr, "[interp] compile len=%zu\n", len);
  }
  Module* m = DecodeModule(cx, bytes, len);
  if (!m) return nullptr;

  JS::RootedObject obj(cx, InterpNewObject(cx, &kModuleClass, JSProto_WasmModule));
  if (!obj) {
    js_delete(m);
    return nullptr;
  }
  JS::SetReservedSlot(obj, 0, JS::PrivateValue(m));
  return obj;
}

// ---- instantiate -----------------------------------------------------------

static bool GetImport(JSContext* cx, JS::HandleObject importObj,
                      const Import& imp, JS::MutableHandleValue out) {
  if (!importObj) {
    JS_ReportErrorASCII(cx, "wasm interp: no import object");
    return false;
  }
  JS::RootedValue modv(cx);
  if (!JS_GetProperty(cx, importObj, imp.module.get(), &modv)) return false;
  if (!modv.isObject()) {
    JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                             JSMSG_WASM_BAD_IMPORT_FIELD, imp.module.get());
    return false;
  }
  JS::RootedObject mo(cx, &modv.toObject());
  return JS_GetProperty(cx, mo, imp.field.get(), out);
}

static Cell EvalInitExpr(Instance* inst, uint8_t kind, uint64_t imm) {
  Cell c;
  c.u64 = 0;
  switch (Op(kind)) {
    case Op::I32Const: c.i32 = int32_t(uint32_t(imm)); break;
    case Op::I64Const: c.i64 = int64_t(imm); break;
    case Op::F32Const: { uint32_t b = uint32_t(imm); memcpy(&c.f32, &b, 4); break; }
    case Op::F64Const: memcpy(&c.f64, &imm, 8); break;
    case Op::GlobalGet: c = inst->globalCells[uint32_t(imm)]; break;
    case Op::RefFunc: c.u64 = PackFunc(inst, uint32_t(imm)); break;
    case Op::RefNull: c.u64 = 0; break;
    default: break;
  }
  return c;
}

JSObject* InstantiateModuleObject(JSContext* cx, JS::HandleObject moduleObj,
                                  JS::HandleObject importObj) {
  Module* m = ModuleFromObject(moduleObj);
  if (!m) {
    JS_ReportErrorASCII(cx, "wasm interp: not a module");
    return nullptr;
  }
  if (!EnsureTracer(cx)) return nullptr;

  Instance* inst = js_new<Instance>();
  if (!inst) { ReportOutOfMemory(cx); return nullptr; }
  inst->cx = cx;
  inst->module = m;
  inst->moduleObj = moduleObj;

  JS::RootedObject instObj(cx, InterpNewObject(cx, &kInstanceClass, JSProto_WasmInstance));
  if (!instObj) { js_delete(inst); return nullptr; }
  JS::SetReservedSlot(instObj, 0, JS::PrivateValue(inst));
  inst->instanceObj = instObj;

  uint32_t totalFuncs = m->funcTypes.length();
  if (!inst->importFuncs.resize(m->numImportFuncs) ||
      !inst->funcWrappers.resize(totalFuncs) ||
      !inst->globalCells.resize(m->globals.length()) ||
      !inst->tableObjs.resize(m->tables.length()) ||
      !inst->dataDropped.resize(m->datas.length()) ||
      !inst->elemDropped.resize(m->elems.length())) {
    ReportOutOfMemory(cx);
    return nullptr;
  }
  // Initialize ALL GC-traced state to safe values BEFORE publishing the
  // instance to the roots tracer below: import resolution calls into JS (which
  // can trigger GC), and the tracer must never read an uninitialized Value or
  // object pointer (resize() does not value-initialize). This was the cause of
  // an OOB crash in InterpTraceRoots when a GC hit mid-instantiation (e.g. an
  // addFunction trampoline instantiated during a request).
  for (size_t i = 0; i < inst->importFuncs.length(); i++) {
    inst->importFuncs[i] = JS::UndefinedValue();
  }
  for (size_t i = 0; i < totalFuncs; i++) inst->funcWrappers[i] = nullptr;
  for (size_t i = 0; i < inst->tableObjs.length(); i++) {
    inst->tableObjs[i] = nullptr;
  }
  for (size_t i = 0; i < inst->globalCells.length(); i++) {
    inst->globalCells[i].u64 =
        (m->globals[i].type == VT::ExternRef) ? JS::NullValue().asRawBits() : 0;
  }

  // Now fully safe-initialized: publish to the live-instance list (GC-traced).
  inst->nextLive = gLiveInstances;
  gLiveInstances = inst;

  // Resolve imports.
  uint32_t fIdx = 0, tIdx = 0, gIdx = 0;
  for (const Import& imp : m->imports) {
    JS::RootedValue v(cx);
    if (!GetImport(cx, importObj, imp, &v)) return nullptr;
    switch (imp.kind) {
      case ExternKind::Func:
        if (!IsCallable(v)) {
          JS_ReportErrorNumberUTF8(cx, GetErrorMessage, nullptr,
                                   JSMSG_WASM_BAD_IMPORT_TYPE, imp.field.get(),
                                   "function");
          return nullptr;
        }
        inst->importFuncs[fIdx++] = v;
        break;
      case ExternKind::Memory:
        if (!v.isObject() || !v.toObject().hasClass(&kMemoryClass)) {
          JS_ReportErrorASCII(cx, "wasm interp: imported memory not a Memory");
          return nullptr;
        }
        inst->memObj = &v.toObject();
        inst->lm = MemoryFromObject(inst->memObj);
        break;
      case ExternKind::Table:
        if (!v.isObject() || !v.toObject().hasClass(&kTableClass)) {
          JS_ReportErrorASCII(cx, "wasm interp: imported table not a Table");
          return nullptr;
        }
        inst->tableObjs[tIdx++] = &v.toObject();
        break;
      case ExternKind::Global: {
        VT gt = m->globals[gIdx].type;
        Cell c;
        c.u64 = 0;
        if (v.isObject() && v.toObject().hasClass(&kGlobalClass)) {
          JS::RootedObject go(cx, &v.toObject());
          JS::RootedValue gv(cx);
          if (!JS_GetProperty(cx, go, "value", &gv)) return nullptr;
          if (!inst->valueToCell(gv, gt, &c)) return nullptr;
        } else if (!inst->valueToCell(v, gt, &c)) {
          return nullptr;
        }
        inst->globalCells[gIdx++] = c;
        break;
      }
    }
  }

  // Create defined memory (single memory model). Shared memory is supported:
  // atomic ops operate on lm->base via real __atomic_* builtins (the base lives
  // in the outer engine's shared heap). NOTE (build 2): cross-thread sharing of
  // the Memory object + grow-without-move still pending; single-instance shared
  // memory works today.
  if (!inst->memObj && !m->memories.empty()) {
    const MemoryDesc& md = m->memories[0];
    JSObject* mo =
        CreateMemoryObject(cx, md.initialPages, md.maximumPages, md.shared);
    if (!mo) return nullptr;
    inst->memObj = mo;
    inst->lm = MemoryFromObject(mo);
  }

  // Create defined tables.
  for (uint32_t i = m->numImportTables; i < m->tables.length(); i++) {
    const TableDesc& td = m->tables[i];
    JSObject* to = CreateTableObject(cx, td.elem, td.initial, td.maximum);
    if (!to) return nullptr;
    inst->tableObjs[i] = to;
  }

  // Evaluate defined globals.
  for (uint32_t i = m->numImportGlobals; i < m->globals.length(); i++) {
    const GlobalDesc& g = m->globals[i];
    inst->globalCells[i] = EvalInitExpr(inst, g.initKind, g.initImm);
  }

  // Apply active element segments.
  for (size_t si = 0; si < m->elems.length(); si++) {
    ElemSeg& es = m->elems[si];
    if (!es.active) continue;
    uint32_t off = uint32_t(EvalInitExpr(inst, es.offKind, es.offImm).i32);
    InstTable* t = TableFromObject(inst->tableObjs[es.tableIndex]);
    if (uint64_t(off) + es.funcIndices.length() > t->length()) {
      JS_ReportErrorASCII(cx, "wasm interp: elem segment out of bounds");
      return nullptr;
    }
    for (size_t k = 0; k < es.funcIndices.length(); k++) {
      uint32_t fi = es.funcIndices[k];
      if (t->elem == VT::FuncRef) {
        t->funcs[off + k] = (fi == UINT32_MAX) ? 0 : PackFunc(inst, fi);
      } else {
        t->refs[off + k] = JS::NullValue();
      }
    }
    // NB: do NOT clearAndFree es.funcIndices here -- the table is per-instance,
    // so EVERY instance of a (cross-thread-shared) module must re-apply this
    // active segment to populate its own table. Freeing the shared module's
    // segment data after the first instance left every pthread worker's table
    // empty (getWasmTableEntry -> null). elemDropped is per-instance (correct).
    inst->elemDropped[si] = true;
  }

  // Apply active data segments.
  for (size_t si = 0; si < m->datas.length(); si++) {
    DataSeg& ds = m->datas[si];
    if (!ds.active) continue;
    uint32_t off = uint32_t(EvalInitExpr(inst, ds.offKind, ds.offImm).i32);
    if (uint64_t(off) + ds.bytes.length() > inst->lm->size) {
      JS_ReportErrorASCII(cx, "wasm interp: data segment out of bounds");
      return nullptr;
    }
    if (ds.bytes.length()) {
      memcpy(inst->lm->base + off, ds.bytes.begin(), ds.bytes.length());
    }
    inst->dataDropped[si] = true;
  }

  // Run the start function.
  if (m->hasStart) {
    if (getenv("GECKO_INTERP_DEBUG")) {
      fprintf(stderr, "[interp] instantiate: running start fn %u (pinned=%d)\n",
              m->startFunc, m->pinned);
    }
    Cell rb[1];
    if (!inst->invoke(m->startFunc, nullptr, rb)) return nullptr;
  }
  if (getenv("GECKO_INTERP_DEBUG")) {
    fprintf(stderr, "[interp] instantiate: done (start ok), building exports\n");
  }

  // Build exports.
  JS::RootedObject exports(cx, JS_NewPlainObject(cx));
  if (!exports) return nullptr;
  for (const Export& ex : m->exports) {
    JS::RootedValue val(cx);
    switch (ex.kind) {
      case ExternKind::Func: {
        JSObject* w = GetOrCreateFuncWrapper(cx, inst, ex.index);
        if (!w) return nullptr;
        val.setObject(*w);
        break;
      }
      case ExternKind::Memory:
        val.setObject(*inst->memObj);
        break;
      case ExternKind::Table:
        val.setObject(*inst->tableObjs[ex.index]);
        break;
      case ExternKind::Global: {
        JSObject* gw = MakeGlobalExport(cx, inst, ex.index);
        if (!gw) return nullptr;
        val.setObject(*gw);
        break;
      }
    }
    if (!JS_DefineProperty(cx, exports, ex.name.get(), val, JSPROP_ENUMERATE)) {
      return nullptr;
    }
  }
  JS::RootedValue exv(cx, JS::ObjectValue(*exports));
  if (!JS_DefineProperty(cx, instObj, "exports", exv, JSPROP_ENUMERATE)) {
    return nullptr;
  }
  return instObj;
}

// ---- cross-thread structured clone ----------------------------------------
// emscripten posts wasmModule + wasmMemory to each pthread worker via
// postMessage (structured clone). The outer engine is -pthread, so its C++ heap
// is shared across worker threads => a Module*/LinearMemory* is valid on every
// thread; we serialize the raw pointer. Shared memory's bytes live in a
// SharedArrayRawBuffer (refcounted) over which the receiving thread reconstructs
// its own SAB. Shared Module/Memory are pinned (leaked) to avoid cross-thread
// refcounting of the C++ structs.
static const uint32_t kInterpCloneTag = 0xffff80f0;  // within Gecko's SCTAG_DOM range
enum class InterpCloneKind : uint32_t { Module = 1, SharedMemory = 2 };

bool InterpCloneWrite(JSContext* cx, JSStructuredCloneWriter* w,
                      JS::HandleObject objIn, bool* handled) {
  *handled = false;
  // The object may be a cross-compartment wrapper (postMessage serializes from a
  // different compartment than the content that created the Memory/Module).
  JSObject* obj = js::CheckedUnwrapStatic(objIn);
  if (!obj) obj = objIn;
  if (getenv("GECKO_INTERP_DEBUG")) {
    fprintf(stderr, "[interp] clone-write obj class=%s mod=%d mem=%d\n",
            obj->getClass()->name, obj->hasClass(&kModuleClass),
            obj->hasClass(&kMemoryClass));
  }
  if (obj->hasClass(&kModuleClass)) {
    *handled = true;
    Module* m = ModuleFromObject(obj);
    m->pinned = true;  // leak: another thread will reference it
    uint64_t ptr = uint64_t(uintptr_t(m));
    return JS_WriteUint32Pair(w, kInterpCloneTag,
                              uint32_t(InterpCloneKind::Module)) &&
           JS_WriteBytes(w, &ptr, sizeof(ptr));
  }
  if (obj->hasClass(&kMemoryClass)) {
    *handled = true;
    LinearMemory* lm = MemoryFromObject(obj);
    if (!lm->shared || !lm->rawbuf) {
      JS_ReportErrorASCII(cx, "wasm interp: cannot clone non-shared memory");
      return false;
    }
    lm->pinned = true;
    // Leak one reference so the backing outlives every thread's wrapper.
    if (!static_cast<SharedArrayRawBuffer*>(lm->rawbuf)->addReference()) {
      JS_ReportErrorASCII(cx, "wasm interp: shared buffer refcount overflow");
      return false;
    }
    uint64_t ptr = uint64_t(uintptr_t(lm));
    return JS_WriteUint32Pair(w, kInterpCloneTag,
                              uint32_t(InterpCloneKind::SharedMemory)) &&
           JS_WriteBytes(w, &ptr, sizeof(ptr));
  }
  return true;  // not ours (handled stays false)
}

JSObject* InterpCloneRead(JSContext* cx, JSStructuredCloneReader* r, uint32_t tag,
                          uint32_t data) {
  if (tag != kInterpCloneTag) return nullptr;
  if (getenv("GECKO_INTERP_DEBUG")) {
    fprintf(stderr, "[interp] clone-read tag=%x kind=%u\n", tag, data);
  }
  uint64_t ptr = 0;
  if (!JS_ReadBytes(r, &ptr, sizeof(ptr))) return nullptr;
  if (data == uint32_t(InterpCloneKind::Module)) {
    Module* m = reinterpret_cast<Module*>(uintptr_t(ptr));
    JS::RootedObject obj(cx, InterpNewObject(cx, &kModuleClass, JSProto_WasmModule));
    if (!obj) return nullptr;
    JS::SetReservedSlot(obj, 0, JS::PrivateValue(m));
    return obj;
  }
  if (data == uint32_t(InterpCloneKind::SharedMemory)) {
    LinearMemory* lm = reinterpret_cast<LinearMemory*>(uintptr_t(ptr));
    auto* raw = static_cast<SharedArrayRawBuffer*>(lm->rawbuf);
    if (!raw->addReference()) {
      JS_ReportErrorASCII(cx, "wasm interp: shared buffer refcount overflow");
      return nullptr;
    }
    // New() adopts the reference (drops it on failure).
    JS::Rooted<FixedLengthSharedArrayBufferObject*> sab(
        cx, SharedArrayBufferObject::New(cx, raw, lm->size));
    if (!sab) return nullptr;
    JS::RootedObject ab(cx, sab);
    return WrapMemoryObject(cx, lm, ab);
  }
  return nullptr;
}

}  // namespace interp
}  // namespace wasm
}  // namespace js

#endif  // wasm_WasmInterpObj_inl_h
