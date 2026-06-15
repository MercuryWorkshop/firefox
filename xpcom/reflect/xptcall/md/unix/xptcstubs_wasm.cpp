/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// wasm32-emscripten xptcall, C++->JS direction (the nsXPTCStubBase replacement).
//
// wasm call_indirect enforces an exact function type at every call site, so the
// native "one Stub# entry, marshal registers in asm" trick can't be ported. We
// instead synthesize, per interface, a vtable whose slots are real wasm
// functions of the right type:
//   slot 0/1/2 -> QueryInterface/AddRef/Release (fixed-signature C++ funcs)
//   slot N>=3  -> a JS shim built with emscripten addFunction(fn, sig), where
//                 sig is computed from the XPT method info to exactly match the
//                 C++ ABI the caller's call_indirect expects. The shim packs its
//                 arguments into a flat buffer and forwards to WasmXPTCStubDispatch,
//                 which rebuilds an nsXPTCMiniVariant[] and calls mOuter->CallMethod.
// Vtables are cached per nsXPTInterfaceInfo (the shims depend only on
// methodIndex+signature, not on the individual proxy), so addFunction is called
// at most once per (interface, method).

#include "xptcprivate.h"

#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <emscripten.h>

#include "mozilla/StaticMutex.h"

// The synthetic proxy object. The vtable pointer MUST be first so that an
// interface pointer to this object dispatches through our table.
struct WasmStub {
  void** vtable;
  nsIXPTCProxy* mOuter;
  const nsXPTInterfaceInfo* mEntry;
};

extern "C" {

// nsISupports slots, implemented directly (fixed signatures, no marshaling).
nsresult WasmStub_QueryInterface(WasmStub* self, const nsID& aIID,
                                 void** aInstancePtr) {
  if (aIID.Equals(self->mEntry->IID())) {
    self->mOuter->AddRef();
    *aInstancePtr = static_cast<void*>(self);
    return NS_OK;
  }
  return self->mOuter->QueryInterface(aIID, aInstancePtr);
}

MozExternalRefCountType WasmStub_AddRef(WasmStub* self) {
  return self->mOuter->AddRef();
}

MozExternalRefCountType WasmStub_Release(WasmStub* self) {
  return self->mOuter->Release();
}

// Called by every method shim. argbuf holds one 8-byte slot per incoming C++
// argument (excluding 'this'): integers/pointers in the low bytes, i64 as 8
// bytes, float as 4 bytes, double as 8 bytes -- matching how the JS shim wrote
// them per the signature. We walk the XPT params (skipping the implicit
// JSContext slot, ignoring a trailing optional-argc slot) into a mini-variant
// array, then dispatch.
EMSCRIPTEN_KEEPALIVE
nsresult WasmXPTCStubDispatch(WasmStub* self, uint32_t methodIndex,
                              uint8_t* argbuf, uint32_t /*nargs*/) {
  const nsXPTMethodInfo* info = nullptr;
  self->mEntry->GetMethodInfo(uint16_t(methodIndex), &info);
  if (!info) {
    return NS_ERROR_UNEXPECTED;
  }

  nsXPTCMiniVariant paramBuffer[PARAM_BUFFER_COUNT];
  uint32_t paramCount = info->ParamCount();
  const uint8_t indexOfJSContext = info->IndexOfJSContext();

  uint32_t slot = 0;  // index into argbuf's 8-byte slots
  for (uint32_t i = 0; i < paramCount; i++) {
    if (i == indexOfJSContext) {
      slot++;  // skip the implicit JSContext* argument
    }
    const nsXPTParamInfo& param = info->Param(i);
    const nsXPTType& type = param.GetType();
    nsXPTCMiniVariant* dp = &paramBuffer[i];
    uint8_t* a = argbuf + size_t(slot) * 8;
    slot++;

    if (param.IsOut() || !type.IsArithmetic()) {
      dp->val.p = *reinterpret_cast<void**>(a);
      continue;
    }
    switch (type) {
      case nsXPTType::T_I8:    dp->val.i8  = int8_t(*reinterpret_cast<int32_t*>(a)); break;
      case nsXPTType::T_I16:   dp->val.i16 = int16_t(*reinterpret_cast<int32_t*>(a)); break;
      case nsXPTType::T_I32:   dp->val.i32 = *reinterpret_cast<int32_t*>(a); break;
      case nsXPTType::T_I64:   dp->val.i64 = *reinterpret_cast<int64_t*>(a); break;
      case nsXPTType::T_U8:    dp->val.u8  = uint8_t(*reinterpret_cast<uint32_t*>(a)); break;
      case nsXPTType::T_U16:   dp->val.u16 = uint16_t(*reinterpret_cast<uint32_t*>(a)); break;
      case nsXPTType::T_U32:   dp->val.u32 = *reinterpret_cast<uint32_t*>(a); break;
      case nsXPTType::T_U64:   dp->val.u64 = *reinterpret_cast<uint64_t*>(a); break;
      case nsXPTType::T_FLOAT: dp->val.f   = *reinterpret_cast<float*>(a); break;
      case nsXPTType::T_DOUBLE:dp->val.d   = *reinterpret_cast<double*>(a); break;
      case nsXPTType::T_BOOL:  dp->val.b   = bool(uint8_t(*reinterpret_cast<uint32_t*>(a))); break;
      case nsXPTType::T_CHAR:  dp->val.c   = char(*reinterpret_cast<int32_t*>(a)); break;
      case nsXPTType::T_WCHAR: dp->val.wc  = char16_t(*reinterpret_cast<uint32_t*>(a)); break;
      default:
        dp->val.p = *reinterpret_cast<void**>(a);
        break;
    }
  }

  return self->mOuter->CallMethod(uint16_t(methodIndex), info, paramBuffer);
}

}  // extern "C"

// Compute the emscripten addFunction signature string for a method: return type
// first ('i' = nsresult), then 'this', then each argument in C++ order.
static char WasmTypeChar(const nsXPTParamInfo& aParam) {
  if (aParam.IsOut()) {
    return 'i';  // out params are pointers
  }
  const nsXPTType& type = aParam.GetType();
  if (!type.IsArithmetic()) {
    return 'i';  // complex types passed by pointer
  }
  switch (type) {
    case nsXPTType::T_I64:
    case nsXPTType::T_U64:
      return 'j';
    case nsXPTType::T_FLOAT:
      return 'f';
    case nsXPTType::T_DOUBLE:
      return 'd';
    default:
      return 'i';
  }
}

static void WasmComputeSig(const nsXPTMethodInfo* aInfo, char* aOut,
                           size_t aMax) {
  size_t n = 0;
  auto put = [&](char c) {
    if (n + 1 < aMax) aOut[n++] = c;
  };
  put('i');  // nsresult return
  put('i');  // this
  uint32_t paramCount = aInfo->ParamCount();
  uint8_t indexOfJSContext = aInfo->IndexOfJSContext();
  for (uint32_t i = 0; i < paramCount; i++) {
    if (aInfo->WantsContext() && i == indexOfJSContext) {
      put('i');  // implicit JSContext*
    }
    put(WasmTypeChar(aInfo->Param(i)));
  }
  if (aInfo->WantsContext() && indexOfJSContext == paramCount) {
    put('i');  // JSContext* after all params (no retval case)
  }
  if (aInfo->WantsOptArgc()) {
    put('i');  // trailing uint8_t argc
  }
  aOut[n] = '\0';
}

// Build (in JS) a method shim of the given signature that forwards to
// WasmXPTCStubDispatch, and return its function-table index.
EM_JS(int, WasmAddStub, (int methodIndex, const char* sigPtr), {
  var sig = UTF8ToString(sigPtr);
  var fn = function() {
    var nargs = arguments.length - 1;  // exclude 'this'
    var thisPtr = arguments[0];
    var buf = _malloc(nargs * 8);
    for (var i = 0; i < nargs; i++) {
      var c = sig.charAt(2 + i);  // sig[0]=ret, sig[1]=this, sig[2..]=args
      var v = arguments[1 + i];
      var off = buf + i * 8;
      if (c == 'd') {
        HEAPF64[off >> 3] = v;
      } else if (c == 'f') {
        HEAPF32[off >> 2] = v;
      } else if (c == 'j') {
        HEAP64[off >> 3] = BigInt(v);
      } else {
        HEAP32[off >> 2] = v;
        HEAP32[(off >> 2) + 1] = 0;
      }
    }
    var ret = _WasmXPTCStubDispatch(thisPtr, methodIndex, buf, nargs);
    _free(buf);
    return ret;
  };
  return addFunction(fn, sig);
});

static mozilla::StaticMutex sVtableMutex MOZ_UNANNOTATED;

static std::unordered_map<const nsXPTInterfaceInfo*, void**>* sVtableCache =
    nullptr;

static void** WasmGetOrBuildVtable(const nsXPTInterfaceInfo* aEntry) {
  mozilla::StaticMutexAutoLock lock(sVtableMutex);
  if (!sVtableCache) {
    sVtableCache = new std::unordered_map<const nsXPTInterfaceInfo*, void**>();
  }
  auto it = sVtableCache->find(aEntry);
  if (it != sVtableCache->end()) {
    return it->second;
  }

  uint32_t methodCount = aEntry->MethodCount();
  void** vtable = static_cast<void**>(malloc(sizeof(void*) * methodCount));
  vtable[0] = reinterpret_cast<void*>(&WasmStub_QueryInterface);
  vtable[1] = reinterpret_cast<void*>(&WasmStub_AddRef);
  vtable[2] = reinterpret_cast<void*>(&WasmStub_Release);
  for (uint32_t mi = 3; mi < methodCount; mi++) {
    const nsXPTMethodInfo* info = nullptr;
    aEntry->GetMethodInfo(uint16_t(mi), &info);
    char sig[PARAM_BUFFER_COUNT + 8];
    if (info) {
      WasmComputeSig(info, sig, sizeof(sig));
    } else {
      strcpy(sig, "ii");
    }
    int idx = WasmAddStub(int(mi), sig);
    vtable[mi] = reinterpret_cast<void*>(intptr_t(idx));
  }

  (*sVtableCache)[aEntry] = vtable;
  return vtable;
}

EXPORT_XPCOM_API(nsresult)
NS_GetXPTCallStub(REFNSIID aIID, nsIXPTCProxy* aOuter,
                  nsISomeInterface** aResult) {
  if (NS_WARN_IF(!aOuter) || NS_WARN_IF(!aResult)) {
    return NS_ERROR_INVALID_ARG;
  }
  const nsXPTInterfaceInfo* iie = nsXPTInterfaceInfo::ByIID(aIID);
  if (!iie || iie->IsBuiltinClass()) {
    return NS_ERROR_FAILURE;
  }
  void** vtable = WasmGetOrBuildVtable(iie);
  WasmStub* stub = static_cast<WasmStub*>(malloc(sizeof(WasmStub)));
  stub->vtable = vtable;
  stub->mOuter = aOuter;
  stub->mEntry = iie;
  *aResult = reinterpret_cast<nsISomeInterface*>(stub);
  return NS_OK;
}

EXPORT_XPCOM_API(void)
NS_DestroyXPTCallStub(nsISomeInterface* aStub) {
  // The vtable is cached/shared per interface; only the per-proxy object is
  // freed here.
  free(static_cast<void*>(aStub));
}

EXPORT_XPCOM_API(size_t)
NS_SizeOfIncludingThisXPTCallStub(const nsISomeInterface* aStub,
                                  mozilla::MallocSizeOf aMallocSizeOf) {
  return aMallocSizeOf(aStub);
}

// nsXPTCStubBase is still a complete type elsewhere; its Stub#/Sentinel# bodies
// are never reached on wasm (we don't instantiate it) but must be defined so the
// class links. Keep them as traps.
#define STUB_ENTRY(n) \
  nsresult nsXPTCStubBase::Stub##n() { abort(); }

#define SENTINEL_ENTRY(n)                        \
  nsresult nsXPTCStubBase::Sentinel##n() {       \
    NS_ERROR("nsXPTCStubBase::Sentinel called"); \
    return NS_ERROR_NOT_IMPLEMENTED;             \
  }

#include "xptcstubsdef.inc"

#undef STUB_ENTRY
#undef SENTINEL_ENTRY
