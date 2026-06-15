/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// wasm32-emscripten xptcall, JS->C++ direction: NS_InvokeByIndex.
//
// wasm call_indirect needs an exact function type at the call site, which we
// don't have at compile time for an arbitrary (object, methodIndex). We sidestep
// that by going through JS: the wasm function table is reachable from JS as
// `wasmTable`, and `wasmTable.get(idx)` yields a callable whose true type is
// applied dynamically. So we marshal the nsXPTCVariant array into a flat buffer
// + a signature string, then let JS read the args back and apply the function.

#include "xptcprivate.h"

#include <cstdlib>
#include <cstdint>
#include <emscripten.h>

// Read args from argbuf per the signature and call wasmTable.get(fp)(that, ...).
// Returns the method's nsresult.
EM_JS(int, WasmInvoke,
      (int fp, int thisPtr, uint8_t* argbuf, const char* sigPtr, int nparams), {
  var sig = UTF8ToString(sigPtr);
  var f = wasmTable.get(fp);
  var args = new Array(nparams + 1);
  args[0] = thisPtr;
  for (var i = 0; i < nparams; i++) {
    var c = sig.charAt(2 + i);  // sig[0]=ret, sig[1]=this, sig[2..]=params
    var off = argbuf + i * 8;
    if (c == 'd') {
      args[i + 1] = HEAPF64[off >> 3];
    } else if (c == 'f') {
      args[i + 1] = HEAPF32[off >> 2];
    } else if (c == 'j') {
      args[i + 1] = HEAP64[off >> 3];
    } else {
      args[i + 1] = HEAP32[off >> 2];
    }
  }
  return f.apply(null, args);
});

extern "C" nsresult NS_InvokeByIndex(nsISupports* that, uint32_t methodIndex,
                                     uint32_t paramCount,
                                     nsXPTCVariant* params) {
  void** vtable = *reinterpret_cast<void***>(that);
  void* fp = vtable[methodIndex];

  char sig[PARAM_BUFFER_COUNT + 8];
  size_t sn = 0;
  sig[sn++] = 'i';  // nsresult return (unused by WasmInvoke, kept for symmetry)
  sig[sn++] = 'i';  // this
  size_t bufCount = paramCount ? paramCount : 1;
  uint8_t* argbuf = static_cast<uint8_t*>(malloc(bufCount * 8));

  for (uint32_t i = 0; i < paramCount; i++) {
    nsXPTCVariant& s = params[i];
    uint8_t* a = argbuf + size_t(i) * 8;
    if (s.IsIndirect()) {
      sig[sn++] = 'i';
      *reinterpret_cast<void**>(a) = &s.val;
      continue;
    }
    switch (s.type) {
      case nsXPTType::T_DOUBLE: sig[sn++] = 'd'; *reinterpret_cast<double*>(a) = s.val.d; break;
      case nsXPTType::T_FLOAT:  sig[sn++] = 'f'; *reinterpret_cast<float*>(a) = s.val.f; break;
      case nsXPTType::T_I64:    sig[sn++] = 'j'; *reinterpret_cast<int64_t*>(a) = s.val.i64; break;
      case nsXPTType::T_U64:    sig[sn++] = 'j'; *reinterpret_cast<uint64_t*>(a) = s.val.u64; break;
      case nsXPTType::T_I8:     sig[sn++] = 'i'; *reinterpret_cast<int32_t*>(a) = s.val.i8; break;
      case nsXPTType::T_I16:    sig[sn++] = 'i'; *reinterpret_cast<int32_t*>(a) = s.val.i16; break;
      case nsXPTType::T_I32:    sig[sn++] = 'i'; *reinterpret_cast<int32_t*>(a) = s.val.i32; break;
      case nsXPTType::T_U8:     sig[sn++] = 'i'; *reinterpret_cast<uint32_t*>(a) = s.val.u8; break;
      case nsXPTType::T_U16:    sig[sn++] = 'i'; *reinterpret_cast<uint32_t*>(a) = s.val.u16; break;
      case nsXPTType::T_U32:    sig[sn++] = 'i'; *reinterpret_cast<uint32_t*>(a) = s.val.u32; break;
      case nsXPTType::T_BOOL:   sig[sn++] = 'i'; *reinterpret_cast<uint32_t*>(a) = s.val.b ? 1 : 0; break;
      case nsXPTType::T_CHAR:   sig[sn++] = 'i'; *reinterpret_cast<int32_t*>(a) = s.val.c; break;
      case nsXPTType::T_WCHAR:  sig[sn++] = 'i'; *reinterpret_cast<uint32_t*>(a) = s.val.wc; break;
      default:                  sig[sn++] = 'i'; *reinterpret_cast<void**>(a) = s.val.p; break;
    }
  }
  sig[sn] = '\0';

  int ret = WasmInvoke(int(intptr_t(fp)), int(intptr_t(that)), argbuf, sig,
                       int(paramCount));
  free(argbuf);
  return nsresult(ret);
}
