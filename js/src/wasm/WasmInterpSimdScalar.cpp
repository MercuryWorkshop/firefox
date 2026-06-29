/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 *
 * Compiled WITHOUT -msimd128 (js/src/wasm/moz.build) so clang cannot emit a SIMD
 * instruction for these scalar loops -- working around a clang 23 miscompile of
 * the composite extmul/avgr intrinsics. See WasmInterpSimdScalar.h.
 */

#include "wasm/WasmInterpSimdScalar.h"

#include <cstring>

namespace js {
namespace wasm {
namespace interp {

void SimdScalarExtmul(const void* ap, const void* bp, void* op,
                      int srcElemBytes, bool srcSigned, int hi) {
  int ndst = 16 / (2 * srcElemBytes);  // 8, 4 or 2
  int base = hi ? ndst : 0;
  if (srcElemBytes == 1) {
    if (srcSigned) {
      const int8_t* a = (const int8_t*)ap;
      const int8_t* b = (const int8_t*)bp;
      int16_t* o = (int16_t*)op;
      for (int i = 0; i < ndst; i++) {
        o[i] = (int16_t)((int)a[base + i] * (int)b[base + i]);
      }
    } else {
      const uint8_t* a = (const uint8_t*)ap;
      const uint8_t* b = (const uint8_t*)bp;
      uint16_t* o = (uint16_t*)op;
      for (int i = 0; i < ndst; i++) {
        o[i] = (uint16_t)((unsigned)a[base + i] * (unsigned)b[base + i]);
      }
    }
  } else if (srcElemBytes == 2) {
    if (srcSigned) {
      const int16_t* a = (const int16_t*)ap;
      const int16_t* b = (const int16_t*)bp;
      int32_t* o = (int32_t*)op;
      for (int i = 0; i < ndst; i++) {
        o[i] = (int32_t)((int)a[base + i] * (int)b[base + i]);
      }
    } else {
      const uint16_t* a = (const uint16_t*)ap;
      const uint16_t* b = (const uint16_t*)bp;
      uint32_t* o = (uint32_t*)op;
      for (int i = 0; i < ndst; i++) {
        o[i] = (uint32_t)((unsigned)a[base + i] * (unsigned)b[base + i]);
      }
    }
  } else {  // srcElemBytes == 4
    if (srcSigned) {
      const int32_t* a = (const int32_t*)ap;
      const int32_t* b = (const int32_t*)bp;
      int64_t* o = (int64_t*)op;
      for (int i = 0; i < ndst; i++) {
        o[i] = (int64_t)a[base + i] * (int64_t)b[base + i];
      }
    } else {
      const uint32_t* a = (const uint32_t*)ap;
      const uint32_t* b = (const uint32_t*)bp;
      uint64_t* o = (uint64_t*)op;
      for (int i = 0; i < ndst; i++) {
        o[i] = (uint64_t)a[base + i] * (uint64_t)b[base + i];
      }
    }
  }
}

void SimdScalarAvgr(const void* ap, const void* bp, void* op, int elemBytes) {
  if (elemBytes == 1) {
    const uint8_t* a = (const uint8_t*)ap;
    const uint8_t* b = (const uint8_t*)bp;
    uint8_t* o = (uint8_t*)op;
    for (int i = 0; i < 16; i++) {
      o[i] = (uint8_t)(((unsigned)a[i] + (unsigned)b[i] + 1u) >> 1);
    }
  } else {  // elemBytes == 2
    const uint16_t* a = (const uint16_t*)ap;
    const uint16_t* b = (const uint16_t*)bp;
    uint16_t* o = (uint16_t*)op;
    for (int i = 0; i < 8; i++) {
      o[i] = (uint16_t)(((unsigned)a[i] + (unsigned)b[i] + 1u) >> 1);
    }
  }
}

void SimdScalarShuffle(const void* ap, const void* bp, const void* idxp,
                       void* op) {
  uint8_t both[32];
  memcpy(both, ap, 16);
  memcpy(both + 16, bp, 16);
  const uint8_t* idx = (const uint8_t*)idxp;
  uint8_t* o = (uint8_t*)op;
  for (int i = 0; i < 16; i++) o[i] = both[idx[i] & 31];
}

}  // namespace interp
}  // namespace wasm
}  // namespace js
