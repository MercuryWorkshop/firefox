/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*-
 * vim: set ts=8 sts=2 et sw=2 tw=80:
 *
 * Scalar (no-SIMD) implementations of the v128 ops that clang 23 MISCOMPILES
 * when the interpreter emits them via the composite <wasm_simd128.h> intrinsics
 * (i16x8/i32x4/i64x2.extmul_low/high, i8x16/i16x8.avgr_u; the composite
 * extend*mul drops/zeros half the result lanes). Inline scalar replacements in
 * WasmInterpRun-inl.h did NOT help: clang re-idiom-recognizes them back into the
 * buggy SIMD instruction even with volatile/optnone/noinline. This TU is compiled
 * WITHOUT -msimd128 (see js/src/wasm/moz.build), so clang cannot emit ANY SIMD
 * instruction here -- the helpers are guaranteed genuinely scalar and correct.
 * Operands are raw 16-byte v128 buffers.
 */

#ifndef wasm_WasmInterpSimdScalar_h
#define wasm_WasmInterpSimdScalar_h

#include <cstdint>

namespace js {
namespace wasm {
namespace interp {

// extmul: out = extend(a) * extend(b), one (2*srcElemBytes)-wide product per
// source lane. srcElemBytes = 1/2/4 (i8/i16/i32 source). hi != 0 selects the
// high half of source lanes. out is 16 bytes.
void SimdScalarExtmul(const void* a, const void* b, void* out, int srcElemBytes,
                      bool srcSigned, int hi);
// avgr_u: out[i] = (a[i] + b[i] + 1) >> 1, unsigned, per elemBytes (1 or 2) lane.
void SimdScalarAvgr(const void* a, const void* b, void* out, int elemBytes);
// i8x16.shuffle: out[i] = (a||b)[idx[i] & 31].
void SimdScalarShuffle(const void* a, const void* b, const void* idx, void* out);

}  // namespace interp
}  // namespace wasm
}  // namespace js

#endif  // wasm_WasmInterpSimdScalar_h
