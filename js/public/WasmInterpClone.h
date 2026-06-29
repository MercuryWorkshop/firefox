/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// Cross-thread structured-clone hooks for the in-process WebAssembly
// interpreter (GECKO_WASM_INTERP, emscripten only). The interpreter's
// Module/Memory wrappers are custom JSClasses the structured-clone core does not
// understand; Gecko's StructuredCloneHolder routes unknown objects here. The
// outer engine is -pthread (shared C++ heap) so the underlying C++ structs are
// valid on every worker thread and we serialize the raw pointer; shared memory's
// bytes live in a refcounted SharedArrayRawBuffer reconstructed on the receiving
// thread. Defined in js/src/wasm/WasmInterpObj-inl.h (compiled into WasmJS.cpp).
// Exposed as a public header only so dom/ (StructuredCloneHolder) can call it.

#ifndef js_WasmInterpClone_h
#define js_WasmInterpClone_h

#include <cstdint>

#include "js/TypeDecls.h"  // JSContext, JSObject, JS::Handle

struct JSStructuredCloneWriter;
struct JSStructuredCloneReader;

namespace js {
namespace wasm {
namespace interp {

// Serialize obj if it is an interp Module/Memory. Sets *handled accordingly;
// returns false only on error (with an exception pending).
bool InterpCloneWrite(JSContext* cx, JSStructuredCloneWriter* w,
                      JS::Handle<JSObject*> obj, bool* handled);

// Reconstruct an interp Module/Memory from a clone tag, or null if the tag is
// not ours (then the caller continues its normal dispatch).
JSObject* InterpCloneRead(JSContext* cx, JSStructuredCloneReader* r,
                          uint32_t tag, uint32_t data);

}  // namespace interp
}  // namespace wasm
}  // namespace js

#endif  // js_WasmInterpClone_h
