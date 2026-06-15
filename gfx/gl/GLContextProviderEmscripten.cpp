/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// GLContextProvider for the wasm32-emscripten (headless) build. Gecko has no
// platform GL provider here (gl_provider="Null", EGL can't dlopen libEGL.so),
// so WebRender's RenderCompositorOGL had no GL context and the engine fell back
// to software RenderDocument. This provider creates a WebGL2 (GLES3) context via
// emscripten's html5_webgl API, backed by the page <canvas>, so WebRender can
// composite on the GPU and present directly to the canvas.
//
// Threading: WebRender creates this on its "Renderer" thread, which does NOT own
// the page canvas's OffscreenCanvas. We therefore use a context proxied to the
// main browser thread (proxyContextToMainThread=ALWAYS), which works from any
// pthread. GL commands run on the main thread; presentation is implicit when
// that thread yields. (A faster path transferring the OffscreenCanvas to the
// Renderer thread is a later optimization.) See gecko-wasm-webgl-canvas notes.

#include "GLContextProvider.h"
#include "GLContext.h"
#include "GLLibraryLoader.h"
#include "mozilla/widget/CompositorWidget.h"
#include "nsString.h"

#include <emscripten.h>
#include <emscripten/html5.h>
#include <emscripten/html5_webgl.h>
#include <cstdio>
#include <cstdlib>
#include <pthread.h>

// The page canvas the embedder renders into (embed-xul/index.html: <canvas id="screen">).
#define MOZ_EMSCRIPTEN_GL_CANVAS "#screen"

namespace mozilla {
namespace gl {

// Canvas passthrough (flag GECKO_GL_PASSTHROUGH): when content requests a
// WebGL/WebGL2 canvas, Gecko's WebGLContext asks GLContextProvider::CreateHeadless
// for an offscreen GL context. We give it a REAL host WebGL2 context on its own
// hidden host <canvas> (the compositor owns "#screen", so content can't share it).
// Content GL commands pass through emscripten to that host context; the result is
// read back to the compositor via SurfaceFactory_Basic (glReadPixels) and uploaded
// by WebRender. Off by default (content WebGL then simply fails to create, as
// before). See gecko-wasm canvas-passthrough notes.
extern "C" __attribute__((weak)) int gecko_gl_passthrough_enabled();
static bool ContentPassthroughEnabled() {
  // The flag is captured on the app thread (embedder main(), where getenv sees the
  // env) and exposed via this weak symbol; getenv() here on the content WebGL
  // worker thread is unreliable (ENV not propagated to all workers).
  return gecko_gl_passthrough_enabled && gecko_gl_passthrough_enabled() != 0;
}

class GLContextEmscripten final : public GLContext {
 public:
  MOZ_DECLARE_REFCOUNTED_VIRTUAL_TYPENAME(GLContextEmscripten, override)

  // Shared creation. `target` is an emscripten canvas selector. `present` =
  // compositor context (explicit-swap offscreen back buffer, blitted to the
  // visible canvas on commit_frame); !present = content/offscreen context (no
  // swap -- WebGLContext drives its own framebuffer and we read it back).
  static RefPtr<GLContextEmscripten> CreateImpl(const GLContextDesc& desc,
                                                const char* target, bool present,
                                                nsACString* const out_failureId) {
    EmscriptenWebGLContextAttributes attrs;
    emscripten_webgl_init_context_attributes(&attrs);
    attrs.majorVersion = 2;  // WebGL2 / GLES3 (WebRender requires GLES3)
    attrs.minorVersion = 0;
    attrs.alpha = EM_TRUE;
    attrs.depth = EM_TRUE;
    attrs.stencil = EM_TRUE;
    attrs.antialias = EM_FALSE;
    attrs.premultipliedAlpha = EM_TRUE;
    attrs.preserveDrawingBuffer = EM_TRUE;
    attrs.failIfMajorPerformanceCaveat = EM_FALSE;
    attrs.enableExtensionsByDefault = EM_TRUE;
    // Compositor: present explicitly; with a proxied context (no OffscreenCanvas)
    // emscripten emulates the swap by blitting an offscreen back buffer to the
    // default framebuffer on emscripten_webgl_commit_frame(). This back buffer is
    // REQUIRED for presentation -- implicit present does NOT work for a proxied
    // context, so without it the canvas stays black. Content contexts render to
    // their own framebuffer (read back for compositing), so they don't swap.
    attrs.explicitSwapControl = present ? EM_TRUE : EM_FALSE;
    attrs.renderViaOffscreenBackBuffer = present ? EM_TRUE : EM_FALSE;
    attrs.proxyContextToMainThread = EMSCRIPTEN_WEBGL_CONTEXT_PROXY_ALWAYS;

    EMSCRIPTEN_WEBGL_CONTEXT_HANDLE ctx =
        emscripten_webgl_create_context(target, &attrs);
    if (ctx <= 0) {
      if (out_failureId) {
        *out_failureId = "FEATURE_FAILURE_EMSCRIPTEN_WEBGL_CREATE"_ns;
      }
      NS_WARNING("GLContextEmscripten: emscripten_webgl_create_context failed");
      return nullptr;
    }

    RefPtr<GLContextEmscripten> gl = new GLContextEmscripten(desc, ctx, present);
    if (!gl->Init()) {
      if (out_failureId) {
        *out_failureId = "FEATURE_FAILURE_EMSCRIPTEN_WEBGL_INIT"_ns;
      }
      return nullptr;
    }
    return gl;
  }

  // Compositor context on the page's visible canvas ("#screen").
  static RefPtr<GLContextEmscripten> Create(const GLContextDesc& desc,
                                            nsACString* const out_failureId) {
    return CreateImpl(desc, MOZ_EMSCRIPTEN_GL_CANVAS, /* present */ true,
                      out_failureId);
  }

  // Content WebGL context (canvas passthrough). Created WORKER-LOCAL on the
  // calling (content) thread via a new OffscreenCanvas, NOT proxied to thread 0.
  // emscripten's GLctx/current-context is per-thread JS state; the compositor's
  // context is proxied to thread 0, so a content context registered on the content
  // thread uses THIS thread's own GLctx and can never clobber the compositor's
  // (which is what broke GPU-mode passthrough). It also runs without the per-call
  // proxy tax. The result is read back (CPU, SurfaceFactory_Basic) for compositing.
  static RefPtr<GLContextEmscripten> CreateContent(
      const GLContextDesc& desc, nsACString* const out_failureId) {
    // NOTE: no top-level commas inside EM_ASM_INT -- the C preprocessor splits
    // macro args on commas (only parens protect them, not braces), so the attrs
    // object is built with assignments rather than a comma-separated literal.
    EMSCRIPTEN_WEBGL_CONTEXT_HANDLE ctx =
        (EMSCRIPTEN_WEBGL_CONTEXT_HANDLE)EM_ASM_INT({
          try {
            var oc = new OffscreenCanvas(300, 150);
            var attrs = {};
            attrs.majorVersion = 2;
            attrs.minorVersion = 0;
            attrs.alpha = true;
            attrs.depth = true;
            attrs.stencil = true;
            attrs.antialias = false;
            attrs.premultipliedAlpha = true;
            attrs.preserveDrawingBuffer = true;
            attrs.failIfMajorPerformanceCaveat = false;
            attrs.enableExtensionsByDefault = true;
            var glctx = oc.getContext("webgl2", attrs);
            if (!glctx) return 0;
            // Register with emscripten's GL on THIS thread (owns it -> local, not
            // proxied). GL.registerContext stamps the current pthread as owner.
            return GL.registerContext(glctx, attrs);
          } catch (e) { return 0; }
        });
    if (ctx <= 0) {
      if (out_failureId) {
        *out_failureId = "FEATURE_FAILURE_EMSCRIPTEN_OFFSCREENCANVAS"_ns;
      }
      NS_WARNING("GLContextEmscripten: content OffscreenCanvas context failed");
      return nullptr;
    }
    RefPtr<GLContextEmscripten> gl =
        new GLContextEmscripten(desc, ctx, /* present */ false);
    if (!gl->Init()) {
      if (out_failureId) {
        *out_failureId = "FEATURE_FAILURE_EMSCRIPTEN_WEBGL_INIT"_ns;
      }
      return nullptr;
    }
    return gl;
  }

  GLContextType GetContextType() const override {
    return GLContextType::Emscripten;
  }

  bool MakeCurrentImpl() const override {
    return emscripten_webgl_make_context_current(mContext) ==
           EMSCRIPTEN_RESULT_SUCCESS;
  }

  bool IsCurrentImpl() const override {
    return emscripten_webgl_get_current_context() == mContext;
  }

  Maybe<SymbolLoader> GetSymbolLoader() const override {
    // emscripten resolves GL entry points by name (requires
    // -sGL_ENABLE_GET_PROC_ADDRESS). Matches SymbolLoader's void*(const char*)
    // ctor; the base GLContext::InitImpl loads the whole symbol table through it.
    return Some(SymbolLoader(emscripten_webgl_get_proc_address));
  }

  bool IsDoubleBuffered() const override { return true; }

  bool SwapBuffers() override {
    // Compositor (offscreen back buffer): blit it to the default framebuffer;
    // presentation happens when the owning thread yields. Content contexts have no
    // explicit-swap back buffer (commit_frame would error), and present via
    // read-back instead, so swapping is a no-op for them.
    if (!mPresent) return true;
    return emscripten_webgl_commit_frame() == EMSCRIPTEN_RESULT_SUCCESS;
  }

  void GetWSIInfo(nsCString* const out) const override {
    out->AppendLiteral("emscripten WebGL2 (proxied to main thread)");
  }

 private:
  GLContextEmscripten(const GLContextDesc& desc,
                      EMSCRIPTEN_WEBGL_CONTEXT_HANDLE ctx, bool present)
      : GLContext(desc, nullptr, false), mContext(ctx), mPresent(present) {}

  ~GLContextEmscripten() {
    MarkDestroyed();
    if (mContext > 0) {
      emscripten_webgl_destroy_context(mContext);
    }
  }

  EMSCRIPTEN_WEBGL_CONTEXT_HANDLE mContext;
  bool mPresent;
};

// -- Provider statics --------------------------------------------------------

already_AddRefed<GLContext> GLContextProviderEmscripten::CreateForCompositorWidget(
    widget::CompositorWidget* aCompositorWidget, bool /*aHardwareWebRender*/,
    bool /*aForceAccelerated*/) {
  GLContextDesc desc;
  desc.isOffscreen = false;
  nsCString failureId;
  RefPtr<GLContext> gl = GLContextEmscripten::Create(desc, &failureId);
  return gl.forget();
}

already_AddRefed<GLContext> GLContextProviderEmscripten::CreateHeadless(
    const GLContextCreateDesc& createDesc, nsACString* const out_failureId) {
  // Offscreen contexts here are content WebGL (canvas.getContext). Only honor them
  // when the canvas passthrough flag is set; otherwise return null (content WebGL
  // is unavailable) rather than fighting the compositor for "#screen".
  if (!ContentPassthroughEnabled()) {
    if (out_failureId) {
      *out_failureId = "FEATURE_FAILURE_EMSCRIPTEN_NO_PASSTHROUGH"_ns;
    }
    return nullptr;
  }
  GLContextDesc desc = {};
  static_cast<GLContextCreateDesc&>(desc) = createDesc;
  desc.isOffscreen = true;
  RefPtr<GLContext> gl = GLContextEmscripten::CreateContent(desc, out_failureId);
  return gl.forget();
}

GLContext* GLContextProviderEmscripten::GetGlobalContext() { return nullptr; }

void GLContextProviderEmscripten::Shutdown() {}

}  // namespace gl
}  // namespace mozilla
