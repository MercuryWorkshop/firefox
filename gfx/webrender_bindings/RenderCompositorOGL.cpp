/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "RenderCompositorOGL.h"

#include "GLContext.h"
#include "GLContextEGL.h"
#include "GLContextProvider.h"
#include "mozilla/gfx/gfxVars.h"
#include "mozilla/gfx/Logging.h"
#include "mozilla/webrender/RenderThread.h"
#include "mozilla/widget/CompositorWidget.h"

#if defined(__EMSCRIPTEN__)
// NB: do NOT #include <emscripten/html5.h> here -- it defines DOM_KEY_LOCATION_*
// macros that collide with mozilla::dom::KeyboardEventBinding's enum of the same
// names (pulled in transitively). Declare just the two canvas-size entry points.
extern "C" {
int emscripten_get_canvas_element_size(const char* target, int* width,
                                       int* height);
int emscripten_set_canvas_element_size(const char* target, int width,
                                       int height);
}
#endif

namespace mozilla::wr {

extern LazyLogModule gRenderThreadLog;
#define LOG(...) MOZ_LOG(gRenderThreadLog, LogLevel::Debug, (__VA_ARGS__))

/* static */
UniquePtr<RenderCompositor> RenderCompositorOGL::Create(
    const RefPtr<widget::CompositorWidget>& aWidget, nsACString& aError) {
  RefPtr<gl::GLContext> gl = RenderThread::Get()->SingletonGL();
  if (!gl) {
    gl = gl::GLContextProvider::CreateForCompositorWidget(
        aWidget, /* aHardwareWebRender */ true, /* aForceAccelerated */ true);
    RenderThread::MaybeEnableGLDebugMessage(gl);
  }
  if (!gl || !gl->MakeCurrent()) {
    gfxCriticalNote << "Failed GL context creation for WebRender: "
                    << gfx::hexa(gl.get());
    return nullptr;
  }
  return MakeUnique<RenderCompositorOGL>(std::move(gl), aWidget);
}

RenderCompositorOGL::RenderCompositorOGL(
    RefPtr<gl::GLContext>&& aGL,
    const RefPtr<widget::CompositorWidget>& aWidget)
    : RenderCompositor(aWidget), mGL(aGL) {
  MOZ_ASSERT(mGL);
  LOG("RenderCompositorOGL::RenderCompositorOGL()");

  mIsEGL = aGL->GetContextType() == mozilla::gl::GLContextType::EGL;
}

RenderCompositorOGL::~RenderCompositorOGL() {
  LOG("RenderCompositorOGL::~RenderCompositorOGL()");

  if (!mGL->MakeCurrent()) {
    gfxCriticalNote
        << "Failed to make render context current during destroying.";
    // Leak resources!
    return;
  }
}

bool RenderCompositorOGL::BeginFrame() {
  if (!mGL->MakeCurrent()) {
    gfxCriticalNote << "Failed to make render context current, can't draw.";
    return false;
  }

#if defined(__EMSCRIPTEN__)
  // The default framebuffer (FB0) is the transferred OffscreenCanvas (#screen)
  // owned by this Renderer thread. The main thread CANNOT resize a transferred
  // OffscreenCanvas, so when the chrome window resizes (gpu_present -> widget
  // Resize -> GetClientSize) the canvas would otherwise stay at its initial size
  // and the scene would render clipped / mispositioned. Resize it here, on the
  // owning thread, to the current widget size before drawing. (Cheap no-op when
  // unchanged; the #glout present canvas is CSS-sized to the viewport so the
  // resized frame displays 1:1.)
  {
    const auto sz = GetBufferSize();
    int cw = 0, ch = 0;
    emscripten_get_canvas_element_size("#screen", &cw, &ch);
    if ((cw != sz.width || ch != sz.height) && sz.width > 0 && sz.height > 0) {
      emscripten_set_canvas_element_size("#screen", sz.width, sz.height);
    }
  }
#endif

  mGL->fBindFramebuffer(LOCAL_GL_FRAMEBUFFER, mGL->GetDefaultFramebuffer());

  return true;
}

RenderedFrameId RenderCompositorOGL::EndFrame(
    const nsTArray<DeviceIntRect>& aDirtyRects) {
  RenderedFrameId frameId = GetNextRenderFrameId();
  if (UsePartialPresent() && aDirtyRects.Length() > 0) {
    gfx::IntRegion bufferInvalid;
    const auto bufferSize = GetBufferSize();
    for (const DeviceIntRect& rect : aDirtyRects) {
      const auto left = std::clamp(rect.min.x, 0, bufferSize.width);
      const auto top = std::clamp(rect.min.y, 0, bufferSize.height);

      const auto right = std::clamp(rect.max.x, 0, bufferSize.width);
      const auto bottom = std::clamp(rect.max.y, 0, bufferSize.height);

      const auto width = right - left;
      const auto height = bottom - top;

      bufferInvalid.OrWith(
          gfx::IntRect(left, (GetBufferSize().height - bottom), width, height));
    }
    gl()->SetDamage(bufferInvalid);
  }
  mGL->SwapBuffers();
  return frameId;
}

void RenderCompositorOGL::Pause() {}

bool RenderCompositorOGL::Resume() { return true; }

LayoutDeviceIntSize RenderCompositorOGL::GetBufferSize() {
  return mWidget->GetClientSize();
}

uint32_t RenderCompositorOGL::GetMaxPartialPresentRects() {
  return gfx::gfxVars::WebRenderMaxPartialPresentRects();
}

bool RenderCompositorOGL::RequestFullRender() { return false; }

bool RenderCompositorOGL::UsePartialPresent() {
  return gfx::gfxVars::WebRenderMaxPartialPresentRects() > 0;
}

bool RenderCompositorOGL::ShouldDrawPreviousPartialPresentRegions() {
  return true;
}

size_t RenderCompositorOGL::GetBufferAge() const {
  if (!StaticPrefs::
          gfx_webrender_allow_partial_present_buffer_age_AtStartup()) {
    return 0;
  }
  return gl()->GetBufferAge();
}

}  // namespace mozilla::wr
