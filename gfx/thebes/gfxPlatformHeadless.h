/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef GFX_PLATFORM_HEADLESS_H
#define GFX_PLATFORM_HEADLESS_H

#include "gfxPlatform.h"
#include "gfxUserFontSet.h"
#include "nsCOMPtr.h"
#include "nsTArray.h"

// A FreeType-based, software-only gfxPlatform for the headless/wasm toolkit.
// Based on gfxAndroidPlatform, minus the Android-runtime (JNI/vsync/codec) bits.
// Fonts come from gfxFT2FontList (directory-based discovery; bundle fonts in a
// "fonts" dir under the process directory on the emscripten virtual FS).
class gfxPlatformHeadless final : public gfxPlatform {
 public:
  gfxPlatformHeadless();
  virtual ~gfxPlatformHeadless();

  static gfxPlatformHeadless* GetPlatform() {
    return (gfxPlatformHeadless*)gfxPlatform::GetPlatform();
  }

  already_AddRefed<gfxASurface> CreateOffscreenSurface(
      const IntSize& aSize, gfxImageFormat aFormat) override;

  gfxImageFormat GetOffscreenFormat() override { return mOffscreenFormat; }

  bool CreatePlatformFontList() override;

  void ReadSystemFontList(mozilla::dom::SystemFontList*) override;

  void GetCommonFallbackFonts(uint32_t aCh, Script aRunScript,
                              FontPresentation aPresentation,
                              nsTArray<const char*>& aFontList) override;

  bool FontHintingEnabled() override { return false; }
  bool RequiresLinearZoom() override { return false; }

  already_AddRefed<mozilla::gfx::VsyncSource> CreateGlobalHardwareVsyncSource()
      override;

  static bool CheckVariationFontSupport() { return true; }

 protected:
  void InitAcceleration() override { gfxPlatform::InitAcceleration(); }

  // wasm/emscripten provides a real GL context (GLContextProviderEmscripten over
  // WebGL2), so enable HW compositing -> WebRender uses the GPU and presents to
  // the page <canvas>, rather than the software RenderDocument fallback.
  bool AccelerateLayersByDefault() override { return true; }

 private:
  gfxImageFormat mOffscreenFormat;
};

#endif /* GFX_PLATFORM_HEADLESS_H */
