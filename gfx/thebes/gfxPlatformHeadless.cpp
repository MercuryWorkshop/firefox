/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "gfxPlatformHeadless.h"

#include "mozilla/gfx/2D.h"
#include "mozilla/CountingAllocatorBase.h"

#include "gfxFT2FontList.h"
#include "gfxImageSurface.h"
#include "gfxTextRun.h"
#include "nsServiceManagerUtils.h"
#include "nsUnicodeProperties.h"
#include "SoftwareVsyncSource.h"

#include "ft2build.h"
#include FT_FREETYPE_H
#include FT_MODULE_H

using namespace mozilla;
using namespace mozilla::dom;
using namespace mozilla::gfx;
using namespace mozilla::unicode;

static FT_Library gPlatformFTLibrary = nullptr;

class FreetypeReporter final : public nsIMemoryReporter,
                               public CountingAllocatorBase<FreetypeReporter> {
 private:
  ~FreetypeReporter() {}

 public:
  NS_DECL_ISUPPORTS

  static void* Malloc(FT_Memory, long size) { return CountingMalloc(size); }
  static void Free(FT_Memory, void* p) { return CountingFree(p); }
  static void* Realloc(FT_Memory, long cur_size, long new_size, void* p) {
    return CountingRealloc(p, new_size);
  }

  NS_IMETHOD CollectReports(nsIHandleReportCallback* aHandleReport,
                            nsISupports* aData, bool aAnonymize) override {
    MOZ_COLLECT_REPORT("explicit/freetype", KIND_HEAP, UNITS_BYTES,
                       MemoryAllocated(), "Memory used by Freetype.");
    return NS_OK;
  }
};

NS_IMPL_ISUPPORTS(FreetypeReporter, nsIMemoryReporter)

static FT_MemoryRec_ sFreetypeMemoryRecord;

gfxPlatformHeadless::gfxPlatformHeadless() {
  sFreetypeMemoryRecord.user = nullptr;
  sFreetypeMemoryRecord.alloc = FreetypeReporter::Malloc;
  sFreetypeMemoryRecord.free = FreetypeReporter::Free;
  sFreetypeMemoryRecord.realloc = FreetypeReporter::Realloc;

  FT_New_Library(&sFreetypeMemoryRecord, &gPlatformFTLibrary);
  FT_Add_Default_Modules(gPlatformFTLibrary);

  Factory::SetFTLibrary(gPlatformFTLibrary);

  RegisterStrongMemoryReporter(MakeAndAddRef<FreetypeReporter>());

  mOffscreenFormat = SurfaceFormat::X8R8G8B8_UINT32;
}

gfxPlatformHeadless::~gfxPlatformHeadless() {
  FT_Done_Library(gPlatformFTLibrary);
  gPlatformFTLibrary = nullptr;
}

already_AddRefed<gfxASurface> gfxPlatformHeadless::CreateOffscreenSurface(
    const IntSize& aSize, gfxImageFormat aFormat) {
  if (!Factory::AllowedSurfaceSize(aSize)) {
    return nullptr;
  }
  RefPtr<gfxASurface> newSurface = new gfxImageSurface(aSize, aFormat);
  return newSurface.forget();
}

void gfxPlatformHeadless::GetCommonFallbackFonts(
    uint32_t aCh, Script aRunScript, FontPresentation aPresentation,
    nsTArray<const char*>& aFontList) {
  if (PrefersColor(aPresentation)) {
    aFontList.AppendElement("Noto Color Emoji");
  }
  // Generic last-resort fallbacks; whichever of these are bundled will be used.
  aFontList.AppendElement("Noto Sans");
  aFontList.AppendElement("Droid Sans Fallback");
}

bool gfxPlatformHeadless::CreatePlatformFontList() {
  return gfxPlatformFontList::Initialize(new gfxFT2FontList);
}

void gfxPlatformHeadless::ReadSystemFontList(
    mozilla::dom::SystemFontList* aFontList) {
  gfxFT2FontList::PlatformFontList()->ReadSystemFontList(aFontList);
}

already_AddRefed<mozilla::gfx::VsyncSource>
gfxPlatformHeadless::CreateGlobalHardwareVsyncSource() {
  // No hardware vsync in wasm; drive layout off a software 60Hz timer.
  RefPtr<VsyncSource> vsyncSource =
      new SoftwareVsyncSource(TimeDuration::FromMilliseconds(1000.0 / 60.0));
  return vsyncSource.forget();
}
