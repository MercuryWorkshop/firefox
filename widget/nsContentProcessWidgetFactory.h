/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef nsContentProcessWidgetFactory_h
#define nsContentProcessWidgetFactory_h

#include "nsISupports.h"
#include "nsComponentManagerUtils.h"
#include "nsServiceManagerUtils.h"
#include "nsXULAppAPI.h"

#define MAKE_COMPONENT_CHOOSER(name_, parent_, content_, constructor_) \
  static already_AddRefed<nsISupports> name_() {                       \
    nsCOMPtr<nsISupports> inst;                                        \
    if (XRE_IsContentProcess()) {                                      \
      inst = constructor_(content_);                                   \
    } else {                                                           \
      inst = constructor_(parent_);                                    \
    }                                                                  \
    return inst.forget();                                              \
  }

#if defined(__EMSCRIPTEN__)
// The cairo-headless toolkit registers no platform clipboard component, so
// @mozilla.org/widget/parent/clipboard;1 is absent. Build the headless clipboard
// directly (single process => always the parent path). Defined in
// widget/headless/HeadlessClipboard.cpp.
namespace mozilla::widget {
already_AddRefed<nsISupports> CreateHeadlessClipboardSupports();
}
static already_AddRefed<nsISupports> nsClipboardSelector() {
  return mozilla::widget::CreateHeadlessClipboardSupports();
}
#elif defined(MOZ_WIDGET_COCOA)
// This should be `do_GetService`, but test_bug466599.xhtml erroneously uses
// `createInstance` rather than `getService`, and passes only because doing so
// bypasses the clipboard contents cache, which does not have the expected
// wrapping.
MAKE_COMPONENT_CHOOSER(nsClipboardSelector,
                       "@mozilla.org/widget/parent/clipboard;1",
                       "@mozilla.org/widget/content/clipboard;1",
                       do_CreateInstance)
#else
MAKE_COMPONENT_CHOOSER(nsClipboardSelector,
                       "@mozilla.org/widget/parent/clipboard;1",
                       "@mozilla.org/widget/content/clipboard;1", do_GetService)
#endif
MAKE_COMPONENT_CHOOSER(nsColorPickerSelector,
                       "@mozilla.org/parent/colorpicker;1",
                       "@mozilla.org/content/colorpicker;1", do_CreateInstance)
MAKE_COMPONENT_CHOOSER(nsFilePickerSelector, "@mozilla.org/parent/filepicker;1",
                       "@mozilla.org/content/filepicker;1", do_CreateInstance)
MAKE_COMPONENT_CHOOSER(nsScreenManagerSelector,
                       "@mozilla.org/gfx/parent/screenmanager;1",
                       "@mozilla.org/gfx/content/screenmanager;1",
                       do_GetService)
MAKE_COMPONENT_CHOOSER(nsDragServiceSelector,
                       "@mozilla.org/widget/parent/dragservice;1",
                       "@mozilla.org/widget/content/dragservice;1",
                       do_GetService)

#undef MAKE_COMPONENT_CHOOSER

#endif  // defined nsContentProcessWidgetFactory_h
