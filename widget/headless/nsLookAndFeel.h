/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#ifndef mozilla_widget_headless_nsLookAndFeel_h
#define mozilla_widget_headless_nsLookAndFeel_h

// The headless-only toolkit has no native widget toolkit to provide a real
// nsLookAndFeel, so the engine's nsLookAndFeel is just the hardcoded
// HeadlessLookAndFeel. nsXPLookAndFeel always instantiates HeadlessLookAndFeel
// at runtime when gfxPlatform::IsHeadless() (always true here); this class only
// exists so the `new nsLookAndFeel()` fallback path compiles and matches the
// `class nsLookAndFeel;` forward declaration in nsXPLookAndFeel.h.
#include "HeadlessLookAndFeel.h"

class nsLookAndFeel final : public mozilla::widget::HeadlessLookAndFeel {
 public:
  nsLookAndFeel() = default;
  ~nsLookAndFeel() override = default;
};

#endif  // mozilla_widget_headless_nsLookAndFeel_h
