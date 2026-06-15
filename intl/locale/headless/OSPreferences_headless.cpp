/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

// Minimal OSPreferences for the headless (wasm engine) build: there is no OS
// locale/regional/date-time source, so report none and let Gecko fall back to
// its default locale (en-US, from prefs) and ICU's default date/time patterns.

#include "OSPreferences.h"

using namespace mozilla::intl;

OSPreferences::OSPreferences() = default;

bool OSPreferences::ReadSystemLocales(nsTArray<nsCString>& aLocaleList) {
  // No OS locale source headless; caller falls back to the default locale.
  return false;
}

bool OSPreferences::ReadRegionalPrefsLocales(nsTArray<nsCString>& aLocaleList) {
  return ReadSystemLocales(aLocaleList);
}

bool OSPreferences::ReadDateTimePattern(DateTimeFormatStyle aDateStyle,
                                        DateTimeFormatStyle aTimeStyle,
                                        const nsACString& aLocale,
                                        nsACString& aRetVal) {
  // No OS date/time customization; ICU defaults apply.
  return false;
}

void OSPreferences::RemoveObservers() {}
