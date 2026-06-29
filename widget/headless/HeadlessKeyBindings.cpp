/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */

#include <cstdio>
#include <cstdlib>
#include "HeadlessKeyBindings.h"
#include "mozilla/ClearOnShutdown.h"
#include "mozilla/Maybe.h"
#include "mozilla/NativeKeyBindingsType.h"
#include "mozilla/TextEvents.h"
#include "mozilla/WritingModes.h"

namespace mozilla {
namespace widget {

HeadlessKeyBindings& HeadlessKeyBindings::GetInstance() {
  static UniquePtr<HeadlessKeyBindings> sInstance;
  if (!sInstance) {
    sInstance.reset(new HeadlessKeyBindings());
    ClearOnShutdown(&sInstance);
  }
  return *sInstance;
}

nsresult HeadlessKeyBindings::AttachNativeKeyEvent(
    WidgetKeyboardEvent& aEvent) {
  // Stub for non-mac platforms.
  return NS_OK;
}

void HeadlessKeyBindings::GetEditCommands(
    NativeKeyBindingsType aType, const WidgetKeyboardEvent& aEvent,
    const Maybe<WritingMode>& aWritingMode, nsTArray<CommandInt>& aCommands) {
  // Headless has no OS key bindings, so editor shortcuts (select-all, clipboard,
  // undo, caret movement) would otherwise do nothing. Provide a portable default
  // set so embeddings get a usable editor. Commands not applicable to a given
  // editor type are skipped by the controller (IsCommandEnabled).
  const bool ctrl = aEvent.IsControl(), shift = aEvent.IsShift(),
             alt = aEvent.IsAlt(), meta = aEvent.IsMeta();
  auto add = [&](Command c) { aCommands.AppendElement(static_cast<CommandInt>(c)); };

  // Identify the letter for Ctrl/Cmd shortcuts. GetEditCommands is queried on the
  // keypress, where a printable key carries its char in mCharCode and mKeyCode is 0;
  // a keydown would carry it in mKeyCode (uppercase ASCII). Normalize to lowercase.
  uint32_t letter = 0;
  if (aEvent.mKeyCode >= 'A' && aEvent.mKeyCode <= 'Z') {
    letter = aEvent.mKeyCode + 32;
  } else if (aEvent.mCharCode) {
    uint32_t c = aEvent.mCharCode | 0x20;  // lowercase
    if (c >= 'a' && c <= 'z') letter = c;
  }
  if ((ctrl || meta) && !alt) {
    switch (letter) {
      case 'a': if (!shift) add(Command::SelectAll); return;
      case 'c': if (!shift) add(Command::Copy); return;
      case 'x': if (!shift) add(Command::Cut); return;
      case 'v': if (!shift) add(Command::Paste); return;
      case 'z': add(shift ? Command::HistoryRedo : Command::HistoryUndo); return;
      case 'y': if (!shift) add(Command::HistoryRedo); return;
      default: break;
    }
  }
  if (!alt && !meta) {
    switch (aEvent.mKeyNameIndex) {
      case KEY_NAME_INDEX_ArrowLeft:
        add(ctrl ? (shift ? Command::SelectWordPrevious : Command::WordPrevious)
                 : (shift ? Command::SelectCharPrevious : Command::CharPrevious));
        return;
      case KEY_NAME_INDEX_ArrowRight:
        add(ctrl ? (shift ? Command::SelectWordNext : Command::WordNext)
                 : (shift ? Command::SelectCharNext : Command::CharNext));
        return;
      case KEY_NAME_INDEX_ArrowUp:
        if (!ctrl) add(shift ? Command::SelectLinePrevious : Command::LinePrevious);
        return;
      case KEY_NAME_INDEX_ArrowDown:
        if (!ctrl) add(shift ? Command::SelectLineNext : Command::LineNext);
        return;
      case KEY_NAME_INDEX_Home:
        if (ctrl) add(shift ? Command::SelectTop : Command::MoveTop);
        else add(shift ? Command::SelectBeginLine : Command::BeginLine);
        return;
      case KEY_NAME_INDEX_End:
        if (ctrl) add(shift ? Command::SelectBottom : Command::MoveBottom);
        else add(shift ? Command::SelectEndLine : Command::EndLine);
        return;
      default: break;
    }
  }
}

}  // namespace widget
}  // namespace mozilla
