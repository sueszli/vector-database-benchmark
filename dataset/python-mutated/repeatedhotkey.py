"""Repeated Hotkeys Test

**What is checked**
This test checks all the controls in a dialog to see if there are controls that
use the same hotkey character.

**How is it checked**
A list of all the hotkeys (converted to uppercase) used in the dialog is
created. Then this list is examined to see if any hotkeys are used more than
once. If any are used more than once a list of all the controls that use this
hotkey are compiled to be used in the bug report.

**When is a bug reported**
If more than one control has the same hotkey then a bug is reported.

**Bug Extra Information**
The bug contains the following extra information
Name	Description
RepeatedHotkey	This is the hotkey that is repeated between the 2 controls
converted to uppercase, String
CharsUsedInDialog	This is a list of all the hotkeys used in the dialog,
String
AllCharsInDialog	This is a list of all the characters in the dialog for
controls that have a hotkeys, String
AvailableInControlS	A list of the available characters for each control.
Any of the characters in this list could be used as the new hotkey without
conflicting with any existing hotkey.

**Is Reference dialog needed**
The reference dialog does not need to be available. If it is available then
for each bug discovered it is checked to see if it is a problem in the
reference dialog.
NOTE: Checking the reference dialog is not so exact here! Only when the
equivalent controls in the reference dialog all have the hotkeys will it be
reported as being in the reference also. I.e. if there are 3 controls with the
same hotkey in the Localised software  then those same controls in the
reference dialog must have the same hotkey for it to be reported as existing
in the reference also.

**False positive bug reports**
There should be very few false positives from this test. Sometimes a control
only has one or 2 characters eg "X:" and it is impossible to avoid a hotkey
clash. Also for Asian languages hotkeys should be the same as the US software
so probably this test should be run on those languages.

**Test Identifier**
The identifier for this test/bug is "RepeatedHotkey"
"""
testname = 'RepeatedHotkey'
from pywinauto.windows.win32defines import SS_NOPREFIX

def RepeatedHotkeyTest(windows):
    if False:
        return 10
    'Return the repeated hotkey errors'
    (hotkeyControls, allChars, hotkeys) = _CollectDialogInfo(windows)
    dlgAvailable = allChars.difference(hotkeys)
    dlgAvailable.difference_update(set('-& _'))
    bugs = []
    for (char, controls) in hotkeyControls.items():
        if len(controls) > 1:
            ctrlsAvailableChars = ''
            for ctrl in controls:
                controlChars = ''
                controlChars = set(ctrl.window_text().lower())
                controlAvailableChars = controlChars.intersection(dlgAvailable)
                controlAvailableChars = '<%s>' % _SetAsString(controlAvailableChars)
                ctrlsAvailableChars += controlAvailableChars
            refCtrls = [ctrl.ref for ctrl in controls if ctrl.ref]
            (refHotkeyControls, refAllChars, refHotkeys) = _CollectDialogInfo(refCtrls)
            isInRef = -1
            if len(refHotkeys) > 1:
                isInRef = 1
            else:
                isInRef = 0
            bugs.append((controls, {'RepeatedHotkey': char, 'CharsUsedInDialog': _SetAsString(hotkeys), 'AllCharsInDialog': _SetAsString(allChars), 'AvailableInControls': ctrlsAvailableChars}, testname, isInRef))
    return bugs

def _CollectDialogInfo(windows):
    if False:
        i = 10
        return i + 15
    'Collect information on the hotkeys in the dialog'
    hotkeyControls = {}
    allChars = ''
    for win in windows:
        if not ImplementsHotkey(win):
            continue
        (pos, char) = GetHotkey(win.window_text())
        if not char:
            continue
        hotkeyControls.setdefault(char.lower(), []).append(win)
        allChars += win.window_text().lower()
    allChars = set(allChars)
    hotkeys = set(hotkeyControls.keys())
    return (hotkeyControls, allChars, hotkeys)

def GetHotkey(text):
    if False:
        i = 10
        return i + 15
    'Return the position and character of the hotkey'
    curEnd = len(text) + 1
    text = text.replace('&&', '__')
    while True:
        pos = text.rfind('&', 0, curEnd)
        if pos in [-1, len(text)]:
            return (-1, '')
        if text[pos - 1] == '&':
            curEnd = pos - 2
            continue
        return (pos, text[pos + 1])

def _SetAsString(settojoin):
    if False:
        return 10
    'Convert the set to a ordered string'
    return ''.join(sorted(settojoin))

def ImplementsHotkey(win):
    if False:
        while True:
            i = 10
    'checks whether a control interprets & character to be a hotkey'
    if win.class_name() == 'Button':
        return True
    elif win.class_name() == 'Static' and (not win.HasStyle(SS_NOPREFIX)):
        return True
    if win.class_name() == 'MenuItem' and win.state() != '2048':
        return True
    return False
RepeatedHotkeyTest.TestsMenus = True