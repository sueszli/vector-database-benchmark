"""Asian Hotkey Format Test

**What is checked**

This test checks whether the format for shortcuts/hotkeys follows the
standards for localised Windows applications. This format is
{localised text}({uppercase hotkey})
so for example if the English control is
"&Help"
the localised control for Asian languages should be
"LocHelp(H)"

**How is it checked**

After checking whether this control displays hotkeys it examines the 1st
string of the control to make sure that the format is correct.
If the reference control is available then it also makes sure that the
hotkey character is the same as the reference.
Controls with a title of less than 4 characters are ignored. This has been
done to avoid false positive bug reports for strings like "&X:".

**When is a bug reported**

A bug is reported when a control has a hotkey and it is not in the correct
format.
Also if the reference control is available a bug will be reported if the
hotkey character is not the same as used in the reference

**Bug Extra Information**

This test produces 2 different types of bug:
BugType: "AsianHotkeyFormat"
There is no extra information associated with this bug type

**BugType: "AsianHotkeyDiffRef"**

There is no extra information associated with this bug type

**Is Reference dialog needed**

The reference dialog is not needed.
If it is unavailable then only bugs of type "AsianHotkeyFormat" will be
reported, bug of type "AsianHotkeyDiffRef" will not be found.

**False positive bug reports**

There should be very few false positive bug reports when testing Asian
software. If a string is very short (eg "&Y:") but is padded with spaces
then it will get reported.

**Test Identifier**

The identifier for this test/bug is "AsianHotkeyTests"
"""
testname = 'AsianHotkeyFormat'
import re
from .repeatedhotkey import ImplementsHotkey, GetHotkey

def AsianHotkeyTest(windows):
    if False:
        return 10
    'Return the repeated hotkey errors'
    bugs = []
    for win in windows:
        if not ImplementsHotkey(win):
            continue
        if _IsAsianHotkeyFormatIncorrect(win.window_text()):
            bugs.append(([win], {}, testname, 0))
    return bugs
_asianHotkeyRE = re.compile('\n    \\(&.\\)      # the hotkey\n    (\n        (\\t.*)|     # tab, and then anything\n        #(\\\\t.*)|   # escaped tab, and then anything\n        (\\(.*\\)     # anything in brackets\n    )|\n    \\s*|            # any whitespace\n    :|              # colon\n    (\\.\\.\\.)|       # elipsis\n    >|              # greater than sign\n    <|              # less than sign\n    (\\n.*)          # newline, and then anything\n    \\s)*$', re.VERBOSE)

def _IsAsianHotkeyFormatIncorrect(text):
    if False:
        for i in range(10):
            print('nop')
    'Check if the format of the hotkey is correct or not'
    (pos, char) = GetHotkey(text)
    if char:
        found = _asianHotkeyRE.search(text)
        if not found:
            return True
    return False
AsianHotkeyTest.TestsMenus = True