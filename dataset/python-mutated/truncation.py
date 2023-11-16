"""Truncation Test

**What is checked**
Checks for controls where the text does not fit in the space provided by the
control.

**How is it checked**
There is a function in windows (DrawText) that allows us to find the size that
certain text will need. We use this function with correct fonts and other
relevant information for the control to be as accurate as possible.

**When is a bug reported**
When the calculated required size for the text is greater than the size of the
space available for displaying the text.

**Bug Extra Information**
The bug contains the following extra information
Name	Description
Strings		The list of the truncated strings as explained above
StringIndices		The list of indices (0 based) that are truncated. This
will often just be 0 but if there are many strings in the control untranslated
it will report ALL the strings e.g. 0,2,5,19,23


**Is Reference dialog needed**
The reference dialog does not need to be available. If it is available then
for each bug discovered it is checked to see if it is a problem in the
reference dialog.

**False positive bug reports**
Certain controls do not display the text that is the title of the control, if
this is not handled in a standard manner by the software then DLGCheck will
report that the string is truncated.

**Test Identifier**
The identifier for this test/bug is "Truncation"
"""
testname = 'Truncation'
import ctypes
import six
from pywinauto.windows import win32defines, win32functions, win32structures

def TruncationTest(windows):
    if False:
        while True:
            i = 10
    'Actually do the test'
    truncations = []
    for win in windows:
        (truncIdxs, truncStrings) = _FindTruncations(win)
        isInRef = -1
        if truncIdxs:
            if win.ref:
                isInRef = 0
                (refTruncIdxs, refTruncStrings) = _FindTruncations(win.ref)
                if refTruncIdxs:
                    isInRef = 1
            truncIdxs = ','.join([six.text_type(index) for index in truncIdxs])
            truncStrings = '"%s"' % ','.join([six.text_type(string) for string in truncStrings])
            truncations.append(([win], {'StringIndices': truncIdxs, 'Strings': truncStrings}, testname, isInRef))
    return truncations

def _FindTruncations(ctrl):
    if False:
        while True:
            i = 10
    'Return the index of the texts that are truncated for this control'
    truncIdxs = []
    truncStrings = []
    for (idx, (text, rect, font, flags)) in enumerate(_GetTruncationInfo(ctrl)):
        if not text:
            continue
        minRect = _GetMinimumRect(text, font, rect, flags)
        if minRect.right > rect.right or minRect.bottom > rect.bottom:
            truncIdxs.append(idx)
            truncStrings.append(text)
    return (truncIdxs, truncStrings)

def _GetMinimumRect(text, font, usableRect, drawFlags):
    if False:
        print('Hello World!')
    'Return the minimum rectangle that the text will fit into\n\n    Uses font, usableRect and drawFlags information to find how\n    to do it accurately.\n    '
    txtDC = win32functions.CreateDC(u'DISPLAY', None, None, None)
    hFontGUI = win32functions.CreateFontIndirect(ctypes.byref(font))
    win32functions.SelectObject(txtDC, hFontGUI)
    modifiedRect = win32structures.RECT(usableRect)
    win32functions.DrawText(txtDC, six.text_type(text), -1, ctypes.byref(modifiedRect), win32defines.DT_CALCRECT | drawFlags)
    win32functions.DeleteObject(hFontGUI)
    win32functions.DeleteDC(txtDC)
    return modifiedRect

def _GroupBoxTruncInfo(win):
    if False:
        for i in range(10):
            print('nop')
    'Return truncation information specific to Button controls'
    lineFormat = win32defines.DT_SINGLELINE
    heightAdj = 4
    widthAdj = 9
    if win.has_style(win32defines.BS_BITMAP) or win.has_style(win32defines.BS_ICON):
        heightAdj = -9000
        widthAdj = -9000
        lineFormat = win32defines.DT_WORDBREAK
    newRect = win.client_rects()[0]
    newRect.right -= widthAdj
    newRect.bottom -= heightAdj
    return [(win.window_text(), newRect, win.font(), lineFormat)]

def _RadioButtonTruncInfo(win):
    if False:
        print('Hello World!')
    'Return truncation information specific to Button controls'
    lineFormat = win32defines.DT_SINGLELINE
    if win.has_style(win32defines.BS_MULTILINE):
        lineFormat = win32defines.DT_WORDBREAK
    widthAdj = 19
    if win.has_style(win32defines.BS_BITMAP) or win.has_style(win32defines.BS_ICON):
        widthAdj = -9000
        lineFormat = win32defines.DT_WORDBREAK
    newRect = win.client_rects()[0]
    newRect.right -= widthAdj
    return [(win.window_text(), newRect, win.font(), lineFormat)]

def _CheckBoxTruncInfo(win):
    if False:
        i = 10
        return i + 15
    'Return truncation information specific to Button controls'
    lineFormat = win32defines.DT_SINGLELINE
    if win.has_style(win32defines.BS_MULTILINE):
        lineFormat = win32defines.DT_WORDBREAK
    widthAdj = 18
    if win.has_style(win32defines.BS_BITMAP) or win.has_style(win32defines.BS_ICON):
        widthAdj = -9000
        lineFormat = win32defines.DT_WORDBREAK
    newRect = win.client_rects()[0]
    newRect.right -= widthAdj
    return [(win.window_text(), newRect, win.font(), lineFormat)]

def _ButtonTruncInfo(win):
    if False:
        while True:
            i = 10
    'Return truncation information specific to Button controls'
    lineFormat = win32defines.DT_SINGLELINE
    if win.has_style(win32defines.BS_MULTILINE):
        lineFormat = win32defines.DT_WORDBREAK
    heightAdj = 4
    widthAdj = 5
    if win.has_style(win32defines.BS_PUSHLIKE):
        widthAdj = 3
        heightAdj = 3
        if win.has_style(win32defines.BS_MULTILINE):
            widthAdj = 9
            heightAdj = 2
    if win.has_style(win32defines.BS_BITMAP) or win.has_style(win32defines.BS_ICON):
        heightAdj = -9000
        widthAdj = -9000
        lineFormat = win32defines.DT_WORDBREAK
    newRect = win.client_rects()[0]
    newRect.right -= widthAdj
    newRect.bottom -= heightAdj
    return [(win.window_text(), newRect, win.font(), lineFormat)]

def _ComboBoxTruncInfo(win):
    if False:
        for i in range(10):
            print('nop')
    'Return truncation information specific to ComboBox controls'
    lineFormat = win32defines.DT_SINGLELINE | win32defines.DT_NOPREFIX
    if win.has_style(win32defines.CBS_DROPDOWN) or win.has_style(win32defines.CBS_DROPDOWNLIST):
        widthAdj = 2
    else:
        widthAdj = 3
    truncData = []
    for title in win.texts():
        newRect = win.client_rects()[0]
        newRect.right -= widthAdj
        truncData.append((title, newRect, win.font(), lineFormat))
    return truncData

def _ComboLBoxTruncInfo(win):
    if False:
        i = 10
        return i + 15
    'Return truncation information specific to ComboLBox controls'
    lineFormat = win32defines.DT_SINGLELINE | win32defines.DT_NOPREFIX
    truncData = []
    for title in win.texts():
        newRect = win.client_rects()[0]
        newRect.right -= 5
        truncData.append((title, newRect, win.font(), lineFormat))
    return truncData

def _ListBoxTruncInfo(win):
    if False:
        print('Hello World!')
    'Return truncation information specific to ListBox controls'
    lineFormat = win32defines.DT_SINGLELINE | win32defines.DT_NOPREFIX
    truncData = []
    for title in win.texts():
        newRect = win.client_rects()[0]
        newRect.right -= 2
        newRect.bottom -= 1
        truncData.append((title, newRect, win.font(), lineFormat))
    return truncData

def _StaticTruncInfo(win):
    if False:
        i = 10
        return i + 15
    'Return truncation information specific to Static controls'
    lineFormat = win32defines.DT_WORDBREAK
    if win.has_style(win32defines.SS_CENTERIMAGE) or win.has_style(win32defines.SS_SIMPLE) or (win.has_style(win32defines.SS_LEFTNOWORDWRAP) and 'WindowsForms' not in win.class_name()):
        lineFormat = win32defines.DT_SINGLELINE
    if win.has_style(win32defines.SS_NOPREFIX):
        lineFormat |= win32defines.DT_NOPREFIX
    return [(win.window_text(), win.client_rects()[0], win.font(), lineFormat)]

def _EditTruncInfo(win):
    if False:
        i = 10
        return i + 15
    'Return truncation information specific to Edit controls'
    lineFormat = win32defines.DT_WORDBREAK | win32defines.DT_NOPREFIX
    if not win.has_style(win32defines.ES_MULTILINE):
        lineFormat |= win32defines.DT_SINGLELINE
    return [(win.window_text(), win.client_rects()[0], win.font(), lineFormat)]

def _DialogTruncInfo(win):
    if False:
        print('Hello World!')
    'Return truncation information specific to Header controls'
    newRect = win.client_rects()[0]
    newRect.top += 5
    newRect.left += 5
    newRect.right -= 5
    if win.has_style(win32defines.WS_THICKFRAME):
        newRect.top += 1
        newRect.left += 1
        newRect.right -= 1
    if win.has_style(win32defines.WS_SYSMENU) and (win.has_exstyle(win32defines.WS_EX_PALETTEWINDOW) or win.has_exstyle(win32defines.WS_EX_TOOLWINDOW)):
        newRect.right -= 15
    elif win.has_style(win32defines.WS_SYSMENU):
        buttons = []
        newRect.right -= 18
        buttons.append('close')
        if not win.has_exstyle(win32defines.WS_EX_DLGMODALFRAME):
            newRect.left += 19
        if win.has_exstyle(win32defines.WS_EX_CONTEXTHELP) and (not (win.has_style(win32defines.WS_MAXIMIZEBOX) and win.has_style(win32defines.WS_MINIMIZEBOX))):
            newRect.right -= 17
            if win.has_style(win32defines.WS_MINIMIZEBOX) or win.has_style(win32defines.WS_MAXIMIZEBOX) or win.has_style(win32defines.WS_GROUP):
                newRect.right -= 3
            buttons.append('help')
        if win.has_style(win32defines.WS_MINIMIZEBOX) or win.has_style(win32defines.WS_MAXIMIZEBOX) or win.has_style(win32defines.WS_GROUP):
            newRect.right -= 32
            buttons.append('min')
            buttons.append('max')
        if buttons:
            diff = 5
            diff += len(buttons) * 16
            if len(buttons) > 1:
                diff += 2
            if 'min' in buttons and 'help' in buttons:
                diff += 4
    return [(win.window_text(), newRect, win.font(), win32defines.DT_SINGLELINE)]

def _StatusBarTruncInfo(win):
    if False:
        for i in range(10):
            print('nop')
    'Return truncation information specific to StatusBar controls'
    truncInfo = _WindowTruncInfo(win)
    for (i, (title, rect, font, flag)) in enumerate(truncInfo):
        rect.bottom -= win.VertBorderWidth
        if i == 0:
            rect.right -= win.HorizBorderWidth
        else:
            rect.right -= win.InterBorderWidth
    return truncInfo

def _HeaderTruncInfo(win):
    if False:
        while True:
            i = 10
    'Return truncation information specific to Header controls'
    truncInfo = _WindowTruncInfo(win)
    for (i, (title, rect, font, flag)) in enumerate(truncInfo):
        rect.right -= 12
    return truncInfo

def _WindowTruncInfo(win):
    if False:
        return 10
    'Return Default truncation information'
    matchedItems = []
    for (i, title) in enumerate(win.texts()):
        if i < len(win.client_rects()):
            rect = win.client_rects()[i]
        else:
            rect = win.client_rects()[0]
        if len(win.fonts()) - 1 < i:
            font = win.font()
        else:
            font = win.fonts()[i]
        matchedItems.append((title, rect, font, win32defines.DT_SINGLELINE))
    return matchedItems
_TruncInfo = {'#32770': _DialogTruncInfo, 'ComboBox': _ComboBoxTruncInfo, 'ComboLBox': _ComboLBoxTruncInfo, 'ListBox': _ListBoxTruncInfo, 'Button': _ButtonTruncInfo, 'CheckBox': _CheckBoxTruncInfo, 'GroupBox': _GroupBoxTruncInfo, 'RadioButton': _RadioButtonTruncInfo, 'Edit': _EditTruncInfo, 'Static': _StaticTruncInfo}

def _GetTruncationInfo(win):
    if False:
        print('Hello World!')
    'helper function to hide non special windows'
    if win.friendly_class_name() in _TruncInfo:
        return _TruncInfo[win.friendly_class_name()](win)
    else:
        return _WindowTruncInfo(win)