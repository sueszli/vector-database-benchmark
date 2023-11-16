"""Compare against reference font test

**What is checked**
This test checks all the parameters of the font for the control against the
font for the reference control. If any value is different then this is reported
as a bug.
Here is a list of all the possible values that are tested:
lfFaceName	The name of the font
lfHeight	The height of the font
lfWidth		Average width of characters
lfEscapement	Angle of text
lfOrientation	Another angle for the text!
lfWeight	How bold the text is
lfItalic	If the font is italic
lfUnderline	If the font is underlined
lfStrikeOut	If the font is struck out
lfCharSet	The character set of the font
lfOutPrecision	The output precision
lfClipPrecision	The clipping precision
lfQuality	The output quality
lfPitchAndFamily	The pitch and family


**How is it checked**
Each property of the font for the control being tested is compared against the
equivalent property of the reference control font for equality.

**When is a bug reported**
For each property of the font that is not identical to the reference font a bug
is reported. So for example if the Font Face has changed and the text is bold
then (at least) 2 bugs will be reported.

**Bug Extra Information**
The bug contains the following extra information
Name	Description
ValueType	What value is incorrect (see above), String
Ref	The reference value converted to a string, String
Loc	The localised value converted to a string, String

**Is Reference dialog needed**
This test will not run if the reference controls are not available.

**False positive bug reports**
Running this test for Asian languages will result in LOTS and LOTS of false
positives, because the font HAS to change for the localised text to display
properly.

**Test Identifier**
The identifier for this test/bug is "CompareToRefFont"
"""
testname = 'CompareToRefFont'
import six
from pywinauto.windows import win32structures
_font_attribs = [field[0] for field in win32structures.LOGFONTW._fields_]

def CompareToRefFontTest(windows):
    if False:
        i = 10
        return i + 15
    'Compare the font to the font of the reference control'
    bugs = []
    for win in windows:
        if not win.ref:
            continue
        for font_attrib in _font_attribs:
            loc_value = getattr(win.font(), font_attrib)
            ref_value = getattr(win.ref.font(), font_attrib)
            if loc_value != ref_value:
                bugs.append(([win], {'ValueType': font_attrib, 'Ref': six.text_type(ref_value), 'Loc': six.text_type(loc_value)}, testname, 0))
    return bugs