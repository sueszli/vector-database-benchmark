"""Miscellaneous Control properties Test

**What is checked**
This checks various values related to a control in windows. The values tested
are
class_name	The class type of the control
style	The Style of the control (GetWindowLong)
exstyle	The Extended Style of the control (GetWindowLong)
help_id	The Help ID of the control (GetWindowLong)
control_id	The Control ID of the control (GetWindowLong)
user_data	The User Data of the control (GetWindowLong)
Visibility	Whether the control is visible or not

**How is it checked**
After retrieving the information for the control we compare it to the same
information from the reference control.

**When is a bug reported**
If the information does not match then a bug is reported.

**Bug Extra Information**
The bug contains the following extra information
Name	Description
ValueType	What value is incorrect (see above), String
Ref	The reference value converted to a string, String
Loc	The localised value converted to a string, String

**Is Reference dialog needed**
This test will not run if the reference controls are not available.

**False positive bug reports**
Some values can change easily without any bug being caused, for example User
Data is actually meant for programmers to store information for the control
and this can change every time the software is run.

**Test Identifier**
The identifier for this test/bug is "MiscValues"
"""
testname = 'MiscValues'
import six

def MiscValuesTest(windows):
    if False:
        while True:
            i = 10
    'Return the bugs from checking miscelaneous values of a control'
    bugs = []
    for win in windows:
        if not win.ref:
            continue
        diffs = {}
        if win.class_name() != win.ref.class_name():
            diffs[u'class_name'] = (win.class_name(), win.ref.class_name())
        if win.style() != win.ref.style():
            diffs[u'style'] = (win.style(), win.ref.style())
        if win.exstyle() != win.ref.exstyle():
            diffs[u'exstyle'] = (win.exstyle(), win.ref.exstyle())
        if win.context_help_id() != win.ref.context_help_id():
            diffs[u'help_id'] = (win.context_help_id(), win.ref.context_help_id())
        if win.control_id() != win.ref.control_id():
            diffs[u'control_id'] = (win.control_id(), win.ref.control_id())
        if win.is_visible() != win.ref.is_visible():
            diffs[u'Visibility'] = (win.is_visible(), win.ref.is_visible())
        if win.user_data() != win.ref.user_data():
            diffs[u'user_data'] = (win.user_data(), win.ref.user_data())
        for (diff, vals) in diffs.items():
            bugs.append(([win], {'ValueType': diff, 'Ref': six.text_type(vals[1]), 'Loc': six.text_type(vals[0])}, testname, 0))
    return bugs