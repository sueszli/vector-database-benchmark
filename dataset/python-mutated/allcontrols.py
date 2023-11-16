"""Get All Controls Test

**What is checked**
This test does no actual testing  it just returns each control.

**How is it checked**
A loop over all the controls in the dialog is made and each control added to
the list of bugs

**When is a bug reported**
For each control.

**Bug Extra Information**
There is no extra information associated with this bug type

**Is Reference dialog needed**
No,but if available the reference control will be returned with the localised
control.

**False positive bug reports**
Not possible

**Test Identifier**
The identifier for this test/bug is "AllControls"
"""
testname = 'AllControls'

def AllControlsTest(windows):
    if False:
        print('Hello World!')
    'Returns just one bug for each control'
    bugs = []
    for win in windows:
        bugs.append(([win], {}, testname, 0))
    return bugs