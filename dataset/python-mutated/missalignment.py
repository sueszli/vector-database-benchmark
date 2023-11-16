"""Missalignment Test

**What is checked**
This test checks that if a set of controls were aligned on a particular axis
in the reference dialog  that they are all aligned on the same axis.

**How is it checked**
A list of all the reference controls that are aligned is created (ie more than
one control with the same Top, Left, Bottom or Right coordinates). These
controls are then analysed in the localised dialog to make sure that they are
all aligned on the same axis.

**When is a bug reported**
A bug is reported when any of the controls that were aligned in the reference
dialog are no longer aligned in the localised control.

**Bug Extra Information**
The bug contains the following extra information
Name	Description
AlignmentType	This is either LEFT, TOP, RIGHT or BOTTOM. It tells you how
the controls were aligned in the reference dialog. String
AlignmentRect	Gives the smallest rectangle that surrounds ALL the controls
concerned in the bug, rectangle

**Is Reference dialog needed**
This test cannot be performed without the reference control. It is required
to see which controls should be aligned.

**False positive bug reports**
It is quite possible that this test reports false positives:
1.	Where the controls only just happen to be aligned in the reference dialog
(by coincidence)
2.	Where the control does not have a clear boundary (for example static
labels or checkboxes)  they may be miss-aligned but it is not noticeable that
they are not.


**Test Identifier**
The identifier for this test/bug is "Missalignment" """
testname = 'Missalignment'
from pywinauto.windows import win32structures

def MissalignmentTest(windows):
    if False:
        i = 10
        return i + 15
    'Run the test on the windows passed in'
    refAlignments = {}
    for win in windows:
        if not win.ref:
            continue
        for side in ('top', 'left', 'right', 'bottom'):
            sideValue = getattr(win.ref.rectangle(), side)
            sideAlignments = refAlignments.setdefault(side, {})
            sideAlignments.setdefault(sideValue, []).append(win)
    bugs = []
    for side in refAlignments:
        for alignment in refAlignments[side]:
            controls = refAlignments[side][alignment]
            sides = [getattr(ctrl.rectangle(), side) for ctrl in controls]
            sides = set(sides)
            if len(sides) > 1:
                overAllRect = win32structures.RECT()
                overAllRect.left = min([ctrl.rectangle().left for ctrl in controls])
                overAllRect.top = min([ctrl.rectangle().top for ctrl in controls])
                overAllRect.right = max([ctrl.rectangle().right for ctrl in controls])
                overAllRect.bottom = max([ctrl.rectangle().bottom for ctrl in controls])
                bugs.append((controls, {'AlignmentType': side.upper(), 'AlignmentRect': overAllRect}, testname, 0))
    return bugs