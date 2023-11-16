"""ComboBox dropped height Test

**What is checked**
It is ensured that the height of the list displayed when the combobox is
dropped down is not less than the height of the reference.

**How is it checked**
The value for the dropped rectangle can be retrieved from windows. The height
of this rectangle is calculated and compared against the reference height.

**When is a bug reported**
If the height of the dropped rectangle for the combobox being checked is less
than the height of the reference one then a bug is reported.

**Bug Extra Information**
There is no extra information associated with this bug type

**Is Reference dialog needed**
The reference dialog is necessary for this test.

**False positive bug reports**
No false bugs should be reported. If the font of the localised control has a
smaller height than the reference then it is possible that the dropped
rectangle could be of a different size.

**Test Identifier**
The identifier for this test/bug is "ComboBoxDroppedHeight"
"""
testname = 'ComboBoxDroppedHeight'

def ComboBoxDroppedHeightTest(windows):
    if False:
        print('Hello World!')
    'Check if each combobox height is the same as the reference'
    bugs = []
    for win in windows:
        if not win.ref:
            continue
        if win.class_name() != 'ComboBox' or win.ref.class_name() != 'ComboBox':
            continue
        if win.dropped_rect().height() != win.ref.dropped_rect().height():
            bugs.append(([win], {}, testname, 0))
    return bugs