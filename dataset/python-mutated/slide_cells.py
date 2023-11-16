"""
# TODO:
- slide-cells-left
- slide-cells-<dir>-n
- slide-cells-selected-<dir>-n
- rename "slide" to "shift"?
"""
from visidata import vd, TableSheet

@TableSheet.api
def slide_cells_right(sheet, row, vcolidx):
    if False:
        i = 10
        return i + 15
    for (oldcol, newcol) in reversed(list(zip(sheet.visibleCols[vcolidx:], sheet.visibleCols[vcolidx + 1:]))):
        newcol.setValue(row, oldcol.getValue(row))
    sheet.visibleCols[vcolidx].setValue(row, None)
TableSheet.addCommand('', 'slide-cells-right', 'slide_cells_right(cursorRow, cursorVisibleColIndex)', '\nShift individual values in current row one visible column to the right, with leftmost cell set to null.\n')
vd.addMenuItems('\n    Edit > Slide > cells > right > slide-cells-right\n')