"""slide rows/columns around"""
import visidata
from visidata import Sheet, moveListItem, vd

@Sheet.api
def slide_col(sheet, colidx, newcolidx):
    if False:
        return 10
    vd.addUndo(moveVisibleCol, sheet, newcolidx, colidx)
    return moveVisibleCol(sheet, colidx, newcolidx)

@Sheet.api
def slide_keycol(sheet, fromKeyColIdx, toKeyColIdx):
    if False:
        while True:
            i = 10
    vd.addUndo(moveKeyCol, sheet, toKeyColIdx, fromKeyColIdx)
    return moveKeyCol(sheet, fromKeyColIdx, toKeyColIdx)

@Sheet.api
def slide_row(sheet, rowidx, newcolidx):
    if False:
        return 10
    vd.addUndo(moveListItem, sheet.rows, newcolidx, rowidx)
    return moveListItem(sheet.rows, rowidx, newcolidx)

def moveKeyCol(sheet, fromKeyColIdx, toKeyColIdx):
    if False:
        i = 10
        return i + 15
    'Move key column to another key column position in sheet.'
    if not 1 <= toKeyColIdx <= len(sheet.keyCols):
        vd.warning('already at edge')
        return fromKeyColIdx - 1
    for col in sheet.keyCols:
        if col.keycol == fromKeyColIdx:
            col.keycol = toKeyColIdx
        elif toKeyColIdx < fromKeyColIdx:
            if toKeyColIdx <= col.keycol < fromKeyColIdx:
                col.keycol += 1
        elif fromKeyColIdx < col.keycol <= toKeyColIdx:
            col.keycol -= 1
    return toKeyColIdx - 1

def moveVisibleCol(sheet, fromVisColIdx, toVisColIdx):
    if False:
        while True:
            i = 10
    'Move visible column to another visible index in sheet.'
    if 0 <= toVisColIdx < sheet.nVisibleCols:
        fromVisColIdx = min(max(fromVisColIdx, 0), sheet.nVisibleCols - 1)
        fromColIdx = sheet.columns.index(sheet.visibleCols[fromVisColIdx])
        if toVisColIdx < len(sheet.keyCols):
            vd.warning('already at edge')
            return fromVisColIdx
        else:
            toColIdx = sheet.columns.index(sheet.visibleCols[toVisColIdx])
        moveListItem(sheet.columns, fromColIdx, toColIdx)
        return toVisColIdx
    else:
        vd.warning('already at edge')
        return fromVisColIdx
Sheet.addCommand('H', 'slide-left', 'sheet.cursorVisibleColIndex = slide_col(cursorVisibleColIndex, cursorVisibleColIndex-1) if not cursorCol.keycol else slide_keycol(cursorCol.keycol, cursorCol.keycol-1)', 'slide current column left')
Sheet.addCommand('L', 'slide-right', 'sheet.cursorVisibleColIndex = slide_col(cursorVisibleColIndex, cursorVisibleColIndex+1) if not cursorCol.keycol else slide_keycol(cursorCol.keycol, cursorCol.keycol+1)', 'slide current column right')
Sheet.addCommand('J', 'slide-down', 'sheet.cursorRowIndex = slide_row(cursorRowIndex, cursorRowIndex+1)', 'slide current row down')
Sheet.addCommand('K', 'slide-up', 'sheet.cursorRowIndex = slide_row(cursorRowIndex, cursorRowIndex-1)', 'slide current row up')
Sheet.addCommand('gH', 'slide-leftmost', 'slide_col(cursorVisibleColIndex, len(keyCols) + 0) if not cursorCol.keycol else slide_keycol(cursorCol.keycol, 1)', 'slide current column all the way to the left of sheet')
Sheet.addCommand('gL', 'slide-rightmost', 'slide_col(cursorVisibleColIndex, nVisibleCols-1) if not cursorCol.keycol else slide_keycol(cursorCol.keycol, len(keyCols))', 'slide current column all the way to the right of sheet')
Sheet.addCommand('gJ', 'slide-bottom', 'slide_row(cursorRowIndex, nRows)', 'slide current row all the way to the bottom of sheet')
Sheet.addCommand('gK', 'slide-top', 'slide_row(cursorRowIndex, 0)', 'slide current row to top of sheet')
Sheet.addCommand('zH', 'slide-left-n', 'slide_col(cursorVisibleColIndex, cursorVisibleColIndex-int(input("slide col left n=", value=1)))', 'slide current column N positions to the left')
Sheet.addCommand('zL', 'slide-right-n', 'slide_col(cursorVisibleColIndex, cursorVisibleColIndex+int(input("slide col left n=", value=1)))', 'slide current column N positions to the right')
Sheet.addCommand('zJ', 'slide-down-n', 'slide_row(cursorRowIndex, cursorRowIndex+int(input("slide row down n=", value=1)))', 'slide current row N positions down')
Sheet.addCommand('zK', 'slide-up-n', 'slide_row(cursorRowIndex, cursorRowIndex-int(input("slide row up n=", value=1)))', 'slide current row N positions up')
Sheet.bindkey('KEY_SLEFT', 'slide-left')
Sheet.bindkey('KEY_SR', 'slide-up')
Sheet.bindkey('kDN', 'slide-down')
Sheet.bindkey('kUP', 'slide-up')
Sheet.bindkey('KEY_SRIGHT', 'slide-right')
Sheet.bindkey('KEY_SF', 'slide-down')
Sheet.bindkey('gKEY_SLEFT', 'slide-leftmost')
Sheet.bindkey('gkDN', 'slide-bottom')
Sheet.bindkey('gkUP', 'slide-top')
Sheet.bindkey('gKEY_SRIGHT', 'slide-rightmost')
vd.addMenuItems('\n    Edit > Slide > Row > up > slide-up\n    Edit > Slide > Row > up N > slide-up-n\n    Edit > Slide > Row > down > slide-down\n    Edit > Slide > Row > down N > slide-down-n\n    Edit > Slide > Row > to top > slide-top\n    Edit > Slide > Row > to bottom > slide-bottom\n    Edit > Slide > Column > left > slide-left\n    Edit > Slide > Column > left N > slide-left-n\n    Edit > Slide > Column > leftmost > slide-leftmost\n    Edit > Slide > Column > right > slide-right\n    Edit > Slide > Column > right N > slide-right-n\n    Edit > Slide > Column > rightmost > slide-rightmost\n')

def make_tester(setup_vdx):
    if False:
        i = 10
        return i + 15

    def t(vdx, golden):
        if False:
            return 10
        global vd
        vd = visidata.vd.resetVisiData()
        vd.runvdx(setup_vdx)
        vd.runvdx(vdx)
        colnames = [c.name for c in vd.sheet.visibleCols]
        assert colnames == golden.split(), ' '.join(colnames)
    return t

def test_slide_keycol_1(vd):
    if False:
        return 10
    t = make_tester('\n            open-file sample_data/sample.tsv\n            +::OrderDate key-col\n            +::Region key-col\n            +::Rep key-col\n        ')
    t('', 'OrderDate Region Rep Item Units Unit_Cost Total')
    t('+::Rep slide-leftmost', 'Rep OrderDate Region Item Units Unit_Cost Total')
    t('+::OrderDate slide-rightmost', 'Region Rep OrderDate Item Units Unit_Cost Total')
    t('+::Rep slide-left', 'OrderDate Rep Region Item Units Unit_Cost Total')
    t('+::OrderDate slide-right', 'Region OrderDate Rep Item Units Unit_Cost Total')
    t('\n        +::Item key-col\n        +::Item slide-left\n        slide-left\n        slide-right\n        slide-right\n        slide-left\n        slide-left\n    ', 'OrderDate Item Region Rep Units Unit_Cost Total')

def test_slide_leftmost(vd):
    if False:
        print('Hello World!')
    t = make_tester('open-file sample_data/benchmark.csv')
    t('+::Paid slide-leftmost', 'Paid Date Customer SKU Item Quantity Unit')
    t = make_tester('\n         open-file sample_data/benchmark.csv\n         +::Date key-col\n    ')
    t('', 'Date Customer SKU Item Quantity Unit Paid')
    t('+::Item slide-leftmost', 'Date Item Customer SKU Quantity Unit Paid')
    t('+::SKU key-col\n         +::Quantity slide-leftmost', 'Date SKU Quantity Customer Item Unit Paid')
    t('+::Date slide-leftmost', 'Date Customer SKU Item Quantity Unit Paid')
    t('+::Item slide-leftmost\n         +::SKU slide-leftmost', 'Date SKU Item Customer Quantity Unit Paid')