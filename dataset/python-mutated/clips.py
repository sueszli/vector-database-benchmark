print('shell/clips.py initializing')
try:
    import common_base as base
except ImportError:
    try:
        import base
    except ImportError:
        import common.lib.base as base
try:
    import common_ctrl as ctrl
except ImportError:
    try:
        import ctrl
    except ImportError:
        import common.lib.ctrl as ctrl
if False:
    try:
        from _stubs import *
    except ImportError:
        from common.lib._stubs import *
_LABEL_COL = 0
_PREVIEW_COL = 1
_LOAD_BUTTON_COL = 2
_EDIT_BUTTON_COL = 3
_BUTTON_COLS = [_LOAD_BUTTON_COL, _EDIT_BUTTON_COL]

class ClipBin(base.Extension):

    def __init__(self, comp):
        if False:
            i = 10
            return i + 15
        super().__init__(comp)

    @property
    def _Clips(self):
        if False:
            i = 10
            return i + 15
        return self.comp.op('./clips')

    def _GetPreviewTOP(self, num):
        if False:
            return 10
        return self.comp.op('./clip_previews/preview__%d' % num)

    @property
    def _Previews(self):
        if False:
            while True:
                i = 10
        return self.comp.op('./clip_previews')

    @property
    def _LoadButtonImage(self):
        if False:
            print('Hello World!')
        return self.comp.op('./load_btn_image')

    @property
    def _EditButtonImage(self):
        if False:
            i = 10
            return i + 15
        return self.comp.op('./edit_btn_image')

    @property
    def _BgColor(self):
        if False:
            print('Hello World!')
        return (0, 0, 0, 1)

    @property
    def _RolloverBgColor(self):
        if False:
            return 10
        return (0.3, 0.3, 0.3, 1)

    @property
    def _ButtonBgColor(self):
        if False:
            for i in range(10):
                print('nop')
        return (0.4, 0.4, 0.4, 1)

    @property
    def _ButtonRolloverBgColor(self):
        if False:
            return 10
        return (0.8, 0.8, 0.8, 1)

    def List_onInitTable(self, listcomp, attribs):
        if False:
            i = 10
            return i + 15
        pass

    def List_onInitCol(self, listcomp, col, attribs):
        if False:
            return 10
        if col == _LABEL_COL:
            attribs.colStretch = True
        elif col == _PREVIEW_COL:
            attribs.colWidth = 100
        elif col in _BUTTON_COLS:
            attribs.colWidth = 30
            attribs.bgColor = self._ButtonBgColor

    def List_onInitRow(self, listcomp, row, attribs):
        if False:
            print('Hello World!')
        if row == 0:
            attribs.leftBorderOutColor = attribs.rightBorderOutColor = attribs.topBorderOutColor = attribs.bottomBorderOutColor = (0.1, 0.1, 0.1, 1)
            attribs.bgColor = (0.9, 0.9, 0.9, 1)
        else:
            attribs.leftBorderOutColor = attribs.rightBorderOutColor = attribs.topBorderOutColor = attribs.bottomBorderOutColor = (0.5, 0.5, 0.5, 1)
            attribs.rowHeight = 80

    def List_onInitCell(self, listcomp, row, col, attribs):
        if False:
            return 10
        if row == 0:
            return
        if col == _LABEL_COL:
            attribs.text = self._Clips[row, 'name']
            attribs.textJustify = JustifyType.TOPLEFT
        elif col == _PREVIEW_COL:
            thumb = self._GetPreviewTOP(row)
            attribs.top = thumb if thumb else ''
            attribs.bgColor = self._BgColor
        elif col == _LOAD_BUTTON_COL:
            attribs.top = self._LoadButtonImage
        elif col == _EDIT_BUTTON_COL:
            attribs.top = self._EditButtonImage

    def List_onRollover(self, listcomp, row, col, prevrow, prevcol):
        if False:
            print('Hello World!')
        previews = self._Previews
        if prevrow and prevrow != -1:
            listcomp.cellAttribs[prevrow, _LABEL_COL].bgColor = self._BgColor
            for btncol in _BUTTON_COLS:
                listcomp.cellAttribs[prevrow, btncol].bgColor = None
        if row and row != -1:
            listcomp.cellAttribs[row, _LABEL_COL].bgColor = self._RolloverBgColor
            for btncol in _BUTTON_COLS:
                listcomp.cellAttribs[row, btncol].bgColor = self._ButtonRolloverBgColor if col == btncol else None
            previews.par.Activeclip = row
        else:
            previews.par.Activeclip = 0

    def List_onSelect(self, listcomp, startrow, startcol, startcoords, endrow, endcol, endcoords, start, end):
        if False:
            for i in range(10):
                print('nop')
        pass

    def List_onHover(self, listcomp, row, col, coords, prevRow, prevCol, prevCoords, dragItems):
        if False:
            print('Hello World!')
        self._LogEvent('List_onHover(row: %r, col: %r, coords: %r, prevRow: %r, prevCol: %r, prevCoords: %r, dragItems: %r)' % (row, col, coords, prevRow, prevCol, prevCoords, dragItems))
        return True

    def List_onDrop(self, listcomp, row, col, coords, prevRow, prevCol, prevCoords, dragItems):
        if False:
            print('Hello World!')
        self._LogEvent('List_onDrop(row: %r, col: %r, coords: %r, prevRow: %r, prevCol: %r, prevCoords: %r, dragItems: %r)' % (row, col, coords, prevRow, prevCol, prevCoords, dragItems))
        return False