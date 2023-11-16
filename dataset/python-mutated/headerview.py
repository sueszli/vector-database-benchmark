from AnyQt.QtCore import Qt, QRect
from AnyQt.QtGui import QBrush, QIcon, QCursor, QPalette, QPainter, QMouseEvent
from AnyQt.QtWidgets import QHeaderView, QStyleOptionHeader, QStyle, QApplication

class HeaderView(QHeaderView):
    """
    A QHeaderView reimplementing `paintSection` to better deal with
    selections in large models.

    In particular:
      * `isColumnSelected`/`isRowSelected` are never queried, only
        `rowIntersectsSelection`/`columnIntersectsSelection` are used.
      * when `highlightSections` is not enabled the selection model is not
        queried at all.
    """

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self.__pressed = -1
        super().__init__(*args, **kwargs)

        def set_pressed(index):
            if False:
                i = 10
                return i + 15
            self.__pressed = index
        self.sectionPressed.connect(set_pressed)
        self.sectionEntered.connect(set_pressed)
        self.setFont(QApplication.font('QHeaderView'))

    def mouseReleaseEvent(self, event: QMouseEvent):
        if False:
            while True:
                i = 10
        self.__pressed = -1
        super().mouseReleaseEvent(event)

    def __sectionIntersectsSelection(self, logicalIndex: int) -> bool:
        if False:
            i = 10
            return i + 15
        selmodel = self.selectionModel()
        if selmodel is None:
            return False
        root = self.rootIndex()
        if self.orientation() == Qt.Horizontal:
            return selmodel.columnIntersectsSelection(logicalIndex, root)
        else:
            return selmodel.rowIntersectsSelection(logicalIndex, root)

    def __isFirstVisibleSection(self, visualIndex):
        if False:
            while True:
                i = 10
        log = self.logicalIndex(visualIndex)
        if log != -1:
            return self.sectionPosition(log) == 0 and self.sectionSize(log) > 0
        else:
            return False

    def __isLastVisibleSection(self, visualIndex):
        if False:
            while True:
                i = 10
        log = self.logicalIndex(visualIndex)
        if log != -1:
            pos = self.sectionPosition(log)
            size = self.sectionSize(log)
            return size > 0 and pos + size == self.length()
        else:
            return False

    def initStyleOptionForIndex(self, option: QStyleOptionHeader, logicalIndex: int) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Similar to initStyleOptionForIndex in Qt 6.0 with the difference that\n        `isSectionSelected` is not used, only `sectionIntersectsSelection`\n        is used (isSectionSelected will scan the entire model column/row\n        when the whole column/row is selected).\n        '
        hover = self.logicalIndexAt(self.mapFromGlobal(QCursor.pos()))
        pressed = self.__pressed
        if self.highlightSections():
            is_selected = self.__sectionIntersectsSelection
        else:
            is_selected = lambda _: False
        state = QStyle.State_None
        if self.isEnabled():
            state |= QStyle.State_Enabled
        if self.window().isActiveWindow():
            state |= QStyle.State_Active
        if self.sectionsClickable():
            if logicalIndex == hover:
                state |= QStyle.State_MouseOver
            if logicalIndex == pressed:
                state |= QStyle.State_Sunken
        if self.highlightSections():
            if is_selected(logicalIndex):
                state |= QStyle.State_On
        if self.isSortIndicatorShown() and self.sortIndicatorSection() == logicalIndex:
            option.sortIndicator = QStyleOptionHeader.SortDown if self.sortIndicatorOrder() == Qt.AscendingOrder else QStyleOptionHeader.SortUp
        style = self.style()
        model = self.model()
        orientation = self.orientation()
        textAlignment = model.headerData(logicalIndex, self.orientation(), Qt.TextAlignmentRole)
        defaultAlignment = self.defaultAlignment()
        textAlignment = textAlignment if isinstance(textAlignment, int) else defaultAlignment
        option.section = logicalIndex
        option.state = QStyle.State(option.state | state)
        option.textAlignment = Qt.Alignment(textAlignment)
        option.iconAlignment = Qt.AlignVCenter
        text = model.headerData(logicalIndex, self.orientation(), Qt.DisplayRole)
        text = str(text) if text is not None else ''
        option.text = text
        icon = model.headerData(logicalIndex, self.orientation(), Qt.DecorationRole)
        try:
            option.icon = QIcon(icon)
        except (TypeError, ValueError):
            pass
        margin = 2 * style.pixelMetric(QStyle.PM_HeaderMargin, None, self)
        headerArrowAlignment = style.styleHint(QStyle.SH_Header_ArrowAlignment, None, self)
        isHeaderArrowOnTheSide = headerArrowAlignment & Qt.AlignVCenter
        if self.isSortIndicatorShown() and self.sortIndicatorSection() == logicalIndex and isHeaderArrowOnTheSide:
            margin += style.pixelMetric(QStyle.PM_HeaderMarkSize, None, self)
        if not option.icon.isNull():
            margin += style.pixelMetric(QStyle.PM_SmallIconSize, None, self)
            margin += style.pixelMetric(QStyle.PM_HeaderMargin, None, self)
        if self.textElideMode() != Qt.ElideNone:
            elideMode = self.textElideMode()
            if hasattr(option, 'textElideMode'):
                option.textElideMode = elideMode
            else:
                option.text = option.fontMetrics.elidedText(option.text, elideMode, option.rect.width() - margin)
        foregroundBrush = model.headerData(logicalIndex, orientation, Qt.ForegroundRole)
        try:
            foregroundBrush = QBrush(foregroundBrush)
        except (TypeError, ValueError):
            pass
        else:
            option.palette.setBrush(QPalette.ButtonText, foregroundBrush)
        backgroundBrush = model.headerData(logicalIndex, orientation, Qt.BackgroundRole)
        try:
            backgroundBrush = QBrush(backgroundBrush)
        except (TypeError, ValueError):
            pass
        else:
            option.palette.setBrush(QPalette.Button, backgroundBrush)
            option.palette.setBrush(QPalette.Window, backgroundBrush)
        visual = self.visualIndex(logicalIndex)
        assert visual != -1
        first = self.__isFirstVisibleSection(visual)
        last = self.__isLastVisibleSection(visual)
        if first and last:
            option.position = QStyleOptionHeader.OnlyOneSection
        elif first:
            option.position = QStyleOptionHeader.Beginning
        elif last:
            option.position = QStyleOptionHeader.End
        else:
            option.position = QStyleOptionHeader.Middle
        option.orientation = orientation
        if self.highlightSections():
            previousSelected = is_selected(self.logicalIndex(visual - 1))
            nextSelected = is_selected(self.logicalIndex(visual + 1))
        else:
            previousSelected = nextSelected = False
        if previousSelected and nextSelected:
            option.selectedPosition = QStyleOptionHeader.NextAndPreviousAreSelected
        elif previousSelected:
            option.selectedPosition = QStyleOptionHeader.PreviousIsSelected
        elif nextSelected:
            option.selectedPosition = QStyleOptionHeader.NextIsSelected
        else:
            option.selectedPosition = QStyleOptionHeader.NotAdjacent

    def paintSection(self, painter, rect, logicalIndex):
        if False:
            return 10
        '\n        Reimplemented from `QHeaderView`.\n        '
        if not rect.isValid():
            return
        oldBO = painter.brushOrigin()
        opt = QStyleOptionHeader()
        opt.rect = rect
        self.initStyleOption(opt)
        oBrushButton = opt.palette.brush(QPalette.Button)
        oBrushWindow = opt.palette.brush(QPalette.Window)
        self.initStyleOptionForIndex(opt, logicalIndex)
        opt.rect = rect
        nBrushButton = opt.palette.brush(QPalette.Button)
        nBrushWindow = opt.palette.brush(QPalette.Window)
        if oBrushButton != nBrushButton or oBrushWindow != nBrushWindow:
            painter.setBrushOrigin(opt.rect.topLeft())
        self.style().drawControl(QStyle.CE_Header, opt, painter, self)
        painter.setBrushOrigin(oldBO)