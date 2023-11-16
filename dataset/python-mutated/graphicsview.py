from types import SimpleNamespace
from typing import Optional, List
from AnyQt.QtCore import Qt, QEvent, QObject, QSizeF
from AnyQt.QtGui import QKeySequence, QTransform
from AnyQt.QtWidgets import QGraphicsView, QGraphicsWidget, QAction, QStyle, QApplication, QSizePolicy
from AnyQt.QtCore import pyqtSignal as Signal, pyqtProperty as Property, pyqtSlot as Slot
from orangecanvas.utils import qsizepolicy_is_expanding, qsizepolicy_is_shrinking
from Orange.widgets.utils.graphicslayoutitem import scaled
__all__ = ['GraphicsWidgetView']

class GraphicsWidgetView(QGraphicsView):
    """
    A Graphics view with a single central QGraphicsWidget which is resized
    fo fit into the view.
    """
    __centralWidget: Optional[QGraphicsWidget] = None
    __fitInView = True
    __aspectMode = Qt.KeepAspectRatio
    __widgetResizable = False
    __zoomFactor = 100

    def __init__(self, *args, widgetResizable=False, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)
        self.__widgetResizable = widgetResizable
        self.__zoomFactor = 100
        zoomin = QAction('Zoom in', self, objectName='zoom-in-action', shortcut=QKeySequence.ZoomIn)
        zoomout = QAction('Zoom out', self, objectName='zoom-out-action', shortcut=QKeySequence.ZoomOut)
        zoomreset = QAction('Actual Size', self, objectName='zoom-reset-action', shortcut=QKeySequence(Qt.ControlModifier | Qt.Key_0))
        fit = QAction('Zoom to fit', self, objectName='zoom-to-fit-action', shortcut=QKeySequence(Qt.ControlModifier | Qt.Key_9), checkable=True)
        if hasattr(QAction, 'setShortcutVisibleInContextMenu'):
            for a in [zoomin, zoomout, zoomreset, fit]:
                a.setShortcutVisibleInContextMenu(True)

        @zoomin.triggered.connect
        def _():
            if False:
                i = 10
                return i + 15
            self.setZoomFactor(self.__zoomFactor + 10)

        @zoomout.triggered.connect
        def _():
            if False:
                for i in range(10):
                    print('nop')
            self.setZoomFactor(self.__zoomFactor - 10)

        @zoomreset.triggered.connect
        def _():
            if False:
                while True:
                    i = 10
            self.__zoomFactor = -1
            self.setZoomFactor(100.0)

        @fit.toggled.connect
        def _(state):
            if False:
                i = 10
                return i + 15
            self.setFitInView(state)
        self.addActions([zoomin, zoomout, zoomreset, fit])
        self._actions = SimpleNamespace(zoomin=zoomin, zoomout=zoomout, zoomreset=zoomreset, fit=fit)

    def viewActions(self) -> List[QAction]:
        if False:
            for i in range(10):
                print('nop')
        return [self._actions.zoomout, self._actions.zoomin, self._actions.zoomreset, self._actions.fit]

    def setZoomFactor(self, factor: float) -> None:
        if False:
            return 10
        '\n        Set the zoom level `factor`\n\n        Parameters\n        ----------\n        factor:\n            Zoom level where 100 is default 50 is half the size and 200 is\n            twice the size\n        '
        if self.__zoomFactor != factor or self.__fitInView:
            self.__fitInView = False
            self._actions.fit.setChecked(False)
            self.__zoomFactor = factor
            self.setTransform(QTransform.fromScale(*(self.__zoomFactor / 100,) * 2))
            self._actions.zoomout.setEnabled(factor >= 20)
            self._actions.zoomin.setEnabled(factor <= 300)
            self.zoomFactorChanged.emit(factor)
            if self.__widgetResizable:
                self._resizeToFit()

    def zoomFactor(self) -> float:
        if False:
            while True:
                i = 10
        '\n        Returns\n        -------\n        factor: float\n            The zoom factor.\n        '
        return self.__zoomFactor
    zoomFactorChanged = Signal(int)
    zoomFactor_ = Property(int, zoomFactor, setZoomFactor, notify=zoomFactorChanged)

    def viewportEvent(self, event: QEvent) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if event.type() == QEvent.Resize:
            self._layout()
        return super().viewportEvent(event)

    def setCentralWidget(self, widget: Optional[QGraphicsWidget]) -> None:
        if False:
            return 10
        "\n        Set the central widget. Previous widget (if set) is unset.\n        The widget needs to be in this view's `scene()`\n        "
        if self.__centralWidget is not None:
            self.__centralWidget.removeEventFilter(self)
            self.__centralWidget.destroyed.disconnect(self.__on_centralWidgetDestroyed)
        self.__centralWidget = widget
        if widget is not None:
            widget.installEventFilter(self)
            widget.destroyed.connect(self.__on_centralWidgetDestroyed)
            self._layout()

    def centralWidget(self) -> Optional[QGraphicsWidget]:
        if False:
            for i in range(10):
                print('nop')
        'Return the central widget.'
        return self.__centralWidget

    @Slot(QObject)
    def __on_centralWidgetDestroyed(self):
        if False:
            while True:
                i = 10
        self.__centralWidget = None

    def widgetResizable(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        "\n        Should the central widget be resized (via .resize()) to match the view.\n        or should the view's scale be updated instead.\n        "
        return self.__widgetResizable

    def setWidgetResizable(self, resizable: bool) -> None:
        if False:
            return 10
        '\n        Parameters\n        ----------\n        resizable: bool\n        '
        if self.__widgetResizable != resizable:
            self.__widgetResizable = resizable
            QApplication.postEvent(self, QEvent(QEvent.LayoutRequest))

    def setFitInView(self, enabled: bool) -> None:
        if False:
            while True:
                i = 10
        if self.__fitInView != enabled:
            self.__fitInView = enabled
            self._actions.fit.setChecked(enabled)
            if enabled:
                if self.__widgetResizable:
                    self._resizeToFit()
                else:
                    self._scaleToFit()

    def setAspectMode(self, mode: Qt.AspectRatioMode) -> None:
        if False:
            while True:
                i = 10
        if self.__aspectMode != mode:
            self.__aspectMode = mode
            if self.__fitInView:
                self._scaleToFit()
            elif self.__widgetResizable:
                self._resizeToFit()

    def eventFilter(self, recv: QObject, event: QEvent) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if event.type() == QEvent.LayoutRequest and recv is self.__centralWidget:
            self._layout()
        return super().eventFilter(recv, event)

    def _layout(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        widget = self.__centralWidget
        if widget is None:
            return
        if self.__widgetResizable:
            self._resizeToFit()
        else:
            self._scaleToFit()

    def _resizeToFit(self):
        if False:
            for i in range(10):
                print('nop')
        widget = self.__centralWidget
        size = self.__viewportContentSize()
        vprect = self.viewport().geometry()
        vprect.setSize(size)
        margins = self.viewportMargins()
        vprect = vprect.marginsRemoved(margins)
        viewrect = self.mapToScene(vprect).boundingRect()
        targetsize = viewrect.size()
        maxsize = widget.maximumSize()
        minsize = widget.minimumSize()
        targetsize = targetsize.expandedTo(minsize).boundedTo(maxsize)
        sh = widget.effectiveSizeHint(Qt.PreferredSize)
        policy = widget.sizePolicy()
        vpolicy = policy.verticalPolicy()
        hpolicy = policy.horizontalPolicy()
        if not self.__fitInView:
            widget.resize(sh.expandedTo(minsize).boundedTo(maxsize))
            return
        width = adjusted_size(sh.width(), targetsize.width(), minsize.width(), maxsize.width(), hpolicy)
        height = adjusted_size(sh.height(), targetsize.height(), minsize.height(), maxsize.height(), vpolicy)
        if policy.hasHeightForWidth():
            constr = QSizeF(width, -1)
            height = adjusted_size(widget.effectiveSizeHint(Qt.PreferredSize, constr).height(), targetsize.height(), widget.effectiveSizeHint(Qt.MinimumSize, constr).height(), widget.effectiveSizeHint(Qt.MaximumSize, constr).height(), QSizePolicy.Fixed)
        widget.resize(QSizeF(width, height))

    def _scaleToFit(self):
        if False:
            for i in range(10):
                print('nop')
        widget = self.__centralWidget
        if widget is None or not self.__fitInView:
            return
        vpsize = self.__viewportContentSize()
        size = widget.size()
        if not size.isEmpty():
            sc = scaled(size, vpsize, self.__aspectMode)
            sx = sc.width() / size.width()
            sy = sc.height() / size.height()
            self.setTransform(QTransform().scale(sx, sy))

    def __viewportContentSize(self):
        if False:
            i = 10
            return i + 15
        msize = self.maximumViewportSize()
        vsbar = self.verticalScrollBar()
        hsbar = self.horizontalScrollBar()
        vsbpolicy = self.verticalScrollBarPolicy()
        hsbpolicy = self.horizontalScrollBarPolicy()
        htransient = hsbar.style().styleHint(QStyle.SH_ScrollBar_Transient, None, hsbar)
        vtransient = vsbar.style().styleHint(QStyle.SH_ScrollBar_Transient, None, vsbar)
        if vsbpolicy == Qt.ScrollBarAsNeeded and (not vtransient):
            msize.setWidth(msize.width() - vsbar.sizeHint().width())
        if hsbpolicy == Qt.ScrollBarAsNeeded and (not htransient):
            msize.setHeight(msize.height() - hsbar.sizeHint().height())
        return msize

def adjusted_size(hint: float, available: float, minimum: float, maximum: float, policy: QSizePolicy.Policy) -> float:
    if False:
        for i in range(10):
            print('nop')
    if policy == QSizePolicy.Fixed:
        return hint
    elif policy == QSizePolicy.Ignored:
        return min(max(available, minimum), maximum)
    size = hint
    if qsizepolicy_is_expanding(policy) and hint < available:
        size = min(max(size, available), maximum)
    if qsizepolicy_is_shrinking(policy) and hint > available:
        size = max(min(size, available), minimum)
    return size

def main(argv=None):
    if False:
        print('Hello World!')
    import sys
    from AnyQt.QtWidgets import QGraphicsScene, QMenu
    from AnyQt.QtGui import QBrush
    app = QApplication(argv or sys.argv)
    scene = QGraphicsScene()
    view = GraphicsWidgetView(scene)
    scene.setParent(view)
    view.setContextMenuPolicy(Qt.CustomContextMenu)

    def context(pos):
        if False:
            for i in range(10):
                print('nop')
        menu = QMenu(view)
        menu.addActions(view.actions())
        a = menu.addAction('Aspect mode')
        am = QMenu(menu)
        am.addAction('Ignore', lambda : view.setAspectMode(Qt.IgnoreAspectRatio))
        am.addAction('Keep', lambda : view.setAspectMode(Qt.KeepAspectRatio))
        am.addAction('Keep by expanding', lambda : view.setAspectMode(Qt.KeepAspectRatioByExpanding))
        a.setMenu(am)
        menu.popup(view.viewport().mapToGlobal(pos))
    view.customContextMenuRequested.connect(context)
    w = QGraphicsWidget()
    w.setPreferredSize(500, 500)
    palette = w.palette()
    palette.setBrush(palette.Window, QBrush(Qt.red, Qt.BDiagPattern))
    w.setPalette(palette)
    w.setAutoFillBackground(True)
    scene.addItem(w)
    view.setCentralWidget(w)
    view.show()
    return app.exec()
if __name__ == '__main__':
    main()