from ..graphicsItems.ViewBox import ViewBox
from ..Qt import QtCore, QtGui, QtWidgets
from ..widgets.GraphicsView import GraphicsView
translate = QtCore.QCoreApplication.translate

class FlowchartGraphicsView(GraphicsView):
    sigHoverOver = QtCore.Signal(object)
    sigClicked = QtCore.Signal(object)

    def __init__(self, widget, *args):
        if False:
            for i in range(10):
                print('nop')
        GraphicsView.__init__(self, *args, useOpenGL=False)
        self._vb = FlowchartViewBox(widget, lockAspect=True, invertY=True)
        self.setCentralItem(self._vb)
        self.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

    def viewBox(self):
        if False:
            for i in range(10):
                print('nop')
        return self._vb

class FlowchartViewBox(ViewBox):

    def __init__(self, widget, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        ViewBox.__init__(self, *args, **kwargs)
        self.widget = widget

    def getMenu(self, ev):
        if False:
            i = 10
            return i + 15
        self._fc_menu = QtWidgets.QMenu()
        self._subMenus = self.getContextMenus(ev)
        for menu in self._subMenus:
            self._fc_menu.addMenu(menu)
        return self._fc_menu

    def getContextMenus(self, ev):
        if False:
            print('Hello World!')
        menu = self.widget.buildMenu(ev.scenePos())
        menu.setTitle(translate('Context Menu', 'Add node'))
        return [menu, ViewBox.getMenu(self, ev)]