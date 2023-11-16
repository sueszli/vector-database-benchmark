__all__ = ['CanvasItem', 'GroupCanvasItem']
import importlib
from .. import ItemGroup, SRTTransform, debug
from .. import functions as fn
from ..graphicsItems.ROI import ROI
from ..Qt import QT_LIB, QtCore, QtWidgets
from . import TransformGuiTemplate_generic as ui_template
translate = QtCore.QCoreApplication.translate

class SelectBox(ROI):

    def __init__(self, scalable=False, rotatable=True):
        if False:
            return 10
        ROI.__init__(self, [0, 0], [1, 1], invertible=True)
        center = [0.5, 0.5]
        if scalable:
            self.addScaleHandle([1, 1], center, lockAspect=True)
            self.addScaleHandle([0, 0], center, lockAspect=True)
        if rotatable:
            self.addRotateHandle([0, 1], center)
            self.addRotateHandle([1, 0], center)

class CanvasItem(QtCore.QObject):
    sigResetUserTransform = QtCore.Signal(object)
    sigTransformChangeFinished = QtCore.Signal(object)
    sigTransformChanged = QtCore.Signal(object)
    "CanvasItem takes care of managing an item's state--alpha, visibility, z-value, transformations, etc. and\n    provides a control widget"
    sigVisibilityChanged = QtCore.Signal(object)
    transformCopyBuffer = None

    def __init__(self, item, **opts):
        if False:
            i = 10
            return i + 15
        defOpts = {'name': None, 'z': None, 'movable': True, 'scalable': False, 'rotatable': True, 'visible': True, 'parent': None}
        defOpts.update(opts)
        self.opts = defOpts
        self.selectedAlone = False
        QtCore.QObject.__init__(self)
        self.canvas = None
        self._graphicsItem = item
        parent = self.opts['parent']
        if parent is not None:
            self._graphicsItem.setParentItem(parent.graphicsItem())
            self._parentItem = parent
        else:
            self._parentItem = None
        z = self.opts['z']
        if z is not None:
            item.setZValue(z)
        self.ctrl = QtWidgets.QWidget()
        self.layout = QtWidgets.QGridLayout()
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.ctrl.setLayout(self.layout)
        self.alphaLabel = QtWidgets.QLabel(translate('CanvasItem', 'Alpha'))
        self.alphaSlider = QtWidgets.QSlider()
        self.alphaSlider.setMaximum(1023)
        self.alphaSlider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.alphaSlider.setValue(1023)
        self.layout.addWidget(self.alphaLabel, 0, 0)
        self.layout.addWidget(self.alphaSlider, 0, 1)
        self.resetTransformBtn = QtWidgets.QPushButton('Reset Transform')
        self.copyBtn = QtWidgets.QPushButton('Copy')
        self.pasteBtn = QtWidgets.QPushButton('Paste')
        self.transformWidget = QtWidgets.QWidget()
        self.transformGui = ui_template.Ui_Form()
        self.transformGui.setupUi(self.transformWidget)
        self.layout.addWidget(self.transformWidget, 3, 0, 1, 2)
        self.transformGui.mirrorImageBtn.clicked.connect(self.mirrorY)
        self.transformGui.reflectImageBtn.clicked.connect(self.mirrorXY)
        self.layout.addWidget(self.resetTransformBtn, 1, 0, 1, 2)
        self.layout.addWidget(self.copyBtn, 2, 0, 1, 1)
        self.layout.addWidget(self.pasteBtn, 2, 1, 1, 1)
        self.alphaSlider.valueChanged.connect(self.alphaChanged)
        self.alphaSlider.sliderPressed.connect(self.alphaPressed)
        self.alphaSlider.sliderReleased.connect(self.alphaReleased)
        self.resetTransformBtn.clicked.connect(self.resetTransformClicked)
        self.copyBtn.clicked.connect(self.copyClicked)
        self.pasteBtn.clicked.connect(self.pasteClicked)
        self.setMovable(self.opts['movable'])
        if 'transform' in self.opts:
            self.baseTransform = self.opts['transform']
        else:
            self.baseTransform = SRTTransform()
            if 'pos' in self.opts and self.opts['pos'] is not None:
                self.baseTransform.translate(self.opts['pos'])
            if 'angle' in self.opts and self.opts['angle'] is not None:
                self.baseTransform.rotate(self.opts['angle'])
            if 'scale' in self.opts and self.opts['scale'] is not None:
                self.baseTransform.scale(self.opts['scale'])
        tr = self.baseTransform.saveState()
        if 'scalable' not in opts and tr['scale'] == (1, 1):
            self.opts['scalable'] = True
        self.selectBox = SelectBox(scalable=self.opts['scalable'], rotatable=self.opts['rotatable'])
        self.selectBox.hide()
        self.selectBox.setZValue(1000000.0)
        self.selectBox.sigRegionChanged.connect(self.selectBoxChanged)
        self.selectBox.sigRegionChangeFinished.connect(self.selectBoxChangeFinished)
        self.itemRotation = QtWidgets.QGraphicsRotation()
        self.itemScale = QtWidgets.QGraphicsScale()
        self._graphicsItem.setTransformations([self.itemRotation, self.itemScale])
        self.tempTransform = SRTTransform()
        self.userTransform = SRTTransform()
        self.resetUserTransform()

    def setMovable(self, m):
        if False:
            for i in range(10):
                print('nop')
        self.opts['movable'] = m
        if m:
            self.resetTransformBtn.show()
            self.copyBtn.show()
            self.pasteBtn.show()
        else:
            self.resetTransformBtn.hide()
            self.copyBtn.hide()
            self.pasteBtn.hide()

    def setCanvas(self, canvas):
        if False:
            while True:
                i = 10
        if canvas is self.canvas:
            return
        if canvas is None:
            self.canvas.removeFromScene(self._graphicsItem)
            self.canvas.removeFromScene(self.selectBox)
        else:
            canvas.addToScene(self._graphicsItem)
            canvas.addToScene(self.selectBox)
        self.canvas = canvas

    def graphicsItem(self):
        if False:
            print('Hello World!')
        'Return the graphicsItem for this canvasItem.'
        return self._graphicsItem

    def parentItem(self):
        if False:
            while True:
                i = 10
        return self._parentItem

    def setParentItem(self, parent):
        if False:
            print('Hello World!')
        self._parentItem = parent
        if parent is not None:
            if isinstance(parent, CanvasItem):
                parent = parent.graphicsItem()
        self.graphicsItem().setParentItem(parent)

    def copyClicked(self):
        if False:
            while True:
                i = 10
        CanvasItem.transformCopyBuffer = self.saveTransform()

    def pasteClicked(self):
        if False:
            print('Hello World!')
        t = CanvasItem.transformCopyBuffer
        if t is None:
            return
        else:
            self.restoreTransform(t)

    def mirrorY(self):
        if False:
            print('Hello World!')
        if not self.isMovable():
            return
        inv = SRTTransform()
        inv.scale(-1, 1)
        self.userTransform = self.userTransform * inv
        self.updateTransform()
        self.selectBoxFromUser()
        self.sigTransformChangeFinished.emit(self)

    def mirrorXY(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.isMovable():
            return
        self.rotate(180.0)

    def hasUserTransform(self):
        if False:
            print('Hello World!')
        return not self.userTransform.isIdentity()

    def ctrlWidget(self):
        if False:
            return 10
        return self.ctrl

    def alphaChanged(self, val):
        if False:
            while True:
                i = 10
        alpha = val / 1023.0
        self._graphicsItem.setOpacity(alpha)

    def setAlpha(self, alpha):
        if False:
            return 10
        self.alphaSlider.setValue(int(fn.clip_scalar(alpha * 1023, 0, 1023)))

    def alpha(self):
        if False:
            while True:
                i = 10
        return self.alphaSlider.value() / 1023.0

    def isMovable(self):
        if False:
            i = 10
            return i + 15
        return self.opts['movable']

    def selectBoxMoved(self):
        if False:
            for i in range(10):
                print('nop')
        'The selection box has moved; get its transformation information and pass to the graphics item'
        self.userTransform = self.selectBox.getGlobalTransform(relativeTo=self.selectBoxBase)
        self.updateTransform()

    def scale(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        self.userTransform.scale(x, y)
        self.selectBoxFromUser()
        self.updateTransform()

    def rotate(self, ang):
        if False:
            while True:
                i = 10
        self.userTransform.rotate(ang)
        self.selectBoxFromUser()
        self.updateTransform()

    def translate(self, x, y):
        if False:
            print('Hello World!')
        self.userTransform.translate(x, y)
        self.selectBoxFromUser()
        self.updateTransform()

    def setTranslate(self, x, y):
        if False:
            print('Hello World!')
        self.userTransform.setTranslate(x, y)
        self.selectBoxFromUser()
        self.updateTransform()

    def setRotate(self, angle):
        if False:
            while True:
                i = 10
        self.userTransform.setRotate(angle)
        self.selectBoxFromUser()
        self.updateTransform()

    def setScale(self, x, y):
        if False:
            while True:
                i = 10
        self.userTransform.setScale(x, y)
        self.selectBoxFromUser()
        self.updateTransform()

    def setTemporaryTransform(self, transform):
        if False:
            return 10
        self.tempTransform = transform
        self.updateTransform()

    def applyTemporaryTransform(self):
        if False:
            print('Hello World!')
        'Collapses tempTransform into UserTransform, resets tempTransform'
        self.userTransform = self.userTransform * self.tempTransform
        self.resetTemporaryTransform()
        self.selectBoxFromUser()

    def resetTemporaryTransform(self):
        if False:
            for i in range(10):
                print('nop')
        self.tempTransform = SRTTransform()
        self.updateTransform()

    def transform(self):
        if False:
            print('Hello World!')
        return self._graphicsItem.transform()

    def updateTransform(self):
        if False:
            print('Hello World!')
        'Regenerate the item position from the base, user, and temp transforms'
        transform = self.baseTransform * self.userTransform * self.tempTransform
        s = transform.saveState()
        self._graphicsItem.setPos(*s['pos'])
        self.itemRotation.setAngle(s['angle'])
        self.itemScale.setXScale(s['scale'][0])
        self.itemScale.setYScale(s['scale'][1])
        self.displayTransform(transform)
        return s

    def displayTransform(self, transform):
        if False:
            for i in range(10):
                print('nop')
        'Updates transform numbers in the ctrl widget.'
        tr = transform.saveState()
        self.transformGui.translateLabel.setText('Translate: (%f, %f)' % (tr['pos'][0], tr['pos'][1]))
        self.transformGui.rotateLabel.setText('Rotate: %f degrees' % tr['angle'])
        self.transformGui.scaleLabel.setText('Scale: (%f, %f)' % (tr['scale'][0], tr['scale'][1]))

    def resetUserTransform(self):
        if False:
            for i in range(10):
                print('nop')
        self.userTransform.reset()
        self.updateTransform()
        self.selectBox.blockSignals(True)
        self.selectBoxToItem()
        self.selectBox.blockSignals(False)
        self.sigTransformChanged.emit(self)
        self.sigTransformChangeFinished.emit(self)

    def resetTransformClicked(self):
        if False:
            for i in range(10):
                print('nop')
        self.resetUserTransform()
        self.sigResetUserTransform.emit(self)

    def restoreTransform(self, tr):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.userTransform = SRTTransform(tr)
            self.updateTransform()
            self.selectBoxFromUser()
            self.sigTransformChanged.emit(self)
            self.sigTransformChangeFinished.emit(self)
        except:
            self.userTransform = SRTTransform()
            debug.printExc('Failed to load transform:')

    def saveTransform(self):
        if False:
            print('Hello World!')
        'Return a dict containing the current user transform'
        return self.userTransform.saveState()

    def selectBoxFromUser(self):
        if False:
            for i in range(10):
                print('nop')
        'Move the selection box to match the current userTransform'
        self.selectBox.blockSignals(True)
        self.selectBox.setState(self.selectBoxBase)
        self.selectBox.applyGlobalTransform(self.userTransform)
        self.selectBox.blockSignals(False)

    def selectBoxToItem(self):
        if False:
            while True:
                i = 10
        "Move/scale the selection box so it fits the item's bounding rect. (assumes item is not rotated)"
        self.itemRect = self._graphicsItem.boundingRect()
        rect = self._graphicsItem.mapRectToParent(self.itemRect)
        self.selectBox.blockSignals(True)
        self.selectBox.setPos([rect.x(), rect.y()])
        self.selectBox.setSize(rect.size())
        self.selectBox.setAngle(0)
        self.selectBoxBase = self.selectBox.getState().copy()
        self.selectBox.blockSignals(False)

    def zValue(self):
        if False:
            i = 10
            return i + 15
        return self.opts['z']

    def setZValue(self, z):
        if False:
            return 10
        self.opts['z'] = z
        if z is not None:
            self._graphicsItem.setZValue(z)

    def selectionChanged(self, sel, multi):
        if False:
            return 10
        '\n        Inform the item that its selection state has changed. \n        ============== =========================================================\n        **Arguments:**\n        sel            (bool) whether the item is currently selected\n        multi          (bool) whether there are multiple items currently \n                       selected\n        ============== =========================================================\n        '
        self.selectedAlone = sel and (not multi)
        self.showSelectBox()
        if self.selectedAlone:
            self.ctrlWidget().show()
        else:
            self.ctrlWidget().hide()

    def showSelectBox(self):
        if False:
            for i in range(10):
                print('nop')
        'Display the selection box around this item if it is selected and movable'
        if self.selectedAlone and self.isMovable() and self.isVisible():
            self.selectBox.show()
        else:
            self.selectBox.hide()

    def hideSelectBox(self):
        if False:
            for i in range(10):
                print('nop')
        self.selectBox.hide()

    def selectBoxChanged(self):
        if False:
            print('Hello World!')
        self.selectBoxMoved()
        self.sigTransformChanged.emit(self)

    def selectBoxChangeFinished(self):
        if False:
            print('Hello World!')
        self.sigTransformChangeFinished.emit(self)

    def alphaPressed(self):
        if False:
            for i in range(10):
                print('nop')
        'Hide selection box while slider is moving'
        self.hideSelectBox()

    def alphaReleased(self):
        if False:
            while True:
                i = 10
        self.showSelectBox()

    def show(self):
        if False:
            for i in range(10):
                print('nop')
        if self.opts['visible']:
            return
        self.opts['visible'] = True
        self._graphicsItem.show()
        self.showSelectBox()
        self.sigVisibilityChanged.emit(self)

    def hide(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.opts['visible']:
            return
        self.opts['visible'] = False
        self._graphicsItem.hide()
        self.hideSelectBox()
        self.sigVisibilityChanged.emit(self)

    def setVisible(self, vis):
        if False:
            i = 10
            return i + 15
        if vis:
            self.show()
        else:
            self.hide()

    def isVisible(self):
        if False:
            while True:
                i = 10
        return self.opts['visible']

    def saveState(self):
        if False:
            print('Hello World!')
        return {'type': self.__class__.__name__, 'name': self.name, 'visible': self.isVisible(), 'alpha': self.alpha(), 'userTransform': self.saveTransform(), 'z': self.zValue(), 'scalable': self.opts['scalable'], 'rotatable': self.opts['rotatable'], 'movable': self.opts['movable']}

    def restoreState(self, state):
        if False:
            while True:
                i = 10
        self.setVisible(state['visible'])
        self.setAlpha(state['alpha'])
        self.restoreTransform(state['userTransform'])
        self.setZValue(state['z'])

class GroupCanvasItem(CanvasItem):
    """
    Canvas item used for grouping others
    """

    def __init__(self, **opts):
        if False:
            while True:
                i = 10
        defOpts = {'movable': False, 'scalable': False}
        defOpts.update(opts)
        item = ItemGroup()
        CanvasItem.__init__(self, item, **defOpts)