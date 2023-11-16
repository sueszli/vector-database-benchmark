from PyQt6.QtCore import QVariantAnimation, QEasingCurve
from PyQt6.QtGui import QVector3D
from UM.Math.Vector import Vector

class CameraAnimation(QVariantAnimation):

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self._camera_tool = None
        self.setDuration(300)
        self.setEasingCurve(QEasingCurve.Type.OutQuad)

    def setCameraTool(self, camera_tool):
        if False:
            return 10
        self._camera_tool = camera_tool

    def setStart(self, start):
        if False:
            print('Hello World!')
        self.setStartValue(QVector3D(start.x, start.y, start.z))

    def setTarget(self, target):
        if False:
            print('Hello World!')
        self.setEndValue(QVector3D(target.x, target.y, target.z))

    def updateCurrentValue(self, value):
        if False:
            print('Hello World!')
        self._camera_tool.setOrigin(Vector(value.x(), value.y(), value.z()))