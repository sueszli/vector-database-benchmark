"""
Created on 2019/10/4
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: MagneticOfSun
@description: 
"""
import math
from PyQt5.QtCore import QFileInfo, QObject, QSize, Qt, QTimer, QLocale
from PyQt5.QtDataVisualization import Q3DCamera, Q3DScatter, Q3DTheme, QAbstract3DGraph, QAbstract3DSeries, QScatter3DSeries, QScatterDataItem
from PyQt5.QtGui import QColor, QLinearGradient, QQuaternion, QVector3D
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QPushButton, QSlider, QSizePolicy, QVBoxLayout, QWidget
country = QLocale.system().country()
if country in (QLocale.China, QLocale.HongKong, QLocale.Taiwan):
    Tr = {'Item rotations example - Magnetic field of the sun': '项目旋转示例-太阳磁场', 'Toggle animation': '开启/关闭 动画', 'Toggle Sun': '显示/隐藏 太阳', 'Field Lines (1 - 128):': '磁场线条数(1 - 128)：', 'Arrows per line (8 - 32):': '箭头数(8 - 32)：'}
else:
    Tr = {}

class ScatterDataModifier(QObject):
    verticalRange = 8.0
    horizontalRange = verticalRange
    ellipse_a = horizontalRange / 3.0
    ellipse_b = verticalRange
    doublePi = math.pi * 2.0
    radiansToDegrees = 360.0 / doublePi
    animationFrames = 30.0

    def __init__(self, scatter):
        if False:
            print('Hello World!')
        super(ScatterDataModifier, self).__init__()
        mesh_dir = QFileInfo(__file__).absolutePath() + '/Data/mesh'
        self.m_graph = scatter
        self.m_rotationTimer = QTimer()
        self.m_fieldLines = 12
        self.m_arrowsPerLine = 16
        self.m_magneticField = QScatter3DSeries()
        self.m_sun = QScatter3DSeries()
        self.m_angleOffset = 0.0
        self.m_angleStep = self.doublePi / self.m_arrowsPerLine / self.animationFrames
        self.m_graph.setShadowQuality(QAbstract3DGraph.ShadowQualityNone)
        self.m_graph.scene().activeCamera().setCameraPreset(Q3DCamera.CameraPresetFront)
        self.m_magneticField.setItemSize(0.2)
        self.m_magneticField.setMesh(QAbstract3DSeries.MeshUserDefined)
        self.m_magneticField.setUserDefinedMesh(mesh_dir + '/narrowarrow.obj')
        fieldGradient = QLinearGradient(0, 0, 16, 1024)
        fieldGradient.setColorAt(0.0, Qt.black)
        fieldGradient.setColorAt(1.0, Qt.white)
        self.m_magneticField.setBaseGradient(fieldGradient)
        self.m_magneticField.setColorStyle(Q3DTheme.ColorStyleRangeGradient)
        self.m_sun.setItemSize(0.2)
        self.m_sun.setName('Sun')
        self.m_sun.setItemLabelFormat('@seriesName')
        self.m_sun.setMesh(QAbstract3DSeries.MeshUserDefined)
        self.m_sun.setUserDefinedMesh(mesh_dir + '/largesphere.obj')
        self.m_sun.setBaseColor(QColor(255, 187, 0))
        self.m_sun.dataProxy().addItem(QScatterDataItem(QVector3D()))
        self.m_graph.addSeries(self.m_magneticField)
        self.m_graph.addSeries(self.m_sun)
        self.m_graph.axisX().setRange(-self.horizontalRange, self.horizontalRange)
        self.m_graph.axisY().setRange(-self.verticalRange, self.verticalRange)
        self.m_graph.axisZ().setRange(-self.horizontalRange, self.horizontalRange)
        self.m_graph.axisX().setSegmentCount(self.horizontalRange)
        self.m_graph.axisZ().setSegmentCount(self.horizontalRange)
        self.m_rotationTimer.timeout.connect(self.triggerRotation)
        self.toggleRotation()
        self.generateData()

    def generateData(self):
        if False:
            for i in range(10):
                print('nop')
        magneticFieldArray = []
        for i in range(self.m_fieldLines):
            horizontalAngle = self.doublePi * i / self.m_fieldLines
            xCenter = self.ellipse_a * math.cos(horizontalAngle)
            zCenter = self.ellipse_a * math.sin(horizontalAngle)
            yRotation = QQuaternion.fromAxisAndAngle(0.0, 1.0, 0.0, horizontalAngle * self.radiansToDegrees)
            for j in range(self.m_arrowsPerLine):
                verticalAngle = self.doublePi * j / self.m_arrowsPerLine + self.m_angleOffset
                xUnrotated = self.ellipse_a * math.cos(verticalAngle)
                y = self.ellipse_b * math.sin(verticalAngle)
                xRotated = xUnrotated * math.cos(horizontalAngle)
                zRotated = xUnrotated * math.sin(horizontalAngle)
                x = xCenter + xRotated
                z = zCenter + zRotated
                zRotation = QQuaternion.fromAxisAndAngle(0.0, 0.0, 1.0, verticalAngle * self.radiansToDegrees)
                totalRotation = yRotation * zRotation
                itm = QScatterDataItem(QVector3D(x, y, z), totalRotation)
                magneticFieldArray.append(itm)
        if self.m_graph.selectedSeries() is self.m_magneticField:
            self.m_graph.clearSelection()
        self.m_magneticField.dataProxy().resetArray(magneticFieldArray)

    def setFieldLines(self, lines):
        if False:
            while True:
                i = 10
        self.m_fieldLines = lines
        self.generateData()

    def setArrowsPerLine(self, arrows):
        if False:
            for i in range(10):
                print('nop')
        self.m_arrowsPerLine = arrows
        self.m_angleOffset = 0.0
        self.m_angleStep = self.doublePi / self.m_arrowsPerLine / self.animationFrames
        self.generateData()

    def triggerRotation(self):
        if False:
            i = 10
            return i + 15
        self.m_angleOffset += self.m_angleStep
        self.generateData()

    def toggleSun(self):
        if False:
            while True:
                i = 10
        self.m_sun.setVisible(not self.m_graph.seriesList()[1].isVisible())

    def toggleRotation(self):
        if False:
            return 10
        if self.m_rotationTimer.isActive():
            self.m_rotationTimer.stop()
        else:
            self.m_rotationTimer.start(15)
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    graph = Q3DScatter()
    container = QWidget.createWindowContainer(graph)
    screenSize = graph.screen().size()
    container.setMinimumSize(QSize(screenSize.width() / 2, screenSize.height() / 1.5))
    container.setMaximumSize(screenSize)
    container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    container.setFocusPolicy(Qt.StrongFocus)
    widget = QWidget()
    hLayout = QHBoxLayout(widget)
    vLayout = QVBoxLayout()
    hLayout.addWidget(container, 1)
    hLayout.addLayout(vLayout)
    widget.setWindowTitle(Tr.get('Item rotations example - Magnetic field of the sun', 'Item rotations example - Magnetic field of the sun'))
    toggleRotationButton = QPushButton(Tr.get('Toggle animation', 'Toggle animation'))
    toggleSunButton = QPushButton(Tr.get('Toggle Sun', 'Toggle Sun'))
    fieldLinesSlider = QSlider(Qt.Horizontal)
    fieldLinesSlider.setTickInterval(1)
    fieldLinesSlider.setMinimum(1)
    fieldLinesSlider.setValue(12)
    fieldLinesSlider.setMaximum(128)
    arrowsSlider = QSlider(Qt.Horizontal)
    arrowsSlider.setTickInterval(1)
    arrowsSlider.setMinimum(8)
    arrowsSlider.setValue(16)
    arrowsSlider.setMaximum(32)
    vLayout.addWidget(toggleRotationButton)
    vLayout.addWidget(toggleSunButton)
    vLayout.addWidget(QLabel(Tr.get('Field Lines (1 - 128):', 'Field Lines (1 - 128):')))
    vLayout.addWidget(fieldLinesSlider)
    vLayout.addWidget(QLabel(Tr.get('Arrows per line (8 - 32):', 'Arrows per line (8 - 32):')))
    vLayout.addWidget(arrowsSlider, 1, Qt.AlignTop)
    modifier = ScatterDataModifier(graph)
    toggleRotationButton.clicked.connect(modifier.toggleRotation)
    toggleSunButton.clicked.connect(modifier.toggleSun)
    fieldLinesSlider.valueChanged.connect(modifier.setFieldLines)
    arrowsSlider.valueChanged.connect(modifier.setArrowsPerLine)
    widget.show()
    sys.exit(app.exec_())