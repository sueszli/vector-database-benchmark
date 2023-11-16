"""
Created on 2017年12月17日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: WorldMap
@description: 
"""
import json
import math
try:
    from PyQt5.QtCore import Qt, QPointF, QRectF
    from PyQt5.QtGui import QColor, QPainter, QPolygonF, QPen, QBrush
    from PyQt5.QtOpenGL import QGLFormat
    from PyQt5.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsPolygonItem
except ImportError:
    from PySide2.QtCore import Qt, QPointF, QRectF
    from PySide2.QtGui import QColor, QPainter, QPolygonF, QPen, QBrush
    from PySide2.QtOpenGL import QGLFormat
    from PySide2.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsPolygonItem

class GraphicsView(QGraphicsView):
    backgroundColor = QColor(31, 31, 47)
    borderColor = QColor(58, 58, 90)

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(GraphicsView, self).__init__(*args, **kwargs)
        self.resize(800, 600)
        self.setBackgroundBrush(self.backgroundColor)
        '\n        #参考 http://doc.qt.io/qt-5/qgraphicsview.html#CacheModeFlag-enum\n        CacheNone                    不使用缓存\n        CacheBackground              缓存背景\n        '
        self.setCacheMode(self.CacheBackground)
        '\n        #参考 http://doc.qt.io/qt-5/qgraphicsview.html#DragMode-enum\n        NoDrag                       什么都没发生; 鼠标事件被忽略。\n        ScrollHandDrag               光标变成指针，拖动鼠标将滚动滚动条。 该模式可以在交互式和非交互式模式下工作。\n        RubberBandDrag               拖动鼠标将设置橡皮筋几何形状，并选择橡皮筋所覆盖的所有项目。 对于非交互式视图，此模式被禁用。\n        '
        self.setDragMode(self.ScrollHandDrag)
        '\n        #参考 http://doc.qt.io/qt-5/qgraphicsview.html#OptimizationFlag-enum\n        DontClipPainter              已过时\n        DontSavePainterState         渲染时，QGraphicsView在渲染背景或前景时以及渲染每个项目时保护painter状态（请参阅QPainter.save()）。 这允许你离开painter处于改变状态（即你可以调用QPainter.setPen()或QPainter.setBrush()，而不需要在绘制之后恢复状态）。 但是，如果项目一致地恢复状态，则应启用此标志以防止QGraphicsView执行相同的操作。\n        DontAdjustForAntialiasing    禁用QGraphicsView的抗锯齿自动调整曝光区域。 在QGraphicsItem.boundingRect()的边界上渲染反锯齿线的项目可能会导致渲染部分线外。 为了防止渲染失真，QGraphicsView在所有方向上将所有曝光区域扩展2个像素。 如果启用此标志，QGraphicsView将不再执行这些调整，最大限度地减少需要重绘的区域，从而提高性能。 一个常见的副作用是，使用抗锯齿功能进行绘制的项目可能会在移动时在画面上留下绘画痕迹。\n        IndirectPainting             从Qt 4.6开始，恢复调用QGraphicsView.drawItems()和QGraphicsScene.drawItems()的旧绘画算法。 仅用于与旧代码的兼容性。\n        '
        self.setOptimizationFlag(self.DontSavePainterState)
        '\n        #参考 http://doc.qt.io/qt-5/qpainter.html#RenderHint-enum\n        Antialiasing                 抗锯齿\n        TextAntialiasing             文本抗锯齿\n        SmoothPixmapTransform        平滑像素变换算法\n        HighQualityAntialiasing      请改用Antialiasing\n        NonCosmeticDefaultPen        已过时\n        Qt4CompatiblePainting        从Qt4移植到Qt5可能有用\n        '
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform)
        if QGLFormat.hasOpenGL():
            self.setRenderHint(QPainter.HighQualityAntialiasing)
        '\n        #当视图被调整大小时，视图如何定位场景。使用这个属性来决定当视口控件的大小改变时，如何在视口中定位场景。 缺省行为NoAnchor在调整大小的过程中不改变场景的位置; 视图的左上角将显示为在调整大小时被锚定。请注意，只有场景的一部分可见（即有滚动条时），此属性的效果才明显。 否则，如果整个场景适合视图，QGraphicsScene使用视图对齐将视景中的场景定位。 \n        #参考 http://doc.qt.io/qt-5/qgraphicsview.html#ViewportAnchor-enum\n        NoAnchor                     视图保持场景的位置不变\n        AnchorViewCenter             视图中心被用作锚点。\n        AnchorUnderMouse             鼠标当前位置被用作锚点\n        '
        self.setResizeAnchor(self.AnchorUnderMouse)
        '\n        Rubber选择模式\n        #参考 http://doc.qt.io/qt-5/qt.html#ItemSelectionMode-enum\n        ContainsItemShape            输出列表仅包含形状完全包含在选择区域内的项目。 不包括与区域轮廓相交的项目。\n        IntersectsItemShape          默认，输出列表包含其形状完全包含在选择区域内的项目以及与区域轮廓相交的项目。\n        ContainsItemBoundingRect     输出列表仅包含边界矩形完全包含在选择区域内的项目。 不包括与区域轮廓相交的项目。\n        IntersectsItemBoundingRect   输出列表包含边界矩形完全包含在选择区域内的项目以及与区域轮廓相交的项目。 这种方法通常用于确定需要重绘的区域。\n        '
        self.setRubberBandSelectionMode(Qt.IntersectsItemShape)
        '\n        #在转换过程中如何定位视图。QGraphicsView使用这个属性决定当变换矩阵改变时如何在视口中定位场景，并且视图的坐标系被变换。 默认行为AnchorViewCenter确保在视图中心的场景点在变换过程中保持不变（例如，在旋转时，场景将围绕视图的中心旋转）。请注意，只有场景的一部分可见（即有滚动条时），此属性的效果才明显。 否则，如果整个场景适合视图，QGraphicsScene使用视图对齐将视景中的场景定位。\n        #参考 http://doc.qt.io/qt-5/qgraphicsview.html#ViewportAnchor-enum\n        NoAnchor                     视图保持场景的位置不变\n        AnchorViewCenter             视图中心被用作锚点。\n        AnchorUnderMouse             鼠标当前位置被用作锚点\n        '
        self.setTransformationAnchor(self.AnchorUnderMouse)
        '\n        #参考 http://doc.qt.io/qt-5/qgraphicsview.html#ViewportUpdateMode-enum\n        FullViewportUpdate           当场景的任何可见部分改变或重新显示时，QGraphicsView将更新整个视口。 当QGraphicsView花费更多的时间来计算绘制的内容（比如重复更新很多小项目）时，这种方法是最快的。 这是不支持部分更新（如QGLWidget）的视口以及需要禁用滚动优化的视口的首选更新模式。\n        MinimalViewportUpdate        QGraphicsView将确定需要重绘的最小视口区域，通过避免重绘未改变的区域来最小化绘图时间。 这是QGraphicsView的默认模式。 虽然这种方法提供了一般的最佳性能，但如果场景中有很多小的可见变化，QGraphicsView最终可能花费更多的时间来寻找最小化的方法。\n        SmartViewportUpdate          QGraphicsView将尝试通过分析需要重绘的区域来找到最佳的更新模式。\n        BoundingRectViewportUpdate   视口中所有更改的边界矩形将被重绘。 这种模式的优点是，QGraphicsView只搜索一个区域的变化，最大限度地减少了花在确定需要重绘的时间。 缺点是还没有改变的地方也需要重新绘制。\n        NoViewportUpdate             当场景改变时，QGraphicsView将永远不会更新它的视口。 预计用户将控制所有更新。 此模式禁用QGraphicsView中的所有（可能较慢）项目可见性测试，适用于要求固定帧速率或视口以其他方式在外部进行更新的场景。\n        '
        self.setViewportUpdateMode(self.SmartViewportUpdate)
        self._scene = QGraphicsScene(-180, -90, 360, 180, self)
        self.setScene(self._scene)
        self.initMap()

    def wheelEvent(self, event):
        if False:
            print('Hello World!')
        if event.modifiers() & Qt.ControlModifier:
            self.scaleView(math.pow(2.0, -event.angleDelta().y() / 240.0))
            return event.accept()
        super(GraphicsView, self).wheelEvent(event)

    def scaleView(self, scaleFactor):
        if False:
            for i in range(10):
                print('nop')
        factor = self.transform().scale(scaleFactor, scaleFactor).mapRect(QRectF(0, 0, 1, 1)).width()
        if factor < 0.07 or factor > 100:
            return
        self.scale(scaleFactor, scaleFactor)

    def initMap(self):
        if False:
            i = 10
            return i + 15
        features = json.load(open('Data/world.json', encoding='utf8')).get('features')
        for feature in features:
            geometry = feature.get('geometry')
            if not geometry:
                continue
            _type = geometry.get('type')
            coordinates = geometry.get('coordinates')
            for coordinate in coordinates:
                if _type == 'Polygon':
                    polygon = QPolygonF([QPointF(latitude, -longitude) for (latitude, longitude) in coordinate])
                    item = QGraphicsPolygonItem(polygon)
                    item.setPen(QPen(self.borderColor, 0))
                    item.setBrush(QBrush(self.backgroundColor))
                    item.setPos(0, 0)
                    self._scene.addItem(item)
                elif _type == 'MultiPolygon':
                    for _coordinate in coordinate:
                        polygon = QPolygonF([QPointF(latitude, -longitude) for (latitude, longitude) in _coordinate])
                        item = QGraphicsPolygonItem(polygon)
                        item.setPen(QPen(self.borderColor, 0))
                        item.setBrush(QBrush(self.backgroundColor))
                        item.setPos(0, 0)
                        self._scene.addItem(item)
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    print('OpenGL Status:', QGLFormat.hasOpenGL())
    view = GraphicsView()
    view.setWindowTitle('世界地图')
    view.show()
    sys.exit(app.exec_())