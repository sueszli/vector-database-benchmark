from AnyQt.QtCore import Qt, QSize, QSizeF
from AnyQt.QtGui import QPixmap
from AnyQt.QtWidgets import QGraphicsScene, QGraphicsView
from orangewidget.tests.base import GuiTest
from Orange.widgets.utils.graphicspixmapwidget import GraphicsPixmapWidget

class TestGraphicsPixmapWidget(GuiTest):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        super().setUp()
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)

    def tearDown(self) -> None:
        if False:
            return 10
        self.scene.clear()
        self.scene.deleteLater()
        self.view.deleteLater()
        del self.scene
        del self.view

    def test_graphicspixmapwidget(self):
        if False:
            print('Hello World!')
        w = GraphicsPixmapWidget()
        self.scene.addItem(w)
        w.setPixmap(QPixmap(100, 100))
        p = w.pixmap()
        self.assertEqual(p.size(), QSize(100, 100))
        self.view.grab()
        w.setScaleContents(True)
        w.setAspectRatioMode(Qt.KeepAspectRatio)
        s = w.sizeHint(Qt.PreferredSize)
        self.assertEqual(s, QSizeF(100.0, 100.0))
        s = w.sizeHint(Qt.PreferredSize, QSizeF(200.0, -1.0))
        self.assertEqual(s, QSizeF(200.0, 200.0))
        s = w.sizeHint(Qt.PreferredSize, QSizeF(-1.0, 200.0))
        self.assertEqual(s, QSizeF(200.0, 200.0))
        self.view.grab()