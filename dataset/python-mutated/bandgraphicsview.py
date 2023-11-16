from PyQt5 import QtWidgets, QtCore, Qt

class BandGraphicsView(QtWidgets.QGraphicsView):

    def resizeEvent(self, event):
        if False:
            while True:
                i = 10
        self.setAlignment(Qt.Qt.AlignCenter)
        self.fitInView(self.scene().itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)
        self.scale(1.3, 1.3)
        self.setViewportMargins(10, 10, 10, 10)