from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QFileDialog

import NNCBIR
import DogColorCBIR
import DogGaborCBIR
import DogHOGCBIR

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self,query_image, image_files,predictor):
        self.predictor = predictor
        QtWidgets.QMainWindow.__init__(self)
        self.setStyleSheet("background-color: white;")
        widget = QWidget(self)
        self.setCentralWidget(widget)
        self.layout = QtWidgets.QGridLayout(widget)

        label = QtWidgets.QLabel(self)
        label.setFont(QFont('Times', 20))
        label.setText("Query Image:")
        self.layout.addWidget(label, 0, 0,alignment =QtCore.Qt.AlignRight)
        btn1 = QtWidgets.QPushButton("Select Query Image")
        btn1.clicked.connect(self.getfiles)
        self.layout.addWidget(btn1, 0, 0)
        self.update_gui( query_image, image_files)

    def update_gui(self, query_image, image_files):
        label = QtWidgets.QLabel(self)
        label.setPixmap(QtGui.QPixmap(query_image).scaled(256, 256, QtCore.Qt.KeepAspectRatio))
        label.setAlignment(QtCore.Qt.AlignCenter)
        self.layout.addWidget(label, 0, 1)

        id = 0
        for row in range(1,3):
            for col in range(3):
                if id < len(image_files):
                    label = QtWidgets.QLabel(self)
                    label.setPixmap(QtGui.QPixmap(image_files[id]).scaled(256, 256, QtCore.Qt.KeepAspectRatio))
                    self.layout.addWidget(label,row, col)
                    id += 1

    def getfiles(self):
        dlg = QFileDialog(self)
        dlg.setFileMode(QFileDialog.AnyFile)
        dlg.setNameFilters(["Images (*.png *.jpg *.jpeg)"])

        if dlg.exec_():
            filenames = dlg.selectedFiles()
            query_image, image_files = self.predictor.predict(filenames[0])
            self.update_gui(query_image, image_files)


if __name__ == "__main__":

    predictor = NNCBIR
    # predictor = DogColorCBIR
    # predictor = DogGaborCBIR
    # predictor = DogHOGCBIR

    query_image,image_files = predictor.predict()

    app = QtWidgets.QApplication(sys.argv)
    Form = MainWindow(query_image,image_files,predictor)
    Form.show()
    sys.exit(app.exec_())

