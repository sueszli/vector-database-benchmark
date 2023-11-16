from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
import json

class ScrollAreaPreview(QtWidgets.QScrollArea):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(ScrollAreaPreview, self).__init__(*args, **kwargs)
        self.setWidgetResizable(True)
        content = QtWidgets.QWidget(self)
        self.setWidget(content)
        lay = QtWidgets.QVBoxLayout(content)
        self.label = QtWidgets.QLabel(content)
        self.label.setWordWrap(True)
        lay.addWidget(self.label)

    def setText(self, text):
        if False:
            for i in range(10):
                print('nop')
        self.label.setText(text)

    def setPixmap(self, pixmap):
        if False:
            for i in range(10):
                print('nop')
        self.label.setPixmap(pixmap)

    def clear(self):
        if False:
            i = 10
            return i + 15
        self.label.clear()

class FileDialogPreview(QtWidgets.QFileDialog):

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(FileDialogPreview, self).__init__(*args, **kwargs)
        self.setOption(self.DontUseNativeDialog, True)
        self.labelPreview = ScrollAreaPreview(self)
        self.labelPreview.setFixedSize(300, 300)
        self.labelPreview.setHidden(True)
        box = QtWidgets.QVBoxLayout()
        box.addWidget(self.labelPreview)
        box.addStretch()
        self.setFixedSize(self.width() + 300, self.height())
        self.layout().addLayout(box, 1, 3, 1, 1)
        self.currentChanged.connect(self.onChange)

    def onChange(self, path):
        if False:
            print('Hello World!')
        if path.lower().endswith('.json'):
            with open(path, 'r') as f:
                data = json.load(f)
                self.labelPreview.setText(json.dumps(data, indent=4, sort_keys=False))
            self.labelPreview.label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
            self.labelPreview.setHidden(False)
        else:
            pixmap = QtGui.QPixmap(path)
            if pixmap.isNull():
                self.labelPreview.clear()
                self.labelPreview.setHidden(True)
            else:
                self.labelPreview.setPixmap(pixmap.scaled(self.labelPreview.width() - 30, self.labelPreview.height() - 30, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
                self.labelPreview.label.setAlignment(QtCore.Qt.AlignCenter)
                self.labelPreview.setHidden(False)