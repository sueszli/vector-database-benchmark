import sys
from ..Qt import QtWidgets
__all__ = ['FileDialog']

class FileDialog(QtWidgets.QFileDialog):

    def __init__(self, *args):
        if False:
            return 10
        QtWidgets.QFileDialog.__init__(self, *args)
        if sys.platform == 'darwin':
            self.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)