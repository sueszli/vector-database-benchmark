from PyInstaller import isolated
hiddenimports = ['PySide2.QtCore', 'PySide2.QtWidgets', 'PySide2.QtGui', 'PySide2.QtSvg']

@isolated.decorate
def conditional_imports():
    if False:
        return 10
    from PySide2 import Qwt5
    out = []
    if hasattr(Qwt5, 'toNumpy'):
        out.append('numpy')
    if hasattr(Qwt5, 'toNumeric'):
        out.append('numeric')
    if hasattr(Qwt5, 'toNumarray'):
        out.append('numarray')
    return out
hiddenimports += conditional_imports()