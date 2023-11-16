"""Show various QStandardPath paths."""
import os
import sys
from PyQt6.QtCore import QT_VERSION_STR, PYQT_VERSION_STR, qVersion, QStandardPaths, QCoreApplication

def print_header():
    if False:
        while True:
            i = 10
    'Show system information.'
    print('Python {}'.format(sys.version))
    print('os.name: {}'.format(os.name))
    print('sys.platform: {}'.format(sys.platform))
    print()
    print('Qt {}, compiled {}'.format(qVersion(), QT_VERSION_STR))
    print('PyQt {}'.format(PYQT_VERSION_STR))
    print()

def print_paths():
    if False:
        return 10
    'Print all QStandardPaths.StandardLocation members.'
    for (name, obj) in vars(QStandardPaths).items():
        if isinstance(obj, QStandardPaths.StandardLocation):
            location = QStandardPaths.writableLocation(obj)
            print('{:25} {}'.format(name, location))

def main():
    if False:
        return 10
    print_header()
    print('No QApplication')
    print('===============')
    print()
    print_paths()
    app = QCoreApplication(sys.argv)
    app.setApplicationName('qapp_name')
    print()
    print('With QApplication')
    print('=================')
    print()
    print_paths()
if __name__ == '__main__':
    main()