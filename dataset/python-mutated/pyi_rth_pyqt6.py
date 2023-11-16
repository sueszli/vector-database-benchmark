def _pyi_rthook():
    if False:
        for i in range(10):
            print('nop')
    import os
    import sys
    from _pyi_rth_utils import is_macos_app_bundle
    pyqt_path = os.path.join(sys._MEIPASS, 'PyQt6', 'Qt6')
    if not os.path.isdir(pyqt_path):
        pyqt_path = os.path.join(sys._MEIPASS, 'PyQt6', 'Qt')
    os.environ['QT_PLUGIN_PATH'] = os.path.join(pyqt_path, 'plugins')
    if is_macos_app_bundle:
        pyqt_path_res = os.path.normpath(os.path.join(sys._MEIPASS, '..', 'Resources', os.path.relpath(pyqt_path, sys._MEIPASS)))
        os.environ['QML2_IMPORT_PATH'] = os.pathsep.join([os.path.join(pyqt_path_res, 'qml'), os.path.join(pyqt_path, 'qml')])
    else:
        os.environ['QML2_IMPORT_PATH'] = os.path.join(pyqt_path, 'qml')
    if sys.platform.startswith('win') and 'PATH' in os.environ:
        os.environ['PATH'] = sys._MEIPASS + os.pathsep + os.environ['PATH']
_pyi_rthook()
del _pyi_rthook