def _pyi_rthook():
    if False:
        while True:
            i = 10
    import os
    import sys
    pyi_data_dir = os.path.join(sys._MEIPASS, 'share')
    xdg_data_dirs = os.environ.get('XDG_DATA_DIRS', None)
    if xdg_data_dirs:
        if pyi_data_dir not in xdg_data_dirs:
            xdg_data_dirs = pyi_data_dir + os.pathsep + xdg_data_dirs
    else:
        xdg_data_dirs = pyi_data_dir
    os.environ['XDG_DATA_DIRS'] = xdg_data_dirs
    del xdg_data_dirs
    del pyi_data_dir
_pyi_rthook()
del _pyi_rthook