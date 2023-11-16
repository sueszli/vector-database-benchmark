from PyInstaller import isolated
from PyInstaller import compat
from PyInstaller.utils import hooks as hookutils

@isolated.decorate
def mpl_data_dir():
    if False:
        i = 10
        return i + 15
    import matplotlib
    return matplotlib.get_data_path()
datas = [(mpl_data_dir(), 'matplotlib/mpl-data')]
binaries = []
if compat.is_win and hookutils.check_requirement('matplotlib >= 3.7.0'):
    (delvewheel_datas, delvewheel_binaries) = hookutils.collect_delvewheel_libs_directory('matplotlib')
    datas += delvewheel_datas
    binaries += delvewheel_binaries