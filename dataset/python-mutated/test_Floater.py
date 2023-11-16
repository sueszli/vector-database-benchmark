import pytest
pytest.importorskip('Pmw')
from direct.tkwidgets.Floater import Floater, FloaterGroup

def test_Floater(tk_toplevel):
    if False:
        return 10
    root = tk_toplevel
    root.title('Pmw Floater demonstration')

    def printVal(val):
        if False:
            i = 10
            return i + 15
        print(val)
    mega1 = Floater(root, command=printVal)
    mega1.pack(side='left', expand=1, fill='x')
    group1 = FloaterGroup(root, dim=4, title='Simple RGBA Panel', labels=('R', 'G', 'B', 'A'), Valuator_min=0.0, Valuator_max=255.0, Valuator_resolution=1.0, command=printVal)