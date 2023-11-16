import pytest
pytest.importorskip('Pmw')
from direct.tkwidgets.EntryScale import EntryScale, EntryScaleGroup

def test_EntryScale(tk_toplevel):
    if False:
        return 10
    root = tk_toplevel
    root.title('Pmw EntryScale demonstration')

    def printVal(val):
        if False:
            for i in range(10):
                print('nop')
        print(val)
    mega1 = EntryScale(root, command=printVal)
    mega1.pack(side='left', expand=1, fill='x')
    group1 = EntryScaleGroup(root, dim=4, title='Simple RGBA Panel', labels=('R', 'G', 'B', 'A'), Valuator_min=0.0, Valuator_max=255.0, Valuator_resolution=1.0, command=printVal)