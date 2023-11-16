import numpy as np
import pyqtgraph as pg

def test_qimage_writethrough():
    if False:
        for i in range(10):
            print('nop')
    (w, h) = (256, 256)
    backstore = np.ones((h, w), dtype=np.uint8)
    ptr0 = backstore.ctypes.data
    fmt = pg.Qt.QtGui.QImage.Format.Format_Grayscale8
    qimg = pg.functions.ndarray_to_qimage(backstore, fmt)

    def get_pointer(obj):
        if False:
            print('Hello World!')
        if hasattr(obj, 'setsize'):
            return int(obj)
        else:
            return np.frombuffer(obj, dtype=np.uint8).ctypes.data
    ptr1 = get_pointer(qimg.constBits())
    assert ptr0 == ptr1
    ptr2 = get_pointer(qimg.bits())
    assert ptr1 == ptr2
    qimg.fill(0)
    assert np.all(backstore == 0)