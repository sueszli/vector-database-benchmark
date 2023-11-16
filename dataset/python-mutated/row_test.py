from common import *

def test_row():
    if False:
        return 10
    ds = vaex.from_items(('x', [1, 2, 3]), ('y', [4, 5, 6]))
    assert ds[0] == [1, 4]
    ds['r'] = ds.x + ds.y
    assert ds[0] == [1, 4, 5]