import os
from perspective import Table

class TestToArrowLZ4(object):

    def test_to_arrow_lz4_roundtrip(self, superstore):
        if False:
            print('Hello World!')
        original_tbl = Table(superstore)
        arrow_uncompressed = original_tbl.view().to_arrow(compression=None)
        tbl = Table(arrow_uncompressed)
        arr = tbl.view().to_arrow(compression='lz4')
        assert len(arr) < len(arrow_uncompressed)
        tbl2 = Table(arr)
        arr2 = tbl2.view().to_arrow(compression=None)
        assert len(arr2) > len(arr)
        tbl3 = Table(arr)
        arr3 = tbl3.view().to_arrow(compression='lz4')
        assert len(arr3) == len(arr)