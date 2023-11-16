import unittest
import numpy as np
from op import Operator
from paddle.base import core

class TestMergeSelectedRows(unittest.TestCase):

    def get_places(self):
        if False:
            while True:
                i = 10
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        return places

    def check_with_place(self, place):
        if False:
            while True:
                i = 10
        scope = core.Scope()
        x_rows = [0, 5, 5, 4, 19]
        out_rows = [0, 4, 5, 19]
        height = 20
        row_numel = 2
        np_array = np.ones((len(x_rows), row_numel)).astype('float32')
        np_array[1, :] = 2.0
        np_array[2, :] = 3.0
        np_array[3, :] = 4.0
        x = scope.var('X').get_selected_rows()
        x.set_rows(x_rows)
        x.set_height(height)
        x_tensor = x.get_tensor()
        x_tensor.set(np_array, place)
        out = scope.var('Out').get_selected_rows()
        op = Operator('merge_selected_rows', X='X', Out='Out')
        op.run(scope, place)
        self.assertEqual(out.rows(), out_rows)
        self.assertEqual(out.height(), height)
        out_array = np.array(out.get_tensor())
        self.assertEqual((4, 2), out_array.shape)
        assert (out_array[0, :] == 1.0).all()
        assert (out_array[1, :] == 4.0).all()
        assert (out_array[2, :] == 5.0).all()
        assert (out_array[3, :] == 1.0).all()

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        for place in self.get_places():
            self.check_with_place(place)
if __name__ == '__main__':
    unittest.main()