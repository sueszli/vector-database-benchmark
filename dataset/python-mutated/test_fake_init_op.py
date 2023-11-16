import unittest
from op import Operator
from paddle.base import core

class TestFakeInitOpSelectedRows(unittest.TestCase):

    def check_with_place(self, place, is_selected_rows):
        if False:
            print('Hello World!')
        scope = core.Scope()
        out_var_name = 'Out'
        if is_selected_rows:
            out_tensor = scope.var(out_var_name).get_selected_rows().get_tensor()
        else:
            out_tensor = scope.var(out_var_name).get_tensor()
        var_shape = [4, 784]
        fake_init_op = Operator('fake_init', Out=out_var_name, shape=var_shape)
        fake_init_op.run(scope, place)
        self.assertEqual(var_shape, out_tensor._get_dims())

    def test_fake_init_selected_rows(self):
        if False:
            print('Hello World!')
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))
        for place in places:
            for is_selected_rows in [True, False]:
                self.check_with_place(place, is_selected_rows)
if __name__ == '__main__':
    unittest.main()