import unittest
import numpy as np
from op import Operator
from op_test import OpTest, check_out_dtype, paddle_static_guard, skip_check_grad_ci
import paddle
import paddle.nn.functional as F
from paddle.base import Program, core, program_guard

class TestLookupTableOp(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'lookup_table'
        table = np.random.random((17, 31)).astype('float64')
        ids = np.random.randint(0, 17, 4).astype('int64')
        ids_expand = np.expand_dims(ids, axis=1)
        self.inputs = {'W': table, 'Ids': ids_expand}
        self.outputs = {'Out': table[ids]}

    def test_check_output(self):
        if False:
            return 10
        self.check_output(check_cinn=True)

    def test_check_grad(self):
        if False:
            print('Hello World!')
        self.check_grad(['W'], 'Out', no_grad_set=set('Ids'), check_cinn=True)

class TestLookupTableOpWithTensorIds(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'lookup_table'
        table = np.random.random((17, 31)).astype('float64')
        ids = np.random.randint(low=0, high=17, size=(2, 4, 5, 1)).astype('int64')
        self.inputs = {'W': table, 'Ids': ids}
        self.outputs = {'Out': table[ids.flatten()].reshape((2, 4, 5, 31))}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_cinn=True)

    def test_check_grad(self):
        if False:
            while True:
                i = 10
        self.check_grad(['W'], 'Out', no_grad_set=set('Ids'), check_cinn=True)

@skip_check_grad_ci(reason="Since paddings are not trainable and fixed in forward,the gradient of paddings makes no sense and we don't test the gradient here.")
class TestLookupTableOpWithPadding(TestLookupTableOp):

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        ids = np.squeeze(self.inputs['Ids'])
        padding_idx = np.random.choice(ids, 1)[0]
        self.outputs['Out'][ids == padding_idx] = np.zeros(31)
        self.attrs = {'padding_idx': int(padding_idx)}
        self.check_output(check_cinn=True)

@skip_check_grad_ci(reason="Since paddings are not trainable and fixed in forward,the gradient of paddings makes no sense and we don't test the gradient here.")
class TestLookupTableOpWithTensorIdsAndPadding(TestLookupTableOpWithTensorIds):

    def test_check_output(self):
        if False:
            while True:
                i = 10
        ids = self.inputs['Ids']
        flatten_idx = ids.flatten()
        padding_idx = np.random.choice(flatten_idx, 1)[0]
        self.outputs['Out'][np.squeeze(ids == padding_idx)] = np.zeros(31)
        self.attrs = {'padding_idx': padding_idx}
        self.check_output(check_cinn=True)

class TestLookupTableWIsSelectedRows(unittest.TestCase):

    def prepare_ids(self, scope, place):
        if False:
            return 10
        ids_tensor = scope.var('Ids').get_tensor()
        ids_array = np.array([[0], [4], [3], [5]]).astype('int64')
        ids_tensor.set(ids_array, place)
        return ids_array

    def prepare_w(self, scope, place):
        if False:
            print('Hello World!')
        rows = [0, 1, 2, 3, 4, 5, 6]
        row_numel = 12
        w_selected_rows = scope.var('W').get_selected_rows()
        w_selected_rows.set_height(len(rows))
        w_selected_rows.set_rows(rows)
        w_array = np.ones((len(rows), row_numel)).astype('float32')
        for i in range(len(rows)):
            w_array[i] *= i
        w_tensor = w_selected_rows.get_tensor()
        w_tensor.set(w_array, place)

    def create_out_tensor(self, scope, place):
        if False:
            print('Hello World!')
        return scope.var('Out').get_tensor()

    def check_result(self, ids_array, result_array):
        if False:
            i = 10
            return i + 15
        for (idx, row) in enumerate(ids_array):
            assert (row[0] == result_array[idx]).all()

    def check_with_place(self, place):
        if False:
            for i in range(10):
                print('nop')
        scope = core.Scope()
        ids_array = self.prepare_ids(scope, place)
        self.prepare_w(scope, place)
        out_tensor = self.create_out_tensor(scope, place)
        lookup_table = Operator('lookup_table', W='W', Ids='Ids', Out='Out')
        lookup_table.run(scope, place)
        result_array = np.array(out_tensor)
        self.check_result(ids_array, result_array)

    def test_w_is_selected_rows(self):
        if False:
            for i in range(10):
                print('nop')
        places = [core.CPUPlace()]
        for place in places:
            self.check_with_place(place)

class TestLookupTableWithTensorIdsWIsSelectedRows(TestLookupTableWIsSelectedRows):

    def prepare_ids(self, scope, place):
        if False:
            while True:
                i = 10
        ids_tensor = scope.var('Ids').get_tensor()
        ids_array = np.random.randint(low=0, high=6, size=(2, 4, 3, 1)).astype('int64')
        ids_tensor.set(ids_array, place)
        return ids_array

    def check_result(self, ids_array, result_array):
        if False:
            for i in range(10):
                print('nop')
        for (idx, row) in np.ndenumerate(ids_array):
            assert (row == result_array[idx]).all()

class TestEmbedOpError(unittest.TestCase):

    def test_errors(self):
        if False:
            return 10
        with paddle_static_guard():
            with program_guard(Program(), Program()):
                input_data = np.random.randint(0, 10, (4, 1)).astype('int64')

                def test_Variable():
                    if False:
                        i = 10
                        return i + 15
                    paddle.static.nn.embedding(input=input_data, size=(10, 64))
                self.assertRaises(TypeError, test_Variable)

                def test_input_dtype():
                    if False:
                        return 10
                    input = paddle.static.data(name='x', shape=[4, 1], dtype='float32')
                    paddle.static.nn.embedding(input=input, size=(10, 64))
                self.assertRaises(TypeError, test_input_dtype)

                def test_param_dtype():
                    if False:
                        i = 10
                        return i + 15
                    input2 = paddle.static.data(name='x2', shape=[4, 1], dtype='int64')
                    paddle.static.nn.embedding(input=input2, size=(10, 64), dtype='int64')
                self.assertRaises(TypeError, test_param_dtype)
                input3 = paddle.static.data(name='x3', shape=[4, 1], dtype='int64')
                paddle.static.nn.embedding(input=input3, size=(10, 64), dtype='float16')

class TestLookupTableOpInt8(OpTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.op_type = 'lookup_table'
        table = np.random.randint(low=-128, high=127, size=(17, 31)).astype('int8')
        ids = np.random.randint(0, 17, 4).astype('int64')
        ids_expand = np.expand_dims(ids, axis=1)
        self.inputs = {'W': table, 'Ids': ids_expand}
        self.outputs = {'Out': table[ids]}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_cinn=True)

    def test_check_grad(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class TestLookupTableOpWithTensorIdsInt8(OpTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.op_type = 'lookup_table'
        table = np.random.randint(low=-128, high=127, size=(17, 31)).astype('int8')
        ids = np.random.randint(low=0, high=17, size=(2, 4, 5, 1)).astype('int64')
        self.inputs = {'W': table, 'Ids': ids}
        self.outputs = {'Out': table[ids.flatten()].reshape((2, 4, 5, 31))}

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_output(check_cinn=True)

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        pass

class TestLookupTableOpWithPaddingInt8(TestLookupTableOpInt8):

    def test_check_output(self):
        if False:
            return 10
        ids = np.squeeze(self.inputs['Ids'])
        padding_idx = np.random.choice(ids, 1)[0]
        self.outputs['Out'][ids == padding_idx] = np.zeros(31)
        self.attrs = {'padding_idx': int(padding_idx)}
        self.check_output(check_cinn=True)

    def test_check_grad(self):
        if False:
            i = 10
            return i + 15
        pass

class TestLookupTableOpWithTensorIdsAndPaddingInt8(TestLookupTableOpWithTensorIdsInt8):

    def test_check_output(self):
        if False:
            for i in range(10):
                print('nop')
        ids = self.inputs['Ids']
        flatten_idx = ids.flatten()
        padding_idx = np.random.choice(flatten_idx, 1)[0]
        self.outputs['Out'][np.squeeze(ids == padding_idx)] = np.zeros(31)
        self.attrs = {'padding_idx': padding_idx}
        self.check_output(check_cinn=True)

    def test_check_grad(self):
        if False:
            print('Hello World!')
        pass

class TestLookupTableWIsSelectedRowsInt8(unittest.TestCase):

    def prepare_ids(self, scope, place):
        if False:
            print('Hello World!')
        ids_tensor = scope.var('Ids').get_tensor()
        ids_array = np.array([[0], [4], [3], [5]]).astype('int64')
        ids_tensor.set(ids_array, place)
        return ids_array

    def prepare_w(self, scope, place):
        if False:
            print('Hello World!')
        rows = [0, 1, 2, 3, 4, 5, 6]
        row_numel = 12
        w_selected_rows = scope.var('W').get_selected_rows()
        w_selected_rows.set_height(len(rows))
        w_selected_rows.set_rows(rows)
        w_array = np.ones((len(rows), row_numel)).astype('int8')
        for i in range(len(rows)):
            w_array[i] *= i
        w_tensor = w_selected_rows.get_tensor()
        w_tensor.set(w_array, place)

    def create_out_tensor(self, scope, place):
        if False:
            i = 10
            return i + 15
        return scope.var('Out').get_tensor()

    def check_result(self, ids_array, result_array):
        if False:
            return 10
        for (idx, row) in enumerate(ids_array):
            assert (row[0] == result_array[idx]).all()

    def check_with_place(self, place):
        if False:
            print('Hello World!')
        scope = core.Scope()
        ids_array = self.prepare_ids(scope, place)
        self.prepare_w(scope, place)
        out_tensor = self.create_out_tensor(scope, place)
        lookup_table = Operator('lookup_table', W='W', Ids='Ids', Out='Out')
        lookup_table.run(scope, place)
        result_array = np.array(out_tensor)
        self.check_result(ids_array, result_array)

    def test_w_is_selected_rows(self):
        if False:
            while True:
                i = 10
        places = [core.CPUPlace()]
        for place in places:
            self.check_with_place(place)

class TestLookupTableWithTensorIdsWIsSelectedRowsInt8(TestLookupTableWIsSelectedRowsInt8):

    def prepare_ids(self, scope, place):
        if False:
            while True:
                i = 10
        ids_tensor = scope.var('Ids').get_tensor()
        ids_array = np.random.randint(low=0, high=6, size=(2, 4, 3, 1)).astype('int64')
        ids_tensor.set(ids_array, place)
        return ids_array

    def check_result(self, ids_array, result_array):
        if False:
            for i in range(10):
                print('nop')
        for (idx, row) in np.ndenumerate(ids_array):
            assert (row == result_array[idx]).all()

@skip_check_grad_ci(reason='Int16 type only be used in test and inference.')
class TestLookupTableOpInt16(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'lookup_table'
        table = np.random.randint(low=-128, high=127, size=(17, 31)).astype('int16')
        ids = np.random.randint(0, 17, 4).astype('int64')
        ids_expand = np.expand_dims(ids, axis=1)
        self.inputs = {'W': table, 'Ids': ids_expand}
        self.outputs = {'Out': table[ids]}

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        self.check_output(check_cinn=True)

@skip_check_grad_ci(reason='Int16 type only be used in test and inference.')
class TestLookupTableOpWithTensorIdsInt16(OpTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.op_type = 'lookup_table'
        table = np.random.randint(low=-128, high=127, size=(17, 31)).astype('int16')
        ids = np.random.randint(low=0, high=17, size=(2, 4, 5, 1)).astype('int64')
        self.inputs = {'W': table, 'Ids': ids}
        self.outputs = {'Out': table[ids.flatten()].reshape((2, 4, 5, 31))}

    def test_check_output(self):
        if False:
            print('Hello World!')
        self.check_output(check_cinn=True)

@skip_check_grad_ci(reason='Int16 type only be used in test and inference.')
class TestLookupTableOpWithPaddingInt16(TestLookupTableOpInt16):

    def test_check_output(self):
        if False:
            i = 10
            return i + 15
        ids = np.squeeze(self.inputs['Ids'])
        padding_idx = np.random.choice(ids, 1)[0]
        self.outputs['Out'][ids == padding_idx] = np.zeros(31)
        self.attrs = {'padding_idx': int(padding_idx)}
        self.check_output(check_cinn=True)

@skip_check_grad_ci(reason='Int16 type only be used in test and inference.')
class TestLookupTableOpWithTensorIdsAndPaddingInt16(TestLookupTableOpWithTensorIdsInt16):

    def test_check_output(self):
        if False:
            print('Hello World!')
        ids = self.inputs['Ids']
        flatten_idx = ids.flatten()
        padding_idx = np.random.choice(flatten_idx, 1)[0]
        self.outputs['Out'][np.squeeze(ids == padding_idx)] = np.zeros(31)
        self.attrs = {'padding_idx': padding_idx}
        self.check_output(check_cinn=True)

class TestLookupTableWIsSelectedRowsInt16(unittest.TestCase):

    def prepare_ids(self, scope, place):
        if False:
            i = 10
            return i + 15
        ids_tensor = scope.var('Ids').get_tensor()
        ids_array = np.array([[0], [4], [3], [5]]).astype('int64')
        ids_tensor.set(ids_array, place)
        return ids_array

    def prepare_w(self, scope, place):
        if False:
            for i in range(10):
                print('nop')
        rows = [0, 1, 2, 3, 4, 5, 6]
        row_numel = 12
        w_selected_rows = scope.var('W').get_selected_rows()
        w_selected_rows.set_height(len(rows))
        w_selected_rows.set_rows(rows)
        w_array = np.ones((len(rows), row_numel)).astype('int16')
        for i in range(len(rows)):
            w_array[i] *= i
        w_tensor = w_selected_rows.get_tensor()
        w_tensor.set(w_array, place)

    def create_out_tensor(self, scope, place):
        if False:
            while True:
                i = 10
        return scope.var('Out').get_tensor()

    def check_result(self, ids_array, result_array):
        if False:
            print('Hello World!')
        for (idx, row) in enumerate(ids_array):
            assert (row[0] == result_array[idx]).all()

    def check_with_place(self, place):
        if False:
            return 10
        scope = core.Scope()
        ids_array = self.prepare_ids(scope, place)
        self.prepare_w(scope, place)
        out_tensor = self.create_out_tensor(scope, place)
        lookup_table = Operator('lookup_table', W='W', Ids='Ids', Out='Out')
        lookup_table.run(scope, place)
        result_array = np.array(out_tensor)
        self.check_result(ids_array, result_array)

    def test_w_is_selected_rows(self):
        if False:
            for i in range(10):
                print('nop')
        places = [core.CPUPlace()]
        for place in places:
            self.check_with_place(place)

class TestLookupTableWithTensorIdsWIsSelectedRowsInt16(TestLookupTableWIsSelectedRowsInt16):

    def prepare_ids(self, scope, place):
        if False:
            i = 10
            return i + 15
        ids_tensor = scope.var('Ids').get_tensor()
        ids_array = np.random.randint(low=0, high=6, size=(2, 4, 3, 1)).astype('int64')
        ids_tensor.set(ids_array, place)
        return ids_array

    def check_result(self, ids_array, result_array):
        if False:
            print('Hello World!')
        for (idx, row) in np.ndenumerate(ids_array):
            assert (row == result_array[idx]).all()

class TestOutDtype(unittest.TestCase):

    def test_dtype(self):
        if False:
            print('Hello World!')
        api_fn = F.embedding
        check_out_dtype(api_fn, in_specs=[([10, 16], 'int64'), ([100, 64],)], expect_dtypes=['float32', 'float64'], target_index=1)
if __name__ == '__main__':
    unittest.main()