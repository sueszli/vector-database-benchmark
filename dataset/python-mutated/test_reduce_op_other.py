from test_reduce_op_new import TestReduceAll

class TestReduceForBool(TestReduceAll):

    def init_attrs(self):
        if False:
            i = 10
            return i + 15
        super().init_attrs()
        self.dtypes = [{'dtype': 'bool'}]
        self.attrs = [{'op_type': 'all', 'keepdim': True}, {'op_type': 'all', 'keepdim': False}, {'op_type': 'any', 'keepdim': True}, {'op_type': 'any', 'keepdim': False}]

class TestReduceAxis(TestReduceAll):

    def init_attrs(self):
        if False:
            for i in range(10):
                print('nop')
        super().init_attrs()
        self.inputs = [{'shape': [1, 512, 1], 'axis': [1]}, {'shape': [1, 1024, 1], 'axis': [1]}, {'shape': [1, 2048, 1], 'axis': [1]}, {'shape': [64, 32, 16, 8, 4], 'axis': [0, 2]}, {'shape': [64, 32, 16, 8, 4], 'axis': [1, 2, 3]}, {'shape': [64, 32, 16, 8, 4], 'axis': []}]
        self.dtypes = [{'dtype': 'float32'}]
        self.attrs = [{'op_type': 'sum', 'keepdim': True}, {'op_type': 'sum', 'keepdim': False}]
if __name__ == '__main__':
    TestReduceForBool().run()
    TestReduceAxis().run()