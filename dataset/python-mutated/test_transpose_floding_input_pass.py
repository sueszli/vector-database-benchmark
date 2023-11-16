import unittest
from pass_test import PassTest

class TestTransposeFoldingInputPass(PassTest):

    def init_input_data(self):
        if False:
            return 10
        'Do not set the shape like [B, N, N].\n        You should set the shape like [B, M, N], where M != N.\n        '
        self.feed_data = {'x': self.random([4, 5, 3], 'float32'), 'y': self.random([4, 5, 6], 'float32')}
        self.folded_num = 1

    def trans_x_func(self, builder, x):
        if False:
            while True:
                i = 10
        return builder.transpose(x, [0, 2, 1])

    def trans_y_func(self, builder, y):
        if False:
            return 10
        return y

    def build_program(self, builder, target):
        if False:
            while True:
                i = 10
        x = builder.create_input(str(self.feed_data['x'].dtype), self.feed_data['x'].shape, 'x')
        y = builder.create_input(str(self.feed_data['y'].dtype), self.feed_data['y'].shape, 'y')
        x_t = self.trans_x_func(builder, x)
        y_t = self.trans_y_func(builder, y)
        out = builder.matmul(x_t, y_t)
        return ([x, y], [out])

    def test_check_results(self):
        if False:
            i = 10
            return i + 15
        self.check_pass_outputs(pass_diff=self.folded_num, test_passes=['TransposeFoldingInput', 'GemmRewriter'])

class TestTransposeFoldingInputPassTransY(TestTransposeFoldingInputPass):

    def init_input_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.feed_data = {'x': self.random([4, 3, 5], 'float32'), 'y': self.random([4, 6, 5], 'float32')}
        self.folded_num = 1

    def trans_x_func(self, builder, x):
        if False:
            i = 10
            return i + 15
        return x

    def trans_y_func(self, builder, y):
        if False:
            return 10
        return builder.transpose(y, [0, 2, 1])

class TestTransposeFoldingInputPassTransXY(TestTransposeFoldingInputPass):

    def init_input_data(self):
        if False:
            print('Hello World!')
        self.feed_data = {'x': self.random([4, 5, 3], 'float32'), 'y': self.random([4, 6, 5], 'float32')}
        self.folded_num = 2

    def trans_x_func(self, builder, x):
        if False:
            return 10
        return builder.transpose(x, [0, 2, 1])

    def trans_y_func(self, builder, y):
        if False:
            print('Hello World!')
        return builder.transpose(y, [0, 2, 1])

class TestTransposeFoldingInputPassWithScale(TestTransposeFoldingInputPass):

    def init_input_data(self):
        if False:
            while True:
                i = 10
        self.feed_data = {'x': self.random([4, 5, 3], 'float32'), 'y': self.random([1, 6, 5], 'float32')}
        self.folded_num = 4

    def trans_x_func(self, builder, x):
        if False:
            i = 10
            return i + 15
        x_s = builder.scale(x, scale=2.0)
        return builder.transpose(x_s, [0, 2, 1])

    def trans_y_func(self, builder, y):
        if False:
            print('Hello World!')
        y_s = builder.scale(y, scale=2.0)
        return builder.transpose(y_s, [0, 2, 1])

class TestTransposeFoldingInputPassWithIdentity(TestTransposeFoldingInputPass):

    def init_input_data(self):
        if False:
            print('Hello World!')
        self.feed_data = {'x': self.random([4, 5, 3], 'float32'), 'y': self.random([1, 6, 5], 'float32')}
        self.folded_num = 4

    def trans_x_func(self, builder, x):
        if False:
            for i in range(10):
                print('nop')
        x_s = builder.scale(x, scale=2.0)
        x_t = builder.transpose(x_s, [0, 2, 1])
        return builder.identity(x_t)

    def trans_y_func(self, builder, y):
        if False:
            i = 10
            return i + 15
        y_s = builder.scale(y, scale=2.0)
        y_t = builder.transpose(y_s, [0, 2, 1])
        return builder.identity(y_t)

class TestTransposeFoldingInputPassWithBroadcastX(TestTransposeFoldingInputPass):

    def init_input_data(self):
        if False:
            while True:
                i = 10
        self.feed_data = {'x': self.random([5, 3], 'float32'), 'y': self.random([4, 6, 5], 'float32')}
        self.folded_num = 5

    def trans_x_func(self, builder, x):
        if False:
            print('Hello World!')
        x_b = builder.broadcast_to(x, [4, 5, 3])
        x_s = builder.scale(x_b, scale=2.0)
        x_t = builder.transpose(x_s, [0, 2, 1])
        return builder.identity(x_t)

    def trans_y_func(self, builder, y):
        if False:
            return 10
        y_s = builder.scale(y, scale=2.0)
        y_t = builder.transpose(y_s, [0, 2, 1])
        return builder.identity(y_t)

class TestTransposeFoldingInputPassWithBroadcastXY(TestTransposeFoldingInputPass):

    def init_input_data(self):
        if False:
            while True:
                i = 10
        self.feed_data = {'x': self.random([5, 3], 'float32'), 'y': self.random([6, 5], 'float32')}
        self.folded_num = 5

    def trans_x_func(self, builder, x):
        if False:
            print('Hello World!')
        x_b = builder.broadcast_to(x, [4, 5, 3])
        x_s = builder.scale(x_b, scale=2.0)
        x_t = builder.transpose(x_s, [0, 2, 1])
        return builder.identity(x_t)

    def trans_y_func(self, builder, y):
        if False:
            i = 10
            return i + 15
        y_b = builder.broadcast_to(y, [4, 6, 5])
        y_s = builder.scale(y_b, scale=2.0)
        y_t = builder.transpose(y_s, [0, 2, 1])
        return builder.identity(y_t)

class TestTransposeFoldingInputPassWithBroadcastAfterTrans(TestTransposeFoldingInputPass):

    def init_input_data(self):
        if False:
            while True:
                i = 10
        self.feed_data = {'x': self.random([5, 3], 'float32'), 'y': self.random([4, 6, 5], 'float32')}
        self.folded_num = 2

    def trans_x_func(self, builder, x):
        if False:
            return 10
        x_t = builder.transpose(x, [1, 0])
        x_b = builder.broadcast_to(x_t, [4, 3, 5])
        return builder.identity(x_b)

    def trans_y_func(self, builder, y):
        if False:
            i = 10
            return i + 15
        y_t = builder.transpose(y, [0, 2, 1])
        return builder.identity(y_t)

class TestTransposeFoldingInputPassInvalidTran(TestTransposeFoldingInputPass):

    def init_input_data(self):
        if False:
            while True:
                i = 10
        self.feed_data = {'x': self.random([3, 4, 5], 'float32'), 'y': self.random([5, 4, 6], 'float32')}
        self.folded_num = 2

    def trans_x_func(self, builder, x):
        if False:
            i = 10
            return i + 15
        x_s = builder.scale(x, scale=2.0)
        x_t = builder.transpose(x_s, [1, 0, 2])
        return builder.identity(x_t)

    def trans_y_func(self, builder, y):
        if False:
            print('Hello World!')
        y_s = builder.scale(y, scale=2.0)
        y_t = builder.transpose(y_s, [1, 0, 2])
        return builder.identity(y_t)

class TestTransposeFoldingInputPassInvalidScale(TestTransposeFoldingInputPass):

    def init_input_data(self):
        if False:
            while True:
                i = 10
        self.feed_data = {'x': self.random([4, 5, 3], 'float32'), 'y': self.random([1, 6, 5], 'float32')}
        self.folded_num = 2

    def trans_x_func(self, builder, x):
        if False:
            print('Hello World!')
        x_s = builder.scale(x, scale=2.0, bias=1.0)
        x_t = builder.transpose(x_s, [0, 2, 1])
        return builder.identity(x_t)

    def trans_y_func(self, builder, y):
        if False:
            return 10
        y_s = builder.scale(y, scale=2.0, bias=1.0)
        y_t = builder.transpose(y_s, [0, 2, 1])
        return builder.identity(y_t)

class TestTransposeFoldingInputPassNoFold(TestTransposeFoldingInputPass):

    def init_input_data(self):
        if False:
            for i in range(10):
                print('nop')
        self.feed_data = {'x': self.random([4, 5, 3], 'float32'), 'y': self.random([4, 6, 5], 'float32')}
        self.folded_num = 0

    def trans_x_func(self, builder, x):
        if False:
            print('Hello World!')
        x_s = builder.scale(x, scale=2.0)
        x_t = builder.transpose(x_s, [0, 2, 1])
        return builder.reshape(x_t, [2, 6, 5])

    def trans_y_func(self, builder, y):
        if False:
            for i in range(10):
                print('nop')
        y_s = builder.scale(y, scale=2.0)
        y_t = builder.transpose(y_s, [0, 2, 1])
        return builder.reshape(y_t, [2, 5, 12])
if __name__ == '__main__':
    unittest.main()