import unittest
from pass_test import PassTest

class TestTransposeFoldingOutputPass(PassTest):

    def init_input_data(self):
        if False:
            print('Hello World!')
        'Do not set the shape like [B, N, N].\n        You should set the shape like [B, M, N], where M != N.\n        '
        self.feed_data = {'x': self.random([4, 3, 5], 'float32'), 'y': self.random([4, 5, 6], 'float32')}

    def expect_folding_number(self):
        if False:
            for i in range(10):
                print('nop')
        return 1

    def trans_out_func(self, builder, out):
        if False:
            print('Hello World!')
        return builder.transpose(out, [0, 2, 1])

    def build_program(self, builder, target):
        if False:
            print('Hello World!')
        x = builder.create_input(str(self.feed_data['x'].dtype), self.feed_data['x'].shape, 'x')
        y = builder.create_input(str(self.feed_data['y'].dtype), self.feed_data['y'].shape, 'y')
        res = builder.matmul(x, y)
        out = self.trans_out_func(builder, res)
        return ([x, y], [out])

    def test_check_results(self):
        if False:
            return 10
        self.check_pass_outputs(pass_diff=self.expect_folding_number(), test_passes=['TransposeFoldingInput', 'GemmRewriter', 'TransposeFoldingOutput', 'GemmRewriter'])

class TestTransposeFoldingOutputPassWithScale(TestTransposeFoldingOutputPass):

    def expect_folding_number(self):
        if False:
            for i in range(10):
                print('nop')
        return 2

    def trans_out_func(self, builder, out):
        if False:
            return 10
        out_s = builder.scale(out, scale=2.0)
        return builder.transpose(out_s, [0, 2, 1])

class TestTransposeFoldingOutputPassWithIdentity(TestTransposeFoldingOutputPass):

    def expect_folding_number(self):
        if False:
            i = 10
            return i + 15
        return 2

    def trans_out_func(self, builder, out):
        if False:
            return 10
        out_i = builder.identity(out)
        out_s = builder.scale(out_i, scale=2.0)
        return builder.transpose(out_s, [0, 2, 1])

class TestTransposeFoldingOutputPassInvlidTrans(TestTransposeFoldingOutputPass):

    def expect_folding_number(self):
        if False:
            i = 10
            return i + 15
        return 1

    def trans_out_func(self, builder, out):
        if False:
            print('Hello World!')
        out_t = builder.transpose(out, [1, 0, 2])
        return builder.scale(out_t, scale=2.0)

class TestTransposeFoldingOutputPassInvlidScale(TestTransposeFoldingOutputPass):

    def expect_folding_number(self):
        if False:
            while True:
                i = 10
        return 1

    def trans_out_func(self, builder, out):
        if False:
            for i in range(10):
                print('nop')
        out_s = builder.scale(out, scale=2.0, bias=1.0)
        return builder.transpose(out_s, [0, 2, 1])

class TestTransposeFoldingOutputPassNoFold(TestTransposeFoldingOutputPass):

    def expect_folding_number(self):
        if False:
            while True:
                i = 10
        return 0

    def trans_out_func(self, builder, out):
        if False:
            return 10
        out_r = builder.reshape(out, [4, 6, 3])
        out_s = builder.scale(out_r, scale=2.0)
        return builder.transpose(out_s, [0, 2, 1])
if __name__ == '__main__':
    unittest.main()