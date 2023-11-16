import unittest
import paddle

class TestBlockRenameVar(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.program = paddle.static.Program()
        self.block = self.program.current_block()
        self.var = self.block.create_var(name='X', shape=[-1, 23, 48], dtype='float32')
        self.op = self.block.append_op(type='abs', inputs={'X': [self.var]}, outputs={'Out': [self.var]})
        self.new_var_name = self.get_new_var_name()

    def get_new_var_name(self):
        if False:
            while True:
                i = 10
        return 'Y'

    def test_rename_var(self):
        if False:
            print('Hello World!')
        self.block._rename_var(self.var.name, self.new_var_name)
        new_var_name_str = self.new_var_name if isinstance(self.new_var_name, str) else self.new_var_name.decode()
        self.assertTrue(new_var_name_str in self.block.vars)

class TestBlockRenameVarStrCase2(TestBlockRenameVar):

    def get_new_var_name(self):
        if False:
            i = 10
            return i + 15
        return 'ABC'

class TestBlockRenameVarBytes(TestBlockRenameVar):

    def get_new_var_name(self):
        if False:
            return 10
        return b'Y'
if __name__ == '__main__':
    unittest.main()