import unittest
from paddle import base

class VarInfo:

    def __init__(self, var_name, var_type, writable):
        if False:
            print('Hello World!')
        self.name = var_name
        self.type = var_type
        self.writable = writable

class TestGlobalVarGetterSetter(unittest.TestCase):

    def test_main(self):
        if False:
            return 10
        var_infos = [VarInfo('FLAGS_free_idle_chunk', bool, False), VarInfo('FLAGS_eager_delete_tensor_gb', float, True)]
        g = base.core.globals()
        for var in var_infos:
            self.assertTrue(var.name in g)
            self.assertTrue(var.name in g.keys())
            value1 = g[var.name]
            value2 = g.get(var.name, None)
            self.assertIsNotNone(value1)
            self.assertEqual(value1, value2)
            self.assertEqual(type(value1), var.type)
            self.assertEqual(type(value2), var.type)
            if var.writable:
                g[var.name] = -1
            else:
                try:
                    g[var.name] = False
                    self.assertTrue(False)
                except:
                    self.assertTrue(True)
        name = '__any_non_exist_name__'
        self.assertFalse(name in g)
        self.assertFalse(name in g.keys())
        self.assertIsNone(g.get(name, None))
        self.assertEqual(g.get(name, -1), -1)
if __name__ == '__main__':
    unittest.main()