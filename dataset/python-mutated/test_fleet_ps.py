import unittest
from paddle.base.framework import default_main_program
from paddle.incubate.distributed.fleet.parameter_server.ir.pserver_pass import _get_optimizer_input_shape
main_program = default_main_program()

class TestFleetPS(unittest.TestCase):

    def test_version(self):
        if False:
            while True:
                i = 10
        from paddle.incubate.distributed.fleet.parameter_server import version
        transpiler = version.is_transpiler()
        self.assertEqual(transpiler, True)

    def test_optimizer_shape(self):
        if False:
            for i in range(10):
                print('nop')
        optimizers = []
        optimizers.append(('adam', 'Moment1', [100, 1], [50, 1]))
        optimizers.append(('adam', 'Moment2', [100, 1], [50, 1]))
        optimizers.append(('adagrad', 'Moment', [100, 1], [50, 1]))
        optimizers.append(('adamax', 'Moment', [100, 1], [50, 1]))
        optimizers.append(('adamax', 'InfNorm', [100, 1], [50, 1]))
        optimizers.append(('momentum', 'Velocity', [100, 1], [50, 1]))
        optimizers.append(('lars_momentum', 'Velocity', [100, 1], [50, 1]))
        optimizers.append(('decayed_adagrad', 'Moment', [100, 1], [50, 1]))
        optimizers.append(('rmsprop', 'Moment', [100, 1], [50, 1]))
        optimizers.append(('rmsprop', 'MeanSquare', [100, 1], [50, 1]))
        optimizers.append(('ftrl', 'SquaredAccumulator', [100, 1], [50, 1]))
        optimizers.append(('ftrl', 'LinearAccumulator', [100, 1], [50, 1]))
        for attrs in optimizers:
            (op_type, varkey, orig_shape, param_shape) = attrs
            new_shape = _get_optimizer_input_shape(op_type, varkey, orig_shape, param_shape)
            self.assertListEqual(new_shape, param_shape)
        optimizers = []
        optimizers.append(('sgd', '', [100, 1], [50, 1]))
        for attrs in optimizers:
            (op_type, varkey, orig_shape, param_shape) = attrs
            new_shape = _get_optimizer_input_shape(op_type, varkey, orig_shape, param_shape)
            self.assertListEqual(new_shape, orig_shape)
        with self.assertRaises(ValueError):
            optimizers = []
            optimizers.append(('new_opti', '', [100, 1], [50, 1]))
            for attrs in optimizers:
                (op_type, varkey, orig_shape, param_shape) = attrs
                _get_optimizer_input_shape(op_type, varkey, orig_shape, param_shape)
if __name__ == '__main__':
    unittest.main()