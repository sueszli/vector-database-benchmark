import os
import numpy as np
import paddle
import paddle.distributed as dist

class SemiAutoParallelTestBase:

    def __init__(self):
        if False:
            return 10
        self._dtype = os.getenv('dtype')
        self._backend = os.getenv('backend')
        self._seed = eval(os.getenv('seed'))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=['x'])

    def check_tensor_eq(self, a, b):
        if False:
            print('Hello World!')
        np1 = a.numpy()
        np2 = b.numpy()
        np.testing.assert_allclose(np1, np2, rtol=1e-05, verbose=True)

    def flatten(self, inputs, terminal_cond):
        if False:
            print('Hello World!')
        '\n        inputs may be single tensor、tuple\n        '
        if terminal_cond(inputs):
            return ([inputs], 'i')
        assert isinstance(inputs, (tuple, list))
        flattened = []
        structure = []
        for i in range(len(inputs)):
            (tmp, tmp_structure) = self.flatten(inputs[i], terminal_cond)
            flattened.extend(tmp)
            structure.append(tmp_structure)
        if isinstance(inputs, tuple):
            structure = tuple(structure)
        return (flattened, structure)

    def unflatten(self, inputs, structure, offset=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        inputs may be single tensor\n        '
        assert isinstance(inputs, list)
        assert offset < len(inputs)
        if structure == 'i':
            offset = offset + 1
            return (inputs[offset - 1], offset)
        assert isinstance(structure, (tuple, list))
        unflattened = []
        for i in range(len(structure)):
            (tmp, offset) = self.unflatten(inputs, structure[i], offset)
            unflattened.append(tmp)
        if isinstance(structure, tuple):
            unflattened = tuple(unflattened)
        return (unflattened, offset)

    def runfunc_and_check(self, inputs_shape, inputs_specs, op_func, with_backward, **kwargs):
        if False:
            print('Hello World!')
        paddle.seed(self._seed)
        np.random.seed(self._seed)
        flat_inputs = []
        flat_dist_inputs = []

        def terminal_cond(x):
            if False:
                i = 10
                return i + 15
            return isinstance(x, list) and all((not isinstance(e, (list, tuple)) for e in x))
        (flat_inputs_specs, inputs_structure) = self.flatten(inputs_specs, terminal_cond)
        (flat_inputs_shape, _) = self.flatten(inputs_shape, terminal_cond)
        assert len(flat_inputs_specs) == len(flat_inputs_shape)
        for (shape, spec) in zip(flat_inputs_shape, flat_inputs_specs):
            input_np = np.random.random(size=shape).astype(self._dtype)
            input = paddle.to_tensor(input_np)
            input.stop_gradient = False
            input_dist_attr = dist.DistAttr(mesh=self._mesh, sharding_specs=spec)
            dist_input = dist.shard_tensor(input, dist_attr=input_dist_attr)
            dist_input.stop_gradient = False
            flat_inputs.append(input)
            flat_dist_inputs.append(dist_input)
        (inputs, _) = self.unflatten(flat_inputs, inputs_structure)
        (dist_inputs, _) = self.unflatten(flat_dist_inputs, inputs_structure)

        def wrap_tuple(e):
            if False:
                print('Hello World!')
            return e if isinstance(e, tuple) else (e,)
        op_inputs = wrap_tuple(inputs)
        op_dist_input = wrap_tuple(dist_inputs)
        out = op_func(*op_inputs, **kwargs)
        dist_out = op_func(*op_dist_input, **kwargs)
        if with_backward:

            def terminal_cond2(x):
                if False:
                    while True:
                        i = 10
                return not isinstance(x, (list, tuple))
            (flat_out, _) = self.flatten(out, terminal_cond2)
            (flat_dist_out, _) = self.flatten(dist_out, terminal_cond2)
            assert len(flat_out) == len(flat_dist_out)
            for (output, dist_output) in zip(flat_out, flat_dist_out):
                self.check_tensor_eq(out, dist_out)
                output.backward()
                dist_output.backward()
            for (x, dist_x) in zip(flat_inputs, flat_dist_inputs):
                self.check_tensor_eq(x.grad, dist_x.grad)
        return (dist_inputs, dist_out)