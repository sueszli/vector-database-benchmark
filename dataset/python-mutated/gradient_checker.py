import os
import numpy as np
from caffe2.python import core, workspace, net_drawer
from caffe2.proto import caffe2_pb2

def getGradientForOp(op):
    if False:
        print('Hello World!')
    return core.GradientRegistry.GetGradientForOp(op, [s + '_grad' for s in op.output])

def _get_grad_blob(grad_map, input_to_check):
    if False:
        print('Hello World!')
    grad_blob = grad_map[input_to_check]
    if isinstance(grad_blob, core.BlobReference):
        return workspace.blobs[grad_blob]
    assert isinstance(grad_blob, core.GradientSlice)
    dense_grad = 'tmp_dense_grad'
    sparse_to_dense_op = core.CreateOperator('SparseToDense', [grad_blob.indices, grad_blob.values, input_to_check], dense_grad)
    workspace.RunOperatorOnce(sparse_to_dense_op)
    return workspace.blobs[dense_grad]

def _get_grad(net, outputs, outputs_with_grad, input_values, inputs_with_grads):
    if False:
        for i in range(10):
            print('nop')
    grad_net = net.Clone(net.Name() + '_copy')
    grad_map = grad_net.AddGradientOperators(outputs_with_grad)
    for (name, value) in (input_values or {}).items():
        workspace.blobs[name] = value
    for input_to_check in inputs_with_grads:
        assert input_to_check in grad_map, '{} has no gradient, cannot check net gradient.'.format(input_to_check)
        assert str(input_to_check) in workspace.blobs
    workspace.RunNetOnce(grad_net)
    forward_results = [(output, workspace.blobs[output]) for output in outputs]
    grads = {input_to_check: _get_grad_blob(grad_map, input_to_check) for input_to_check in inputs_with_grads}
    return (forward_results, grads, grad_net)

def _assert_close(value1, value2, threshold, err_msg=''):
    if False:
        while True:
            i = 10
    np.testing.assert_allclose(value1, value2, atol=threshold, rtol=threshold, err_msg=err_msg)
    delta = np.abs(value1 - value2).flatten()
    return (np.mean(delta), max(delta))

class NetGradientChecker:

    @staticmethod
    def CompareNets(nets, outputs, outputs_with_grad_ids, inputs_with_grads, input_values=None, threshold=1e-07, print_net_images=False):
        if False:
            print('Hello World!')

        def _get_output_with_grad_names(net_outputs):
            if False:
                while True:
                    i = 10
            return [net_outputs[i] for i in outputs_with_grad_ids]
        if print_net_images:
            for (i, net) in enumerate(nets):
                png = net_drawer.GetPydotGraph(net).create_png()
                with open('caffe2_net_forward_' + str(i) + net.Name() + '.png', 'wb') as f:
                    f.write(png)
        results = [_get_grad(net, net_outputs, _get_output_with_grad_names(net_outputs), input_values, inputs_with_grads) for (net, net_outputs) in zip(nets, outputs)]
        if print_net_images:
            (_, _, backward_nets) = zip(*results)
            for (i, net) in enumerate(backward_nets):
                png = net_drawer.GetPydotGraph(net).create_png()
                with open('caffe2_net_' + str(i) + net.Name() + '.png', 'wb') as f:
                    f.write(png)
        (first_net_results, first_net_grads, _) = results[0]
        for (net_results, net_grads, _) in results[1:]:
            assert len(net_results) == len(first_net_results)
            for (idx, ((blob1, blob_value1), (blob2, blob_value2))) in enumerate(zip(first_net_results, net_results)):
                _assert_close(blob_value1, blob_value2, threshold, err_msg='Different forward pass results for output id {}. Corresponding output blobs: {} and {}'.format(idx, blob1, blob2))
            assert net_grads.keys() == first_net_grads.keys()
            for (blob, blob_grad_value) in net_grads.items():
                _assert_close(first_net_grads[blob], blob_grad_value, threshold, err_msg='Different gradients for input {}'.format(blob))

    @staticmethod
    def Check(net, outputs_with_grad, input_values, input_to_check, step_size=0.0001, threshold=0.05, print_net=True):
        if False:
            print('Hello World!')
        (net_results, net_grads, full_net) = _get_grad(net, [], outputs_with_grad, input_values, [input_to_check])
        analytic_grad = net_grads[input_to_check]

        def GetLoss(new_value):
            if False:
                return 10
            workspace.blobs[input_to_check] = new_value
            workspace.RunNetOnce(full_net)
            return sum([workspace.blobs[output] for output in outputs_with_grad]).sum()

        def GetValue(dim, delta):
            if False:
                i = 10
                return i + 15
            input_value = input_values[input_to_check].copy()
            input_value.flat[dim] += delta
            return input_value
        grad_estimate = np.zeros_like(input_values[input_to_check])
        for dim in range(input_values[input_to_check].size):
            pos_loss = GetLoss(GetValue(dim, step_size))
            neg_loss = GetLoss(GetValue(dim, -step_size))
            grad_estimate.flat[dim] = (pos_loss - neg_loss) / step_size / 2
        err_msg = 'Error in gradient check for net_copy {}'.format(net.Name())
        if print_net:
            err_msg += ': {}'.format(net.Proto())
        return _assert_close(analytic_grad, grad_estimate, threshold, err_msg)

class GradientChecker:
    """A gradient checker in Python.

    This is not the most efficient way to check gradients, as the Python
    interface will involve a lot of copies back and forth operations. Use at your
    own risk.
    """

    def __init__(self, stepsize, threshold, device_option=None, workspace_name='gradient_check', input_device_options=None):
        if False:
            while True:
                i = 10
        self._stepsize = stepsize
        self._threshold = threshold
        self._device_option = device_option or caffe2_pb2.DeviceOption()
        self._workspace_name = workspace_name
        if input_device_options is None:
            self._input_device_options = {}
        else:
            self._input_device_options = input_device_options

    def GetLossAndGrad(self, op, grad_ops, inputs, input_names, input_to_check, grad_name, outputs_with_grads):
        if False:
            return 10
        for i in range(len(inputs)):
            workspace.FeedBlob(input_names[i], inputs[i], self._input_device_options.get(input_names[i], self._device_option))
        x = inputs[input_to_check]
        workspace.RunOperatorOnce(op)
        loss = 0.0
        for idx in outputs_with_grads:
            name = op.output[idx]
            arr = workspace.FetchBlob(name)
            loss += (arr ** 2).sum()
            workspace.FeedBlob(name + '_grad', arr, self._device_option)
        loss /= 2.0
        workspace.RunOperatorsOnce(grad_ops)
        if isinstance(grad_name, core.GradientSlice):
            workspace.FeedBlob('zeros', np.zeros_like(x, dtype=np.float32))
            workspace.FeedBlob('ones', np.ones(1, dtype=np.float32))
            gv_cpu_op = core.CreateOperator('EnsureCPUOutput', grad_name.values, grad_name.values + '_cpu', device_option=self._device_option)
            gi_cpu_op = core.CreateOperator('EnsureCPUOutput', grad_name.indices, grad_name.indices + '_cpu', device_option=self._device_option)
            sparse_to_dense_op = core.CreateOperator('ScatterWeightedSum', ['zeros', 'ones', grad_name.indices + '_cpu', grad_name.values + '_cpu', 'ones'], 'zeros')
            workspace.RunOperatorOnce(gv_cpu_op)
            workspace.RunOperatorOnce(gi_cpu_op)
            workspace.RunOperatorOnce(sparse_to_dense_op)
            grad = workspace.FetchBlob('zeros')
        else:
            grad = workspace.FetchBlob(grad_name)
        return (loss, grad)

    def CheckSimple(self, op, inputs, input_to_check, outputs_with_grads, grad_ops=None, input_device_options=None, ensure_outputs_are_inferred=False):
        if False:
            for i in range(10):
                print('nop')
        'Checks the operator in a very simple fashion by stacking a sum of\n        squares on the top.\n\n        Inputs:\n          op: the operator to be checked.\n          inputs: the input data in numpy arrays.\n          input_to_check: an index specifying which input blob we should\n              check.\n          outputs_with_grads: indices specifying which output blobs will we\n              need to check gradients with. For these outputs, we will collect a\n              squared sum and also feed in their gradients.\n          grad_operator: the gradient operator. If not given, we will get the\n              gradient operator from the gradient registry.\n          input_device_options: an optional mapping from input names to\n              DeviceOptions (to override the default DeviceOption)\n          ensure_outputs_are_inferred: if set will assert that the gradient output\n              shapes matches the inferred shapes\n        Outputs:\n          boolean: True if it passes, False if it does not pass.\n        '
        old_ws_name = workspace.CurrentWorkspace()
        if self._workspace_name != old_ws_name:
            workspace.SwitchWorkspace(self._workspace_name, True)
        op.device_option.CopyFrom(self._device_option)
        if grad_ops is None:
            (grad_ops, g_input) = getGradientForOp(op)
        _input_device_options = input_device_options or core.InferOpBlobDevicesAsDict(op)[0]
        for (i, arr) in enumerate(inputs):
            workspace.FeedBlob(op.input[i], arr, _input_device_options.get(op.input[i], self._device_option))
        grad_name = g_input[input_to_check]
        (loss, grad) = self.GetLossAndGrad(op, grad_ops, inputs, op.input, input_to_check, grad_name, outputs_with_grads)
        grad_estimate = np.zeros_like(inputs[input_to_check])
        if grad_estimate.shape != grad.shape:
            raise Exception('Mismatched gradient shapes: estimated ({}), grad ({})'.format(grad_estimate.shape, grad.shape))
        if ensure_outputs_are_inferred:
            self._assertInferTensorChecks(op, grad_ops)
        full_grad_check = os.getenv('CAFFE2_FULL_GRAD_CHECK') == '1'
        dims_to_check = inputs[input_to_check].size
        for current_dim in range(dims_to_check):
            if not full_grad_check and current_dim >= 3 and (current_dim + 3 < dims_to_check):
                grad_estimate.flat[current_dim] = grad.flat[current_dim]
                continue
            inputs[input_to_check].flat[current_dim] += self._stepsize
            (pos_loss, _) = self.GetLossAndGrad(op, grad_ops, inputs, op.input, input_to_check, grad_name, outputs_with_grads)
            inputs[input_to_check].flat[current_dim] -= self._stepsize * 2
            (neg_loss, _) = self.GetLossAndGrad(op, grad_ops, inputs, op.input, input_to_check, grad_name, outputs_with_grads)
            inputs[input_to_check].flat[current_dim] += self._stepsize
            grad_estimate.flat[current_dim] = (pos_loss - neg_loss) / self._stepsize / 2
        fail_mat = ~np.isclose(grad, grad_estimate, atol=self._threshold, rtol=self._threshold)
        if np.any(fail_mat):
            idx = np.flatnonzero(fail_mat)
            print('Failed. [idx, grad, grad_estimate] are:')
            print(np.vstack([idx, grad.flat[idx], grad_estimate.flat[idx]]).T)
            ret = False
        else:
            ret = True
        if self._workspace_name != old_ws_name:
            workspace.ResetWorkspace()
            workspace.SwitchWorkspace(old_ws_name)
        return (ret, grad, grad_estimate)

    def _assertInferTensorChecks(self, op, grad_ops):
        if False:
            for i in range(10):
                print('nop')
        tmp_net = caffe2_pb2.NetDef()
        tmp_net.op.extend([op])
        tmp_net.op.extend(grad_ops)
        (inferred_shapes, inferred_types) = workspace.InferShapesAndTypes([tmp_net], nets_proto=True)
        outputs = set()
        for grad_op in grad_ops:
            outputs.update(grad_op.output)
        for output in outputs:
            if output not in inferred_shapes:
                raise Exception('expected output {} to be inferred'.format(output))
            blob = workspace.FetchBlob(output)
            correct_shape = list(blob.shape)
            inferred_shape = list(inferred_shapes[output])
            if correct_shape != inferred_shape:
                raise Exception('Mismatched inferred shape: want({}), got({})'.format(correct_shape, inferred_shape))
            if type(blob) is np.ndarray:
                if blob.dtype == np.dtype('float64'):
                    correct_type = caffe2_pb2.TensorProto.DOUBLE
                elif blob.dtype == np.dtype('float32'):
                    correct_type = caffe2_pb2.TensorProto.FLOAT
                elif blob.dtype == np.dtype('int32'):
                    correct_type = caffe2_pb2.TensorProto.INT32
                elif blob.dtype == np.dtype('int64'):
                    correct_type = caffe2_pb2.TensorProto.INT64
                else:
                    correct_type = 'unknown {}'.format(np.dtype)
            else:
                correct_type = str(type(blob))
            inferred_type = inferred_types[output]
            if correct_type != inferred_type:
                raise Exception('Mismatched inferred type: want({}), got({})'.format(correct_type, inferred_type))