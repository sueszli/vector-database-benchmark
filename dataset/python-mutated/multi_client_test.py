"""Tests for DTensor multi-client setup."""
import os
from absl import flags
import numpy as np
from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import api as d_api
from tensorflow.dtensor.python import config as d_config
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import layout as d_layout
from tensorflow.dtensor.python import mesh_util
from tensorflow.dtensor.python.tests import multi_client_test_util
from tensorflow.dtensor.python.tests import test_backend_util
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.eager import backprop
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test as tf_test
_MODEL_DIM_SIZE = flags.DEFINE_integer('model_dim_size', 4, 'Size of the model dimension.')
_BATCH_DIM = 'batch'
_MODEL_DIM = 'model'
_BATCH_SIZE = 8
_STEPS = 5
_LR = 0.001

@polymorphic_function.function
def _run_step(inputs, w, b, k):
    if False:
        print('Hello World!')
    with backprop.GradientTape() as g:
        g.watch([w, b])
        logits = nn_ops.conv2d_v2(inputs, k, strides=[1, 1, 1, 1], padding='SAME')
        logits = array_ops.reshape(logits, [logits.shape[0], -1])
        logits = math_ops.matmul(logits, w)
        logits = logits + b
        loss = math_ops.reduce_sum(logits, axis=[0, 1])
    (gw, gb) = g.gradient(loss, [w, b])
    for (v, v_grad) in zip([w, b], [gw, gb]):
        v.assign_sub(_LR * v_grad)
    return (gw, gb, loss)

def init_var(mesh):
    if False:
        while True:
            i = 10
    w_initializer = stateless_random_ops.stateless_random_normal([28 * 28, 16], seed=[0, 1])
    b_initializer = stateless_random_ops.stateless_random_normal([16], seed=[0, 2])
    k_initializer = stateless_random_ops.stateless_random_normal([3, 3, 1, 1], seed=[0, 3])
    n_w = variables.Variable(w_initializer)
    n_b = variables.Variable(b_initializer)
    n_k = variables.Variable(k_initializer)
    w_initializer_on_mesh = d_api.copy_to_mesh(w_initializer, d_layout.Layout.replicated(mesh, rank=2))
    b_initializer_on_mesh = d_api.copy_to_mesh(b_initializer, d_layout.Layout.replicated(mesh, rank=1))
    k_initializer_on_mesh = d_api.copy_to_mesh(k_initializer, d_layout.Layout.replicated(mesh, rank=4))
    w = d_variable.DVariable(d_api.relayout(w_initializer_on_mesh, d_layout.Layout(['unsharded', _MODEL_DIM], mesh)))
    b = d_variable.DVariable(d_api.relayout(b_initializer_on_mesh, d_layout.Layout([_MODEL_DIM], mesh)))
    k = d_variable.DVariable(k_initializer_on_mesh)
    return ((n_w, n_b, n_k), (w, b, k))

class DTensorMNISTTest(tf_test.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(DTensorMNISTTest, self).setUp()
        if config.list_physical_devices('GPU'):
            device_type = 'GPU'
        elif test_util.is_tpu_present():
            device_type = 'TPU'
        else:
            device_type = 'CPU'
        local_devices = d_config.local_devices(device_type)
        num_devices = len(local_devices)
        global_device_ids = test_util.create_device_ids_array((d_config.num_clients() * num_devices // _MODEL_DIM_SIZE.value, _MODEL_DIM_SIZE.value))
        device_ids_list = np.ravel(global_device_ids).tolist()
        index = d_config.client_id() * num_devices
        local_device_ids = device_ids_list[index:index + num_devices]
        self.mesh = d_layout.Mesh([_BATCH_DIM, _MODEL_DIM], global_device_ids=global_device_ids, local_device_ids=local_device_ids, local_devices=local_devices)

    def tearDown(self):
        if False:
            while True:
                i = 10
        mesh_util.barrier(self.mesh)
        super().tearDown()

    def test_mnist(self):
        if False:
            return 10

        def train():
            if False:
                return 10
            input_layout = d_layout.Layout.batch_sharded(self.mesh, _BATCH_DIM, rank=4)
            ((n_w, n_b, n_k), (w, b, k)) = init_var(self.mesh)
            for i in range(_STEPS):
                data = stateless_random_ops.stateless_random_normal([_BATCH_SIZE, 28, 28, 1], seed=[0, i])
                (g_nw, g_nb, n_loss) = _run_step(data.numpy(), n_w, n_b, n_k)
                input_image = d_api.copy_to_mesh(data, layout=d_layout.Layout.replicated(self.mesh, rank=4))
                input_image = d_api.relayout(input_image, layout=input_layout)
                with ops.device_v2(self.mesh.local_devices()[0]):
                    (gw, gb, loss) = _run_step(input_image, w, b, k)
            gw = d_api.relayout(gw, d_layout.Layout.replicated(self.mesh, rank=2))
            w = d_api.relayout(w, d_layout.Layout.replicated(self.mesh, rank=2))
            gb = d_api.relayout(gb, d_layout.Layout.replicated(self.mesh, rank=1))
            b = d_api.relayout(b, d_layout.Layout.replicated(self.mesh, rank=1))
            return ((n_loss, g_nw, g_nb, n_w, n_b), (loss, gw, gb, w, b))
        ((n_loss, g_nw, g_nb, n_w, n_b), (loss, gw, gb, w, b)) = train()
        self.assertAllClose(n_loss, loss, atol=0.0005)
        self.assertAllClose(g_nw, gw, atol=1e-05)
        self.assertAllClose(g_nb, gb, atol=1e-05)
        self.assertAllClose(n_w, w, atol=1e-05)
        self.assertAllClose(n_b, b, atol=1e-05)

    def test_copy_to_mesh(self):
        if False:
            return 10
        layout = d_layout.Layout([_BATCH_DIM, 'unsharded'], self.mesh)
        host_layout = d_layout.Layout(layout.sharding_specs, self.mesh.host_mesh())
        x = d_api.pack([array_ops.ones((2, 2), dtype=dtypes.float32)] * len(self.mesh.local_devices()), layout)

        @polymorphic_function.function
        def d2h(x):
            if False:
                print('Hello World!')
            return d_api.copy_to_mesh(x, host_layout)

        @polymorphic_function.function
        def h2d(x):
            if False:
                print('Hello World!')
            return d_api.copy_to_mesh(x, layout)
        y = d2h(x)
        ys = d_api.unpack(y)
        for i in ys:
            self.assertAllClose(i, array_ops.ones((2, 2)), atol=1e-05)
        z = h2d(y)
        zs = d_api.unpack(z)
        for i in zs:
            self.assertAllClose(i, array_ops.ones((2, 2)), atol=1e-05)

def client_config_function(config_params):
    if False:
        print('Hello World!')
    num_clients = config_params['num_clients']
    dtensor_client_id = config_params['client_id']
    dtensor_jobs = config_params['worker_jobs']
    num_devices = config_params['num_devices']
    if num_clients != 0:
        os.environ[d_config._DT_CLIENT_ID] = f'{dtensor_client_id}'
        os.environ[d_config._DT_JOB_NAME] = 'worker'
        os.environ[d_config._DT_JOBS] = ','.join(dtensor_jobs)
    if config.list_physical_devices('GPU'):
        device_type = 'GPU'
    elif test_util.is_tpu_present():
        device_type = 'TPU'
    else:
        device_type = 'CPU'
    test_util.reset_context()
    if device_type != 'TPU':
        test_util.reset_logical_devices(device_type, num_devices)
    accelerator_util.initialize_accelerator_system(device_type, enable_coordination_service=True)
    logical_devices = test_util.list_local_logical_devices(device_type)
    assert len(logical_devices) == num_devices, (logical_devices, f'Test is mis-configured: expecting {num_devices} logical_devices.')
if __name__ == '__main__':
    test_backend_util.handle_test_main(multi_client_test_util.multi_client_main, client_config_function)