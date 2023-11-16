"""Test for distribution_lib.py."""
import os
from unittest import mock
import numpy as np
import pytest
import tensorflow as tf
from keras import backend
from keras import testing
from keras.backend import distribution_lib as backend_dlib
from keras.distribution import distribution_lib

@pytest.mark.skipif(backend.backend() != 'jax', reason='Only JAX has the backend to mock at the moment')
@mock.patch.object(backend_dlib, 'initialize', return_value=None)
class MultiProcessInitializeTest(testing.TestCase):

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        super().tearDown()
        os.environ.clear()

    def test_initialize_with_explicit_param(self, mock_backend_initialize):
        if False:
            for i in range(10):
                print('nop')
        job_addresses = '10.0.0.1:1234,10.0.0.2:2345'
        num_processes = 2
        current_process_id = 0
        distribution_lib.initialize(job_addresses, num_processes, current_process_id)
        mock_backend_initialize.assert_called_once_with(job_addresses, num_processes, current_process_id)

    def test_initialize_with_env_vars(self, mock_backend_initialize):
        if False:
            print('Hello World!')
        job_addresses = '10.0.0.1:1234,10.0.0.2:2345'
        num_processes = 2
        current_process_id = 0
        os.environ['KERAS_DISTRIBUTION_JOB_ADDRESSES'] = job_addresses
        os.environ['KERAS_DISTRIBUTION_NUM_PROCESSES'] = str(num_processes)
        os.environ['KERAS_DISTRIBUTION_PROCESS_ID'] = str(current_process_id)
        distribution_lib.initialize()
        mock_backend_initialize.assert_called_once_with(job_addresses, num_processes, current_process_id)

    def test_init_with_nones(self, mock_backend_initialize):
        if False:
            for i in range(10):
                print('nop')
        distribution_lib.initialize()
        mock_backend_initialize.assert_called_once_with(None, None, None)

class DeviceMeshTest(testing.TestCase):

    def test_mesh_creation(self):
        if False:
            print('Hello World!')
        devices = [f'cpu:{i}' for i in range(8)]
        shape = (4, 2)
        axis_names = ['batch', 'model']
        mesh = distribution_lib.DeviceMesh(shape, axis_names, devices)
        self.assertEqual(mesh.shape, shape)
        self.assertEqual(mesh.axis_names, axis_names)
        self.assertEqual(mesh.devices.shape, shape)

    def test_input_validation(self):
        if False:
            return 10
        devices = [f'cpu:{i}' for i in range(4)]
        with self.assertRaisesRegex(ValueError, 'Shape and axis_names cannot be empty'):
            distribution_lib.DeviceMesh((4,), '', devices)
        with self.assertRaisesRegex(ValueError, 'Shape and axis_names should have same size'):
            distribution_lib.DeviceMesh((4, 2), ['batch'], devices)
        with self.assertRaisesRegex(ValueError, 'Shape does not match the number of devices'):
            distribution_lib.DeviceMesh((4, 2), ['batch', 'model'], devices)

class TensorLayoutTest(testing.TestCase):

    def setUp(self):
        if False:
            return 10
        self.mesh = distribution_lib.DeviceMesh((4, 2), ['data', 'model'], [f'cpu:{i}' for i in range(8)])

    def test_tensor_layout_creation(self):
        if False:
            print('Hello World!')
        axes = ('data', None)
        layout = distribution_lib.TensorLayout(axes, self.mesh)
        self.assertEqual(layout.device_mesh, self.mesh)
        self.assertEqual(layout.axes, axes)

    def test_tensor_layout_validation(self):
        if False:
            i = 10
            return i + 15
        axes = ('data', 'unknown', None)
        with self.assertRaisesRegex(ValueError, 'Invalid axis names for Layout'):
            distribution_lib.TensorLayout(axes, self.mesh)

    def test_lazy_device_mesh_injection(self):
        if False:
            print('Hello World!')
        axes = ('data', None)
        layout = distribution_lib.TensorLayout(axes, None)
        self.assertIsNone(layout.device_mesh)
        self.assertEqual(layout.axes, axes)
        layout.device_mesh = self.mesh
        self.assertEqual(layout.device_mesh, self.mesh)
        self.assertEqual(layout.axes, axes)

    def test_lazy_device_mesh_validation(self):
        if False:
            print('Hello World!')
        axes = ('data', 'unknown', None)
        layout = distribution_lib.TensorLayout(axes, None)
        self.assertIsNone(layout.device_mesh)
        self.assertEqual(layout.axes, axes)
        with self.assertRaisesRegex(ValueError, 'Invalid axis names for Layout'):
            layout.device_mesh = self.mesh

class DistributionTest(testing.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        devices = [f'cpu:{i}' for i in range(8)]
        shape = (4, 2)
        axis_names = ['batch', 'model']
        self.device_mesh = distribution_lib.DeviceMesh(shape, axis_names, devices)

    def test_init_with_device_mesh(self):
        if False:
            for i in range(10):
                print('nop')
        distribution = distribution_lib.Distribution(self.device_mesh)
        self.assertIs(distribution.device_mesh, self.device_mesh)

    def test_scope(self):
        if False:
            i = 10
            return i + 15
        distribution_1 = distribution_lib.Distribution(self.device_mesh)
        distribution_2 = distribution_lib.Distribution(self.device_mesh)
        self.assertIsNone(distribution_lib.distribution())
        with distribution_1.scope():
            self.assertIs(distribution_lib.distribution(), distribution_1)
            with distribution_2.scope():
                self.assertIs(distribution_lib.distribution(), distribution_2)
            self.assertIs(distribution_lib.distribution(), distribution_1)
        self.assertIsNone(distribution_lib.distribution())

@pytest.mark.skipif(backend.backend() != 'jax', reason='Only JAX has the proper backend distribution lib')
class DataParallelDistributionTest(testing.TestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.devices = [f'cpu:{i}' for i in range(8)]
        shape = (8,)
        axis_names = ['data']
        self.device_mesh = distribution_lib.DeviceMesh(shape, axis_names, self.devices)

    def test_create_with_device_mesh(self):
        if False:
            for i in range(10):
                print('nop')
        distribution = distribution_lib.DataParallel(device_mesh=self.device_mesh)
        device_mesh = distribution.device_mesh
        self.assertEqual(len(device_mesh.devices), 8)
        self.assertEqual(device_mesh.axis_names, ['data'])
        self.assertEqual(distribution._batch_dim_name, 'data')
        self.assertFalse(distribution._is_multi_process)
        self.assertEqual(distribution._process_id, 0)
        self.assertEqual(distribution._num_process, 1)

    def test_create_with_devices(self):
        if False:
            while True:
                i = 10
        distribution = distribution_lib.DataParallel(devices=self.devices)
        device_mesh = distribution.device_mesh
        self.assertEqual(len(device_mesh.devices), 8)
        self.assertEqual(device_mesh.axis_names, ['batch'])
        self.assertEqual(distribution._batch_dim_name, 'batch')

    @mock.patch.object(distribution_lib, 'list_devices', return_value=[f'cpu:{i}' for i in range(8)])
    def test_create_with_list_devices(self, mock_list_devices):
        if False:
            while True:
                i = 10
        distribution = distribution_lib.DataParallel()
        mock_list_devices.assert_called_once()
        device_mesh = distribution.device_mesh
        self.assertEqual(len(device_mesh.devices), 8)
        self.assertEqual(device_mesh.axis_names, ['batch'])
        self.assertEqual(distribution._batch_dim_name, 'batch')

    def test_get_data_layout(self):
        if False:
            return 10
        distribution = distribution_lib.DataParallel(device_mesh=self.device_mesh)
        data = np.arange(16).reshape((4, 2, 2))
        data_layout = distribution.get_data_layout(data.shape)
        self.assertIs(data_layout.device_mesh, self.device_mesh)
        self.assertEqual(data_layout.axes, ('data', None, None))

    def test_get_variable_layout(self):
        if False:
            return 10
        distribution = distribution_lib.DataParallel(device_mesh=self.device_mesh)
        variable = backend.Variable(initializer=[1, 2, 3])
        variable_layout = distribution.get_variable_layout(variable)
        self.assertIs(variable_layout.device_mesh, self.device_mesh)
        self.assertEqual(variable_layout.axes, (None,))

    def test_get_tensor_layout(self):
        if False:
            print('Hello World!')
        distribution = distribution_lib.DataParallel(device_mesh=self.device_mesh)
        path = 'path/to/tensor'
        tensor_layout = distribution.get_tensor_layout(path)
        self.assertIsNone(tensor_layout)

    def test_distribute_dataset(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = tf.data.Dataset.range(8)
        distribution = distribution_lib.DataParallel(device_mesh=self.device_mesh)
        distributed_dataset = distribution.distribute_dataset(dataset)
        self.assertIs(dataset, distributed_dataset)

@pytest.mark.skipif(backend.backend() != 'jax', reason='Only JAX has the proper backend distribution lib')
class ModelParallelDistributionTest(testing.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.devices = [f'cpu:{i}' for i in range(8)]
        shape = (2, 4)
        axis_names = ['data', 'model']
        self.device_mesh = distribution_lib.DeviceMesh(shape, axis_names, self.devices)

    def test_distribute_weights(self):
        if False:
            i = 10
            return i + 15
        layout_map = distribution_lib.LayoutMap(self.device_mesh)
        layout_map['.*kernel'] = distribution_lib.TensorLayout([None, 'model'])
        layout_map['.*bias'] = distribution_lib.TensorLayout(['model'])
        distribution = distribution_lib.ModelParallel(self.device_mesh, layout_map, batch_dim_name='data')
        kernel = backend.Variable(initializer=np.arange(8, 4), name='kernel')
        bias = backend.Variable(initializer=np.arange(4), name='bias')
        rng_seed = backend.Variable(initializer=[0, 1], name='seed')
        kernel_layout = distribution.get_variable_layout(kernel)
        self.assertIs(kernel_layout.device_mesh, self.device_mesh)
        self.assertEqual(kernel_layout.axes, (None, 'model'))
        bias_layout = distribution.get_variable_layout(bias)
        self.assertIs(bias_layout.device_mesh, self.device_mesh)
        self.assertEqual(bias_layout.axes, ('model',))
        rng_seed_layout = distribution.get_variable_layout(rng_seed)
        self.assertIs(rng_seed_layout.device_mesh, self.device_mesh)
        self.assertEqual(rng_seed_layout.axes, (None,))

    def test_distribute_data(self):
        if False:
            for i in range(10):
                print('nop')
        layout_map = distribution_lib.LayoutMap(self.device_mesh)
        distribution = distribution_lib.ModelParallel(self.device_mesh, layout_map, batch_dim_name='data')
        data = np.arange(16).reshape((4, 2, 2))
        data_layout = distribution.get_data_layout(data.shape)
        self.assertIs(data_layout.device_mesh, self.device_mesh)
        self.assertEqual(data_layout.axes, ('data', None, None))

    def test_get_tensor_layout(self):
        if False:
            while True:
                i = 10
        layout_map = distribution_lib.LayoutMap(self.device_mesh)
        layout_map['.*kernel'] = distribution_lib.TensorLayout([None, 'model'])
        layout_map['.*bias'] = distribution_lib.TensorLayout(['model'])
        layout_map['/model/layer/tensor'] = ('data', None)
        distribution = distribution_lib.ModelParallel(self.device_mesh, layout_map, batch_dim_name='data')
        layout = distribution.get_tensor_layout('/model/layer/tensor')
        self.assertIs(layout.device_mesh, self.device_mesh)
        self.assertEqual(layout.axes, ('data', None))
        layout = distribution.get_tensor_layout('/model/layer/other_tensor')
        self.assertIsNone(layout)

    def test_distribute_dataset(self):
        if False:
            while True:
                i = 10
        dataset = tf.data.Dataset.range(8)
        distribution = distribution = distribution_lib.ModelParallel(self.device_mesh, {}, batch_dim_name='data')
        distributed_dataset = distribution.distribute_dataset(dataset)
        self.assertIs(dataset, distributed_dataset)

class LayoutMapTest(testing.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.devices = [f'cpu:{i}' for i in range(8)]
        shape = (4, 2)
        axis_names = ['data', 'model']
        self.device_mesh = distribution_lib.DeviceMesh(shape, axis_names, self.devices)
        self.sharded_2d = distribution_lib.TensorLayout([None, 'model'])
        self.sharded_1d = distribution_lib.TensorLayout(['model'])
        self.replicated_2d = distribution_lib.TensorLayout([None, None])
        self.replicated_1d = distribution_lib.TensorLayout([None])

    def test_add(self):
        if False:
            for i in range(10):
                print('nop')
        layout_map = distribution_lib.LayoutMap(self.device_mesh)
        layout_map['dense/kernel'] = self.sharded_2d
        layout_map['dense/bias'] = self.sharded_1d
        layout_map['conv/bias'] = ('model',)
        self.assertLen(layout_map, 3)
        kernel_layout = layout_map['dense/kernel']
        self.assertEqual(kernel_layout.axes, (None, 'model'))
        self.assertIs(kernel_layout.device_mesh, self.device_mesh)
        bias_layout = layout_map['dense/bias']
        self.assertEqual(bias_layout.axes, ('model',))
        self.assertIs(bias_layout.device_mesh, self.device_mesh)
        conv_bias_layout = layout_map['conv/bias']
        self.assertEqual(conv_bias_layout.axes, ('model',))
        self.assertIs(bias_layout.device_mesh, self.device_mesh)
        with self.assertRaisesRegex(ValueError, 'dense/kernel already exist'):
            layout_map['dense/kernel'] = self.sharded_2d
        with self.assertRaisesRegex(ValueError, 'should be a TensorLayout'):
            layout_map['conv.kernel'] = ['a', 'b']

    def test_get(self):
        if False:
            print('Hello World!')
        layout_map = distribution_lib.LayoutMap(self.device_mesh)
        layout_map['dense/kernel'] = self.sharded_2d
        layout_map['dense/bias'] = self.sharded_1d
        layout_map['dense.*kernel'] = self.replicated_2d
        layout_map['dense.*bias'] = self.replicated_1d
        layout_map['bias'] = self.sharded_1d
        self.assertEqual(layout_map['dense/kernel'], self.sharded_2d)
        self.assertEqual(layout_map['dense/bias'], self.sharded_1d)
        self.assertEqual(layout_map['dense_2/kernel'], self.replicated_2d)
        with self.assertRaisesRegex(ValueError, "Path 'dense_2/bias' matches multiple layout"):
            layout_map['dense_2/bias']
        self.assertIsNone(layout_map['conv2d/kernel'])
        self.assertEqual(layout_map['conv2d/bias'], self.sharded_1d)

    def test_delete(self):
        if False:
            return 10
        layout_map = distribution_lib.LayoutMap(self.device_mesh)
        layout_map['dense/kernel'] = self.sharded_2d
        layout_map['dense/bias'] = self.sharded_1d
        self.assertEqual(layout_map.pop('dense/kernel'), self.sharded_2d)
        with self.assertRaises(KeyError):
            layout_map.pop('.*bias')
        del layout_map['dense/bias']
        self.assertLen(layout_map, 0)

    def test_len(self):
        if False:
            return 10
        layout_map = distribution_lib.LayoutMap(self.device_mesh)
        self.assertLen(layout_map, 0)
        layout_map['dense/kernel'] = self.sharded_2d
        layout_map['dense/bias'] = self.sharded_1d
        self.assertLen(layout_map, 2)

    def test_iter(self):
        if False:
            for i in range(10):
                print('nop')
        layout_map = distribution_lib.LayoutMap(self.device_mesh)
        layout_map['dense/kernel'] = self.sharded_2d
        layout_map['dense/bias'] = self.sharded_1d
        self.assertEqual(list(layout_map.keys()), ['dense/kernel', 'dense/bias'])
        keys = []
        values = []
        for (k, v) in layout_map.items():
            keys.append(k)
            values.append(v)
        self.assertEqual(keys, ['dense/kernel', 'dense/bias'])
        self.assertEqual(values, [self.sharded_2d, self.sharded_1d])