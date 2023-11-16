"""Tests for tensorflow_hub.keras_layer."""
import json
import os
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
version_fn = getattr(tf.keras, 'version', None)
if version_fn and version_fn().startswith('3.'):
    import tf_keras
    from tf_keras.api._v2 import keras as tf_keras_v2
else:
    tf_keras = tf.keras
    tf_keras_v2 = tf.compat.v2.keras

def _skip_if_no_tf_asset(test_case):
    if False:
        for i in range(10):
            print('nop')
    if not hasattr(tf.saved_model, 'Asset'):
        test_case.skipTest('Your TensorFlow version (%s) looks too old for creating SavedModels  with assets.' % tf.__version__)

def _json_cycle(x):
    if False:
        for i in range(10):
            print('nop')
    return json.loads(json.dumps(x))

def _save_half_plus_one_model(export_dir, save_from_keras=False):
    if False:
        print('Hello World!')
    'Writes Hub-style SavedModel to compute y = wx + 1, with w trainable.'
    inp = tf_keras_v2.layers.Input(shape=(1,), dtype=tf.float32)
    times_w = tf_keras_v2.layers.Dense(units=1, kernel_initializer=tf_keras_v2.initializers.Constant([[0.5]]), kernel_regularizer=tf_keras_v2.regularizers.l2(0.01), use_bias=False)
    plus_1 = tf_keras_v2.layers.Dense(units=1, kernel_initializer=tf_keras_v2.initializers.Constant([[1.0]]), bias_initializer=tf_keras_v2.initializers.Constant([1.0]), trainable=False)
    outp = plus_1(times_w(inp))
    model = tf_keras_v2.Model(inp, outp)
    if save_from_keras:
        tf.saved_model.save(model, export_dir)
        return

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 1), dtype=tf.float32)])
    def call_fn(inputs):
        if False:
            i = 10
            return i + 15
        return model(inputs, training=False)
    obj = tf.train.Checkpoint()
    obj.__call__ = call_fn
    obj.variables = model.trainable_variables + model.non_trainable_variables
    assert len(obj.variables) == 3, 'Expect 2 kernels and 1 bias.'
    obj.trainable_variables = [times_w.kernel]
    assert len(model.losses) == 1, 'Expect 1 regularization loss.'
    obj.regularization_losses = [tf.function(lambda : model.losses[0], input_signature=[])]
    tf.saved_model.save(obj, export_dir)

def _save_half_plus_one_hub_module_v1(path):
    if False:
        i = 10
        return i + 15
    'Writes a model in TF1 Hub format to compute y = wx + 1, with w trainable.'

    def half_plus_one():
        if False:
            i = 10
            return i + 15
        x = tf.compat.v1.placeholder(shape=(None, 1), dtype=tf.float32)
        times_w = tf_keras_v2.layers.Dense(units=1, kernel_initializer=tf_keras_v2.initializers.Constant([[0.5]]), kernel_regularizer=tf_keras_v2.regularizers.l2(0.01), use_bias=False)
        plus_1 = tf_keras_v2.layers.Dense(units=1, kernel_initializer=tf_keras_v2.initializers.Constant([[1.0]]), bias_initializer=tf_keras_v2.initializers.Constant([1.0]), trainable=False)
        y = plus_1(times_w(x))
        hub.add_signature(inputs=x, outputs=y)
    spec = hub.create_module_spec(half_plus_one)
    _export_module_spec_with_init_weights(spec, path)

def _save_2d_text_embedding(export_dir, save_from_keras=False):
    if False:
        for i in range(10):
            print('nop')
    'Writes SavedModel to compute y = length(text)*w, with w trainable.'

    class StringLengthLayer(tf_keras_v2.layers.Layer):

        def call(self, inputs):
            if False:
                i = 10
                return i + 15
            return tf.strings.length(inputs)
    inp = tf_keras_v2.layers.Input(shape=(1,), dtype=tf.string)
    text_length = StringLengthLayer()
    times_w = tf_keras_v2.layers.Dense(units=2, kernel_initializer=tf_keras_v2.initializers.Constant([0.1, 0.3]), kernel_regularizer=tf_keras_v2.regularizers.l2(0.01), use_bias=False)
    outp = times_w(text_length(inp))
    model = tf_keras_v2.Model(inp, outp)
    if save_from_keras:
        tf.saved_model.save(model, export_dir)
        return

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 1), dtype=tf.string)])
    def call_fn(inputs):
        if False:
            print('Hello World!')
        return model(inputs, training=False)
    obj = tf.train.Checkpoint()
    obj.__call__ = call_fn
    obj.variables = model.trainable_variables + model.non_trainable_variables
    assert len(obj.variables) == 1, 'Expect 1 weight, received {}.'.format(len(obj.variables))
    obj.trainable_variables = [times_w.kernel]
    assert len(model.losses) == 1, 'Expect 1 regularization loss, received {}.'.format(len(model.losses))
    obj.regularization_losses = [tf.function(lambda : model.losses[0], input_signature=[])]
    tf.saved_model.save(obj, export_dir)

def _tensors_names_set(tensor_sequence):
    if False:
        print('Hello World!')
    'Converts tensor sequence to a set of tensor references.'
    return {t.name for t in tensor_sequence}

def _save_batch_norm_model(export_dir, save_from_keras=False):
    if False:
        for i in range(10):
            print('nop')
    'Writes a Hub-style SavedModel with a batch norm layer.'
    inp = tf_keras_v2.layers.Input(shape=(1,), dtype=tf.float32)
    bn = tf_keras_v2.layers.BatchNormalization(momentum=0.8)
    outp = bn(inp)
    model = tf_keras_v2.Model(inp, outp)
    if save_from_keras:
        tf.saved_model.save(model, export_dir)
        return

    @tf.function
    def call_fn(inputs, training=False):
        if False:
            for i in range(10):
                print('nop')
        return model(inputs, training=training)
    for training in (True, False):
        call_fn.get_concrete_function(tf.TensorSpec((None, 1), tf.float32), training=training)
    obj = tf.train.Checkpoint()
    obj.__call__ = call_fn
    obj.trainable_variables = [bn.beta, bn.gamma]
    assert _tensors_names_set(obj.trainable_variables) == _tensors_names_set(model.trainable_variables)
    obj.variables = [bn.beta, bn.gamma, bn.moving_mean, bn.moving_variance]
    assert _tensors_names_set(obj.variables) == _tensors_names_set(model.trainable_variables + model.non_trainable_variables)
    obj.regularization_losses = []
    assert not model.losses
    tf.saved_model.save(obj, export_dir)

def _get_batch_norm_vars(imported):
    if False:
        while True:
            i = 10
    'Returns the 4 variables of an imported batch norm model in sorted order.'
    expected_suffixes = ['beta', 'gamma', 'moving_mean', 'moving_variance']
    variables = sorted(imported.variables, key=lambda v: v.name)
    names = [v.name for v in variables]
    assert len(variables) == 4
    assert all((name.endswith(suffix + ':0') for (name, suffix) in zip(names, expected_suffixes)))
    return variables

def _save_model_with_hparams(export_dir):
    if False:
        i = 10
        return i + 15
    'Writes a Hub-style SavedModel to compute y = ax + b with hparams a, b.'

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 1), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.float32)])
    def call_fn(x, a=1.0, b=0.0):
        if False:
            while True:
                i = 10
        return tf.add(tf.multiply(a, x), b)
    obj = tf.train.Checkpoint()
    obj.__call__ = call_fn
    tf.saved_model.save(obj, export_dir)

def _save_model_with_custom_attributes(export_dir, temp_dir, save_from_keras=False):
    if False:
        print('Hello World!')
    'Writes a Hub-style SavedModel with a custom attributes.'
    f = lambda a: tf.strings.to_number(a, tf.int64)
    if save_from_keras:
        inp = tf_keras_v2.layers.Input(shape=(1,), dtype=tf.string)
        outp = tf_keras_v2.layers.Lambda(f)(inp)
        model = tf_keras_v2.Model(inp, outp)
    else:
        model = tf.train.Checkpoint()
        model.__call__ = tf.function(input_signature=[tf.TensorSpec(shape=(None, 1), dtype=tf.string)])(f)
    asset_source_file_name = os.path.join(temp_dir, 'number.txt')
    tf.io.gfile.makedirs(temp_dir)
    with tf.io.gfile.GFile(asset_source_file_name, 'w') as f:
        f.write('12345\n')
    model.sample_input = tf.saved_model.Asset(asset_source_file_name)
    model.sample_output = tf.Variable([[12345]], dtype=tf.int64)
    tf.saved_model.save(model, export_dir)
    tf.io.gfile.remove(asset_source_file_name)
    return export_dir

def _save_model_with_dict_input_output(export_dir):
    if False:
        print('Hello World!')
    'Writes SavedModel using dicts to compute x+y, x+2y and maybe x-y.'

    @tf.function
    def call_fn(d, return_dict=False):
        if False:
            while True:
                i = 10
        x = d['x']
        y = d['y']
        sigma = tf.concat([tf.add(x, y), tf.add(x, 2 * y)], axis=-1)
        if return_dict:
            return dict(sigma=sigma, delta=tf.subtract(x, y))
        else:
            return sigma
    d_spec = dict(x=tf.TensorSpec(shape=(None, 1), dtype=tf.float32), y=tf.TensorSpec(shape=(None, 1), dtype=tf.float32))
    for return_dict in (False, True):
        call_fn.get_concrete_function(d_spec, return_dict=return_dict)
    obj = tf.train.Checkpoint()
    obj.__call__ = call_fn
    tf.saved_model.save(obj, export_dir)

def _save_model_with_obscurely_shaped_list_output(export_dir):
    if False:
        while True:
            i = 10
    'Writes SavedModel with hard-to-predict output shapes.'

    def broadcast_obscurely_to(input_tensor, shape):
        if False:
            for i in range(10):
                print('nop')
        'Like tf.broadcast_to(), but hostile to static shape propagation.'
        obscured_shape = tf.cast(tf.cast(shape, tf.float32) + 0.1 * tf.sin(tf.random.uniform((), -3, +3)) + 0.3, tf.int32)
        return tf.broadcast_to(input_tensor, obscured_shape)

    @tf.function(input_signature=[tf.TensorSpec(shape=(None, 1), dtype=tf.float32)])
    def call_fn(x):
        if False:
            i = 10
            return i + 15
        batch_size = tf.shape(x)[0]
        return [broadcast_obscurely_to(tf.reshape(i * x, [batch_size] + [1] * i), tf.concat([[batch_size], [i] * i], axis=0)) for i in range(1, 4)]
    obj = tf.train.Checkpoint()
    obj.__call__ = call_fn
    tf.saved_model.save(obj, export_dir)

def _save_plus_one_saved_model_v2(path, save_from_keras=False):
    if False:
        for i in range(10):
            print('nop')
    'Writes Hub-style SavedModel that increments the input by one.'
    if save_from_keras:
        raise NotImplementedError()
    obj = tf.train.Checkpoint()

    @tf.function(input_signature=[tf.TensorSpec(None, dtype=tf.float32)])
    def plus_one(x):
        if False:
            for i in range(10):
                print('nop')
        return x + 1
    obj.__call__ = plus_one
    tf.saved_model.save(obj, path)

def _save_plus_one_saved_model_v2_keras_default_callable(path):
    if False:
        i = 10
        return i + 15
    'Writes Hub-style SavedModel that increments the input by one.'
    obj = tf.train.Checkpoint()

    @tf.function(input_signature=[tf.TensorSpec(None, dtype=tf.float32)])
    def plus_one(x):
        if False:
            while True:
                i = 10
        return x + 1

    @tf.function(input_signature=[tf.TensorSpec(None, dtype=tf.float32), tf.TensorSpec((), dtype=tf.bool)])
    def keras_default(x, training=False):
        if False:
            for i in range(10):
                print('nop')
        if training:
            return x + 1
        return x
    obj.__call__ = keras_default
    obj.plus_one = plus_one
    tf.saved_model.save(obj, path, signatures={'plus_one': obj.plus_one})

def _save_plus_one_hub_module_v1(path):
    if False:
        for i in range(10):
            print('nop')
    'Writes a model in TF1 Hub format that increments the input by one.'

    def plus_one():
        if False:
            while True:
                i = 10
        x = tf.compat.v1.placeholder(dtype=tf.float32, name='x')
        y = x + 1
        hub.add_signature(inputs=x, outputs=y)
    spec = hub.create_module_spec(plus_one)
    _export_module_spec_with_init_weights(spec, path)

def _export_module_spec_with_init_weights(spec, path):
    if False:
        for i in range(10):
            print('nop')
    'Initializes initial weights of a TF1.x HubModule and saves it.'
    with tf.compat.v1.Graph().as_default():
        module = hub.Module(spec, trainable=True)
        with tf.compat.v1.Session() as session:
            session.run(tf.compat.v1.global_variables_initializer())
            module.export(path, session)

def _dispatch_model_format(model_format, saved_model_fn, hub_module_fn, *args):
    if False:
        for i in range(10):
            print('nop')
    'Dispatches the correct save function based on the model format.'
    if model_format == 'TF2SavedModel_SavedRaw':
        saved_model_fn(*args, save_from_keras=False)
    elif model_format == 'TF2SavedModel_SavedFromKeras':
        saved_model_fn(*args, save_from_keras=True)
    elif model_format == 'TF1HubModule':
        hub_module_fn(*args)
    else:
        raise ValueError('Unrecognized format: ' + format)

class KerasTest(tf.test.TestCase, parameterized.TestCase):
    """Tests KerasLayer in an all-Keras environment."""

    @parameterized.parameters('TF2SavedModel_SavedRaw', 'TF2SavedModel_SavedFromKeras')
    def testHalfPlusOneRetraining(self, model_format):
        if False:
            while True:
                i = 10
        export_dir = os.path.join(self.get_temp_dir(), 'half-plus-one')
        _dispatch_model_format(model_format, _save_half_plus_one_model, _save_half_plus_one_hub_module_v1, export_dir)
        inp = tf_keras_v2.layers.Input(shape=(1,), dtype=tf.float32)
        imported = hub.KerasLayer(export_dir, trainable=True)
        outp = imported(inp)
        model = tf_keras_v2.Model(inp, outp)
        self.assertAllEqual(model(np.array([[0.0], [8.0], [10.0], [12.0]], dtype=np.float32)), np.array([[1.0], [5.0], [6.0], [7.0]], dtype=np.float32))
        self.assertAllEqual(model.losses, np.array([0.0025], dtype=np.float32))
        self.assertEqual(len(model.trainable_weights), 1)
        self.assertEqual(model.trainable_weights[0].shape.rank, 2)
        self.assertEqual(len(model.non_trainable_weights), 2)
        self.assertCountEqual([v.shape.rank for v in model.non_trainable_weights], [2, 1])
        self.assertNoCommonElements(_tensors_names_set(model.trainable_weights), _tensors_names_set(model.non_trainable_weights))
        model.compile(tf_keras_v2.optimizers.SGD(0.002), 'mean_squared_error', run_eagerly=True)
        x = [[9.0], [10.0], [11.0]] * 10
        y = [[xi[0] / 2.0 + 6] for xi in x]
        model.fit(np.array(x), np.array(y), batch_size=len(x), epochs=10, verbose=2)
        self.assertAllEqual(model(np.array([[0.0]], dtype=np.float32)), np.array([[1.0]], dtype=np.float32))
        self.assertAllClose(model(np.array([[10.0]], dtype=np.float32)), np.array([[11.0]], dtype=np.float32), atol=0.0, rtol=0.03)
        self.assertAllClose(model.losses, np.array([0.01], dtype=np.float32), atol=0.0, rtol=0.06)

    @parameterized.parameters('TF2SavedModel_SavedRaw', 'TF2SavedModel_SavedFromKeras')
    def testRegularizationLoss(self, model_format):
        if False:
            while True:
                i = 10
        export_dir = os.path.join(self.get_temp_dir(), 'half-plus-one')
        _dispatch_model_format(model_format, _save_half_plus_one_model, _save_half_plus_one_hub_module_v1, export_dir)
        inp = tf_keras_v2.layers.Input(shape=(1,), dtype=tf.float32)
        imported = hub.KerasLayer(export_dir, trainable=False)
        outp = imported(inp)
        model = tf_keras_v2.Model(inp, outp)
        self.assertAllEqual(model.losses, np.array([0.0], dtype=np.float32))
        imported.trainable = True
        self.assertAllEqual(model.losses, np.array([0.0025], dtype=np.float32))
        imported.trainable = False
        self.assertAllEqual(model.losses, np.array([0.0], dtype=np.float32))
        imported.trainable = True
        self.assertAllEqual(model.losses, np.array([0.0025], dtype=np.float32))

    @parameterized.named_parameters(('SavedRaw', False), ('SavedFromKeras', True))
    def testBatchNormRetraining(self, save_from_keras):
        if False:
            return 10
        'Tests imported batch norm with trainable=True.'
        export_dir = os.path.join(self.get_temp_dir(), 'batch-norm')
        _save_batch_norm_model(export_dir, save_from_keras=save_from_keras)
        inp = tf_keras_v2.layers.Input(shape=(1,), dtype=tf.float32)
        imported = hub.KerasLayer(export_dir, trainable=True)
        (var_beta, var_gamma, var_mean, var_variance) = _get_batch_norm_vars(imported)
        outp = imported(inp)
        model = tf_keras_v2.Model(inp, outp)
        model.compile(tf_keras_v2.optimizers.SGD(0.1), 'mean_squared_error', run_eagerly=True)
        x = [[11.0], [12.0], [13.0]]
        y = [[2 * xi[0]] for xi in x]
        model.fit(np.array(x), np.array(y), batch_size=len(x), epochs=100)
        self.assertAllClose(var_mean.numpy(), np.array([12.0]))
        self.assertAllClose(var_beta.numpy(), np.array([24.0]))
        self.assertAllClose(model(np.array(x, np.float32)), np.array(y))
        for _ in range(100):
            self.assertAllClose(model(np.array([[10.0], [20.0], [30.0]], np.float32)), np.array([[20.0], [40.0], [60.0]]))
        self.assertAllClose(var_mean.numpy(), np.array([12.0]))
        self.assertAllClose(var_beta.numpy(), np.array([24.0]))

    @parameterized.named_parameters(('SavedRaw', False), ('SavedFromKeras', True))
    def testBatchNormFreezing(self, save_from_keras):
        if False:
            print('Hello World!')
        'Tests imported batch norm with trainable=False.'
        export_dir = os.path.join(self.get_temp_dir(), 'batch-norm')
        _save_batch_norm_model(export_dir, save_from_keras=save_from_keras)
        inp = tf_keras_v2.layers.Input(shape=(1,), dtype=tf.float32)
        imported = hub.KerasLayer(export_dir, trainable=False)
        (var_beta, var_gamma, var_mean, var_variance) = _get_batch_norm_vars(imported)
        dense = tf_keras_v2.layers.Dense(units=1, kernel_initializer=tf_keras_v2.initializers.Constant([[1.5]]), use_bias=False)
        outp = dense(imported(inp))
        model = tf_keras_v2.Model(inp, outp)
        self.assertAllClose(var_beta.numpy(), np.array([0.0]))
        self.assertAllClose(var_gamma.numpy(), np.array([1.0]))
        self.assertAllClose(var_mean.numpy(), np.array([0.0]))
        self.assertAllClose(var_variance.numpy(), np.array([1.0]))
        model.compile(tf_keras_v2.optimizers.SGD(0.1), 'mean_squared_error', run_eagerly=True)
        x = [[1.0], [2.0], [3.0]]
        y = [[2 * xi[0]] for xi in x]
        model.fit(np.array(x), np.array(y), batch_size=len(x), epochs=20)
        self.assertAllClose(var_beta.numpy(), np.array([0.0]))
        self.assertAllClose(var_gamma.numpy(), np.array([1.0]))
        self.assertAllClose(var_mean.numpy(), np.array([0.0]))
        self.assertAllClose(var_variance.numpy(), np.array([1.0]))
        self.assertAllClose(model(np.array(x, np.float32)), np.array(y))

    @parameterized.named_parameters(('SavedRaw', False), ('SavedFromKeras', True))
    def testCustomAttributes(self, save_from_keras):
        if False:
            return 10
        'Tests custom attributes (Asset and Variable) on a SavedModel.'
        _skip_if_no_tf_asset(self)
        base_dir = os.path.join(self.get_temp_dir(), 'custom-attributes')
        export_dir = os.path.join(base_dir, 'model')
        temp_dir = os.path.join(base_dir, 'scratch')
        _save_model_with_custom_attributes(export_dir, temp_dir, save_from_keras=save_from_keras)
        imported = hub.KerasLayer(export_dir)
        expected_outputs = imported.resolved_object.sample_output.value().numpy()
        asset_path = imported.resolved_object.sample_input.asset_path.numpy()
        with tf.io.gfile.GFile(asset_path) as f:
            inputs = tf.constant([[f.read()]], dtype=tf.string)
        actual_outputs = imported(inputs).numpy()
        self.assertAllEqual(expected_outputs, actual_outputs)

    @parameterized.named_parameters(('NoOutputShapes', False), ('WithOutputShapes', True))
    def testInputOutputDict(self, pass_output_shapes):
        if False:
            print('Hello World!')
        'Tests use of input/output dicts.'
        export_dir = os.path.join(self.get_temp_dir(), 'with-dicts')
        _save_model_with_dict_input_output(export_dir)
        x_in = tf_keras_v2.layers.Input(shape=(1,), dtype=tf.float32)
        y_in = tf_keras_v2.layers.Input(shape=(1,), dtype=tf.float32)
        dict_in = dict(x=x_in, y=y_in)
        kwargs = dict(arguments=dict(return_dict=True))
        if pass_output_shapes:
            kwargs['output_shape'] = dict(sigma=(2,), delta=(1,))
        imported = hub.KerasLayer(export_dir, **kwargs)
        dict_out = imported(dict_in)
        delta_out = dict_out['delta']
        sigma_out = dict_out['sigma']
        concat_out = tf_keras_v2.layers.concatenate([delta_out, sigma_out])
        model = tf_keras_v2.Model(dict_in, [delta_out, sigma_out, concat_out])
        x = np.array([[11.0], [22.0], [33.0]], dtype=np.float32)
        y = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        outputs = model(dict(x=x, y=y))
        self.assertLen(outputs, 3)
        (delta, sigma, concat) = [x.numpy() for x in outputs]
        self.assertAllClose(delta, np.array([[10.0], [20.0], [30.0]]))
        self.assertAllClose(sigma, np.array([[12.0, 13.0], [24.0, 26.0], [36.0, 39.0]]))
        self.assertAllClose(concat, np.array([[10.0, 12.0, 13.0], [20.0, 24.0, 26.0], [30.0, 36.0, 39.0]]))
        config = imported.get_config()
        new_layer = hub.KerasLayer.from_config(_json_cycle(config))
        if pass_output_shapes:
            self.assertEqual(new_layer._output_shape, imported._output_shape)
        else:
            self.assertFalse(hasattr(new_layer, '_output_shape'))

    @parameterized.named_parameters(('NoOutputShapes', False), ('WithOutputShapes', True))
    def testOutputShapeList(self, pass_output_shapes):
        if False:
            print('Hello World!')
        export_dir = os.path.join(self.get_temp_dir(), 'obscurely-shaped')
        _save_model_with_obscurely_shaped_list_output(export_dir)
        kwargs = {}
        if pass_output_shapes:
            kwargs['output_shape'] = [[1], [2, 2], [3, 3, 3]]
        inp = tf_keras_v2.layers.Input(shape=(1,), dtype=tf.float32)
        imported = hub.KerasLayer(export_dir, **kwargs)
        outp = imported(inp)
        model = tf_keras_v2.Model(inp, outp)
        x = np.array([[1.0], [10.0]], dtype=np.float32)
        outputs = model(x)
        self.assertLen(outputs, 3)
        (single, double, triple) = [x.numpy() for x in outputs]
        self.assertAllClose(single, np.array([[1.0], [10.0]]))
        self.assertAllClose(double, np.array([[[2.0, 2.0], [2.0, 2.0]], [[20.0, 20.0], [20.0, 20.0]]]))
        self.assertAllClose(triple, np.array([[[[3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]], [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]], [[3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [3.0, 3.0, 3.0]]], [[[30.0, 30.0, 30.0], [30.0, 30.0, 30.0], [30.0, 30.0, 30.0]], [[30.0, 30.0, 30.0], [30.0, 30.0, 30.0], [30.0, 30.0, 30.0]], [[30.0, 30.0, 30.0], [30.0, 30.0, 30.0], [30.0, 30.0, 30.0]]]]))
        config = imported.get_config()
        new_layer = hub.KerasLayer.from_config(_json_cycle(config))
        if pass_output_shapes:
            self.assertEqual(new_layer._output_shape, imported._output_shape)
        else:
            self.assertFalse(hasattr(new_layer, '_output_shape'))

    @parameterized.named_parameters(('SavedRaw', False), ('SavedFromKeras', True))
    def testComputeOutputShape(self, save_from_keras):
        if False:
            i = 10
            return i + 15
        export_dir = os.path.join(self.get_temp_dir(), 'half-plus-one')
        _save_half_plus_one_model(export_dir, save_from_keras=save_from_keras)
        layer = hub.KerasLayer(export_dir)
        self.assertEqual([10, 1], layer.compute_output_shape(tuple([10, 1])).as_list())
        layer.get_config()

    @parameterized.named_parameters(('SavedRaw', False), ('SavedFromKeras', True))
    def testComputeOutputShapeDifferentDtypes(self, save_from_keras):
        if False:
            i = 10
            return i + 15
        export_dir = os.path.join(self.get_temp_dir(), '2d-text-embed')
        _save_2d_text_embedding(export_dir, save_from_keras=save_from_keras)
        layer = hub.KerasLayer(export_dir, output_shape=(2,))
        self.assertEqual([None, 2], layer.compute_output_shape((None, 1)).as_list())
        self.assertEqual([3, 2], layer.compute_output_shape((3, 1)).as_list())

    @parameterized.named_parameters(('SavedRaw', False), ('SavedFromKeras', True))
    def testResaveWithMixedPrecision(self, save_from_keras):
        if False:
            return 10
        'Tests importing a float32 model then saving it with mixed_float16.'
        (major, minor, _) = tf.version.VERSION.split('.')
        if not tf.executing_eagerly() or (int(major), int(minor)) < (2, 4):
            self.skipTest('Test uses non-experimental mixed precision API, which is only available in TF 2.4 or above')
        export_dir1 = os.path.join(self.get_temp_dir(), 'mixed-precision')
        export_dir2 = os.path.join(self.get_temp_dir(), 'mixed-precision2')
        _save_2d_text_embedding(export_dir1, save_from_keras=save_from_keras)
        try:
            tf_keras_v2.mixed_precision.set_global_policy('mixed_float16')
            inp = tf_keras_v2.layers.Input(shape=(1,), dtype=tf.string)
            imported = hub.KerasLayer(export_dir1, trainable=True)
            outp = imported(inp)
            model = tf_keras_v2.Model(inp, outp)
            model.compile(tf_keras_v2.optimizers.SGD(0.002, momentum=0.001), 'mean_squared_error', run_eagerly=True)
            x = [['a'], ['aa'], ['aaa']]
            y = [len(xi) for xi in x]
            model.fit(x, y)
            tf.saved_model.save(model, export_dir2)
        finally:
            tf_keras_v2.mixed_precision.set_global_policy('float32')

    def testComputeOutputShapeNonEager(self):
        if False:
            for i in range(10):
                print('nop')
        export_dir = os.path.join(self.get_temp_dir(), 'half-plus-one')
        _save_half_plus_one_hub_module_v1(export_dir)
        with tf.compat.v1.Graph().as_default():
            layer = hub.KerasLayer(export_dir, output_shape=(1,))
            self.assertEqual([None, 1], layer.compute_output_shape((None, 1)).as_list())
            self.assertEqual([3, 1], layer.compute_output_shape((3, 1)).as_list())

    @parameterized.named_parameters(('SavedRaw', False), ('SavedFromKeras', True))
    def testGetConfigFromConfig(self, save_from_keras):
        if False:
            while True:
                i = 10
        export_dir = os.path.join(self.get_temp_dir(), 'half-plus-one')
        _save_half_plus_one_model(export_dir, save_from_keras=save_from_keras)
        layer = hub.KerasLayer(export_dir)
        in_value = np.array([[10.0]], dtype=np.float32)
        result = layer(in_value).numpy()
        config = layer.get_config()
        new_layer = hub.KerasLayer.from_config(_json_cycle(config))
        new_result = new_layer(in_value).numpy()
        self.assertEqual(result, new_result)

    def testGetConfigFromConfigWithHParams(self):
        if False:
            print('Hello World!')
        if tf.__version__ == '2.0.0-alpha0':
            self.skipTest('b/127938157 broke use of default hparams')
        export_dir = os.path.join(self.get_temp_dir(), 'with-hparams')
        _save_model_with_hparams(export_dir)
        layer = hub.KerasLayer(export_dir, arguments=dict(a=10.0))
        in_value = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        expected_result = np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
        result = layer(in_value).numpy()
        self.assertAllEqual(expected_result, result)
        config = layer.get_config()
        new_layer = hub.KerasLayer.from_config(_json_cycle(config))
        new_result = new_layer(in_value).numpy()
        self.assertAllEqual(result, new_result)

    @parameterized.named_parameters(('SavedRaw', False), ('SavedFromKeras', True))
    def testSaveModelConfig(self, save_from_keras):
        if False:
            return 10
        export_dir = os.path.join(self.get_temp_dir(), 'half-plus-one')
        _save_half_plus_one_model(export_dir, save_from_keras=save_from_keras)
        model = tf_keras_v2.Sequential([hub.KerasLayer(export_dir)])
        in_value = np.array([[10.0]], dtype=np.float32)
        result = model(in_value).numpy()
        json_string = model.to_json()
        new_model = tf_keras_v2.models.model_from_json(json_string, custom_objects={'KerasLayer': hub.KerasLayer})
        new_result = new_model(in_value).numpy()
        self.assertEqual(result, new_result)

class KerasLayerTest(tf.test.TestCase, parameterized.TestCase):
    """Unit tests for KerasLayer."""

    @parameterized.parameters('TF1HubModule', 'TF2SavedModel_SavedRaw')
    def test_load_with_defaults(self, model_format):
        if False:
            for i in range(10):
                print('nop')
        export_dir = os.path.join(self.get_temp_dir(), 'plus_one_' + model_format)
        _dispatch_model_format(model_format, _save_plus_one_saved_model_v2, _save_plus_one_hub_module_v1, export_dir)
        (inputs, expected_outputs) = (10.0, 11.0)
        layer = hub.KerasLayer(export_dir)
        output = layer(inputs)
        self.assertEqual(output, expected_outputs)

    @parameterized.parameters(('TF1HubModule', None, None, True), ('TF1HubModule', None, None, False), ('TF1HubModule', 'default', None, True), ('TF1HubModule', None, 'default', False), ('TF1HubModule', 'default', 'default', False))
    def test_load_legacy_hub_module_v1_with_signature(self, model_format, signature, output_key, as_dict):
        if False:
            for i in range(10):
                print('nop')
        export_dir = os.path.join(self.get_temp_dir(), 'plus_one_' + model_format)
        _dispatch_model_format(model_format, _save_plus_one_saved_model_v2, _save_plus_one_hub_module_v1, export_dir)
        (inputs, expected_outputs) = (10.0, 11.0)
        layer = hub.KerasLayer(export_dir, signature=signature, output_key=output_key, signature_outputs_as_dict=as_dict)
        output = layer(inputs)
        if as_dict:
            self.assertEqual(output, {'default': expected_outputs})
        else:
            self.assertEqual(output, expected_outputs)

    @parameterized.parameters(('TF2SavedModel_SavedRaw', None, None, False), ('TF2SavedModel_SavedRaw', 'serving_default', None, True), ('TF2SavedModel_SavedRaw', 'serving_default', 'output_0', False))
    def test_load_callable_saved_model_v2_with_signature(self, model_format, signature, output_key, as_dict):
        if False:
            while True:
                i = 10
        export_dir = os.path.join(self.get_temp_dir(), 'plus_one_' + model_format)
        _dispatch_model_format(model_format, _save_plus_one_saved_model_v2, _save_plus_one_hub_module_v1, export_dir)
        (inputs, expected_outputs) = (10.0, 11.0)
        layer = hub.KerasLayer(export_dir, signature=signature, output_key=output_key, signature_outputs_as_dict=as_dict)
        output = layer(inputs)
        if as_dict:
            self.assertIsInstance(output, dict)
            self.assertEqual(output['output_0'], expected_outputs)
        else:
            self.assertEqual(output, expected_outputs)

    def test_load_callable_keras_default_saved_model_v2_with_signature(self):
        if False:
            while True:
                i = 10
        export_dir = os.path.join(self.get_temp_dir(), 'plus_one_keras_default')
        _save_plus_one_saved_model_v2_keras_default_callable(export_dir)
        (inputs, expected_outputs) = (10.0, 11.0)
        layer = hub.KerasLayer(export_dir, signature='plus_one', signature_outputs_as_dict=True)
        output = layer(inputs)
        self.assertIsInstance(output, dict)
        self.assertEqual(output['output_0'], expected_outputs)

    @parameterized.parameters(('TF1HubModule', None, None, True), ('TF1HubModule', None, None, False), ('TF1HubModule', 'default', None, True), ('TF1HubModule', None, 'default', False), ('TF1HubModule', 'default', 'default', False), ('TF2SavedModel_SavedRaw', None, None, False), ('TF2SavedModel_SavedRaw', 'serving_default', None, True), ('TF2SavedModel_SavedRaw', 'serving_default', 'output_0', False))
    def test_keras_layer_get_config(self, model_format, signature, output_key, as_dict):
        if False:
            return 10
        export_dir = os.path.join(self.get_temp_dir(), 'plus_one_' + model_format)
        _dispatch_model_format(model_format, _save_plus_one_saved_model_v2, _save_plus_one_hub_module_v1, export_dir)
        inputs = 10.0
        layer = hub.KerasLayer(export_dir, signature=signature, output_key=output_key, signature_outputs_as_dict=as_dict)
        outputs = layer(inputs)
        config = layer.get_config()
        new_layer = hub.KerasLayer.from_config(_json_cycle(config))
        new_outputs = new_layer(inputs)
        self.assertEqual(outputs, new_outputs)

    def test_keras_layer_fails_if_signature_output_not_specified(self):
        if False:
            print('Hello World!')
        export_dir = os.path.join(self.get_temp_dir(), 'saved_model_v2_mini')
        _save_plus_one_saved_model_v2(export_dir, save_from_keras=False)
        with self.assertRaisesRegex(ValueError, 'When using a signature, either output_key or signature_outputs_as_dict=True should be set.'):
            hub.KerasLayer(export_dir, signature='serving_default')

    def test_keras_layer_fails_if_with_outputs_as_dict_but_no_signature(self):
        if False:
            i = 10
            return i + 15
        export_dir = os.path.join(self.get_temp_dir(), 'saved_model_v2_mini')
        _save_plus_one_saved_model_v2(export_dir, save_from_keras=False)
        with self.assertRaisesRegex(ValueError, 'signature_outputs_as_dict is only valid if specifying a signature *'):
            hub.KerasLayer(export_dir, signature_outputs_as_dict=True)

    def test_keras_layer_fails_if_saved_model_v2_with_tags(self):
        if False:
            return 10
        export_dir = os.path.join(self.get_temp_dir(), 'saved_model_v2_mini')
        _save_plus_one_saved_model_v2(export_dir, save_from_keras=False)
        with self.assertRaises(ValueError):
            hub.KerasLayer(export_dir, signature=None, tags=['train'])

    def test_keras_layer_fails_if_setting_both_output_key_and_as_dict(self):
        if False:
            i = 10
            return i + 15
        export_dir = os.path.join(self.get_temp_dir(), 'saved_model_v2_mini')
        _save_plus_one_saved_model_v2(export_dir, save_from_keras=False)
        with self.assertRaisesRegex(ValueError, 'When using a signature, either output_key or signature_outputs_as_dict=True should be set.'):
            hub.KerasLayer(export_dir, signature='default', signature_outputs_as_dict=True, output_key='output')

    def test_keras_layer_fails_if_output_is_not_dict(self):
        if False:
            i = 10
            return i + 15
        export_dir = os.path.join(self.get_temp_dir(), 'saved_model_v2_mini')
        _save_plus_one_saved_model_v2(export_dir, save_from_keras=False)
        layer = hub.KerasLayer(export_dir, output_key='output_0')
        with self.assertRaisesRegex(ValueError, 'Specifying `output_key` is forbidden if output type *'):
            layer(10.0)

    def test_keras_layer_fails_if_output_key_not_in_layer_outputs(self):
        if False:
            print('Hello World!')
        export_dir = os.path.join(self.get_temp_dir(), 'hub_module_v1_mini')
        _save_plus_one_hub_module_v1(export_dir)
        layer = hub.KerasLayer(export_dir, output_key='unknown')
        with self.assertRaisesRegex(ValueError, 'KerasLayer output does not contain the output key*'):
            layer(10.0)

    def test_keras_layer_fails_if_hub_module_trainable(self):
        if False:
            return 10
        export_dir = os.path.join(self.get_temp_dir(), 'hub_module_v1_mini')
        _save_plus_one_hub_module_v1(export_dir)
        layer = hub.KerasLayer(export_dir, trainable=True)
        with self.assertRaisesRegex(ValueError, 'trainable.*=.*True.*unsupported'):
            layer(10.0)

    def test_keras_layer_fails_if_signature_trainable(self):
        if False:
            while True:
                i = 10
        export_dir = os.path.join(self.get_temp_dir(), 'saved_model_v2_mini')
        _save_plus_one_saved_model_v2(export_dir, save_from_keras=False)
        layer = hub.KerasLayer(export_dir, signature='serving_default', signature_outputs_as_dict=True, trainable=True)
        layer.trainable = True
        with self.assertRaisesRegex(ValueError, 'trainable.*=.*True.*unsupported'):
            layer(10.0)

    def test_keras_layer_logs_if_training_zero_variables(self):
        if False:
            return 10
        path = os.path.join(self.get_temp_dir(), 'zero-variables')
        _save_model_with_hparams(path)
        layer = hub.KerasLayer(path, trainable=True)
        if hasattr(self, 'assertLogs'):
            with self.assertLogs(level='ERROR') as logs:
                layer([[10.0]])
                layer([[10.0]])
            self.assertLen(logs.records, 1)
            self.assertRegexpMatches(logs.records[0].msg, 'zero trainable weights')
        else:
            layer([[10.0]])
            layer([[10.0]])
if __name__ == '__main__':
    tf.compat.v1.enable_v2_behavior()
    tf.test.main()