"""Utilities to use Modules as feature columns."""
import collections
import tensorflow as tf
from tensorflow_hub import image_util
from tensorflow_hub import module
from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_v2

class DenseFeatureColumn(feature_column._DenseColumn, feature_column_v2.DenseColumn):

    @property
    def dtype(self):
        if False:
            print('Hello World!')
        return tf.float32
_MODULE_RESOURCE_STRING = 'module'

def text_embedding_column(key, module_spec, trainable=False):
    if False:
        i = 10
        return i + 15
    'Uses a Module to construct a dense representation from a text feature.\n\n  TODO(b/131678043): This does not work yet with TF2.\n\n  This feature column can be used on an input feature whose values are strings\n  of arbitrary size.\n\n  The result of this feature column is the result of passing its `input`\n  through the module `m` instantiated from `module_spec`, as per\n  `result = m(input)`. The `result` must have dtype float32 and shape\n  `[batch_size, num_features]` with a known value of num_features.\n\n  Example:\n\n  ```python\n    comment = hub.text_embedding_column("comment", "/tmp/text-module")\n    feature_columns = [comment, ...]\n    ...\n    features = {\n      "comment": np.array(["wow, much amazing", "so easy", ...]),\n      ...\n    }\n    labels = np.array([[1], [0], ...])\n    # If running TF 2.x, use `tf.compat.v1.estimator.inputs.numpy_input_fn`\n    input_fn = tf.estimator.inputs.numpy_input_fn(features, labels,\n                                                  shuffle=True)\n    estimator = tf.estimator.DNNClassifier(hidden_units, feature_columns)\n    estimator.train(input_fn, max_steps=100)\n  ```\n\n  Args:\n    key: A string or `_FeatureColumn` identifying the text feature.\n    module_spec: A ModuleSpec defining the Module to instantiate or a path where\n      to load a ModuleSpec via `load_module_spec`\n    trainable: Whether or not the Module is trainable. False by default, meaning\n      the pre-trained weights are frozen. This is different from the ordinary\n      tf.feature_column.embedding_column(), but that one is intended for\n      training from scratch.\n\n  Returns:\n    `_DenseColumn` that converts from text input.\n\n  Raises:\n     ValueError: if module_spec is not suitable for use in this feature column.\n  '
    return _TextEmbeddingColumn(key=key, module_spec_path=module_spec, trainable=trainable)

def _check_module_is_text_embedding(module_spec):
    if False:
        return 10
    'Raises ValueError if `module_spec` is not a text-embedding module.\n\n  Args:\n    module_spec: A `ModuleSpec` to test.\n\n  Raises:\n    ValueError: if `module_spec` default signature is not compatible with\n    Tensor(string, shape=(?,)) -> Tensor(float32, shape=(?,K)).\n  '
    issues = []
    input_info_dict = module_spec.get_input_info_dict()
    if len(input_info_dict) != 1:
        issues.append('Module default signature must require only one input')
    else:
        (input_info,) = input_info_dict.values()
        input_shape = input_info.get_shape()
        if not (input_info.dtype == tf.string and input_shape.ndims == 1 and (input_shape.as_list() == [None])):
            issues.append('Module default signature must have only one input tf.Tensor(shape=(?,), dtype=string)')
    output_info_dict = module_spec.get_output_info_dict()
    if 'default' not in output_info_dict:
        issues.append("Module default signature must have a 'default' output.")
    else:
        output_info = output_info_dict['default']
        output_shape = output_info.get_shape()
        if not (output_info.dtype == tf.float32 and output_shape.ndims == 2 and (not output_shape.as_list()[0]) and output_shape.as_list()[1]):
            issues.append("Module default signature must have a 'default' output of tf.Tensor(shape=(?,K), dtype=float32).")
    if issues:
        raise ValueError('Module is not a text-embedding: %r' % issues)

class _TextEmbeddingColumn(DenseFeatureColumn, collections.namedtuple('_ModuleEmbeddingColumn', ('key', 'module_spec_path', 'trainable'))):
    """Returned by text_embedding_column(). Do not use directly."""

    def __init__(self, key, module_spec_path, trainable):
        if False:
            i = 10
            return i + 15
        self.module_spec = module.as_module_spec(self.module_spec_path)
        _check_module_is_text_embedding(self.module_spec)
        super().__init__()

    @property
    def _is_v2_column(self):
        if False:
            i = 10
            return i + 15
        return True

    @property
    def parents(self):
        if False:
            for i in range(10):
                print('nop')
        "See 'FeatureColumn` base class."
        return [self.key]

    @property
    def name(self):
        if False:
            return 10
        'Returns string. Used for variable_scope and naming.'
        if not hasattr(self, '_name'):
            key_name = self.key if isinstance(self.key, str) else self.key.name
            self._name = '{}_hub_module_embedding'.format(key_name)
        return self._name

    def create_state(self, state_manager):
        if False:
            i = 10
            return i + 15
        'Imports the module along with all variables.'
        trainable = self.trainable and state_manager._trainable
        m = module.Module(self.module_spec, trainable=trainable)
        state_manager.add_resource(self, _MODULE_RESOURCE_STRING, m)

    def _transform_feature(self, inputs):
        if False:
            i = 10
            return i + 15
        'Returns intermediate representation (usually a `Tensor`).'
        return inputs.get(self.key)

    def transform_feature(self, transformation_cache, state_manager):
        if False:
            for i in range(10):
                print('nop')
        return transformation_cache.get(self.key, state_manager)

    @property
    def _parse_example_spec(self):
        if False:
            print('Hello World!')
        'Returns a `tf.Example` parsing spec as dict.'
        return self.parse_example_spec

    @property
    def parse_example_spec(self):
        if False:
            while True:
                i = 10
        'Returns a `tf.Example` parsing spec as dict.'
        return {self.key: tf.compat.v1.FixedLenFeature([1], tf.string)}

    @property
    def _variable_shape(self):
        if False:
            for i in range(10):
                print('nop')
        '`TensorShape` of `_get_dense_tensor`, without batch dimension.'
        return self.variable_shape

    @property
    def variable_shape(self):
        if False:
            return 10
        '`TensorShape` of `_get_dense_tensor`, without batch dimension.'
        return self.module_spec.get_output_info_dict()['default'].get_shape()[1:]

    def _get_dense_tensor_for_input_tensor(self, input_tensor, text_module):
        if False:
            while True:
                i = 10
        text_batch = tf.reshape(input_tensor, shape=[-1])
        return text_module(text_batch)

    def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
        if False:
            print('Hello World!')
        'Returns a `Tensor`.'
        del weight_collections
        input_tensor = inputs.get(self)
        text_module = module.Module(self.module_spec, trainable=self.trainable and trainable)
        return self._get_dense_tensor_for_input_tensor(input_tensor, text_module)

    def get_dense_tensor(self, transformation_cache, state_manager):
        if False:
            return 10
        'Returns a `Tensor`.'
        input_tensor = transformation_cache.get(self, state_manager)
        text_module = state_manager.get_resource(self, _MODULE_RESOURCE_STRING)
        return self._get_dense_tensor_for_input_tensor(input_tensor, text_module)

    def get_config(self):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(self.module_spec_path, str):
            raise NotImplementedError('Can only generate a valid config for `hub.text_embedding_column`that uses a string `module_spec`.\n\nGot `type(module_spec)`: {}'.format(type(self.module_spec_path)))
        config = dict(zip(self._fields, self))
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None, columns_by_name=None):
        if False:
            while True:
                i = 10
        copied_config = config.copy()
        return cls(**copied_config)

def image_embedding_column(key, module_spec, image_size=None):
    if False:
        for i in range(10):
            print('nop')
    'Uses a Module to get a dense 1-D representation from the pixels of images.\n\n  TODO(b/131678043): This does not work yet with TF2.\n\n  This feature column can be used on images, represented as float32 tensors of\n  RGB pixel data in the range [0,1]. This can be read from a numeric_column()\n  if the tf.Example input data happens to have decoded images, all with the\n  same shape [height, width, 3]. More commonly, the input_fn will have code to\n  explicitly decode images, resize them (possibly after performing data\n  augmentation such as random crops etc.), and provide a batch of shape\n  [batch_size, height, width, 3].\n\n  The result of this feature column is the result of passing its `input`\n  through the module `m` instantiated from `module_spec`, as per\n  `result = m({"images": input})`. The `result` must have dtype float32 and\n  shape `[batch_size, num_features]` with a known value of num_features.\n\n  Example:\n\n  ```python\n    image_column = hub.image_embedding_column("embeddings", "/tmp/image-module")\n    feature_columns = [image_column, ...]\n    estimator = tf.estimator.LinearClassifier(feature_columns, ...)\n    height, width = hub.get_expected_image_size(image_column.module_spec)\n    input_fn = ...  # Provides "embeddings" with shape [None, height, width, 3].\n    estimator.train(input_fn, ...)\n  ```\n\n  Args:\n    key: A string or `_FeatureColumn` identifying the input image data.\n    module_spec: A string handle or a `ModuleSpec` identifying the module.\n    image_size: Optional. If specified it should be a tuple of image height and\n        width to use with the module. Note that it depends on the module on\n        whether the default size can be overridden and what the permissible\n        values are.\n\n  Returns:\n    `_DenseColumn` that converts from pixel data.\n\n  Raises:\n     ValueError: if module_spec is not suitable for use in this feature column.\n  '
    if isinstance(image_size, list):
        image_size = tuple(image_size)
    return _ImageEmbeddingColumn(key=key, module_spec_path=module_spec, image_size=image_size)

def _check_module_is_image_embedding(module_spec, check_image_size):
    if False:
        for i in range(10):
            print('nop')
    'Raises ValueError if `module_spec` is not usable as image embedding.\n\n  Args:\n    module_spec: A `_ModuleSpec` to test.\n    check_image_size: Whether to check for compatibility with\n        get_expected_image_size.\n\n  Raises:\n    ValueError: if `module_spec` default signature is not compatible with\n        mappingan "images" input to a Tensor(float32, shape=(_,K)).\n  '
    issues = []
    input_info_dict = module_spec.get_input_info_dict()
    if list(input_info_dict.keys()) != ['images'] or input_info_dict['images'].dtype != tf.float32:
        issues.append("Module 'default' signature must require a single input, which must have type float32 and name 'images'.")
    else:
        try:
            if check_image_size:
                image_util.get_expected_image_size(module_spec)
        except ValueError as e:
            issues.append('Module does not support hub.get_expected_image_size(); original error was:\n' + str(e))
    output_info_dict = module_spec.get_output_info_dict()
    if 'default' not in output_info_dict:
        issues.append("Module 'default' signature must have a 'default' output.")
    else:
        output_type = output_info_dict['default'].dtype
        output_shape = output_info_dict['default'].get_shape()
        if not (output_type == tf.float32 and output_shape.ndims == 2 and output_shape.dims[1].value):
            issues.append("Module 'default' signature must have a 'default' output of tf.Tensor(shape=(_,K), dtype=float32).")
    if issues:
        raise ValueError('Module is not usable as image embedding: %r' % issues)

class _ImageEmbeddingColumn(DenseFeatureColumn, collections.namedtuple('_ImageEmbeddingColumn', ('key', 'module_spec_path', 'image_size'))):
    """Returned by image_embedding_column(). Do not use directly."""

    def __init__(self, key, module_spec_path, image_size):
        if False:
            print('Hello World!')
        self.module_spec = module.as_module_spec(self.module_spec_path)
        _check_module_is_image_embedding(self.module_spec, check_image_size=self.image_size is None)
        super().__init__()

    @property
    def _is_v2_column(self):
        if False:
            return 10
        return True

    @property
    def parents(self):
        if False:
            print('Hello World!')
        "See 'FeatureColumn` base class."
        return [self.key]

    @property
    def name(self):
        if False:
            print('Hello World!')
        'Returns string. Used for variable_scope and naming.'
        if not hasattr(self, '_name'):
            key_name = self.key if isinstance(self.key, str) else self.key.name
            self._name = '{}_hub_module_embedding'.format(key_name)
        return self._name

    def create_state(self, state_manager):
        if False:
            while True:
                i = 10
        'Imports the module along with all variables.'
        m = module.Module(self.module_spec)
        state_manager.add_resource(self, _MODULE_RESOURCE_STRING, m)

    def _transform_feature(self, inputs):
        if False:
            print('Hello World!')
        'Returns intermediate representation (usually a `Tensor`).'
        return inputs.get(self.key)

    def transform_feature(self, transformation_cache, state_manager):
        if False:
            for i in range(10):
                print('nop')
        return transformation_cache.get(self.key, state_manager)

    @property
    def _parse_example_spec(self):
        if False:
            return 10
        'Returns a `tf.Example` parsing spec as dict.'
        return self.parse_example_spec

    @property
    def parse_example_spec(self):
        if False:
            while True:
                i = 10
        'Returns a `tf.Example` parsing spec as dict.'
        if self.image_size:
            (height, width) = self.image_size
        else:
            (height, width) = image_util.get_expected_image_size(self.module_spec)
        input_shape = [height, width, 3]
        return {self.key: tf.compat.v1.FixedLenFeature(input_shape, tf.float32)}

    @property
    def _variable_shape(self):
        if False:
            while True:
                i = 10
        '`TensorShape` of `_get_dense_tensor`, without batch dimension.'
        return self.variable_shape

    @property
    def variable_shape(self):
        if False:
            i = 10
            return i + 15
        '`TensorShape` of `_get_dense_tensor`, without batch dimension.'
        return self.module_spec.get_output_info_dict()['default'].get_shape()[1:]

    def _get_dense_tensor_for_images(self, images, image_module):
        if False:
            print('Hello World!')
        return image_module({'images': images})

    def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
        if False:
            print('Hello World!')
        del weight_collections, trainable
        images = inputs.get(self)
        image_module = module.Module(self.module_spec)
        return self._get_dense_tensor_for_images(images, image_module)

    def get_dense_tensor(self, transformation_cache, state_manager):
        if False:
            while True:
                i = 10
        images = transformation_cache.get(self, state_manager)
        image_module = state_manager.get_resource(self, _MODULE_RESOURCE_STRING)
        return self._get_dense_tensor_for_images(images, image_module)

    def get_config(self):
        if False:
            i = 10
            return i + 15
        if not isinstance(self.module_spec_path, str):
            raise NotImplementedError('Can only generate a valid config for `hub.image_embedding_column`that uses a string `module_spec`.\n\nGot `type(module_spec)`: {}'.format(type(self.module_spec_path)))
        config = dict(zip(self._fields, self))
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None, columns_by_name=None):
        if False:
            return 10
        copied_config = config.copy()
        return cls(**copied_config)

def sparse_text_embedding_column(key, module_spec, combiner, default_value, trainable=False):
    if False:
        while True:
            i = 10
    'Uses a Module to construct dense representations from sparse text features.\n\n  TODO(b/131678043): This does not work yet with TF2.\n\n  The input to this feature column is a batch of multiple strings with\n  arbitrary size, assuming the input is a SparseTensor.\n\n  This type of feature column is typically suited for modules that operate on\n  pre-tokenized text to produce token level embeddings which are combined with\n  the combiner into a text embedding. The combiner always treats the tokens as a\n  bag of words rather than a sequence.\n\n  The output (i.e., transformed input layer) is a DenseTensor, with shape\n  [batch_size, num_embedding_dim].\n\n  For Example:\n\n  ```python\n    comment = hub.sparse_text_embedding_column("comment", "/tmp/text_module")\n    feature_columns = [comment, ...]\n    ...\n    features = {\n      "comment": tf.SparseTensor(indices=[[0, 0], [1, 2]],\n                                 values=[\'sparse\', \'embedding\'],\n                                 dense_shape=[3, 4]),\n      ...\n    }\n    estimator = tf.estimator.DNNClassifier(hidden_units, feature_columns)\n  ```\n\n  Args:\n    key: A string or `_FeatureColumn` identifying the text feature.\n    module_spec: A string handle or a `_ModuleSpec` identifying the module.\n    combiner: a string specifying reducing op for embeddings in the same\n      Example. Currently, \'mean\', \'sqrtn\', \'sum\' are supported. Using\n      combiner=None is undefined.\n    default_value: default value for Examples where the text feature is empty.\n      Note, it\'s recommended to have default_value consistent OOV tokens, in\n      case there was special handling of OOV in the text module. If None, the\n      text feature is assumed be non-empty for each Example.\n    trainable: Whether or not the Module is trainable. False by default, meaning\n      the pre-trained weights are frozen. This is different from the ordinary\n      tf.feature_column.embedding_column(), but that one is intended for\n      training from scratch.\n\n  Returns:\n    `_DenseColumn` that converts from text input.\n\n  Raises:\n     ValueError: if module_spec is not suitable for use in this feature column.\n     ValueError: if combiner not in (\'mean\', \'sqrtn\', \'sum\').\n  '
    module_spec = module.as_module_spec(module_spec)
    _check_module_is_text_embedding(module_spec)
    if combiner not in ('mean', 'sqrtn', 'sum'):
        raise ValueError("combiner must be 'mean', 'sqrtn' or 'sum': %r" % combiner)
    return _SparseTextEmbeddingColumn(key=key, module_spec=module_spec, trainable=trainable, default_value=default_value, combiner=combiner)

class _SparseTextEmbeddingColumn(DenseFeatureColumn, collections.namedtuple('_ModuleEmbeddingColumn', ('key', 'combiner', 'module_spec', 'default_value', 'trainable'))):
    """Returned by sparse_text_embedding_column(). Do not use directly."""

    @property
    def _is_v2_column(self):
        if False:
            return 10
        return True

    @property
    def parents(self):
        if False:
            for i in range(10):
                print('nop')
        "See 'FeatureColumn` base class."
        return [self.key]

    @property
    def name(self):
        if False:
            while True:
                i = 10
        'Returns string. Used for variable_scope and naming.'
        if not hasattr(self, '_name'):
            key_name = self.key if isinstance(self.key, str) else self.key.name
            self._name = '{}_hub_module_embedding'.format(key_name)
        return self._name

    def _transform_feature(self, inputs):
        if False:
            print('Hello World!')
        'Returns intermediate representation (usually a `Tensor`).'
        return inputs.get(self.key)

    def transform_feature(self, transformation_cache, state_manager):
        if False:
            while True:
                i = 10
        return transformation_cache.get(self.key, state_manager)

    @property
    def _parse_example_spec(self):
        if False:
            return 10
        'Returns a `tf.Example` parsing spec as dict.'
        return self.parse_example_spec

    @property
    def parse_example_spec(self):
        if False:
            while True:
                i = 10
        'Returns a `tf.Example` parsing spec as dict.'
        return {self.key: tf.compat.v1.VarLenFeature(tf.string)}

    @property
    def _variable_shape(self):
        if False:
            while True:
                i = 10
        '`TensorShape` of `_get_dense_tensor`, without batch dimension.'
        return self.variable_shape

    @property
    def variable_shape(self):
        if False:
            print('Hello World!')
        '`TensorShape` of `_get_dense_tensor`, without batch dimension.'
        return self.module_spec.get_output_info_dict()['default'].get_shape()[1:]

    def _get_dense_tensor_for_inputs(self, text_batch, trainable):
        if False:
            i = 10
            return i + 15
        m = module.Module(self.module_spec, trainable=self.trainable and trainable)
        if self.default_value is not None:
            text_batch = tf.sparse.fill_empty_rows(text_batch, self.default_value)[0]
        embedded_tokens = m(text_batch.values)
        embedding_ids = tf.SparseTensor(indices=text_batch.indices, values=tf.range(tf.shape(text_batch.indices)[0], dtype=tf.int32), dense_shape=text_batch.dense_shape)
        return tf.nn.embedding_lookup_sparse(params=embedded_tokens, sp_ids=embedding_ids, sp_weights=None, combiner=self.combiner)

    def _get_dense_tensor(self, inputs, weight_collections=None, trainable=None):
        if False:
            while True:
                i = 10
        'Returns a `Tensor`.'
        del weight_collections
        text_batch = inputs.get(self)
        return self._get_dense_tensor_for_inputs(text_batch, self.trainable and trainable)

    def get_dense_tensor(self, transformation_cache, state_manager):
        if False:
            return 10
        'Returns a `Tensor`.'
        input_tensor = transformation_cache.get(self, state_manager)
        return self._get_dense_tensor_for_inputs(input_tensor, self.trainable)