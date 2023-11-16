import sys
import tensorflow as tf
from tensorflow.python.data.util import nest
from tensorflow.python.framework import tensor_shape
from nvidia.dali import types
from nvidia.dali import internal as _internal
from nvidia.dali.external_source import _is_external_source, _has_external_source
from nvidia.dali.external_source import _is_external_source_with_callback
from nvidia.dali._utils.external_source_impl import _get_generator_from_source_desc
from nvidia.dali._utils.external_source_impl import _cycle_enabled
from distutils.version import LooseVersion
import warnings
from nvidia.dali_tf_plugin import dali_tf_plugin
from collections.abc import Mapping, Iterable
_dali_tf_module = dali_tf_plugin.load_dali_tf_plugin()
_dali_tf = _dali_tf_module.dali
_dali_tf.__doc__ = _dali_tf.__doc__ + '\n\n    Please keep in mind that TensorFlow allocates almost all available device memory by default.\n    This might cause errors in DALI due to insufficient memory. On how to change this behaviour\n    please look into the TensorFlow documentation, as it may differ based on your use case.\n'
_experimental_dataset_docstring = 'Experimental variant of\n:class:`~nvidia.dali.plugin.tf.DALIDataset`. This dataset adds support for input tf.data.Datasets.\nSupport for input tf.data.Datasets is available only for TensorFlow 2.4.1 and newer.\n\n**Input dataset specification**\n\nEach of the input datasets must be mapped to a :meth:`~nvidia.dali.fn.external_source` operator\nthat will represent the input to the DALI pipeline. In the pipeline the input is represented as\nthe ``name`` parameter of :meth:`~nvidia.dali.fn.external_source`. Input datasets must be provided\nas a mapping from that ``name`` to the dataset object via the ``input_datasets`` dictionary\nargument of DALIDatasetWithInputs.\n\n**Per-sample and batch mode**\n\nThe input datasets can operate in per-sample mode or in batch mode.\n\nIn per-sample mode, the values produced by the source dataset are interpreted\nas individual samples. The batch dimension is absent. For example, a 640x480 RGB image would\nhave a shape ``[480, 640, 3]``.\n\nIn batch mode, the tensors produced by the source dataset are interpreted as batches,\nwith an additional outer dimension denoting the samples in the batch. For example, a batch of\nten 640x480 RGB images would have a shape ``[10, 480, 640, 3]``.\n\nIn both cases (per-sample and batch mode), the layout of those inputs should be denoted as "HWC".\n\nIn per-sample mode DALIDataset will query the inputs dataset ``batch_size``-times to build a batch\nthat would be fed into the DALI Pipeline.\nIn per-sample mode, each sample produced by the input dataset can have a different shape,\nbut the number of dimension and the layout must remain constant.\n\n**External Source with** ``source`` **parameter**\n\nThis experimental DALIDataset accepts pipelines with :meth:`~nvidia.dali.fn.external_source`\nnodes that have ``source`` parameter specified.\nIn that case, the ``source`` will be converted automatically into appropriate\n``tf.data.Dataset.from_generator`` dataset with correct placement and\n``tf.data.experimental.copy_to_device`` directives.\n\nThose nodes can also work in per-sample or in batch mode. The data in batch mode must be\na dense, uniform tensor (each sample has the same dimensions). Only CPU data is accepted.\n\nThis allows TensorFlow DALIDataset to work with most Pipelines that have External Source\n``source`` already specified.\n\n.. warning::\n    This class is experimental and its API might change without notice.\n\n.. note::\n    External source nodes with ``num_outputs`` specified to any number are not\n    supported - this means that callbacks with multiple (tuple) outputs are not supported.\n\n.. note::\n    External source ``cycle`` policy ``\'raise\'`` is not supported - the dataset is not restartable.\n\n.. note::\n    External source ``cuda_stream`` parameter is ignored - ``source`` is supposed to return\n    CPU data and tf.data.Dataset inputs are handled internally.\n\n.. note::\n    External source ``use_copy_kernel`` and ``blocking`` parameters are ignored.\n\n.. note::\n    Setting ``no_copy`` on the external source nodes when defining the pipeline is considered\n    a no-op when used with DALI Dataset. The ``no_copy`` option is handled internally\n    and enabled automatically if possible.\n\n.. note::\n    Parallel execution of external source callback provided via ``source`` is not supported.\n    The callback is executed via TensorFlow ``tf.data.Dataset.from_generator`` - the ``parallel``\n    and ``prefetch_queue_depth`` parameters are ignored.\n\n\nThe operator adds additional parameters to the ones supported by the\n:class:`~nvidia.dali.plugin.tf.DALIDataset`:\n\nParameters\n----------\n    input_datasets : dict[str, tf.data.Dataset] or\n                     dict[str, nvidia.dali.plugin.tf.experimental.Input]\n        input datasets to the DALI Pipeline. It must be provided as a dictionary mapping from\n        the names of the ``External Source`` nodes to the datasets objects or to the\n        :meth:`~nvidia.dali.plugin.tf.experimental.Input` wrapper.\n\n        For example::\n\n            {\n                \'tensor_input\': tf.data.Dataset.from_tensors(tensor).repeat(),\n                \'generator_input\': tf.data.Dataset.from_generator(some_generator)\n            }\n\n        can be passed as ``input_datasets`` for Pipeline like::\n\n            @pipeline_def\n            def external_source_pipe():\n                input_0 = fn.external_source(name=\'tensor_input\')\n                input_1 = fn.external_source(name=\'generator_input\')\n                return fn.resize(input_1, resize_x=input_0)\n\n        Entries that use ``tf.data.Dataset`` directly, like::\n\n            {\n                \'input\': tf.data.Dataset.from_tensors(tensor)\n            }\n\n        are equivalent to following specification using\n        ``nvidia.dali.plugin.tf.experimental.Input``::\n\n            {\n                \'input\' : nvidia.dali.plugin.tf.experimental.Input(\n                              dataset=tf.data.Dataset.from_tensors(tensor),\n                              layout=None,\n                              batch=False)\n            }\n\n        This means that inputs, specified as ``tf.data.Dataset`` directly, are considered\n        sample inputs.\n\n        .. warning::\n            Input dataset must be placed on the same device as ``DALIDatasetWithInputs``.\n            If the input has different placement (for instance, input is placed on CPU, while\n            ``DALIDatasetWithInputs`` is placed on GPU) the ``tf.data.experimental.copy_to_device``\n            with GPU argument must be first applied to input.\n'
_experimental_input_docstring = 'Wrapper for an input passed to DALIDataset.\nAllows to pass additional options that can override some of the ones specified\nin the External Source node in the Python Pipeline object.\nPassing None indicates, that the value should be looked up in the pipeline definition.\n\nParameters\n----------\ndataset : tf.data.Dataset\n    The dataset used as an input\nlayout : str, optional, default = None\n    Layout of the input. If None, the layout will be taken from the corresponding\n    External Source node in the Python Pipeline object. If both are provided,\n     the layouts must be the same.\n    If neither is provided, empty layout will be used.\nbatch: bool, optional, default = False\n    Batch mode of a given input. If None, the batch mode will be taken from the\n    corresponding External Source node in the Python Pipeline object.\n\n    If the ``batch = False``, the input dataset is considered sample input.\n\n    If the ``batch = True``, the input dataset is expected to return batches.\n'

def serialize_pipeline(pipeline):
    if False:
        print('Hello World!')
    try:
        return pipeline.serialize()
    except RuntimeError as e:
        raise RuntimeError('Error during pipeline initialization. Note that some operators (e.g. Python Operators) cannot be used with TensorFlow Dataset API and DALIIterator.') from e

def DALIIteratorWrapper(pipeline=None, serialized_pipeline=None, sparse=[], shapes=[], dtypes=[], batch_size=-1, prefetch_queue_depth=2, **kwargs):
    if False:
        print('Hello World!')
    '\n  TF Plugin Wrapper\n\n  This operator works in the same way as DALI TensorFlow plugin, with the exception that it also\n  accepts Pipeline objects as an input, which are serialized internally. For more information,\n  see :meth:`nvidia.dali.plugin.tf.DALIRawIterator`.\n  '
    if type(prefetch_queue_depth) is dict:
        exec_separated = True
        cpu_prefetch_queue_depth = prefetch_queue_depth['cpu_size']
        gpu_prefetch_queue_depth = prefetch_queue_depth['gpu_size']
    elif type(prefetch_queue_depth) is int:
        exec_separated = False
        cpu_prefetch_queue_depth = -1
        gpu_prefetch_queue_depth = prefetch_queue_depth
    if serialized_pipeline is None:
        serialized_pipeline = serialize_pipeline(pipeline)
    if (not isinstance(shapes, Iterable) or len(shapes) == 0) and batch_size == -1:
        raise Exception('shapes and batch_size arguments cannot be empty, please provide at leas one shape argument element with the BATCH size or set batch_size')
    if len(sparse) > 0 and sparse[0] and (batch_size == -1):
        if isinstance(shapes[0], Iterable) and len(shapes[0]) == 1:
            shapes[0] = (shapes[0][0], 1)
        else:
            shapes[0] = (shapes[0], 1)
    new_dtypes = []
    new_shapes = []
    for i in range(len(dtypes)):
        if i < len(sparse) and sparse[i]:
            new_dtypes.append(tf.int64)
            new_dtypes.append(dtypes[i])
            new_dtypes.append(tf.int64)
            if len(shapes) > i and len(shapes[i]) > 0:
                new_shapes.append((shapes[i][0], 1))
                new_shapes.append(shapes[i][0])
            else:
                new_shapes.append(())
                new_shapes.append(())
            new_shapes.append(())
        else:
            new_dtypes.append(dtypes[i])
            if len(shapes) > i:
                new_shapes.append(shapes[i])
    out = _dali_tf(serialized_pipeline=serialized_pipeline, shapes=new_shapes, dtypes=new_dtypes, sparse=sparse, batch_size=batch_size, exec_separated=exec_separated, gpu_prefetch_queue_depth=gpu_prefetch_queue_depth, cpu_prefetch_queue_depth=cpu_prefetch_queue_depth, **kwargs)
    new_out = []
    j = 0
    for i in range(len(dtypes)):
        if i < len(sparse) and sparse[i]:
            new_out.append(tf.SparseTensor(indices=out[j], values=out[j + 1], dense_shape=out[j + 2]))
            j += 3
        else:
            new_out.append(out[j])
            j += 1
    return new_out

def DALIIterator():
    if False:
        return 10
    return DALIIteratorWrapper

def DALIRawIterator():
    if False:
        return 10
    return _dali_tf

def _get_tf_version():
    if False:
        for i in range(10):
            print('nop')
    return LooseVersion(tf.__version__)
MIN_TENSORFLOW_VERSION = LooseVersion('1.15')

def dataset_compatible_tensorflow():
    if False:
        for i in range(10):
            print('nop')
    'Returns ``True`` if current TensorFlow version is compatible with DALIDataset.'
    return LooseVersion(tf.__version__) >= MIN_TENSORFLOW_VERSION

def dataset_inputs_compatible_tensorflow():
    if False:
        for i in range(10):
            print('nop')
    'Returns ``True`` if the current TensorFlow version is compatible with\n    experimental.DALIDatasetWithInputs and input Datasets can be used with DALI.\n    '
    return LooseVersion(tf.__version__) >= LooseVersion('2.4.1')

def dataset_distributed_compatible_tensorflow():
    if False:
        for i in range(10):
            print('nop')
    'Returns ``True`` if the tf.distribute APIs for current TensorFlow version are compatible\n    with DALIDataset.\n    '
    return LooseVersion(tf.__version__) >= LooseVersion('2.5.0')

def _get_experimental():
    if False:
        return 10
    current_module = sys.modules[__name__]
    experimental = _internal.get_submodule(current_module, 'experimental')
    return experimental

def _insert_experimental_member(member, name):
    if False:
        return 10
    experimental_module = _get_experimental()
    member.__module__ = experimental_module
    setattr(experimental_module, name, member)

def _get_external_source_param(input_name, input_value, name_es_map, param_name):
    if False:
        i = 10
        return i + 15
    'Get value of the parameter `param_name` specified for the External Source node\n       named `input_name`. It can be specified either via `input_value` or in the op instance\n       passed in `name_es_map`.\n       Not `None` value in `input_value` overwrites the one specified in the Operator instances.\n       Otherwise, the one from pipeline definition (the op instance) is used.\n\n    Parameters\n    ----------\n    input_name : str\n        Name of the input\n    input_value : Input, optional\n        Description of the input\n    name_es_map : dict[str, ExternalSource]\n        Mapping from the External Source names to operator nodes.\n    param_name : str\n        name of the parameter we want to access\n    '

    def get_param_from_pipe(input_name, name_es_map, param_name):
        if False:
            while True:
                i = 10
        es_op = name_es_map[input_name]
        try:
            return getattr(es_op, '_' + param_name)
        except AttributeError:
            return getattr(es_op._op, '_' + param_name, None)
    if input_value is None or getattr(input_value, param_name) is None:
        return get_param_from_pipe(input_name, name_es_map, param_name)
    else:
        return getattr(input_value, param_name)

def _get_signature(dtype, shape):
    if False:
        return 10
    return tf.TensorSpec(shape=shape, dtype=dtype)

def _get_current_device_spec():
    if False:
        for i in range(10):
            print('nop')
    'Best guess at checking the current device string in eager and graph mode.\n\n    Using callable in `with tf.device(...)` for Graph mode will probably break it.\n    The graph in use is assumed to be current default graph.\n    '
    if tf.executing_eagerly():
        dummy_context_manager = tf.device(None)
        context = dummy_context_manager._ctx
        return context.device_spec
    else:
        g = tf.compat.v1.get_default_graph()
        spec = g._device_function_stack.peek_top_obj()
        return tf.DeviceSpec.from_string(spec.display_name)
if dataset_compatible_tensorflow():
    from tensorflow.python.framework import ops
    from tensorflow.python.data.ops import dataset_ops
    from tensorflow.python.data.util import structure
    import functools

    def dataset_options():
        if False:
            i = 10
            return i + 15
        options = tf.data.Options()
        options.experimental_optimization.apply_default_optimizations = False
        if hasattr(options.experimental_optimization, 'autotune'):
            options.experimental_optimization.autotune = False
        else:
            options.autotune.enabled = False
        return options

    class _DALIDatasetV2(dataset_ops.DatasetV2):

        def __init__(self, pipeline, output_dtypes=None, output_shapes=None, fail_on_device_mismatch=True, *, input_datasets=None, batch_size=1, num_threads=4, device_id=0, exec_separated=False, prefetch_queue_depth=2, cpu_prefetch_queue_depth=2, gpu_prefetch_queue_depth=2, dtypes=None, shapes=None):
            if False:
                return 10
            output_shapes = self._handle_deprecation(output_shapes, shapes, 'shapes')
            output_dtypes = self._handle_deprecation(output_dtypes, dtypes, 'dtypes')
            if not self._check_dtypes(output_dtypes, tf.DType):
                raise TypeError(f'`output_dtypes` should be provided as single tf.DType value or a tuple of tf.DType values. Got value `{output_dtypes}` of the type `{type(output_dtypes)}`.')
            if output_shapes is None:
                output_shapes = nest.map_structure(lambda _: tensor_shape.TensorShape(None), output_dtypes)
            else:
                output_shapes = nest.map_structure_up_to(output_dtypes, tensor_shape.as_shape, output_shapes)
            if not isinstance(output_dtypes, tuple):
                output_dtypes = (output_dtypes,)
                output_shapes = (output_shapes,)
            output_classes = nest.map_structure(lambda _: ops.Tensor, output_dtypes)
            self._pipeline_instance = pipeline
            self._pipeline_serialized = serialize_pipeline(pipeline)
            self._batch_size = batch_size
            self._num_threads = num_threads
            if device_id is None:
                device_id = types.CPU_ONLY_DEVICE_ID
            self._device_id = device_id
            self._exec_separated = exec_separated
            self._prefetch_queue_depth = prefetch_queue_depth
            self._cpu_prefetch_queue_depth = cpu_prefetch_queue_depth
            self._gpu_prefetch_queue_depth = gpu_prefetch_queue_depth
            self._output_shapes = output_shapes
            self._output_dtypes = output_dtypes
            self._fail_on_device_mismatch = fail_on_device_mismatch
            self._setup_inputs(input_datasets)
            self._structure = structure.convert_legacy_structure(self._output_dtypes, self._output_shapes, output_classes)
            super(_DALIDatasetV2, self).__init__(self._as_variant_tensor())

        def _input_lists_from_input_datasets(self, input_datasets, name_es_map):
            if False:
                for i in range(10):
                    print('nop')
            'Extract the input specification from the input_datasets dictionary.\n\n            Validate if the inputs exist in the pipeline and the types are correct\n\n            Returns\n            -------\n            list, list, list, list\n                input_datasets, input_names, input_layouts, input_batched\n            '
            if input_datasets is None:
                return ([], [], [], [])

            def _get_dataset(value):
                if False:
                    for i in range(10):
                        print('nop')
                if isinstance(value, dataset_ops.DatasetV2):
                    return value
                else:
                    return value.dataset
            in_datasets_list = []
            in_names_list = []
            in_layouts_list = []
            in_batched_list = []
            error_str = '`input_datasets` must be a dictionary that maps input names (the `name` specified for External Source node in DALI pipeline) to input datasets objects (`tf.data.Dataset`) or `nvidia.dali.plugin.tf.experimental.Input` wrapper objects'
            if not isinstance(input_datasets, Mapping):
                raise TypeError(error_str + f', got: `{input_datasets}` of type: {{type(input_datasets)}} instead.')
            for (input_name, input_value) in input_datasets.items():
                if not isinstance(input_name, str):
                    raise TypeError(error_str + f'. Expected the keys (representing the input names) to be of type `str`, got: `{input_name}` of type: {input_name} instead.')
                is_dataset_only = isinstance(input_value, dataset_ops.DatasetV2)
                experimental = _get_experimental()
                if not is_dataset_only and (not isinstance(input_value, experimental.Input)):
                    raise TypeError(error_str + f'. Expected the values of the dictionary (representing the inputs) to be of type `tf.data.Dataset` or `nvidia.dali.plugin.tf.Input` got: `{input_value}` of type: {type(input_value)} instead.')
                if input_name not in name_es_map.keys():
                    raise ValueError(f"Did not find an External Source placeholder node with name='{input_name}' in the provided pipeline - required by the name specified in the `input_datasets`. Names of available placeholder External Source nodes are: {list(name_es_map.keys())}. Placeholder nodes cannot have `source` argument specified.")
                in_names_list.append(input_name)
                in_datasets_list.append(_get_dataset(input_value))
                if is_dataset_only:
                    as_input = experimental.Input(input_value, layout=None, batch=False)
                else:
                    as_input = input_value
                layout = _get_external_source_param(input_name, as_input, name_es_map, 'layout')
                in_layouts_list.append(layout or '')
                batched = _get_external_source_param(input_name, as_input, name_es_map, 'batch')
                in_batched_list.append(batched if batched is not None else True)
            return (in_datasets_list, in_names_list, in_layouts_list, in_batched_list)

        def _input_lists_from_source(self, callbacked_es_map):
            if False:
                print('Hello World!')
            dali_device_spec = _get_current_device_spec()
            is_dali_on_gpu = dali_device_spec.device_type == 'GPU'
            in_datasets_list = []
            in_names_list = []
            in_layouts_list = []
            in_batched_list = []
            for (input_name, external_source) in callbacked_es_map.items():
                in_names_list.append(input_name)
                layout = _get_external_source_param(input_name, None, callbacked_es_map, 'layout')
                in_layouts_list.append(layout or '')
                batched = _get_external_source_param(input_name, None, callbacked_es_map, 'batch')
                in_batched_list.append(batched if batched is not None else True)
                source_desc = external_source._op._source_desc
                if source_desc.cycle == 'raise':
                    raise NotImplementedError(f"External Source node: '{input_name}' got argument cycle='raise' which is not supported.")
                with tf.device('/cpu:0'):
                    (tf_gen, dtype, shape) = _get_generator_from_source_desc(source_desc, self._batch_size, external_source._batch)
                    signature = _get_signature(dtype, shape)
                    dataset = tf.data.Dataset.from_generator(tf_gen, output_signature=signature)
                    if _cycle_enabled(source_desc.cycle):
                        dataset = dataset.repeat()
                    if is_dali_on_gpu:
                        dataset = dataset.apply(tf.data.experimental.copy_to_device(dali_device_spec.to_string()))
                    in_datasets_list.append(dataset)
            return (in_datasets_list, in_names_list, in_layouts_list, in_batched_list)

        def _setup_inputs(self, input_datasets):
            if False:
                i = 10
                return i + 15
            'Verify the input specification and assign it to private members in\n            normalized form.\n            '
            has_es = _has_external_source(self._pipeline_instance)
            if input_datasets is None and (not has_es):
                self._input_datasets = ()
                self._input_names = ()
                self._input_layouts = ()
                self._input_batched = ()
                return
            self._assert_pipeline_instance()
            if input_datasets is None:
                input_datasets = {}
            (name_es_map, callbacked_es_map) = self._get_name_es_instance_map()
            inputs_from_dict = self._input_lists_from_input_datasets(input_datasets, name_es_map)
            inputs_from_source = self._input_lists_from_source(callbacked_es_map)
            if not input_datasets.keys().isdisjoint(callbacked_es_map.keys()):
                overlapped = input_datasets.keys().intersection(callbacked_es_map.keys())
                raise ValueError(f'Double specification of External Source input is not allowed. External Source nodes named: `{overlapped}` got inputs specified via `input_datasets` DALIDataset argument and ExternalSource `source` argument at the same time.')
            non_matched = set(name_es_map.keys()) - set(input_datasets.keys()) - set(callbacked_es_map.keys())
            if len(non_matched) != 0:
                raise ValueError(f'Found External Source nodes in the Pipeline, that were not assigned any inputs. Nodes without inputs: \n{list(non_matched)}.\nNodes that were assigned inputs:\n{list(input_datasets.keys())}.')
            self._input_datasets = tuple(inputs_from_dict[0] + inputs_from_source[0])
            self._input_names = tuple(inputs_from_dict[1] + inputs_from_source[1])
            self._input_layouts = tuple(inputs_from_dict[2] + inputs_from_source[2])
            self._input_batched = tuple((int(b) for b in inputs_from_dict[3] + inputs_from_source[3]))

        def _assert_pipeline_instance(self):
            if False:
                print('Hello World!')
            'Ensure that the pipeline is built, and check if the Python part is available.\n            '
            self._pipeline_instance.build()
            if not self._pipeline_instance._py_graph_built and self._pipeline_instance._built:
                raise ValueError('Deserialized pipelines cannot be used with `input_datasets`. Please provide a pipeline that was created directly in Python and not recreated from serialized one.')

        def _assert_correct_external_sources(self, external_source):
            if False:
                return 10
            'Validate that the external source nodes used are properly configured'
            if external_source._op._num_outputs is not None:
                raise ValueError('Found placeholder External Source node (without `source` argument) in the Pipeline that was created with `num_outputs` `num_outputs` parameter. Only single-output (with `num_outputs=None`), named (with `name` argument specified) External Source nodes are supported as inputs placeholders for DALIDataset integration. Alternatively, External Source can be used with `source` parameter specified.')
            if external_source._op._name is None:
                raise ValueError('Found placeholder External Source node (without `source` argument) in the Pipeline that was not named (no `name` argument set). Only single-output (with `num_outputs=None`), named (with `name` argument specified) External Source nodes are supported as inputs placeholders for DALIDataset integration. Alternatively, External Source can be used with `source` parameter specified.')

        def _get_name_es_instance_map(self):
            if False:
                for i in range(10):
                    print('nop')
            'Return mappings between name of External Source and the op.\n\n            Returns\n            -------\n            mapping for placeholders nodes, mapping for nodes with Python source\n                Two mappings are returned, separating the placeholder nodes without a `source`\n                and nodes that got a `source` parameter.\n            '
            name_es = {}
            name_es_with_callback = {}
            for op in self._pipeline_instance._ops:
                if _is_external_source_with_callback(op):
                    name_es_with_callback[op.name] = op
                elif _is_external_source(op):
                    self._assert_correct_external_sources(op)
                    name_es[op._op._name] = op
            return (name_es, name_es_with_callback)

        def _check_dtypes(self, values, expected_elem_type):
            if False:
                print('Hello World!')
            "Check whether `values` is instance of `expected_elem_type` or tuple of\n             `expected_elem_type`.\n            TF doesn't treat list as a nesting type, but as a Tensor.\n            "
            if isinstance(values, expected_elem_type):
                return True
            elif isinstance(values, tuple) and all((isinstance(elem, expected_elem_type) for elem in values)):
                return True
            else:
                return False

        def _handle_deprecation(self, supported_arg, deprecated_arg, name):
            if False:
                i = 10
                return i + 15
            if deprecated_arg is not None:
                if supported_arg is not None:
                    raise ValueError('Usage of `{name}` is deprecated in favor of `output_{name}`. Both arguments were provided, but only `output_{name}` should be provided.'.format(name=name))
                warnings.warn('Use of argument `{name}` is deprecated. Please use `output_{name}` instead. `output_{name}` should be provided as a tuple or a single value.'.format(name=name), Warning, stacklevel=2)
                if isinstance(deprecated_arg, list):
                    return tuple(deprecated_arg)
                return deprecated_arg
            else:
                return supported_arg

        @property
        def element_spec(self):
            if False:
                for i in range(10):
                    print('nop')
            return self._structure

        @property
        def _element_structure(self):
            if False:
                return 10
            return self._structure

        def _inputs(self):
            if False:
                for i in range(10):
                    print('nop')
            return nest.flatten(self._input_datasets)

        def _as_variant_tensor(self):
            if False:
                i = 10
                return i + 15
            return _dali_tf_module.dali_dataset(nest.map_structure(lambda d: d._variant_tensor, self._input_datasets), input_names=self._input_names, input_layouts=self._input_layouts, input_batched=self._input_batched, pipeline=self._pipeline_serialized, batch_size=self._batch_size, num_threads=self._num_threads, device_id=self._device_id, exec_separated=self._exec_separated, prefetch_queue_depth=self._prefetch_queue_depth, cpu_prefetch_queue_depth=self._cpu_prefetch_queue_depth, gpu_prefetch_queue_depth=self._gpu_prefetch_queue_depth, output_shapes=self._output_shapes, output_dtypes=self._output_dtypes, fail_on_device_mismatch=self._fail_on_device_mismatch)
    if _get_tf_version() < LooseVersion('2.0'):

        class _DALIDatasetImpl(dataset_ops.DatasetV1Adapter):

            @functools.wraps(_DALIDatasetV2.__init__)
            def __init__(self, pipeline, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                self._wrapped = _DALIDatasetV2(pipeline, **kwargs)
                super(_DALIDatasetImpl, self).__init__(self._wrapped)
    else:
        _DALIDatasetImpl = _DALIDatasetV2
    _experimental_kwargs = ['input_datasets']

    class DALIDataset(dataset_ops._OptionsDataset):

        @functools.wraps(_DALIDatasetV2.__init__)
        def __init__(self, pipeline, **kwargs):
            if False:
                print('Hello World!')
            for disallowed_kwarg in _experimental_kwargs:
                if disallowed_kwarg in kwargs.keys():
                    raise TypeError(f"__init__() got an unexpected keyword argument '{disallowed_kwarg}'. Dataset inputs are allowed only in 'experimental.DALIDatasetWithInputs'.")
            if _has_external_source(pipeline):
                raise ValueError("DALIDataset got a DALI pipeline containing External Source operator nodes. External Source nodes can be used to express placeholders for tf.data.Dataset inputs to DALI or to run user-provided Python code via `source` parameter. Support for Dataset inputs and External Source's `source` is allowed only in 'experimental.DALIDatasetWithInputs'.")
            dataset_impl = _DALIDatasetImpl(pipeline, **kwargs)
            super(DALIDataset, self).__init__(dataset_impl, dataset_options())
else:

    class DALIDataset:

        def __init__(self, pipeline, output_dtypes=None, output_shapes=None, fail_on_device_mismatch=True, *, batch_size=1, num_threads=4, device_id=0, exec_separated=False, prefetch_queue_depth=2, cpu_prefetch_queue_depth=2, gpu_prefetch_queue_depth=2, dtypes=None, shapes=None):
            if False:
                print('Hello World!')
            raise RuntimeError('DALIDataset is not supported for detected version of TensorFlow. DALIDataset supports versions: 1.15, 2.x family')
if dataset_inputs_compatible_tensorflow():

    def _load_experimental_dataset():
        if False:
            i = 10
            return i + 15

        class DALIDatasetWithInputs(dataset_ops._OptionsDataset):

            @functools.wraps(_DALIDatasetV2.__init__)
            def __init__(self, pipeline, **kwargs):
                if False:
                    return 10
                dataset_impl = _DALIDatasetImpl(pipeline, **kwargs)
                super(DALIDatasetWithInputs, self).__init__(dataset_impl, dataset_options())
        DALIDatasetWithInputs.__doc__ = _experimental_dataset_docstring
        _insert_experimental_member(DALIDatasetWithInputs, 'DALIDatasetWithInputs')

        class Input:

            def __init__(self, dataset, *, layout=None, batch=False):
                if False:
                    return 10
                if not isinstance(dataset, dataset_ops.DatasetV2):
                    raise TypeError('The inputs specified to DALIDataset must be instances of type `tf.data.Dataset` got: `{}` of type: {} instead.'.format(dataset, type(dataset)))
                self.dataset = dataset
                self.layout = layout
                self.batch = batch
        Input.__doc__ = _experimental_input_docstring
        _insert_experimental_member(Input, 'Input')
    _load_experimental_dataset()
else:

    def _load_experimental_dataset():
        if False:
            for i in range(10):
                print('nop')

        class DALIDatasetWithInputs:

            def __init__(self, *args, **kwargs):
                if False:
                    print('Hello World!')
                raise RuntimeError('experimental.DALIDatasetWithInputs is not supported for detected version of TensorFlow. DALIDataset supports versions: 2.4.1 and above.')
        DALIDatasetWithInputs.__doc__ = _experimental_dataset_docstring
        _insert_experimental_member(DALIDatasetWithInputs, 'DALIDatasetWithInputs')

        class Input:

            def __init__(self, *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                pass
        Input.__doc__ = _experimental_input_docstring
        _insert_experimental_member(Input, 'Input')
    _load_experimental_dataset()
DALIDataset.__doc__ = 'Creates a ``DALIDataset`` compatible with\n    `tf.data.Dataset <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`_ from a DALI\n    pipeline. It supports TensorFlow 1.15 and 2.x family.\n\n    ``DALIDataset`` can be placed on CPU and GPU.\n\n    Please keep in mind that TensorFlow allocates almost all available device memory by default.\n    This might cause errors in DALI due to insufficient memory. On how to change this behaviour\n    please look into the TensorFlow documentation, as it may differ based on your use case.\n\n    .. warning::\n       Most TensorFlow Datasets have only CPU variant. To process GPU-placed ``DALIDataset`` by\n       other TensorFlow dataset you need to first copy it back to CPU using explicit\n       ``tf.data.experimental.copy_to_device`` - roundtrip from CPU to GPU back to CPU would\n       probably degrade performance a lot and is thus discouraged.\n\n       Additionally, it is advised to not use datasets like ``repeat()`` or similar after\n       ``DALIDataset``, which may interfere with DALI memory allocations and prefetching.\n\n    Parameters\n    ----------\n    pipeline : :class:`nvidia.dali.Pipeline`\n        defining the data processing to be performed.\n    output_dtypes: tf.DType or tuple of tf.DType, default = None\n        expected output types\n    output_shapes: tuple of shapes, optional, default = None\n        expected output shapes. If provided, must match arity of the ``output_dtypes``.\n        When set to None, DALI will infer the shapes on its own.\n        Individual shapes can be also set to None or contain None to indicate unknown dimensions.\n        If specified must be compatible with shape returned from DALI Pipeline\n        and with ``batch_size`` argument which will be the outermost dimension of returned tensors.\n        In case of ``batch_size = 1`` it can be omitted in the shape.\n        DALI Dataset will try to match requested shape by squeezing 1-sized dimensions\n        from shape obtained from Pipeline.\n    fail_on_device_mismatch : bool, optional, default = True\n        When set to ``True`` runtime check will be performed to ensure DALI device and TF device\n        are both CPU or both GPU. In some contexts this check might be inaccurate. When set to\n         ``False`` will skip the check but print additional logs to check the devices. Keep in mind\n        that this may allow hidden GPU to CPU copies in the workflow and impact performance.\n    batch_size : int, optional, default = 1\n        batch size of the pipeline.\n    num_threads : int, optional, default = 4\n        number of CPU threads used by the pipeline.\n    device_id : int, optional, default = 0\n        id of GPU used by the pipeline.\n        A None value for this parameter means that DALI should not use GPU nor CUDA runtime.\n        This limits the pipeline to only CPU operators but allows it to run on any\n        CPU capable machine.\n    exec_separated : bool, optional, default = False\n        Whether to execute the pipeline in a way that enables\n        overlapping CPU and GPU computation, typically resulting\n        in faster execution speed, but larger memory consumption.\n    prefetch_queue_depth : int, optional, default = 2\n        depth of the executor queue. Deeper queue makes DALI more\n        resistant to uneven execution time of each batch, but it also\n        consumes more memory for internal buffers.\n        Value will be used with ``exec_separated`` set to ``False``.\n    cpu_prefetch_queue_depth : int, optional, default = 2\n        depth of the executor cpu queue. Deeper queue makes DALI more\n        resistant to uneven execution time of each batch, but it also\n        consumes more memory for internal buffers.\n        Value will be used with ``exec_separated`` set to ``True``.\n    gpu_prefetch_queue_depth : int, optional, default = 2\n        depth of the executor gpu queue. Deeper queue makes DALI more\n        resistant to uneven execution time of each batch, but it also\n        consumes more memory for internal buffers.\n        Value will be used with ``exec_separated`` set to ``True``.\n\n    Returns\n    -------\n    ``DALIDataset`` object based on DALI pipeline and compatible with ``tf.data.Dataset`` API.\n\n    '
DALIIterator.__doc__ = DALIIteratorWrapper.__doc__
DALIRawIterator.__doc__ = _dali_tf.__doc__