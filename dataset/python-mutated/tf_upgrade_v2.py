"""Upgrader for Python scripts from 1.* TensorFlow to 2.0 TensorFlow."""
import ast
import copy
import functools
import sys
import pasta
from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2
from tensorflow.tools.compatibility import reorders_v2

class UnaliasedTFImport(ast_edits.AnalysisResult):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.log_level = ast_edits.ERROR
        self.log_message = 'The tf_upgrade_v2 script detected an unaliased `import tensorflow`. The script can only run when importing with `import tensorflow as tf`.'

class VersionedTFImport(ast_edits.AnalysisResult):

    def __init__(self, version):
        if False:
            print('Hello World!')
        self.log_level = ast_edits.INFO
        self.log_message = 'Not upgrading symbols because `tensorflow.' + version + '` was directly imported as `tf`.'
compat_v1_import = VersionedTFImport('compat.v1')
compat_v2_import = VersionedTFImport('compat.v2')

class TFAPIImportAnalysisSpec(ast_edits.APIAnalysisSpec):

    def __init__(self):
        if False:
            print('Hello World!')
        self.symbols_to_detect = {}
        self.imports_to_detect = {('tensorflow', None): UnaliasedTFImport(), ('tensorflow.compat.v1', 'tf'): compat_v1_import, ('tensorflow.compat.v2', 'tf'): compat_v2_import}

class CompatV1ImportReplacer(ast.NodeVisitor):
    """AST Visitor that replaces `import tensorflow.compat.v1 as tf`.

  Converts `import tensorflow.compat.v1 as tf` to `import tensorflow as tf`
  """

    def visit_Import(self, node):
        if False:
            i = 10
            return i + 15
        'Handle visiting an import node in the AST.\n\n    Args:\n      node: Current Node\n    '
        for import_alias in node.names:
            if import_alias.name == 'tensorflow.compat.v1' and import_alias.asname == 'tf':
                import_alias.name = 'tensorflow'
        self.generic_visit(node)

class TFAPIChangeSpec(ast_edits.NoUpdateSpec):
    """List of maps that describe what changed in the API."""

    def __init__(self, import_rename=False, upgrade_compat_v1_import=False):
        if False:
            print('Hello World!')
        self.upgrade_compat_v1_import = upgrade_compat_v1_import
        self.function_keyword_renames = {'tf.test.assert_equal_graph_def': {'checkpoint_v2': None, 'hash_table_shared_name': None}, 'tf.autograph.to_code': {'arg_types': None, 'arg_values': None, 'indentation': None}, 'tf.autograph.to_graph': {'arg_types': None, 'arg_values': None}, 'tf.nn.embedding_lookup': {'validate_indices': None}, 'tf.image.sample_distorted_bounding_box': {'seed2': None}, 'tf.gradients': {'colocate_gradients_with_ops': None}, 'tf.hessians': {'colocate_gradients_with_ops': None}, '*.minimize': {'colocate_gradients_with_ops': None}, '*.compute_gradients': {'colocate_gradients_with_ops': None}, 'tf.cond': {'strict': None, 'fn1': 'true_fn', 'fn2': 'false_fn'}, 'tf.argmin': {'dimension': 'axis'}, 'tf.argmax': {'dimension': 'axis'}, 'tf.arg_min': {'dimension': 'axis'}, 'tf.arg_max': {'dimension': 'axis'}, 'tf.math.argmin': {'dimension': 'axis'}, 'tf.math.argmax': {'dimension': 'axis'}, 'tf.image.crop_and_resize': {'box_ind': 'box_indices'}, 'tf.extract_image_patches': {'ksizes': 'sizes'}, 'tf.image.extract_image_patches': {'ksizes': 'sizes'}, 'tf.image.resize': {'align_corners': None}, 'tf.image.resize_images': {'align_corners': None}, 'tf.expand_dims': {'dim': 'axis'}, 'tf.batch_to_space': {'block_size': 'block_shape'}, 'tf.space_to_batch': {'block_size': 'block_shape'}, 'tf.nn.space_to_batch': {'block_size': 'block_shape'}, 'tf.constant': {'verify_shape': 'verify_shape_is_now_always_true'}, 'tf.convert_to_tensor': {'preferred_dtype': 'dtype_hint'}, 'tf.nn.softmax_cross_entropy_with_logits': {'dim': 'axis'}, 'tf.nn.softmax_cross_entropy_with_logits_v2': {'dim': 'axis'}, 'tf.linalg.l2_normalize': {'dim': 'axis'}, 'tf.linalg.norm': {'keep_dims': 'keepdims'}, 'tf.norm': {'keep_dims': 'keepdims'}, 'tf.load_file_system_library': {'library_filename': 'library_location'}, 'tf.count_nonzero': {'input_tensor': 'input', 'keep_dims': 'keepdims', 'reduction_indices': 'axis'}, 'tf.math.count_nonzero': {'input_tensor': 'input', 'keep_dims': 'keepdims', 'reduction_indices': 'axis'}, 'tf.nn.erosion2d': {'kernel': 'filters', 'rates': 'dilations'}, 'tf.math.l2_normalize': {'dim': 'axis'}, 'tf.math.log_softmax': {'dim': 'axis'}, 'tf.math.softmax': {'dim': 'axis'}, 'tf.nn.l2_normalize': {'dim': 'axis'}, 'tf.nn.log_softmax': {'dim': 'axis'}, 'tf.nn.moments': {'keep_dims': 'keepdims'}, 'tf.nn.pool': {'dilation_rate': 'dilations'}, 'tf.nn.separable_conv2d': {'rate': 'dilations'}, 'tf.nn.depthwise_conv2d': {'rate': 'dilations'}, 'tf.nn.softmax': {'dim': 'axis'}, 'tf.nn.sufficient_statistics': {'keep_dims': 'keepdims'}, 'tf.debugging.assert_all_finite': {'t': 'x', 'msg': 'message'}, 'tf.verify_tensor_all_finite': {'t': 'x', 'msg': 'message'}, 'tf.sparse.add': {'thresh': 'threshold'}, 'tf.sparse_add': {'thresh': 'threshold'}, 'tf.sparse.concat': {'concat_dim': 'axis', 'expand_nonconcat_dim': 'expand_nonconcat_dims'}, 'tf.sparse_concat': {'concat_dim': 'axis', 'expand_nonconcat_dim': 'expand_nonconcat_dims'}, 'tf.sparse.split': {'split_dim': 'axis'}, 'tf.sparse_split': {'split_dim': 'axis'}, 'tf.sparse.reduce_max': {'reduction_axes': 'axis', 'keep_dims': 'keepdims'}, 'tf.sparse_reduce_max': {'reduction_axes': 'axis', 'keep_dims': 'keepdims'}, 'tf.sparse.reduce_sum': {'reduction_axes': 'axis', 'keep_dims': 'keepdims'}, 'tf.sparse_reduce_sum': {'reduction_axes': 'axis', 'keep_dims': 'keepdims'}, 'tf.nn.max_pool_with_argmax': {'Targmax': 'output_dtype'}, 'tf.nn.max_pool': {'value': 'input'}, 'tf.nn.avg_pool': {'value': 'input'}, 'tf.nn.avg_pool2d': {'value': 'input'}, 'tf.multinomial': {'output_dtype': 'dtype'}, 'tf.random.multinomial': {'output_dtype': 'dtype'}, 'tf.reverse_sequence': {'seq_dim': 'seq_axis', 'batch_dim': 'batch_axis'}, 'tf.nn.batch_norm_with_global_normalization': {'t': 'input', 'm': 'mean', 'v': 'variance'}, 'tf.nn.dilation2d': {'filter': 'filters', 'rates': 'dilations'}, 'tf.nn.conv3d': {'filter': 'filters'}, 'tf.zeros_like': {'tensor': 'input'}, 'tf.ones_like': {'tensor': 'input'}, 'tf.nn.conv2d_transpose': {'value': 'input', 'filter': 'filters'}, 'tf.nn.conv3d_transpose': {'value': 'input', 'filter': 'filters'}, 'tf.nn.convolution': {'filter': 'filters', 'dilation_rate': 'dilations'}, 'tf.gfile.Exists': {'filename': 'path'}, 'tf.gfile.Remove': {'filename': 'path'}, 'tf.gfile.Stat': {'filename': 'path'}, 'tf.gfile.Glob': {'filename': 'pattern'}, 'tf.gfile.MkDir': {'dirname': 'path'}, 'tf.gfile.MakeDirs': {'dirname': 'path'}, 'tf.gfile.DeleteRecursively': {'dirname': 'path'}, 'tf.gfile.IsDirectory': {'dirname': 'path'}, 'tf.gfile.ListDirectory': {'dirname': 'path'}, 'tf.gfile.Copy': {'oldpath': 'src', 'newpath': 'dst'}, 'tf.gfile.Rename': {'oldname': 'src', 'newname': 'dst'}, 'tf.gfile.Walk': {'in_order': 'topdown'}, 'tf.random.stateless_multinomial': {'output_dtype': 'dtype'}, 'tf.string_to_number': {'string_tensor': 'input'}, 'tf.strings.to_number': {'string_tensor': 'input'}, 'tf.string_to_hash_bucket': {'string_tensor': 'input'}, 'tf.strings.to_hash_bucket': {'string_tensor': 'input'}, 'tf.reduce_all': {'reduction_indices': 'axis', 'keep_dims': 'keepdims'}, 'tf.math.reduce_all': {'reduction_indices': 'axis', 'keep_dims': 'keepdims'}, 'tf.reduce_any': {'reduction_indices': 'axis', 'keep_dims': 'keepdims'}, 'tf.math.reduce_any': {'reduction_indices': 'axis', 'keep_dims': 'keepdims'}, 'tf.reduce_min': {'reduction_indices': 'axis', 'keep_dims': 'keepdims'}, 'tf.math.reduce_min': {'reduction_indices': 'axis', 'keep_dims': 'keepdims'}, 'tf.reduce_max': {'reduction_indices': 'axis', 'keep_dims': 'keepdims'}, 'tf.math.reduce_max': {'reduction_indices': 'axis', 'keep_dims': 'keepdims'}, 'tf.reduce_sum': {'reduction_indices': 'axis', 'keep_dims': 'keepdims'}, 'tf.math.reduce_sum': {'reduction_indices': 'axis', 'keep_dims': 'keepdims'}, 'tf.reduce_mean': {'reduction_indices': 'axis', 'keep_dims': 'keepdims'}, 'tf.math.reduce_mean': {'reduction_indices': 'axis', 'keep_dims': 'keepdims'}, 'tf.reduce_prod': {'reduction_indices': 'axis', 'keep_dims': 'keepdims'}, 'tf.math.reduce_prod': {'reduction_indices': 'axis', 'keep_dims': 'keepdims'}, 'tf.reduce_logsumexp': {'reduction_indices': 'axis', 'keep_dims': 'keepdims'}, 'tf.math.reduce_logsumexp': {'reduction_indices': 'axis', 'keep_dims': 'keepdims'}, 'tf.reduce_join': {'keep_dims': 'keepdims', 'reduction_indices': 'axis'}, 'tf.strings.reduce_join': {'keep_dims': 'keepdims', 'reduction_indices': 'axis'}, 'tf.squeeze': {'squeeze_dims': 'axis'}, 'tf.nn.weighted_moments': {'keep_dims': 'keepdims'}, 'tf.nn.conv1d': {'value': 'input', 'use_cudnn_on_gpu': None}, 'tf.nn.conv2d': {'filter': 'filters', 'use_cudnn_on_gpu': None}, 'tf.nn.conv2d_backprop_input': {'use_cudnn_on_gpu': None, 'input_sizes': 'output_shape', 'out_backprop': 'input', 'filter': 'filters'}, 'tf.contrib.summary.audio': {'tensor': 'data', 'family': None}, 'tf.contrib.summary.create_file_writer': {'name': None}, 'tf.contrib.summary.generic': {'name': 'tag', 'tensor': 'data', 'family': None}, 'tf.contrib.summary.histogram': {'tensor': 'data', 'family': None}, 'tf.contrib.summary.image': {'tensor': 'data', 'bad_color': None, 'max_images': 'max_outputs', 'family': None}, 'tf.contrib.summary.scalar': {'tensor': 'data', 'family': None}, 'tf.nn.weighted_cross_entropy_with_logits': {'targets': 'labels'}, 'tf.decode_raw': {'bytes': 'input_bytes'}, 'tf.io.decode_raw': {'bytes': 'input_bytes'}, 'tf.contrib.framework.load_variable': {'checkpoint_dir': 'ckpt_dir_or_file'}}
        all_renames_v2.add_contrib_direct_import_support(self.function_keyword_renames)
        self.symbol_renames = all_renames_v2.symbol_renames
        self.import_rename = import_rename
        if self.import_rename:
            self.import_renames = {'tensorflow': ast_edits.ImportRename('tensorflow.compat.v2', excluded_prefixes=['tensorflow.contrib', 'tensorflow.flags', 'tensorflow.compat.v1', 'tensorflow.compat.v2', 'tensorflow.google'])}
        else:
            self.import_renames = {}
        self.change_to_function = {}
        self.reordered_function_names = {'tf.io.serialize_sparse', 'tf.io.serialize_many_sparse', 'tf.argmax', 'tf.argmin', 'tf.batch_to_space', 'tf.cond', 'tf.nn.space_to_batch', 'tf.boolean_mask', 'tf.convert_to_tensor', 'tf.nn.conv1d', 'tf.nn.conv2d', 'tf.nn.conv2d_backprop_input', 'tf.nn.ctc_beam_search_decoder', 'tf.nn.moments', 'tf.nn.convolution', 'tf.nn.crelu', 'tf.nn.weighted_moments', 'tf.nn.pool', 'tf.nn.separable_conv2d', 'tf.nn.depthwise_conv2d', 'tf.multinomial', 'tf.random.multinomial', 'tf.pad', 'tf.quantize_v2', 'tf.feature_column.categorical_column_with_vocabulary_file', 'tf.shape', 'tf.size', 'tf.random.poisson', 'tf.sparse.add', 'tf.sparse_add', 'tf.sparse.concat', 'tf.sparse_concat', 'tf.sparse.segment_mean', 'tf.sparse.segment_sqrt_n', 'tf.sparse.segment_sum', 'tf.sparse_matmul', 'tf.sparse.reduce_max', 'tf.sparse_reduce_max', 'tf.io.decode_csv', 'tf.strings.length', 'tf.strings.reduce_join', 'tf.strings.substr', 'tf.substr', 'tf.transpose', 'tf.tuple', 'tf.parse_example', 'tf.parse_single_example', 'tf.io.parse_example', 'tf.io.parse_single_example', 'tf.while_loop', 'tf.reduce_all', 'tf.math.reduce_all', 'tf.reduce_any', 'tf.math.reduce_any', 'tf.reduce_min', 'tf.math.reduce_min', 'tf.reduce_max', 'tf.math.reduce_max', 'tf.reduce_sum', 'tf.math.reduce_sum', 'tf.reduce_mean', 'tf.math.reduce_mean', 'tf.reduce_prod', 'tf.math.reduce_prod', 'tf.reduce_logsumexp', 'tf.math.reduce_logsumexp', 'tf.reduce_join', 'tf.confusion_matrix', 'tf.math.confusion_matrix', 'tf.math.in_top_k', 'tf.nn.depth_to_space', 'tf.nn.embedding_lookup', 'tf.nn.embedding_lookup_sparse', 'tf.nn.in_top_k', 'tf.nn.space_to_depth', 'tf.test.assert_equal_graph_def', 'tf.linalg.norm', 'tf.norm', 'tf.reverse_sequence', 'tf.sparse_split', 'tf.nn.softmax_cross_entropy_with_logits', 'tf.nn.fractional_avg_pool', 'tf.nn.fractional_max_pool', 'tf.image.sample_distorted_bounding_box', 'tf.gradients', 'tf.hessians', 'tf.nn.max_pool', 'tf.nn.avg_pool', 'tf.estimator.LinearClassifier', 'tf.estimator.LinearRegressor', 'tf.estimator.DNNLinearCombinedClassifier', 'tf.estimator.DNNLinearCombinedRegressor', 'tf.estimator.DNNRegressor', 'tf.estimator.DNNClassifier', 'tf.estimator.BaselineClassifier', 'tf.estimator.BaselineRegressor', 'tf.initializers.uniform_unit_scaling', 'tf.uniform_unit_scaling_initializer', 'tf.data.experimental.TensorStructure', 'tf.data.experimental.SparseTensorStructure', 'tf.data.experimental.RaggedTensorStructure', 'tf.data.experimental.TensorArrayStructure', 'tf.debugging.assert_all_finite', 'tf.gather_nd'}
        self.manual_function_reorders = {'tf.contrib.summary.audio': ['name', 'tensor', 'sample_rate', 'max_outputs', 'family', 'step'], 'tf.contrib.summary.create_file_writer': ['logdir', 'max_queue', 'flush_millis', 'filename_suffix', 'name'], 'tf.contrib.summary.generic': ['name', 'tensor', 'metadata', 'family', 'step'], 'tf.contrib.summary.histogram': ['name', 'tensor', 'family', 'step'], 'tf.contrib.summary.image': ['name', 'tensor', 'bad_color', 'max_images', 'family', 'step'], 'tf.contrib.summary.scalar': ['name', 'tensor', 'family', 'step']}
        self.function_reorders = dict(reorders_v2.reorders)
        self.function_reorders.update(self.manual_function_reorders)
        decay_function_comment = (ast_edits.INFO, 'To use learning rate decay schedules with TensorFlow 2.0, switch to the schedules in `tf.keras.optimizers.schedules`.\n')
        assert_return_type_comment = (ast_edits.INFO, '<function name> has been changed to return None, the data argument has been removed, and arguments have been reordered.\nThe calls have been converted to compat.v1 for safety (even though  they may already have been correct).')
        assert_rank_comment = (ast_edits.INFO, '<function name> has been changed to return None, and the data and summarize arguments have been removed.\nThe calls have been converted to compat.v1 for safety (even though  they may already have been correct).')
        contrib_layers_layer_norm_comment = (ast_edits.WARNING, '(Manual edit required) `tf.contrib.layers.layer_norm` has been deprecated, and its implementation has been integrated with `tf.keras.layers.LayerNormalization` in TensorFlow 2.0. Note that, the default value of `epsilon` is changed to `1e-3` in the new API from `1e-12`, and this may introduce numerical differences. Please check the new API and use that instead.')
        contrib_estimator_head_comment = (ast_edits.WARNING, '(Manual edit required) `tf.contrib.estimator.*_head` has been deprecated, and its implementation has been integrated with `tf.estimator.*Head` in TensorFlow 2.0. Please check the new API and use that instead.')
        initializers_no_dtype_comment = (ast_edits.INFO, 'Initializers no longer have the dtype argument in the constructor or partition_info argument in the __call__ method.\nThe calls have been converted to compat.v1 for safety (even though they may already have been correct).')
        metrics_comment = (ast_edits.INFO, 'tf.metrics have been replaced with object oriented versions in TF 2.0 and after. The metric function calls have been converted to compat.v1 for backward compatibility. Please update these calls to the TF 2.0 versions.')
        losses_comment = (ast_edits.INFO, 'tf.losses have been replaced with object oriented versions in TF 2.0 and after. The loss function calls have been converted to compat.v1 for backward compatibility. Please update these calls to the TF 2.0 versions.')
        deprecate_partition_strategy_comment = (ast_edits.WARNING, "`partition_strategy` has been removed from <function name>.  The 'div' strategy will be used by default.")
        uniform_unit_scaling_initializer_comment = (ast_edits.ERROR, 'uniform_unit_scaling_initializer has been removed. Please use tf.initializers.variance_scaling instead with distribution=uniform to get equivalent behaviour.')
        export_saved_model_renamed = (ast_edits.ERROR, '(Manual edit required) Please rename the method export_savedmodel() to export_saved_model(). Two things to note:\n\t(1) The argument strip_default_attributes has been removed. The function will always strip the default attributes from ops. If this breaks your code, please switch to tf.compat.v1.estimator.Estimator.\n\t(2) This change only effects core estimator. If you are using tf.contrib.learn.Estimator, please switch to using core estimator.')
        summary_api_comment = (ast_edits.INFO, 'The TF 1.x summary API cannot be automatically migrated to TF 2.0, so symbols have been converted to tf.compat.v1.summary.* and must be migrated manually. Typical usage will only require changes to the summary writing logic, not to individual calls like scalar(). For examples of the new summary API, see the Effective TF 2.0 migration document or check the TF 2.0 TensorBoard tutorials.')
        contrib_summary_comment = (ast_edits.WARNING, 'tf.contrib.summary.* functions have been migrated best-effort to tf.compat.v2.summary.* equivalents where possible, but the resulting code is not guaranteed to work, so please check carefully. For more information about the new summary API, see the Effective TF 2.0 migration document or check the updated TensorBoard tutorials.')
        contrib_summary_family_arg_comment = (ast_edits.WARNING, "<function name> replacement does not accept a 'family' argument; instead regular name scoping should be used. This call site specifies a family argument that has been removed on conversion, so the emitted tag names may be incorrect without manual editing.")
        contrib_create_file_writer_comment = (ast_edits.WARNING, 'tf.contrib.summary.create_file_writer() has been ported to the new tf.compat.v2.summary.create_file_writer(), which no longer re-uses existing event files for the same logdir; instead it always opens a new writer/file. The python writer objects must be re-used explicitly if the reusing behavior is desired.')
        contrib_summary_record_every_n_comment = (ast_edits.ERROR, '(Manual edit required) tf.contrib.summary.record_summaries_every_n_global_steps(n, step) should be replaced by a call to tf.compat.v2.summary.record_if() with the argument `lambda: tf.math.equal(0, global_step % n)` (or in graph mode, the lambda body can be used directly). If no global step was passed, instead use tf.compat.v1.train.get_or_create_global_step().')
        contrib_summary_graph_comment = (ast_edits.ERROR, '(Manual edit required) tf.contrib.summary.graph() has no direct equivalent in TF 2.0 because manual graph construction has been superseded by use of tf.function. To log tf.function execution graphs to the summary writer, use the new tf.compat.v2.summary.trace_* functions instead.')
        contrib_summary_import_event_comment = (ast_edits.ERROR, '(Manual edit required) tf.contrib.summary.import_event() has no direct equivalent in TF 2.0. For a similar experimental feature, try tf.compat.v2.summary.experimental.write_raw_pb() which also accepts serialized summary protocol buffer input, but for tf.Summary protobufs rather than tf.Events.')
        keras_default_save_format_comment = (ast_edits.WARNING, "(This warning is only applicable if the code saves a tf.Keras model) Keras model.save now saves to the Tensorflow SavedModel format by default, instead of HDF5. To continue saving to HDF5, add the argument save_format='h5' to the save() function.")
        distribute_strategy_api_changes = "If you're using the strategy with a custom training loop, note the following changes in methods: make_dataset_iterator->experimental_distribute_dataset, experimental_make_numpy_iterator->experimental_make_numpy_dataset, extended.call_for_each_replica->run, reduce requires an axis argument, unwrap->experimental_local_results experimental_initialize and experimental_finalize no longer needed "
        contrib_mirrored_strategy_warning = (ast_edits.ERROR, '(Manual edit required) tf.contrib.distribute.MirroredStrategy has been migrated to tf.distribute.MirroredStrategy. Things to note: Constructor arguments have changed. If you are using MirroredStrategy with Keras training framework, the input provided to `model.fit` will be assumed to have global batch size and split across the replicas. ' + distribute_strategy_api_changes)
        core_mirrored_strategy_warning = (ast_edits.WARNING, '(Manual edit may be required) tf.distribute.MirroredStrategy API has changed. ' + distribute_strategy_api_changes)
        contrib_one_device_strategy_warning = (ast_edits.ERROR, '(Manual edit required) tf.contrib.distribute.OneDeviceStrategy has been migrated to tf.distribute.OneDeviceStrategy. ' + distribute_strategy_api_changes)
        contrib_tpu_strategy_warning = (ast_edits.ERROR, '(Manual edit required) tf.contrib.distribute.TPUStrategy has been migrated to tf.distribute.TPUStrategy. Note the slight changes in constructor. ' + distribute_strategy_api_changes)
        contrib_collective_strategy_warning = (ast_edits.ERROR, '(Manual edit required) tf.contrib.distribute.CollectiveAllReduceStrategy has been migrated to tf.distribute.experimental.MultiWorkerMirroredStrategy. Note the changes in constructor. ' + distribute_strategy_api_changes)
        contrib_ps_strategy_warning = (ast_edits.ERROR, '(Manual edit required) tf.contrib.distribute.ParameterServerStrategy has been migrated to tf.compat.v1.distribute.experimental.ParameterServerStrategy (multi machine) and tf.distribute.experimental.CentralStorageStrategy (one machine). Note the changes in constructors. ' + distribute_strategy_api_changes)
        keras_experimental_export_comment = (ast_edits.WARNING, "tf.keras.experimental.export_saved_model and tf.keras.experimental.load_from_saved_model have been deprecated.Please use model.save(path, save_format='tf') (or alternatively tf.keras.models.save_model), and tf.keras.models.load_model(path) instead.")
        saved_model_load_warning = (ast_edits.WARNING, 'tf.saved_model.load works differently in 2.0 compared to 1.0. See migration information in the documentation of tf.compat.v1.saved_model.load.\nThe calls have been converted to compat.v1.')
        self.function_warnings = {'*.export_savedmodel': export_saved_model_renamed, '*.save': keras_default_save_format_comment, 'tf.assert_equal': assert_return_type_comment, 'tf.assert_none_equal': assert_return_type_comment, 'tf.assert_negative': assert_return_type_comment, 'tf.assert_positive': assert_return_type_comment, 'tf.assert_non_negative': assert_return_type_comment, 'tf.assert_non_positive': assert_return_type_comment, 'tf.assert_near': assert_return_type_comment, 'tf.assert_less': assert_return_type_comment, 'tf.assert_less_equal': assert_return_type_comment, 'tf.assert_greater': assert_return_type_comment, 'tf.assert_greater_equal': assert_return_type_comment, 'tf.assert_integer': assert_return_type_comment, 'tf.assert_type': assert_return_type_comment, 'tf.assert_scalar': assert_return_type_comment, 'tf.assert_rank': assert_rank_comment, 'tf.assert_rank_at_least': assert_rank_comment, 'tf.assert_rank_in': assert_rank_comment, 'tf.contrib.layers.layer_norm': contrib_layers_layer_norm_comment, 'tf.contrib.estimator.binary_classification_head': contrib_estimator_head_comment, 'tf.contrib.estimator.logistic_regression_head': contrib_estimator_head_comment, 'tf.contrib.estimator.multi_class_head': contrib_estimator_head_comment, 'tf.contrib.estimator.multi_head': contrib_estimator_head_comment, 'tf.contrib.estimator.multi_label_head': contrib_estimator_head_comment, 'tf.contrib.estimator.poisson_regression_head': contrib_estimator_head_comment, 'tf.contrib.estimator.regression_head': contrib_estimator_head_comment, 'tf.contrib.saved_model.load_keras_model': keras_experimental_export_comment, 'tf.contrib.saved_model.save_keras_model': keras_experimental_export_comment, 'tf.contrib.summary.all_summary_ops': contrib_summary_comment, 'tf.contrib.summary.audio': contrib_summary_comment, 'tf.contrib.summary.create_file_writer': contrib_create_file_writer_comment, 'tf.contrib.summary.generic': contrib_summary_comment, 'tf.contrib.summary.graph': contrib_summary_graph_comment, 'tf.contrib.summary.histogram': contrib_summary_comment, 'tf.contrib.summary.import_event': contrib_summary_import_event_comment, 'tf.contrib.summary.image': contrib_summary_comment, 'tf.contrib.summary.record_summaries_every_n_global_steps': contrib_summary_record_every_n_comment, 'tf.contrib.summary.scalar': contrib_summary_comment, 'tf.debugging.assert_equal': assert_return_type_comment, 'tf.debugging.assert_greater': assert_return_type_comment, 'tf.debugging.assert_greater_equal': assert_return_type_comment, 'tf.debugging.assert_integer': assert_return_type_comment, 'tf.debugging.assert_less': assert_return_type_comment, 'tf.debugging.assert_less_equal': assert_return_type_comment, 'tf.debugging.assert_near': assert_return_type_comment, 'tf.debugging.assert_negative': assert_return_type_comment, 'tf.debugging.assert_non_negative': assert_return_type_comment, 'tf.debugging.assert_non_positive': assert_return_type_comment, 'tf.debugging.assert_none_equal': assert_return_type_comment, 'tf.debugging.assert_positive': assert_return_type_comment, 'tf.debugging.assert_type': assert_return_type_comment, 'tf.debugging.assert_scalar': assert_return_type_comment, 'tf.debugging.assert_rank': assert_rank_comment, 'tf.debugging.assert_rank_at_least': assert_rank_comment, 'tf.debugging.assert_rank_in': assert_rank_comment, 'tf.train.exponential_decay': decay_function_comment, 'tf.train.piecewise_constant_decay': decay_function_comment, 'tf.train.polynomial_decay': decay_function_comment, 'tf.train.natural_exp_decay': decay_function_comment, 'tf.train.inverse_time_decay': decay_function_comment, 'tf.train.cosine_decay': decay_function_comment, 'tf.train.cosine_decay_restarts': decay_function_comment, 'tf.train.linear_cosine_decay': decay_function_comment, 'tf.train.noisy_linear_cosine_decay': decay_function_comment, 'tf.nn.embedding_lookup': deprecate_partition_strategy_comment, 'tf.nn.embedding_lookup_sparse': deprecate_partition_strategy_comment, 'tf.nn.nce_loss': deprecate_partition_strategy_comment, 'tf.nn.safe_embedding_lookup_sparse': deprecate_partition_strategy_comment, 'tf.nn.sampled_softmax_loss': deprecate_partition_strategy_comment, 'tf.keras.estimator.model_to_estimator': (ast_edits.WARNING, "Estimators from <function name> will save object-based checkpoints (format used by `keras_model.save_weights` and `keras_model.load_weights`) by default in 2.0. To continue saving name-based checkpoints, set `checkpoint_format='saver'`."), 'tf.keras.experimental.export_saved_model': keras_experimental_export_comment, 'tf.keras.experimental.load_from_saved_model': keras_experimental_export_comment, 'tf.keras.initializers.Zeros': initializers_no_dtype_comment, 'tf.keras.initializers.zeros': initializers_no_dtype_comment, 'tf.keras.initializers.Ones': initializers_no_dtype_comment, 'tf.keras.initializers.ones': initializers_no_dtype_comment, 'tf.keras.initializers.Constant': initializers_no_dtype_comment, 'tf.keras.initializers.constant': initializers_no_dtype_comment, 'tf.keras.initializers.VarianceScaling': initializers_no_dtype_comment, 'tf.keras.initializers.Orthogonal': initializers_no_dtype_comment, 'tf.keras.initializers.orthogonal': initializers_no_dtype_comment, 'tf.keras.initializers.Identity': initializers_no_dtype_comment, 'tf.keras.initializers.identity': initializers_no_dtype_comment, 'tf.keras.initializers.glorot_uniform': initializers_no_dtype_comment, 'tf.keras.initializers.glorot_normal': initializers_no_dtype_comment, 'tf.initializers.zeros': initializers_no_dtype_comment, 'tf.zeros_initializer': initializers_no_dtype_comment, 'tf.initializers.ones': initializers_no_dtype_comment, 'tf.ones_initializer': initializers_no_dtype_comment, 'tf.initializers.constant': initializers_no_dtype_comment, 'tf.constant_initializer': initializers_no_dtype_comment, 'tf.initializers.random_uniform': initializers_no_dtype_comment, 'tf.random_uniform_initializer': initializers_no_dtype_comment, 'tf.initializers.random_normal': initializers_no_dtype_comment, 'tf.random_normal_initializer': initializers_no_dtype_comment, 'tf.initializers.truncated_normal': initializers_no_dtype_comment, 'tf.truncated_normal_initializer': initializers_no_dtype_comment, 'tf.initializers.variance_scaling': initializers_no_dtype_comment, 'tf.variance_scaling_initializer': initializers_no_dtype_comment, 'tf.initializers.orthogonal': initializers_no_dtype_comment, 'tf.orthogonal_initializer': initializers_no_dtype_comment, 'tf.initializers.identity': initializers_no_dtype_comment, 'tf.glorot_uniform_initializer': initializers_no_dtype_comment, 'tf.initializers.glorot_uniform': initializers_no_dtype_comment, 'tf.glorot_normal_initializer': initializers_no_dtype_comment, 'tf.initializers.glorot_normal': initializers_no_dtype_comment, 'tf.losses.absolute_difference': losses_comment, 'tf.losses.add_loss': losses_comment, 'tf.losses.compute_weighted_loss': losses_comment, 'tf.losses.cosine_distance': losses_comment, 'tf.losses.get_losses': losses_comment, 'tf.losses.get_regularization_loss': losses_comment, 'tf.losses.get_regularization_losses': losses_comment, 'tf.losses.get_total_loss': losses_comment, 'tf.losses.hinge_loss': losses_comment, 'tf.losses.huber_loss': losses_comment, 'tf.losses.log_loss': losses_comment, 'tf.losses.mean_pairwise_squared_error': losses_comment, 'tf.losses.mean_squared_error': losses_comment, 'tf.losses.sigmoid_cross_entropy': losses_comment, 'tf.losses.softmax_cross_entropy': losses_comment, 'tf.losses.sparse_softmax_cross_entropy': losses_comment, 'tf.metrics.accuracy': metrics_comment, 'tf.metrics.auc': metrics_comment, 'tf.metrics.average_precision_at_k': metrics_comment, 'tf.metrics.false_negatives': metrics_comment, 'tf.metrics.false_negatives_at_thresholds': metrics_comment, 'tf.metrics.false_positives': metrics_comment, 'tf.metrics.false_positives_at_thresholds': metrics_comment, 'tf.metrics.mean': metrics_comment, 'tf.metrics.mean_absolute_error': metrics_comment, 'tf.metrics.mean_cosine_distance': metrics_comment, 'tf.metrics.mean_iou': metrics_comment, 'tf.metrics.mean_per_class_accuracy': metrics_comment, 'tf.metrics.mean_relative_error': metrics_comment, 'tf.metrics.mean_squared_error': metrics_comment, 'tf.metrics.mean_tensor': metrics_comment, 'tf.metrics.percentage_below': metrics_comment, 'tf.metrics.precision': metrics_comment, 'tf.metrics.precision_at_k': metrics_comment, 'tf.metrics.precision_at_thresholds': metrics_comment, 'tf.metrics.precision_at_top_k': metrics_comment, 'tf.metrics.recall': metrics_comment, 'tf.metrics.recall_at_k': metrics_comment, 'tf.metrics.recall_at_thresholds': metrics_comment, 'tf.metrics.recall_at_top_k': metrics_comment, 'tf.metrics.root_mean_squared_error': metrics_comment, 'tf.metrics.sensitivity_at_specificity': metrics_comment, 'tf.metrics.sparse_average_precision_at_k': metrics_comment, 'tf.metrics.sparse_precision_at_k': metrics_comment, 'tf.metrics.specificity_at_sensitivity': metrics_comment, 'tf.metrics.true_negatives': metrics_comment, 'tf.metrics.true_negatives_at_thresholds': metrics_comment, 'tf.metrics.true_positives': metrics_comment, 'tf.metrics.true_positives_at_thresholds': metrics_comment, 'tf.get_variable': (ast_edits.WARNING, '<function name> returns ResourceVariables by default in 2.0, which have well-defined semantics and are stricter about shapes. You can disable this behavior by passing use_resource=False, or by calling tf.compat.v1.disable_resource_variables().'), 'tf.pywrap_tensorflow': (ast_edits.ERROR, '<function name> cannot be converted automatically. `tf.pywrap_tensorflow` will not be distributed with TensorFlow 2.0, please consider an alternative in public TensorFlow APIs.'), 'tf.contrib.distribute.MirroredStrategy': contrib_mirrored_strategy_warning, 'tf.distribute.MirroredStrategy': core_mirrored_strategy_warning, 'tf.contrib.distribute.OneDeviceStrategy': contrib_one_device_strategy_warning, 'tf.contrib.distribute.TPUStrategy': contrib_tpu_strategy_warning, 'tf.contrib.distribute.CollectiveAllReduceStrategy': contrib_collective_strategy_warning, 'tf.contrib.distribute.ParameterServerStrategy': contrib_ps_strategy_warning, 'tf.summary.FileWriter': summary_api_comment, 'tf.summary.FileWriterCache': summary_api_comment, 'tf.summary.Summary': summary_api_comment, 'tf.summary.audio': summary_api_comment, 'tf.summary.histogram': summary_api_comment, 'tf.summary.image': summary_api_comment, 'tf.summary.merge': summary_api_comment, 'tf.summary.merge_all': summary_api_comment, 'tf.summary.scalar': summary_api_comment, 'tf.summary.tensor_summary': summary_api_comment, 'tf.summary.text': summary_api_comment, 'tf.saved_model.load': saved_model_load_warning, 'tf.saved_model.loader.load': saved_model_load_warning}
        all_renames_v2.add_contrib_direct_import_support(self.function_warnings)
        for (symbol, replacement) in all_renames_v2.addons_symbol_mappings.items():
            warning = (ast_edits.WARNING, '(Manual edit required) `{}` has been migrated to `{}` in TensorFlow Addons. The API spec may have changed during the migration. Please see https://github.com/tensorflow/addons for more info.'.format(symbol, replacement))
            self.function_warnings[symbol] = warning
        self.function_arg_warnings = {'tf.nn.conv1d': {('use_cudnn_on_gpu', 4): (ast_edits.WARNING, 'use_cudnn_on_gpu has been removed, behavior is now equivalentto setting it to True.')}, 'tf.nn.conv2d': {('use_cudnn_on_gpu', 4): (ast_edits.WARNING, 'use_cudnn_on_gpu has been removed, behavior is now equivalentto setting it to True.')}, 'tf.nn.conv2d_backprop_filter': {('use_cudnn_on_gpu', 5): (ast_edits.WARNING, 'use_cudnn_on_gpu has been removed, behavior is now equivalentto setting it to True.')}, 'tf.nn.conv2d_backprop_input': {('use_cudnn_on_gpu', 5): (ast_edits.WARNING, 'use_cudnn_on_gpu has been removed, behavior is now equivalentto setting it to True.')}, 'tf.gradients': {('colocate_gradients_with_ops', 4): (ast_edits.INFO, "tf.gradients no longer takes 'colocate_gradients_with_ops' argument, it behaves as if it was set to True.")}, 'tf.hessians': {('colocate_gradients_with_ops', 3): (ast_edits.INFO, "tf.hessians no longer takes 'colocate_gradients_with_ops' argument, it behaves as if it was set to True.")}, '*.minimize': {('colocate_gradients_with_ops', 5): (ast_edits.INFO, "Optimizer.minimize no longer takes 'colocate_gradients_with_ops' argument, it behaves as if it was set to True.")}, '*.compute_gradients': {('colocate_gradients_with_ops', 4): (ast_edits.INFO, "Optimizer.compute_gradients no longer takes 'colocate_gradients_with_ops' argument, it behaves as if it was set to True.")}, 'tf.cond': {('strict', 3): (ast_edits.WARNING, "tf.cond no longer takes 'strict' argument, it behaves as if was set to True.")}, 'tf.contrib.summary.audio': {('family', 4): contrib_summary_family_arg_comment}, 'tf.contrib.summary.create_file_writer': {('name', 4): (ast_edits.WARNING, "tf.contrib.summary.create_file_writer() no longer supports implicit writer re-use based on shared logdirs or resource names; this call site passed a 'name' argument that has been removed. The new tf.compat.v2.summary.create_file_writer() replacement has a 'name' parameter but the semantics are the usual ones to name the op itself and do not control writer re-use; writers must be manually re-used if desired.")}, 'tf.contrib.summary.generic': {('name', 0): (ast_edits.WARNING, "tf.contrib.summary.generic() takes a 'name' argument for the op name that also determines the emitted tag (prefixed by any active name scopes), but tf.compat.v2.summary.write(), which replaces it, separates these into 'tag' and 'name' arguments. The 'name' argument here has been converted to 'tag' to preserve a meaningful tag, but any name scopes will not be reflected in the tag without manual editing."), ('family', 3): contrib_summary_family_arg_comment}, 'tf.contrib.summary.histogram': {('family', 2): contrib_summary_family_arg_comment}, 'tf.contrib.summary.image': {('bad_color', 2): (ast_edits.WARNING, "tf.contrib.summary.image no longer takes the 'bad_color' argument; caller must now preprocess if needed. This call site specifies a bad_color argument so it cannot be converted safely."), ('family', 4): contrib_summary_family_arg_comment}, 'tf.contrib.summary.scalar': {('family', 2): contrib_summary_family_arg_comment}, 'tf.image.resize': {('align_corners', 3): (ast_edits.WARNING, 'align_corners is not supported by tf.image.resize, the new default transformation is close to what v1 provided. If you require exactly the same transformation as before, use compat.v1.image.resize.')}, 'tf.image.resize_bilinear': {('align_corners', 2): (ast_edits.WARNING, 'align_corners is not supported by tf.image.resize, the new default transformation is close to what v1 provided. If you require exactly the same transformation as before, use compat.v1.image.resize_bilinear.')}, 'tf.image.resize_area': {('align_corners', 2): (ast_edits.WARNING, 'align_corners is not supported by tf.image.resize, the new default transformation is close to what v1 provided. If you require exactly the same transformation as before, use compat.v1.image.resize_area.')}, 'tf.image.resize_bicubic': {('align_corners', 2): (ast_edits.WARNING, 'align_corners is not supported by tf.image.resize, the new default transformation is close to what v1 provided. If you require exactly the same transformation as before, use compat.v1.image.resize_bicubic.')}, 'tf.image.resize_nearest_neighbor': {('align_corners', 2): (ast_edits.WARNING, 'align_corners is not supported by tf.image.resize, the new default transformation is close to what v1 provided. If you require exactly the same transformation as before, use compat.v1.image.resize_nearest_neighbor.')}}
        all_renames_v2.add_contrib_direct_import_support(self.function_arg_warnings)
        canned_estimator_msg_optimizer = 'tf.keras.optimizers.* only, so the call was converted to compat.v1. Please note that tf.train.Optimizers have one-to-one correspondents in tf.keras.optimizers, so you may be able to convert to the new optimizers directly (See https://www.tensorflow.org/api_docs/python/tf/keras/optimizers). Checkpoint compatibility is not guaranteed, but there is a checkpoint converter tool that you can use.'
        canned_estimator_msg = 'no longer takes `input_layer_partitioner` arg, and it supports ' + canned_estimator_msg_optimizer
        self.function_transformers = {'*.make_initializable_iterator': _iterator_transformer, '*.make_one_shot_iterator': _iterator_transformer, 'tf.nn.dropout': _dropout_transformer, 'tf.to_bfloat16': _cast_transformer, 'tf.to_complex128': _cast_transformer, 'tf.to_complex64': _cast_transformer, 'tf.to_double': _cast_transformer, 'tf.to_float': _cast_transformer, 'tf.to_int32': _cast_transformer, 'tf.to_int64': _cast_transformer, 'tf.nn.softmax_cross_entropy_with_logits': _softmax_cross_entropy_with_logits_transformer, 'tf.image.extract_glimpse': _extract_glimpse_transformer, 'tf.image.resize_area': _image_resize_transformer, 'tf.image.resize_bicubic': _image_resize_transformer, 'tf.image.resize_bilinear': _image_resize_transformer, 'tf.image.resize_nearest_neighbor': _image_resize_transformer, 'tf.nn.fractional_avg_pool': _pool_seed_transformer, 'tf.nn.fractional_max_pool': _pool_seed_transformer, 'tf.name_scope': _name_scope_transformer, 'tf.strings.split': _string_split_rtype_transformer, 'tf.estimator.BaselineEstimator': functools.partial(_rename_if_arg_found_transformer, arg_name='optimizer', message='tf.estimator.BaselineEstimator supports ' + canned_estimator_msg_optimizer), 'tf.estimator.BaselineClassifier': functools.partial(_rename_if_arg_found_and_add_loss_reduction_transformer, arg_names=['optimizer'], message='tf.estimator.BaselineClassifier supports ' + canned_estimator_msg_optimizer), 'tf.estimator.BaselineRegressor': functools.partial(_rename_if_arg_found_and_add_loss_reduction_transformer, arg_names=['input_layer_partitioner', 'optimizer'], message='tf.estimator.BaselineRegressor supports ' + canned_estimator_msg_optimizer), 'tf.estimator.DNNEstimator': functools.partial(_rename_if_any_arg_found_transformer, arg_names=['input_layer_partitioner', 'optimizer'], message='tf.estimator.DNNEstimator no longer takes input_layer_partitioner, so the call was converted to compat.v1.'), 'tf.estimator.DNNClassifier': functools.partial(_rename_if_arg_found_and_add_loss_reduction_transformer, arg_names=['input_layer_partitioner', 'optimizer'], message='tf.estimator.DNNClassifier ' + canned_estimator_msg), 'tf.estimator.DNNRegressor': functools.partial(_rename_if_arg_found_and_add_loss_reduction_transformer, arg_names=['input_layer_partitioner', 'optimizer'], message='tf.estimator.DNNRegressor ' + canned_estimator_msg), 'tf.estimator.LinearEstimator': functools.partial(_rename_if_any_arg_found_transformer, arg_names=['input_layer_partitioner', 'optimizer'], message='tf.estimator.LinearEstimator ' + canned_estimator_msg), 'tf.estimator.LinearClassifier': functools.partial(_rename_if_arg_found_and_add_loss_reduction_transformer, arg_names=['input_layer_partitioner', 'optimizer'], message='tf.estimator.LinearClassifier ' + canned_estimator_msg), 'tf.estimator.LinearRegressor': functools.partial(_rename_if_arg_found_and_add_loss_reduction_transformer, arg_names=['input_layer_partitioner', 'optimizer'], message='tf.estimator.LinearRegressor ' + canned_estimator_msg), 'tf.estimator.DNNLinearCombinedEstimator': functools.partial(_rename_if_any_arg_found_transformer, arg_names=['input_layer_partitioner', 'dnn_optimizer', 'linear_optimizer'], message='tf.estimator.DNNLinearCombinedEstimator ' + canned_estimator_msg), 'tf.estimator.DNNLinearCombinedClassifier': functools.partial(_rename_if_arg_found_and_add_loss_reduction_transformer, arg_names=['input_layer_partitioner', 'dnn_optimizer', 'linear_optimizer'], message='tf.estimator.DNNLinearCombinedClassifier ' + canned_estimator_msg), 'tf.estimator.DNNLinearCombinedRegressor': functools.partial(_rename_if_arg_found_and_add_loss_reduction_transformer, arg_names=['input_layer_partitioner', 'dnn_optimizer', 'linear_optimizer'], message='tf.estimator.DNNLinearCombinedRegressor ' + canned_estimator_msg), 'tf.device': functools.partial(_rename_if_arg_found_transformer, arg_name='device_name', arg_ok_predicate=_is_ast_str, remove_if_ok=False, message='tf.device no longer takes functions as an argument. We could not determine that the argument value is a string, so the call was converted to compat.v1.'), 'tf.zeros_like': functools.partial(_rename_if_arg_found_transformer, arg_name='optimize', arg_ok_predicate=_is_ast_true, remove_if_ok=True, message='tf.zeros_like no longer takes an optimize argument, and behaves as if optimize=True. This call site specifies something other than optimize=True, so it was converted to compat.v1.'), 'tf.ones_like': functools.partial(_rename_if_arg_found_transformer, arg_name='optimize', arg_ok_predicate=_is_ast_true, remove_if_ok=True, message='tf.ones_like no longer takes an optimize argument, and behaves as if optimize=True. This call site specifies something other than optimize=True, so it was converted to compat.v1.'), 'tf.while_loop': functools.partial(_rename_if_arg_found_transformer, arg_name='return_same_structure', arg_ok_predicate=_is_ast_true, remove_if_ok=True, message="tf.while_loop no longer takes 'return_same_structure' argument and behaves as if return_same_structure=True. This call site specifies something other than return_same_structure=True, so it was converted to compat.v1."), 'tf.nn.ctc_beam_search_decoder': functools.partial(_rename_if_arg_found_transformer, arg_name='merge_repeated', arg_ok_predicate=_is_ast_false, remove_if_ok=True, message="tf.nn.ctc_beam_search_decoder no longer takes the 'merge_repeated' argument and behaves as if merge_repeated=False. This call site specifies something other than merge_repeated=False, so it was converted to compat.v1."), 'tf.nn.dilation2d': functools.partial(_add_argument_transformer, arg_name='data_format', arg_value_ast=ast.Str('NHWC')), 'tf.nn.erosion2d': functools.partial(_add_argument_transformer, arg_name='data_format', arg_value_ast=ast.Str('NHWC')), 'tf.contrib.summary.always_record_summaries': functools.partial(_add_summary_recording_cond_transformer, cond='True'), 'tf.contrib.summary.audio': _add_summary_step_transformer, 'tf.contrib.summary.generic': _add_summary_step_transformer, 'tf.contrib.summary.histogram': _add_summary_step_transformer, 'tf.contrib.summary.image': _add_summary_step_transformer, 'tf.contrib.summary.never_record_summaries': functools.partial(_add_summary_recording_cond_transformer, cond='False'), 'tf.contrib.summary.scalar': _add_summary_step_transformer, 'tf.contrib.layers.l1_regularizer': _contrib_layers_l1_regularizer_transformer, 'tf.contrib.layers.l2_regularizer': _contrib_layers_l2_regularizer_transformer, 'tf.contrib.layers.xavier_initializer': _contrib_layers_xavier_initializer_transformer, 'tf.contrib.layers.xavier_initializer_conv2d': _contrib_layers_xavier_initializer_transformer, 'tf.contrib.layers.variance_scaling_initializer': _contrib_layers_variance_scaling_initializer_transformer, 'tf.initializers.uniform_unit_scaling': _add_uniform_scaling_initializer_transformer, 'tf.uniform_unit_scaling_initializer': _add_uniform_scaling_initializer_transformer, 'slim.l1_regularizer': _contrib_layers_l1_regularizer_transformer, 'slim.l2_regularizer': _contrib_layers_l2_regularizer_transformer, 'slim.xavier_initializer': _contrib_layers_xavier_initializer_transformer, 'slim.xavier_initializer_conv2d': _contrib_layers_xavier_initializer_transformer, 'slim.variance_scaling_initializer': _contrib_layers_variance_scaling_initializer_transformer, 'tf.keras.models.save_model': functools.partial(_add_argument_transformer, arg_name='save_format', arg_value_ast=ast.Str('h5'))}
        all_renames_v2.add_contrib_direct_import_support(self.function_transformers)
        self.module_deprecations = module_deprecations_v2.MODULE_DEPRECATIONS

    def preprocess(self, root_node, after_compat_v1_upgrade=False):
        if False:
            print('Hello World!')
        visitor = ast_edits.PastaAnalyzeVisitor(TFAPIImportAnalysisSpec())
        visitor.visit(root_node)
        detections = set(visitor.results)
        if compat_v1_import in detections and self.upgrade_compat_v1_import and (not after_compat_v1_upgrade):
            CompatV1ImportReplacer().visit(root_node)
            return self.preprocess(root_node, after_compat_v1_upgrade=True)
        if detections:
            self.function_handle = {}
            self.function_reorders = {}
            self.function_keyword_renames = {}
            self.symbol_renames = {}
            self.function_warnings = {}
            self.change_to_function = {}
            self.module_deprecations = module_deprecations_v2.MODULE_DEPRECATIONS
            self.function_transformers = {}
            self.import_renames = {}
        return (root_node, visitor.log, visitor.warnings_and_errors)

    def clear_preprocessing(self):
        if False:
            while True:
                i = 10
        self.__init__(import_rename=self.import_rename, upgrade_compat_v1_import=self.upgrade_compat_v1_import)

def _is_ast_str(node):
    if False:
        while True:
            i = 10
    'Determine whether this node represents a string.'
    allowed_types = [ast.Str]
    if hasattr(ast, 'Bytes'):
        allowed_types += [ast.Bytes]
    if hasattr(ast, 'JoinedStr'):
        allowed_types += [ast.JoinedStr]
    if hasattr(ast, 'FormattedValue'):
        allowed_types += [ast.FormattedValue]
    return isinstance(node, allowed_types)

def _is_ast_true(node):
    if False:
        print('Hello World!')
    if hasattr(ast, 'NameConstant'):
        return isinstance(node, ast.NameConstant) and node.value is True
    else:
        return isinstance(node, ast.Name) and node.id == 'True'

def _is_ast_false(node):
    if False:
        return 10
    if hasattr(ast, 'NameConstant'):
        return isinstance(node, ast.NameConstant) and node.value is False
    else:
        return isinstance(node, ast.Name) and node.id == 'False'

def _rename_if_arg_found_transformer(parent, node, full_name, name, logs, arg_name=None, arg_ok_predicate=None, remove_if_ok=False, message=None):
    if False:
        i = 10
        return i + 15
    'Replaces the given call with tf.compat.v1 if the given arg is found.\n\n  This requires the function to be called with all named args, so for using\n  this transformer, the function should also be added to renames.\n\n  If the arg is not found, the call site is left alone.\n\n  If the arg is found, and if arg_ok_predicate is given, it is called with\n  the ast Expression representing the argument value found. If it returns\n  True, the function is left alone.\n\n  If the arg is found, arg_ok_predicate is not None and returns ok, and\n  remove_if_ok is True, the argument is removed from the call.\n\n  Otherwise, `compat.v1` is inserted between tf and the function name.\n\n  Args:\n    parent: Parent of node.\n    node: ast.Call node to maybe modify.\n    full_name: full name of function to modify\n    name: name of function to modify\n    logs: list of logs to append to\n    arg_name: name of the argument to look for\n    arg_ok_predicate: predicate callable with the ast of the argument value,\n      returns whether the argument value is allowed.\n    remove_if_ok: remove the argument if present and ok as determined by\n      arg_ok_predicate.\n    message: message to print if a non-ok arg is found (and hence, the function\n      is renamed to its compat.v1 version).\n\n  Returns:\n    node, if it was modified, else None.\n  '
    (arg_present, arg_value) = ast_edits.get_arg_value(node, arg_name)
    if not arg_present:
        return
    if arg_ok_predicate and arg_ok_predicate(arg_value):
        if remove_if_ok:
            for (i, kw) in enumerate(node.keywords):
                if kw.arg == arg_name:
                    node.keywords.pop(i)
                    logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Removed argument %s for function %s' % (arg_name, full_name or name)))
                    break
            return node
        else:
            return
    new_name = full_name.replace('tf.', 'tf.compat.v1.', 1)
    node.func = ast_edits.full_name_node(new_name)
    logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Renaming %s to %s because argument %s is present. %s' % (full_name, new_name, arg_name, message if message is not None else '')))
    return node

def _add_argument_transformer(parent, node, full_name, name, logs, arg_name, arg_value_ast):
    if False:
        for i in range(10):
            print('nop')
    'Adds an argument (as a final kwarg arg_name=arg_value_ast).'
    node.keywords.append(ast.keyword(arg=arg_name, value=arg_value_ast))
    logs.append((ast_edits.INFO, node.lineno, node.col_offset, "Adding argument '%s' to call to %s." % (pasta.dump(node.keywords[-1]), full_name or name)))
    return node

def _iterator_transformer(parent, node, full_name, name, logs):
    if False:
        for i in range(10):
            print('nop')
    'Transform iterator methods to compat function calls.'
    if full_name and (full_name.startswith('tf.compat.v1.data') or full_name.startswith('tf.data')):
        return
    if not isinstance(node.func, ast.Attribute):
        return
    node.args = [node.func.value] + node.args
    node.func.value = ast_edits.full_name_node('tf.compat.v1.data')
    logs.append((ast_edits.WARNING, node.lineno, node.col_offset, 'Changing dataset.%s() to tf.compat.v1.data.%s(dataset). Please check this transformation.\n' % (name, name)))
    return node

def _dropout_transformer(parent, node, full_name, name, logs):
    if False:
        print('Hello World!')
    'Replace keep_prob with 1-rate.'

    def _replace_keep_prob_node(parent, old_value):
        if False:
            while True:
                i = 10
        'Replaces old_value with 1-(old_value).'
        one = ast.Num(n=1)
        one.lineno = 0
        one.col_offset = 0
        new_value = ast.BinOp(left=one, op=ast.Sub(), right=old_value)
        pasta.ast_utils.replace_child(parent, old_value, new_value)
        ast.copy_location(new_value, old_value)
        pasta.base.formatting.set(old_value, 'prefix', '(')
        pasta.base.formatting.set(old_value, 'suffix', ')')
    for keep_prob in node.keywords:
        if keep_prob.arg == 'keep_prob':
            logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Changing keep_prob arg of tf.nn.dropout to rate\n'))
            keep_prob.arg = 'rate'
            _replace_keep_prob_node(keep_prob, keep_prob.value)
            return node
    if len(node.args) < 2:
        logs.append((ast_edits.ERROR, node.lineno, node.col_offset, 'tf.nn.dropout called without arguments, so automatic fix was disabled. tf.nn.dropout has changed the semantics of the second argument.'))
    else:
        rate_arg = ast.keyword(arg='rate', value=node.args[1])
        _replace_keep_prob_node(rate_arg, rate_arg.value)
        node.keywords.append(rate_arg)
        del node.args[1]
        logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Changing keep_prob arg of tf.nn.dropout to rate, and recomputing value.\n'))
        return node

def _cast_transformer(parent, node, full_name, name, logs):
    if False:
        while True:
            i = 10
    'Transforms to_int and to_float to cast(..., dtype=...).'
    dtype_str = name[3:]
    if dtype_str == 'float':
        dtype_str = 'float32'
    elif dtype_str == 'double':
        dtype_str = 'float64'
    new_arg = ast.keyword(arg='dtype', value=ast.Attribute(value=ast.Name(id='tf', ctx=ast.Load()), attr=dtype_str, ctx=ast.Load()))
    if len(node.args) == 2:
        name_arg = ast.keyword(arg='name', value=node.args[-1])
        node.args = node.args[:-1]
        node.keywords.append(name_arg)
    new_arg.value.lineno = node.lineno
    new_arg.value.col_offset = node.col_offset + 100
    node.keywords.append(new_arg)
    if isinstance(node.func, ast.Attribute):
        node.func.attr = 'cast'
    else:
        assert isinstance(node.func, ast.Name)
        node.func.id = 'cast'
    logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Changed %s call to tf.cast(..., dtype=tf.%s).' % (full_name, dtype_str)))
    return node

def _softmax_cross_entropy_with_logits_transformer(parent, node, full_name, name, logs):
    if False:
        i = 10
        return i + 15
    'Wrap labels argument with stop_gradients.'

    def _wrap_label(parent, old_value):
        if False:
            return 10
        'Wrap labels with tf.stop_gradient.'
        already_stop_grad = isinstance(old_value, ast.Call) and isinstance(old_value.func, ast.Attribute) and (old_value.func.attr == 'stop_gradient') and isinstance(old_value.func.value, ast.Name) and (old_value.func.value.id == 'tf')
        if already_stop_grad:
            return False
        try:
            new_value = ast.Call(ast.Name(id='tf.stop_gradient', ctx=ast.Load()), [old_value], [])
        except TypeError:
            new_value = ast.Call(ast.Name(id='tf.stop_gradient', ctx=ast.Load()), [old_value], [], None, None)
        pasta.ast_utils.replace_child(parent, old_value, new_value)
        ast.copy_location(new_value, old_value)
        return True
    for karg in node.keywords:
        if karg.arg == 'labels':
            if _wrap_label(karg, karg.value):
                logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Changing labels arg of tf.nn.softmax_cross_entropy_with_logits to tf.stop_gradient(labels). Please check this transformation.\n'))
            return node
    return node

def _image_resize_transformer(parent, node, full_name, name, logs):
    if False:
        i = 10
        return i + 15
    'Transforms image.resize_* to image.resize(..., method=*, ...).'
    resize_method = name[7:].upper()
    new_arg = ast.keyword(arg='method', value=ast.Attribute(value=ast.Attribute(value=ast.Attribute(value=ast.Name(id='tf', ctx=ast.Load()), attr='image', ctx=ast.Load()), attr='ResizeMethod', ctx=ast.Load()), attr=resize_method, ctx=ast.Load()))
    if len(node.args) == 4:
        pos_arg = ast.keyword(arg='preserve_aspect_ratio', value=node.args[-1])
        node.args = node.args[:-1]
        node.keywords.append(pos_arg)
    if len(node.args) == 3:
        pos_arg = ast.keyword(arg='align_corners', value=node.args[-1])
        node.args = node.args[:-1]
    new_keywords = []
    for kw in node.keywords:
        if kw.arg != 'align_corners':
            new_keywords.append(kw)
    node.keywords = new_keywords
    new_arg.value.lineno = node.lineno
    new_arg.value.col_offset = node.col_offset + 100
    node.keywords.append(new_arg)
    if isinstance(node.func, ast.Attribute):
        node.func.attr = 'resize'
    else:
        assert isinstance(node.func, ast.Name)
        node.func.id = 'resize'
    logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Changed %s call to tf.image.resize(..., method=tf.image.ResizeMethod.%s).' % (full_name, resize_method)))
    return node

def _pool_seed_transformer(parent, node, full_name, name, logs):
    if False:
        for i in range(10):
            print('nop')
    'Removes seed2 and deterministic, and adds non-zero seed if needed.'
    seed_arg = None
    deterministic = False
    modified = False
    new_keywords = []
    for kw in node.keywords:
        if sys.version_info[:2] >= (3, 5) and isinstance(kw, ast.Starred):
            pass
        elif kw.arg == 'seed':
            seed_arg = kw
        elif kw.arg == 'seed2' or kw.arg == 'deterministic':
            lineno = getattr(kw, 'lineno', node.lineno)
            col_offset = getattr(kw, 'col_offset', node.col_offset)
            logs.append((ast_edits.INFO, lineno, col_offset, 'Removed argument %s for function %s' % (kw.arg, full_name or name)))
            if kw.arg == 'deterministic':
                if not _is_ast_false(kw.value):
                    deterministic = True
            modified = True
            continue
        new_keywords.append(kw)
    if deterministic:
        if seed_arg is None:
            new_keywords.append(ast.keyword(arg='seed', value=ast.Num(42)))
            logs.add((ast_edits.INFO, node.lineno, node.col_offset, 'Adding seed=42 to call to %s since determinism was requested' % (full_name or name)))
        else:
            logs.add((ast_edits.WARNING, node.lineno, node.col_offset, 'The deterministic argument is deprecated for %s, pass a non-zero seed for determinism. The deterministic argument is present, possibly not False, and the seed is already set. The converter cannot determine whether it is nonzero, please check.'))
    if modified:
        node.keywords = new_keywords
        return node
    else:
        return

def _extract_glimpse_transformer(parent, node, full_name, name, logs):
    if False:
        while True:
            i = 10

    def _replace_uniform_noise_node(parent, old_value):
        if False:
            while True:
                i = 10
        "Replaces old_value with 'uniform' or 'gaussian'."
        uniform = ast.Str(s='uniform')
        gaussian = ast.Str(s='gaussian')
        new_value = ast.IfExp(body=uniform, test=old_value, orelse=gaussian)
        pasta.ast_utils.replace_child(parent, old_value, new_value)
        ast.copy_location(new_value, old_value)
        pasta.base.formatting.set(new_value.test, 'prefix', '(')
        pasta.base.formatting.set(new_value.test, 'suffix', ')')
    for uniform_noise in node.keywords:
        if uniform_noise.arg == 'uniform_noise':
            logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Changing uniform_noise arg of tf.image.extract_glimpse to noise, and recomputing value. Please check this transformation.\n'))
            uniform_noise.arg = 'noise'
            value = 'uniform' if uniform_noise.value else 'gaussian'
            _replace_uniform_noise_node(uniform_noise, uniform_noise.value)
            return node
    if len(node.args) >= 5:
        _replace_uniform_noise_node(node, node.args[5])
        logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Changing uniform_noise arg of tf.image.extract_glimpse to noise, and recomputing value.\n'))
        return node

def _add_summary_step_transformer(parent, node, full_name, name, logs):
    if False:
        return 10
    'Adds a step argument to the summary API call if not specified.\n\n  The inserted argument value is tf.compat.v1.train.get_or_create_global_step().\n  '
    for keyword_arg in node.keywords:
        if keyword_arg.arg == 'step':
            return node
    default_value = 'tf.compat.v1.train.get_or_create_global_step()'
    ast_value = ast.parse(default_value).body[0].value
    del ast_value.lineno
    node.keywords.append(ast.keyword(arg='step', value=ast_value))
    logs.append((ast_edits.WARNING, node.lineno, node.col_offset, "Summary API writing function %s now requires a 'step' argument; inserting default of %s." % (full_name or name, default_value)))
    return node

def _add_summary_recording_cond_transformer(parent, node, full_name, name, logs, cond):
    if False:
        return 10
    'Adds cond argument to tf.contrib.summary.xxx_record_summaries().\n\n  This is in anticipation of them being renamed to tf.summary.record_if(), which\n  requires the cond argument.\n  '
    node.args.append(pasta.parse(cond))
    logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Adding `%s` argument to %s in anticipation of it being renamed to tf.compat.v2.summary.record_if()' % (cond, full_name or name)))
    return node

def _add_loss_reduction_transformer(parent, node, full_name, name, logs):
    if False:
        i = 10
        return i + 15
    'Adds a loss_reduction argument if not specified.\n\n  Default value for tf.estimator.*Classifier and tf.estimator.*Regressor\n  loss_reduction argument changed to SUM_OVER_BATCH_SIZE. So, we update\n  existing calls to use the old default value `tf.keras.losses.Reduction.SUM`.\n\n  Note: to apply this transformation, symbol must be added\n  to reordered_function_names above.\n  '
    for keyword_arg in node.keywords:
        if keyword_arg.arg == 'loss_reduction':
            return node
    default_value = 'tf.keras.losses.Reduction.SUM'
    ast_value = pasta.parse(default_value)
    node.keywords.append(ast.keyword(arg='loss_reduction', value=ast_value))
    logs.append((ast_edits.INFO, node.lineno, node.col_offset, '%s: Default value of loss_reduction has been changed to SUM_OVER_BATCH_SIZE; inserting old default value %s.\n' % (full_name or name, default_value)))
    return node

def _rename_if_any_arg_found_transformer(parent, node, full_name, name, logs, arg_names=None, arg_ok_predicate=None, remove_if_ok=False, message=None):
    if False:
        return 10
    'Replaces the given call with tf.compat.v1 if any of the arg_names is found.\n\n  Args:\n    parent: Parent of node.\n    node: ast.Call node to modify.\n    full_name: full name of function to modify.\n    name: name of function to modify.\n    logs: list of logs to append to.\n    arg_names: list of names of the argument to look for.\n    arg_ok_predicate: predicate callable with the ast of the argument value,\n      returns whether the argument value is allowed.\n    remove_if_ok: remove the argument if present and ok as determined by\n      arg_ok_predicate.\n    message: message to print if a non-ok arg is found (and hence, the function\n      is renamed to its compat.v1 version).\n\n  Returns:\n    node, if it was modified, else None.\n  '
    for arg_name in arg_names:
        rename_node = _rename_if_arg_found_transformer(parent, node, full_name, name, logs, arg_name, arg_ok_predicate, remove_if_ok, message)
        node = rename_node if rename_node else node
    return node

def _rename_if_arg_found_and_add_loss_reduction_transformer(parent, node, full_name, name, logs, arg_names=None, arg_ok_predicate=None, remove_if_ok=False, message=None):
    if False:
        print('Hello World!')
    'Combination of _rename_if_arg_found and _add_loss_reduction transformers.\n\n  Args:\n    parent: Parent of node.\n    node: ast.Call node to maybe modify.\n    full_name: full name of function to modify\n    name: name of function to modify\n    logs: list of logs to append to\n    arg_names: list of names of the argument to look for\n    arg_ok_predicate: predicate callable with the ast of the argument value,\n      returns whether the argument value is allowed.\n    remove_if_ok: remove the argument if present and ok as determined by\n      arg_ok_predicate.\n    message: message to print if a non-ok arg is found (and hence, the function\n      is renamed to its compat.v1 version).\n\n  Returns:\n    node, if it was modified, else None.\n  '
    node = _add_loss_reduction_transformer(parent, node, full_name, name, logs)
    for arg_name in arg_names:
        rename_node = _rename_if_arg_found_transformer(parent, node, full_name, name, logs, arg_name, arg_ok_predicate, remove_if_ok, message)
        node = rename_node if rename_node else node
    return node

def _add_uniform_scaling_initializer_transformer(parent, node, full_name, name, logs):
    if False:
        return 10
    'Updates references to uniform_unit_scaling_initializer.\n\n  Transforms:\n  tf.uniform_unit_scaling_initializer(factor, seed, dtype) to\n  tf.compat.v1.keras.initializers.VarianceScaling(\n      scale=factor, distribution="uniform", seed=seed)\n\n  Note: to apply this transformation, symbol must be added\n  to reordered_function_names above.\n  '
    for keyword_arg in node.keywords:
        if keyword_arg.arg == 'factor':
            keyword_arg.arg = 'scale'
    distribution_value = '"uniform"'
    ast_value = pasta.parse(distribution_value)
    node.keywords.append(ast.keyword(arg='distribution', value=ast_value))
    lineno = node.func.value.lineno
    col_offset = node.func.value.col_offset
    node.func.value = ast_edits.full_name_node('tf.compat.v1.keras.initializers')
    node.func.value.lineno = lineno
    node.func.value.col_offset = col_offset
    node.func.attr = 'VarianceScaling'
    return node

def _contrib_layers_xavier_initializer_transformer(parent, node, full_name, name, logs):
    if False:
        return 10
    'Updates references to contrib.layers.xavier_initializer.\n\n  Transforms:\n  tf.contrib.layers.xavier_initializer(uniform, seed, dtype) to\n  tf.compat.v1.keras.initializers.VarianceScaling(\n      scale=1.0, mode="fan_avg",\n      distribution=("uniform" if uniform else "truncated_normal"),\n      seed=seed, dtype=dtype)\n\n  Returns: The new node\n  '

    def _get_distribution(old_value):
        if False:
            while True:
                i = 10
        'Returns an AST matching the following:\n    ("uniform" if (old_value) else "truncated_normal")\n    '
        dist = pasta.parse('"uniform" if old_value else "truncated_normal"')
        ifexpr = dist.body[0].value
        pasta.ast_utils.replace_child(ifexpr, ifexpr.test, old_value)
        pasta.base.formatting.set(dist, 'prefix', '(')
        pasta.base.formatting.set(dist, 'suffix', ')')
        return dist
    found_distribution = False
    for keyword_arg in node.keywords:
        if keyword_arg.arg == 'uniform':
            found_distribution = True
            keyword_arg.arg = 'distribution'
            old_value = keyword_arg.value
            new_value = _get_distribution(keyword_arg.value)
            pasta.ast_utils.replace_child(keyword_arg, old_value, new_value)
            pasta.base.formatting.set(keyword_arg.value, 'prefix', '(')
            pasta.base.formatting.set(keyword_arg.value, 'suffix', ')')
    new_keywords = []
    scale = pasta.parse('1.0')
    new_keywords.append(ast.keyword(arg='scale', value=scale))
    mode = pasta.parse('"fan_avg"')
    new_keywords.append(ast.keyword(arg='mode', value=mode))
    if len(node.args) >= 1:
        found_distribution = True
        dist = _get_distribution(node.args[0])
        new_keywords.append(ast.keyword(arg='distribution', value=dist))
    if not found_distribution:
        uniform_dist = pasta.parse('"uniform"')
        new_keywords.append(ast.keyword(arg='distribution', value=uniform_dist))
    if len(node.args) >= 2:
        new_keywords.append(ast.keyword(arg='seed', value=node.args[1]))
    if len(node.args) >= 3:
        new_keywords.append(ast.keyword(arg='dtype', value=node.args[2]))
    node.args = []
    node.keywords = new_keywords + node.keywords
    lineno = node.func.value.lineno
    col_offset = node.func.value.col_offset
    node.func.value = ast_edits.full_name_node('tf.compat.v1.keras.initializers')
    node.func.value.lineno = lineno
    node.func.value.col_offset = col_offset
    node.func.attr = 'VarianceScaling'
    logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Changing tf.contrib.layers xavier initializer to a tf.compat.v1.keras.initializers.VarianceScaling and converting arguments.\n'))
    return node

def _contrib_layers_variance_scaling_initializer_transformer(parent, node, full_name, name, logs):
    if False:
        print('Hello World!')
    'Updates references to contrib.layers.variance_scaling_initializer.\n\n  Transforms:\n  tf.contrib.layers.variance_scaling_initializer(\n    factor, mode, uniform, seed, dtype\n  ) to\n  tf.compat.v1.keras.initializers.VarianceScaling(\n      scale=factor, mode=mode.lower(),\n      distribution=("uniform" if uniform else "truncated_normal"),\n      seed=seed, dtype=dtype)\n\n  And handles the case where no factor is provided and scale needs to be\n  set to 2.0 to match contrib\'s default instead of tf.keras.initializer\'s\n  default of 1.0\n  '

    def _replace_distribution(parent, old_value):
        if False:
            while True:
                i = 10
        'Replaces old_value: ("uniform" if (old_value) else "truncated_normal")'
        new_value = pasta.parse('"uniform" if old_value else "truncated_normal"')
        ifexpr = new_value.body[0].value
        pasta.ast_utils.replace_child(ifexpr, ifexpr.test, old_value)
        pasta.ast_utils.replace_child(parent, old_value, new_value)
        pasta.base.formatting.set(new_value, 'prefix', '(')
        pasta.base.formatting.set(new_value, 'suffix', ')')

    def _replace_mode(parent, old_value):
        if False:
            i = 10
            return i + 15
        'Replaces old_value with (old_value).lower().'
        new_value = pasta.parse('mode.lower()')
        mode = new_value.body[0].value.func
        pasta.ast_utils.replace_child(mode, mode.value, old_value)
        pasta.ast_utils.replace_child(parent, old_value, new_value)
        pasta.base.formatting.set(old_value, 'prefix', '(')
        pasta.base.formatting.set(old_value, 'suffix', ')')
    found_scale = False
    for keyword_arg in node.keywords:
        if keyword_arg.arg == 'factor':
            keyword_arg.arg = 'scale'
            found_scale = True
        if keyword_arg.arg == 'mode':
            _replace_mode(keyword_arg, keyword_arg.value)
        if keyword_arg.arg == 'uniform':
            keyword_arg.arg = 'distribution'
            _replace_distribution(keyword_arg, keyword_arg.value)
    if len(node.args) >= 1:
        found_scale = True
    if len(node.args) >= 2:
        _replace_mode(node, node.args[1])
    if len(node.args) >= 3:
        _replace_distribution(node, node.args[2])
    if not found_scale:
        scale_value = pasta.parse('2.0')
        node.keywords = [ast.keyword(arg='scale', value=scale_value)] + node.keywords
    lineno = node.func.value.lineno
    col_offset = node.func.value.col_offset
    node.func.value = ast_edits.full_name_node('tf.compat.v1.keras.initializers')
    node.func.value.lineno = lineno
    node.func.value.col_offset = col_offset
    node.func.attr = 'VarianceScaling'
    logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Changing tf.contrib.layers.variance_scaling_initializer to a tf.compat.v1.keras.initializers.VarianceScaling and converting arguments.\n'))
    return node

def _contrib_layers_l1_regularizer_transformer(parent, node, full_name, name, logs):
    if False:
        for i in range(10):
            print('nop')
    "Replace slim l1 regularizer with Keras one.\n\n  This entails renaming the 'scale' arg to 'l' and dropping any\n  provided scope arg.\n  "
    scope_keyword = None
    for keyword in node.keywords:
        if keyword.arg == 'scale':
            logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Renaming scale arg of regularizer\n'))
            keyword.arg = 'l'
        if keyword.arg == 'scope':
            scope_keyword = keyword
    if scope_keyword:
        logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Dropping scope arg from tf.contrib.layers.l1_regularizer, because it is unsupported in tf.keras.regularizers.l1\n'))
        node.keywords.remove(scope_keyword)
    if len(node.args) > 1:
        node.args = node.args[:1]
        logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Dropping scope arg from tf.contrib.layers.l1_regularizer, because it is unsupported in tf.keras.regularizers.l1\n'))
    lineno = node.func.value.lineno
    col_offset = node.func.value.col_offset
    node.func.value = ast_edits.full_name_node('tf.keras.regularizers')
    node.func.value.lineno = lineno
    node.func.value.col_offset = col_offset
    node.func.attr = 'l1'
    return node

def _contrib_layers_l2_regularizer_transformer(parent, node, full_name, name, logs):
    if False:
        return 10
    'Replace slim l2 regularizer with Keras one, with l=0.5*scale.\n\n  Also drops the scope argument.\n  '

    def _replace_scale_node(parent, old_value):
        if False:
            for i in range(10):
                print('nop')
        'Replaces old_value with 0.5*(old_value).'
        half = ast.Num(n=0.5)
        half.lineno = 0
        half.col_offset = 0
        new_value = ast.BinOp(left=half, op=ast.Mult(), right=old_value)
        pasta.ast_utils.replace_child(parent, old_value, new_value)
        pasta.base.formatting.set(old_value, 'prefix', '(')
        pasta.base.formatting.set(old_value, 'suffix', ')')
    scope_keyword = None
    for keyword in node.keywords:
        if keyword.arg == 'scale':
            keyword.arg = 'l'
            _replace_scale_node(keyword, keyword.value)
        if keyword.arg == 'scope':
            scope_keyword = keyword
    if len(node.args) >= 1:
        _replace_scale_node(node, node.args[0])
    if scope_keyword:
        logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Dropping scope arg from tf.contrib.layers.l2_regularizer, because it is unsupported in tf.keras.regularizers.l2\n'))
        node.keywords.remove(scope_keyword)
    if len(node.args) > 1:
        node.args = node.args[:1]
        logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Dropping scope arg from tf.contrib.layers.l2_regularizer, because it is unsupported in tf.keras.regularizers.l2\n'))
    logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Multiplying scale arg of tf.contrib.layers.l2_regularizer by half to what tf.keras.regularizers.l2 expects.\n'))
    lineno = node.func.value.lineno
    col_offset = node.func.value.col_offset
    node.func.value = ast_edits.full_name_node('tf.keras.regularizers')
    node.func.value.lineno = lineno
    node.func.value.col_offset = col_offset
    node.func.attr = 'l2'
    return node

def _name_scope_transformer(parent, node, full_name, name, logs):
    if False:
        print('Hello World!')
    "Fix name scope invocation to use 'default_name' and omit 'values' args."
    (name_found, name) = ast_edits.get_arg_value(node, 'name', 0)
    (default_found, default_name) = ast_edits.get_arg_value(node, 'default_name', 1)
    if name_found and pasta.dump(name) != 'None':
        logs.append((ast_edits.INFO, node.lineno, node.col_offset, '`name` passed to `name_scope`. Because you may be re-entering an existing scope, it is not safe to convert automatically,  the v2 name_scope does not support re-entering scopes by name.\n'))
        new_name = 'tf.compat.v1.name_scope'
        logs.append((ast_edits.INFO, node.func.lineno, node.func.col_offset, 'Renamed %r to %r' % (full_name, new_name)))
        new_name_node = ast_edits.full_name_node(new_name, node.func.ctx)
        ast.copy_location(new_name_node, node.func)
        pasta.ast_utils.replace_child(node, node.func, new_name_node)
        return node
    if default_found:
        logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Using default_name as name in call to name_scope.\n'))
        node.args = []
        node.keywords = [ast.keyword(arg='name', value=default_name)]
        return node
    logs.append((ast_edits.ERROR, node.lineno, node.col_offset, 'name_scope call with neither name nor default_name cannot be converted properly.'))

def _rename_to_compat_v1(node, full_name, logs, reason):
    if False:
        i = 10
        return i + 15
    new_name = full_name.replace('tf.', 'tf.compat.v1.', 1)
    return _rename_func(node, full_name, new_name, logs, reason)

def _rename_func(node, full_name, new_name, logs, reason):
    if False:
        return 10
    logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Renamed %r to %r: %s' % (full_name, new_name, reason)))
    new_name_node = ast_edits.full_name_node(new_name, node.func.ctx)
    ast.copy_location(new_name_node, node.func)
    pasta.ast_utils.replace_child(node, node.func, new_name_node)
    return node

def _string_split_transformer(parent, node, full_name, name, logs):
    if False:
        i = 10
        return i + 15
    'Update tf.string_split arguments: skip_empty, sep, result_type, source.'
    for (i, kw) in enumerate(node.keywords):
        if kw.arg == 'skip_empty':
            if _is_ast_false(kw.value):
                logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'removed argument skip_empty for tf.string_split.'))
                node.keywords.pop(i)
                break
            else:
                return _rename_to_compat_v1(node, full_name, logs, "tf.string_split's replacement no longer takes the skip_empty argument.")
    found_sep = False
    for (i, kw) in enumerate(node.keywords):
        if kw.arg == 'sep':
            found_sep = True
            if isinstance(kw.value, ast.Str):
                if kw.value.s == '':
                    node = _rename_func(node, full_name, 'tf.strings.bytes_split', logs, 'Splitting bytes is not handled by tf.strings.bytes_split().')
                    node.keywords.pop(i)
            else:
                return _rename_to_compat_v1(node, full_name, logs, "The semantics for tf.string_split's sep parameter have changed when sep is the empty string; but sep is not a string literal, so we can't tell if it's an empty string.")
    if not found_sep:
        return _rename_to_compat_v1(node, full_name, logs, "The semantics for tf.string_split's sep parameter have changed when sep unspecified: it now splits on all whitespace, not just the space character.")
    return _string_split_rtype_transformer(parent, node, full_name, name, logs)

def _string_split_rtype_transformer(parent, node, full_name, name, logs):
    if False:
        i = 10
        return i + 15
    'Update tf.strings.split arguments: result_type, source.'
    need_to_sparse = True
    for (i, kw) in enumerate(node.keywords):
        if kw.arg == 'result_type':
            if isinstance(kw.value, ast.Str) and kw.value.s in ('RaggedTensor', 'SparseTensor'):
                logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Removed argument result_type=%r for function %s' % (kw.value.s, full_name or name)))
                node.keywords.pop(i)
                if kw.value.s == 'RaggedTensor':
                    need_to_sparse = False
            else:
                return _rename_to_compat_v1(node, full_name, logs, '%s no longer takes the result_type parameter.' % full_name)
            break
    for (i, kw) in enumerate(node.keywords):
        if kw.arg == 'source':
            kw.arg = 'input'
    if need_to_sparse:
        if isinstance(parent, ast.Attribute) and parent.attr == 'to_sparse':
            return
        logs.append((ast_edits.INFO, node.lineno, node.col_offset, 'Adding call to RaggedTensor.to_sparse() to result of strings.split, since it now returns a RaggedTensor.'))
        node = ast.Attribute(value=copy.deepcopy(node), attr='to_sparse')
        try:
            node = ast.Call(node, [], [])
        except TypeError:
            node = ast.Call(node, [], [], None, None)
    return node