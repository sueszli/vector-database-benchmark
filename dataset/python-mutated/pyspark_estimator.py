import os
import types
import logging
import tempfile
import shutil
from bigdl.dllib.utils.common import get_node_and_core_number
from bigdl.dllib.utils.file_utils import is_local_path
from bigdl.dllib.utils.utils import get_node_ip
from bigdl.orca.data.file import is_file, exists, get_remote_file_to_local, get_remote_files_with_prefix_to_local, put_local_file_to_remote, put_local_files_with_prefix_to_remote
from bigdl.orca.learn.tf2.spark_runner import SparkRunner
from bigdl.orca.learn.utils import find_free_port, find_ip_and_free_port
from bigdl.orca.learn.utils import maybe_dataframe_to_xshards, dataframe_to_xshards, convert_predict_xshards_to_dataframe, make_data_creator, load_model, save_model, process_xshards_of_pandas_dataframe, add_predict_to_pd_xshards, update_predict_xshards
from bigdl.orca.learn.log_monitor import start_log_server, stop_log_server
from bigdl.orca import OrcaContext
from bigdl.orca.data.shard import SparkXShards
from bigdl.dllib.utils.log4Error import invalidInputError
from pyspark.sql.dataframe import DataFrame
from typing import TYPE_CHECKING, Any, Dict, List, Callable, Union, Optional
if TYPE_CHECKING:
    import numpy as np
    from pyspark.sql.dataframe import DataFrame as SparkDataFrame
    from tensorflow.python.saved_model.save_options import SaveOptions
    from tensorflow.python.keras.callbacks import Callback
    from tensorflow.python.keras.engine.training import Model
logger = logging.getLogger(__name__)

def parse_model_dir(model_dir: str) -> str:
    if False:
        return 10
    if model_dir and model_dir.startswith('dbfs:/'):
        model_dir = '/dbfs/' + model_dir[len('dbfs:/'):]
    return model_dir

class SparkTFEstimator:

    def __init__(self, model_creator: Optional[Callable]=None, config: Optional[Dict]=None, compile_args_creator: Optional[Callable]=None, verbose: bool=False, workers_per_node: int=1, model_dir: Optional[str]=None, log_to_driver: bool=True, **kwargs) -> None:
        if False:
            print('Hello World!')
        self.model_creator = model_creator
        self.compile_args_creator = compile_args_creator
        self.config = {} if config is None else config
        self.verbose = verbose
        sc = OrcaContext.get_spark_context()
        (num_node, num_core) = get_node_and_core_number()
        self.num_workers = num_node * workers_per_node
        self.total_cores = num_node * num_core
        self.workerRDD = sc.parallelize(list(range(self.total_cores * 4)), self.total_cores * 4).repartition(self.num_workers)
        if 'inter_op_parallelism' not in self.config:
            self.config['inter_op_parallelism'] = 1
        if 'intra_op_parallelism' not in self.config:
            self.config['intra_op_parallelism'] = num_core // workers_per_node
        self.model_weights = None
        self.optimizer_weights = None
        self.load_path = None
        self.load_params = None
        if 'batch_size' in self.config:
            invalidInputError(False, 'Please do not specify batch_size in config. Input batch_size in the fit/evaluate function of the estimator instead.')
        self.model_dir = parse_model_dir(model_dir)
        master = sc.getConf().get('spark.master')
        if not master.startswith('local'):
            logger.info('For cluster mode, make sure to use shared filesystem path as model directory.')
        self.application_id = sc.applicationId
        self.ip = get_node_ip()
        self.port = find_free_port()
        is_local = sc.master.startswith('local')
        self.need_to_log_to_driver = not is_local and log_to_driver
        if self.need_to_log_to_driver:
            self.log_server_thread = start_log_server(self.ip, self.port)

    def _get_cluster_info(self, sc):
        if False:
            while True:
                i = 10

        def get_worker_address(iter):
            if False:
                return 10
            worker_ip = get_node_ip()
            worker_port = find_free_port()
            addresses = find_ip_and_free_port(iter)
            res = [(f'{worker_ip}:{worker_port}', address) for address in addresses]
            return res
        worker_info = self.workerRDD.barrier().mapPartitions(get_worker_address).collect()
        address_info = [info[0] for info in worker_info]
        cluster_info = [info[1] for info in worker_info]
        return (address_info, cluster_info)

    def fit(self, data: Union['SparkXShards', 'SparkDataFrame', Callable], epochs: int=1, batch_size: int=32, verbose: Union[str, int]=1, callbacks: Optional[List['Callback']]=None, validation_data: Union['SparkXShards', 'SparkDataFrame', Callable, None]=None, class_weight: Optional[Dict[int, float]]=None, initial_epoch: int=0, steps_per_epoch: Optional[int]=None, validation_steps: Optional[int]=None, validation_freq: int=1, data_config: Optional[Dict]=None, feature_cols: Optional[List[str]]=None, label_cols: Optional[List[str]]=None) -> Dict:
        if False:
            return 10
        '\n        Train this tensorflow model with train data.\n        :param data: train data. It can be XShards, Spark DataFrame or creator function which\n               returns Iter or DataLoader.\n               If data is XShards, each partition can be a Pandas DataFrame or a dictionary of\n               {\'x\': feature, \'y\': label}, where feature(label) is a numpy array or a tuple of\n               numpy arrays.\n        :param epochs: Number of epochs to train the model. Default: 1.\n        :param batch_size: Total batch size for all workers used for training. Each worker\'s batch\n               size would be this value divide the total number of workers. Default: 32.\n        :param verbose: Prints output of one model if true.\n        :param callbacks: List of Keras compatible callbacks to apply during training.\n        :param validation_data: validation data. Validation data type should be the same\n               as train data.\n        :param class_weight: Optional dictionary mapping class indices (integers) to a weight\n               (float) value, used for weighting the loss function. This can be useful to tell\n               the model to "pay more attention" to samples from an under-represented class.\n        :return:\n        '
        if not isinstance(data, types.FunctionType):
            invalidInputError(isinstance(batch_size, int) and batch_size > 0, 'batch_size should be a positive integer')
        elif batch_size:
            invalidInputError(isinstance(batch_size, int) and batch_size > 0, 'batch_size should be a positive integer')
        if batch_size:
            local_batch_size = batch_size // self.num_workers
            if local_batch_size <= 0:
                local_batch_size = 1
        else:
            local_batch_size = None
        sc = OrcaContext.get_spark_context()
        if self.model_weights:
            weights = sc.broadcast(self.model_weights)
        else:
            weights = None
        if self.optimizer_weights:
            opt_weights = sc.broadcast(self.optimizer_weights)
        else:
            opt_weights = None
        init_params = dict(model_creator=self.model_creator, model_load=self.load_path, compile_args_creator=self.compile_args_creator, config=self.config, verbose=self.verbose, size=self.num_workers, model_weights=weights, optimizer_weights=opt_weights, mode='fit', cluster_info=self._get_cluster_info(sc), model_dir=self.model_dir, application_id=self.application_id, need_to_log_to_driver=self.need_to_log_to_driver, driver_ip=self.ip, driver_port=self.port)
        if self.load_params is not None:
            init_params.update(self.load_params)
        params = dict(epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=callbacks, class_weight=class_weight, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, validation_freq=validation_freq, data_config=data_config)
        if isinstance(data, SparkXShards):
            data = data.to_lazy()
            if validation_data is not None and isinstance(validation_data, SparkXShards):
                validation_data = validation_data.to_lazy()
        if isinstance(data, DataFrame) or isinstance(data, SparkXShards):
            if data.rdd.getNumPartitions() != self.num_workers:
                data = data.repartition(self.num_workers)
            if validation_data is not None:
                invalidInputError(isinstance(validation_data, DataFrame) or isinstance(validation_data, SparkXShards), 'validation_data should have the same type with train data')
                if validation_data.rdd.getNumPartitions() != self.num_workers:
                    validation_data = validation_data.repartition(self.num_workers)
        (data, validation_data) = maybe_dataframe_to_xshards(data, validation_data, feature_cols, label_cols, mode='fit', num_workers=self.num_workers, accept_str_col=True, shard_size=local_batch_size)
        if isinstance(data, SparkXShards):
            if data._get_class_name() == 'pandas.core.frame.DataFrame':
                (data, validation_data) = process_xshards_of_pandas_dataframe(data, feature_cols, label_cols, validation_data, 'fit')
            if validation_data is None:

                def transform_func(iter, init_param, param):
                    if False:
                        print('Hello World!')
                    partition_data = list(iter)
                    param['data_creator'] = make_data_creator(partition_data)
                    return SparkRunner(**init_param).step(**param)
                res = data.rdd.barrier().mapPartitions(lambda iter: transform_func(iter, init_params, params)).collect()
            else:

                def transform_func(iter, init_param, param):
                    if False:
                        print('Hello World!')
                    data_tuple_list = list(iter)
                    data_list = [x for data_tuple in data_tuple_list for x in data_tuple[0]]
                    valid_list = [x for data_tuple in data_tuple_list for x in data_tuple[1]]
                    param['data_creator'] = make_data_creator(data_list)
                    param['validation_data_creator'] = make_data_creator(valid_list)
                    return SparkRunner(**init_param).step(**param)
                train_rdd = data.rdd.mapPartitions(lambda iter: [list(iter)])
                val_rdd = validation_data.rdd.mapPartitions(lambda iter: [list(iter)])
                res = train_rdd.zip(val_rdd).barrier().mapPartitions(lambda iter: transform_func(iter, init_params, params)).collect()
        else:
            params['data_creator'] = data
            params['validation_data_creator'] = validation_data

            def transform_func(iter, init_param, param):
                if False:
                    for i in range(10):
                        print('nop')
                return SparkRunner(**init_param).step(**param)
            res = self.workerRDD.barrier().mapPartitions(lambda iter: transform_func(iter, init_params, params)).collect()
        result = self._update_weights(res)
        return result

    def evaluate(self, data: Union['SparkXShards', 'SparkDataFrame', Callable], batch_size: int=32, num_steps: Optional[int]=None, verbose: Union[str, int]=1, sample_weight: Optional['np.ndarray']=None, callbacks: Optional[List['Callback']]=None, data_config: Optional[Dict]=None, feature_cols: Optional[List[str]]=None, label_cols: Optional[List[str]]=None) -> Dict:
        if False:
            for i in range(10):
                print('nop')
        "\n        Evaluates the model on the validation data set.\n        :param data: evaluate data. It can be XShards, Spark DataFrame or creator function which\n               returns Iter or DataLoader.\n               If data is XShards, each partition can be a Pandas DataFrame or a dictionary of\n               {'x': feature, 'y': label}, where feature(label) is a numpy array or a tuple of\n               numpy arrays.\n        :param batch_size: Total batch size for all workers used for evaluation. Each worker's batch\n               size would be this value divide the total number of workers. Default: 32.\n        :param verbose: Prints output of one model if true.\n        :param callbacks: List of Keras compatible callbacks to apply during evaluation.\n        :return: validation result\n        "
        if not isinstance(data, types.FunctionType):
            invalidInputError(isinstance(batch_size, int) and batch_size > 0, 'batch_size should be a positive integer')
        elif batch_size:
            invalidInputError(isinstance(batch_size, int) and batch_size > 0, 'batch_size should be a positive integer')
        if batch_size:
            local_batch_size = batch_size // self.num_workers
            if local_batch_size <= 0:
                local_batch_size = 1
        else:
            local_batch_size = None
        sc = OrcaContext.get_spark_context()
        logger.info('Starting validation step.')
        if isinstance(data, SparkXShards):
            data = data.to_lazy()
        if isinstance(data, DataFrame) or isinstance(data, SparkXShards):
            if data.rdd.getNumPartitions() != self.num_workers:
                data = data.repartition(self.num_workers)
        (data, _) = maybe_dataframe_to_xshards(data, validation_data=None, feature_cols=feature_cols, label_cols=label_cols, mode='evaluate', num_workers=self.num_workers, accept_str_col=True, shard_size=local_batch_size)
        if self.model_weights:
            weights = sc.broadcast(self.model_weights)
        else:
            weights = None
        init_params = dict(model_creator=self.model_creator, model_load=self.load_path, compile_args_creator=self.compile_args_creator, config=self.config, verbose=self.verbose, size=self.num_workers, model_weights=weights, mode='evaluate', cluster_info=self._get_cluster_info(sc), model_dir=self.model_dir, application_id=self.application_id, need_to_log_to_driver=self.need_to_log_to_driver, driver_ip=self.ip, driver_port=self.port)
        if self.load_params is not None:
            init_params.update(self.load_params)
        params = dict(batch_size=batch_size, verbose=verbose, sample_weight=sample_weight, steps=num_steps, callbacks=callbacks, data_config=data_config)
        if isinstance(data, SparkXShards):
            if data._get_class_name() == 'pandas.core.frame.DataFrame':
                data = process_xshards_of_pandas_dataframe(data, feature_cols, label_cols)

            def transform_func(iter, init_param, param):
                if False:
                    print('Hello World!')
                partition_data = list(iter)
                param['data_creator'] = make_data_creator(partition_data)
                return SparkRunner(**init_param).validate(**param)
            res = data.rdd.barrier().mapPartitions(lambda iter: transform_func(iter, init_params, params)).collect()
        else:
            params['data_creator'] = data

            def transform_func(iter, init_param, param):
                if False:
                    while True:
                        i = 10
                return SparkRunner(**init_param).validate(**param)
            res = self.workerRDD.barrier().mapPartitions(lambda iter: transform_func(iter, init_params, params)).collect()
        return res[0]

    def predict(self, data: Union['SparkXShards', 'SparkDataFrame'], batch_size: Optional[int]=32, verbose: Union[str, int]=1, steps: Optional[int]=None, callbacks: Optional[List['Callback']]=None, data_config: Optional[Dict]=None, feature_cols: Optional[List[str]]=None, output_cols: Optional[List[str]]=None) -> Union['SparkXShards', 'SparkDataFrame']:
        if False:
            for i in range(10):
                print('nop')
        "\n        Predict the input data\n        :param data: predict input data.  It can be XShards or Spark DataFrame.\n               If data is XShards, each partition can be a Pandas DataFrame or a dictionary of\n               {'x': feature}, where feature is a numpy array or a tuple of numpy arrays.\n        :param batch_size: Total batch size for all workers used for evaluation. Each worker's batch\n               size would be this value divide the total number of workers. Default: 32.\n        :param verbose: Prints output of one model if true.\n        :param steps: Total number of steps (batches of samples) before declaring the prediction\n               round finished. Ignored with the default value of None.\n        :param callbacks: List of Keras compatible callbacks to apply during prediction.\n        :param data_config: An optional dictionary that can be passed to data creator function.\n        :param feature_cols: Feature column name(s) of data. Only used when data is a Spark\n               DataFrame or an XShards of Pandas DataFrame. Default: None.\n        :param output_cols: Column name(s) of the model output data. Only used when data is\n               a Spark DataFrame, note the order of column name(s) should be consistent with the\n               model output data. Default: None.\n        :return:\n        "
        if batch_size:
            invalidInputError(isinstance(batch_size, int) and batch_size > 0, 'batch_size should be a positive integer')
            local_batch_size = batch_size // self.num_workers
            if local_batch_size <= 0:
                local_batch_size = 1
        else:
            local_batch_size = None
        logger.info('Starting predict step.')
        sc = OrcaContext.get_spark_context()
        if self.model_weights:
            weights = sc.broadcast(self.model_weights)
        else:
            weights = None
        init_params = dict(model_creator=self.model_creator, model_load=self.load_path, compile_args_creator=self.compile_args_creator, config=self.config, verbose=self.verbose, size=self.num_workers, model_weights=weights, mode='predict', cluster_info=None, model_dir=self.model_dir, application_id=self.application_id, need_to_log_to_driver=self.need_to_log_to_driver, driver_ip=self.ip, driver_port=self.port)
        if self.load_params is not None:
            init_params.update(self.load_params)
        params = dict(verbose=verbose, batch_size=batch_size, steps=steps, callbacks=callbacks, data_config=data_config, output_cols=output_cols)

        def transform_func(iter, init_param, param):
            if False:
                i = 10
                return i + 15
            partition_data = list(iter)
            param['data_creator'] = make_data_creator(partition_data)
            return SparkRunner(**init_param).predict(**param)
        if isinstance(data, DataFrame):
            (xshards, _) = dataframe_to_xshards(data, validation_data=None, feature_cols=feature_cols, label_cols=None, mode='predict', accept_str_col=True, shard_size=local_batch_size)

            def transform_func(iter, init_param, param):
                if False:
                    for i in range(10):
                        print('nop')
                param['data_creator'] = make_data_creator(iter)
                return SparkRunner(**init_param).predict(**param)
            pred_shards = SparkXShards.lazy(xshards.rdd.mapPartitions(lambda iter: transform_func(iter, init_params, params)))
            result = convert_predict_xshards_to_dataframe(data, pred_shards, output_cols)
        elif isinstance(data, SparkXShards):
            xshards = data.to_lazy()
            if xshards._get_class_name() == 'pandas.core.frame.DataFrame':
                xshards = process_xshards_of_pandas_dataframe(xshards, feature_cols)
                pred_shards = SparkXShards.lazy(xshards.rdd.mapPartitions(lambda iter: transform_func(iter, init_params, params)))
                result = add_predict_to_pd_xshards(data, pred_shards)
            else:
                pred_shards = SparkXShards.lazy(xshards.rdd.mapPartitions(lambda iter: transform_func(iter, init_params, params)))
                result = update_predict_xshards(data, pred_shards)
            data.uncache()
        else:
            invalidInputError(False, 'Only XShards or Spark DataFrame are supported for predict')
        return result

    def save_weights(self, filepath: str, overwrite: bool=True, save_format: Optional[str]=None) -> None:
        if False:
            i = 10
            return i + 15
        "\n        Save model weights at the provided path.\n        :param filepath: String or PathLike, path to the file to save the weights to.\n        When saving in TensorFlow format, this is the prefix used for checkpoint files\n        (multiple files are generated). Note that the '.h5' suffix causes weights to be\n        saved in HDF5 format. It can be local, hdfs, or s3 filepath.\n        :param overwrite: Whether to silently overwrite any existing file at the target location,\n        or provide the user with a manual prompt.\n        :param save_format: Either 'tf' or 'h5'.\n        A filepath ending in '.h5' or '.keras' will default to HDF5 if save_format is None.\n        Otherwise None defaults to 'tf'.\n        "
        model = self.get_model()
        if is_local_path(filepath):
            model.save_weights(filepath, overwrite, save_format)
        else:
            file_name = os.path.basename(filepath)
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, file_name)
            try:
                model.save_weights(temp_path, overwrite, save_format)
                if save_format == 'h5' or filepath.endswith('.h5') or filepath.endswith('.keras'):
                    put_local_file_to_remote(temp_path, filepath)
                else:
                    remote_dir = os.path.dirname(filepath)
                    put_local_files_with_prefix_to_remote(temp_path, remote_dir)
            finally:
                shutil.rmtree(temp_dir)

    def load_weights(self, filepath: str, by_name: bool=False) -> None:
        if False:
            print('Hello World!')
        '\n        Load tensorflow keras model weights in this estimator.\n\n        :param filepath: keras model weights save path.\n        :param by_name: Boolean, whether to load weights by name or by topological\n               order. Only topological loading is supported for weight files in\n               TensorFlow format.\n        '
        model = self.get_model(set_weights=False)
        if is_file(filepath):
            if is_local_path(filepath):
                model.load_weights(filepath, by_name)
            else:
                file_name = os.path.basename(filepath)
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, file_name)
                try:
                    get_remote_file_to_local(filepath, temp_path)
                    model.load_weights(temp_path, by_name)
                finally:
                    shutil.rmtree(temp_dir)
        elif is_local_path(filepath):
            model.load_weights(filepath, by_name)
        else:
            temp_dir = tempfile.mkdtemp()
            try:
                prefix = os.path.basename(filepath)
                get_remote_files_with_prefix_to_local(filepath, temp_dir)
                model.load_weights(os.path.join(temp_dir, prefix), by_name)
            finally:
                shutil.rmtree(temp_dir)
        self.model_weights = model.get_weights()

    def save(self, filepath: str, overwrite: bool=True, include_optimizer: bool=True, save_format: Optional[str]=None, signatures: Optional[str]=None, options: Optional['SaveOptions']=None) -> None:
        if False:
            return 10
        "\n        Saves the model to Tensorflow SavedModel or a single HDF5 file.\n\n        :param filepath: String, PathLike, path to SavedModel or H5 file to save the\n            model. It can be local/hdfs/s3 filepath\n        :param overwrite: Whether to silently overwrite any existing file at the\n            target location, or provide the user with a manual prompt.\n        :param include_optimizer: If True, save optimizer's state together.\n        :param save_format: Either `'tf'` or `'h5'`, indicating whether to save the\n            model to Tensorflow SavedModel or HDF5. Defaults to 'tf' in TF 2.X,\n            and 'h5' in TF 1.X.\n        :param signatures: Signatures to save with the SavedModel. Applicable to the\n            'tf' format only. Please see the `signatures` argument in\n            `tf.saved_model.save` for details.\n        :param options: (only applies to SavedModel format)\n            `tf.saved_model.SaveOptions` object that specifies options for\n            saving to SavedModel.\n        "
        if self.model_dir is not None and exists(self._model_saved_path):
            model = load_model(self._model_saved_path)
        else:
            model = self.get_model()
        save_model(model, filepath, overwrite=overwrite, include_optimizer=include_optimizer, save_format=save_format, signatures=signatures, options=options)

    def load(self, filepath: str, custom_objects: Optional[Dict]=None, compile: bool=True) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Loads a model saved via `estimator.save()\n\n        :param filepath: (str) Path of saved model.\n        :param custom_objects: Optional dictionary mapping names\n          (strings) to custom classes or functions to be\n          considered during deserialization.\n        :param compile: Boolean, whether to compile the model after loading.\n        :param options: Optional `tf.saved_model.LoadOptions` object that specifies\n        options for loading from SavedModel.\n\n        '
        sc = OrcaContext.get_spark_context()
        self.load_params = dict(filepath=filepath, custom_objects=custom_objects, compile=compile)
        model = load_model(**self.load_params)
        self.model_weights = model.get_weights()
        if model.optimizer is not None:
            if hasattr(model.optimizer, 'get_weights'):
                self.optimizer_weights = model.optimizer.get_weights()
            else:
                self.optimizer_weights = [var.numpy() for var in model.optimizer.variables()]
        if self.model_creator is None:
            self.load_path = filepath
            if is_file(self.load_path):
                sc.addFile(self.load_path, recursive=False)
            else:
                sc.addFile(self.load_path, recursive=True)
        if self.model_dir is not None:
            save_model(model, self._model_saved_path, save_format='h5', filemode=438)

    def get_model(self, set_weights: bool=True) -> 'Model':
        if False:
            print('Hello World!')
        '\n        Returns the learned model.\n\n        :return: the learned model.\n        '
        if self.model_creator is not None:
            model = self.model_creator(self.config)
        elif self.load_params is not None:
            model = load_model(**self.load_params)
        else:
            invalidInputError(False, 'Please load a saved model when model_creator is None.')
        if set_weights:
            if self.optimizer_weights is not None:
                import tensorflow as tf
                grad_vars = model.trainable_weights
                zero_grads = [tf.zeros_like(w) for w in grad_vars]
                model.optimizer.apply_gradients(zip(zero_grads, grad_vars))
                model.optimizer.set_weights(self.optimizer_weights)
            if self.model_weights is not None:
                model.set_weights(self.model_weights)
        return model

    @property
    def _model_saved_path(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return os.path.join(self.model_dir, '{}_model.h5'.format(self.application_id))

    def shutdown(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Shutdown estimator and release resources.\n        '
        if self.need_to_log_to_driver:
            stop_log_server(self.log_server_thread, self.ip, self.port)

    def _update_weights(self, res):
        if False:
            i = 10
            return i + 15
        if self.model_dir is not None:
            result = res
            try:
                temp_dir = tempfile.mkdtemp()
                get_remote_file_to_local(os.path.join(self.model_dir, 'state.pkl'), os.path.join(temp_dir, 'state.pkl'))
                from bigdl.orca.common import SafePickle
                with open(os.path.join(temp_dir, 'state.pkl'), 'rb') as f:
                    state = SafePickle.load(f)
                    self.model_weights = state['weights']
            finally:
                shutil.rmtree(temp_dir)
        else:
            result = res[0]
            states = res[1]
            self.model_weights = states['weights']
            self.optimizer_weights = states['opt_weights']
        return result