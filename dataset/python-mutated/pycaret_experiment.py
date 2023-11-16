import inspect
import os
import warnings
from collections import defaultdict
from typing import Any, BinaryIO, Callable, Dict, Optional, Union
import cloudpickle
import pandas as pd
import pycaret.internal.patches.sklearn
import pycaret.internal.patches.yellowbrick
import pycaret.internal.persistence
from pycaret.internal.logging import get_logger
from pycaret.utils.constants import DATAFRAME_LIKE
from pycaret.utils.generic import LazyExperimentMapping
LOGGER = get_logger()

class _PyCaretExperiment:
    _attributes_to_not_save = ['data', 'test_data', 'data_func']

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self._ml_usecase = None
        self._available_plots = {}
        self._variable_keys = set()
        self.exp_id = None
        self.gpu_param = False
        self.n_jobs_param = -1
        self.logger = LOGGER
        self._master_model_container = []
        self.data = None
        self.target_param = None
        self.idx = [None, None]
        self.fold_generator = None
        self.pipeline = None
        self._display_container = None
        self._fxs = defaultdict(list)
        self._setup_ran = False
        self._setup_params = None
        self._remote = False

    def _pack_for_remote(self) -> dict:
        if False:
            i = 10
            return i + 15
        'Serialize local member variables and send to remote. Note we should not use\n        ``__getstate__`` here because it will be hard to maintain.\n        We are using a different mechanism that is more resistant to further\n        code change. This private method is for parallel processing.\n        '
        return {'_setup_params': self._setup_params, '_remote': True}

    def _unpack_at_remote(self, data: dict) -> None:
        if False:
            i = 10
            return i + 15
        'Deserialize member variables at remote to reconstruct the experiment.\n        This private method is for parallel processing.\n        '
        for (k, v) in data.items():
            setattr(self, k, v)

    def _register_setup_params(self, params: dict) -> None:
        if False:
            print('Hello World!')
        'Register the parameters used to call ``setup`` at local machine.\n        This information will be sent to remote workers to re-setup the experiments.\n        This private method is for parallel processing.\n        '
        self._setup_params = {k: v for (k, v) in params.items() if k != 'self' and v is not None}

    @property
    def _property_keys(self) -> set:
        if False:
            for i in range(10):
                print('nop')
        return {n for n in dir(self) if not n.startswith('_') and isinstance(getattr(self.__class__, n, None), property)}

    @property
    def gpu_n_jobs_param(self) -> int:
        if False:
            print('Hello World!')
        return self.n_jobs_param if not self.gpu_param else 1

    @property
    def variables(self) -> dict:
        if False:
            return 10
        return LazyExperimentMapping(self)

    @property
    def is_multiclass(self) -> bool:
        if False:
            return 10
        '\n        Method to check if the problem is multiclass.\n        '
        return False

    @property
    def variable_and_property_keys(self) -> set:
        if False:
            while True:
                i = 10
        return self._variable_keys.union(self._property_keys)

    def _check_environment(self) -> None:
        if False:
            print('Hello World!')
        self.logger.info('Checking environment')
        from platform import machine, platform, python_build, python_version
        self.logger.info(f'python_version: {python_version()}')
        self.logger.info(f'python_build: {python_build()}')
        self.logger.info(f'machine: {machine()}')
        self.logger.info(f'platform: {platform()}')
        import psutil
        self.logger.info(f'Memory: {psutil.virtual_memory()}')
        self.logger.info(f'Physical Core: {psutil.cpu_count(logical=False)}')
        self.logger.info(f'Logical Core: {psutil.cpu_count(logical=True)}')
        from pycaret.utils._show_versions import show_versions
        self.logger.info('Checking libraries')
        self.logger.info(show_versions(logger=self.logger))

    def setup(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        return

    def _check_setup_ran(self):
        if False:
            for i in range(10):
                print('nop')
        'Checks to see if setup has been run or not. If it has not been run, then\n        an error is raised. Useful for operations that require setup to be run before\n        they can be executed. e.g. in some experiments, setup must be run first before\n        plotting can be done.\n\n        Raises\n        ------\n        RuntimeError\n            If setup has not been run.\n        '
        if not self._setup_ran:
            raise RuntimeError('This function/method requires the users to run setup() first.\nMore info: https://pycaret.gitbook.io/docs/get-started/quickstart')

    def deploy_model(self, model, model_name: str, authentication: dict, platform: str='aws'):
        if False:
            print('Hello World!')
        return None

    def save_model(self, model, model_name: str, model_only: bool=False, verbose: bool=True, **kwargs):
        if False:
            while True:
                i = 10
        return None

    def load_model(self, model_name: str, platform: Optional[str]=None, authentication: Optional[Dict[str, str]]=None, verbose: bool=True):
        if False:
            for i in range(10):
                print('nop')
        "\n        This function loads a previously saved transformation pipeline and model\n        from the current active directory into the current python environment.\n        Load object must be a pickle file.\n\n        Example\n        -------\n        >>> saved_lr = load_model('lr_model_23122019')\n\n        This will load the previously saved model in saved_lr variable. The file\n        must be in the current directory.\n\n        Parameters\n        ----------\n        model_name : str, default = none\n            Name of pickle file to be passed as a string.\n\n        platform: str, default = None\n            Name of platform, if loading model from cloud. Current available options are:\n            'aws', 'gcp' and 'azure'.\n\n        authentication : dict\n            dictionary of applicable authentication tokens.\n\n            When platform = 'aws':\n            {'bucket' : 'Name of Bucket on S3'}\n\n            When platform = 'gcp':\n            {'project': 'gcp_pycaret', 'bucket' : 'pycaret-test'}\n\n            When platform = 'azure':\n            {'container': 'pycaret-test'}\n\n        verbose: bool, default = True\n            Success message is not printed when verbose is set to False.\n\n        Returns\n        -------\n        Model Object\n\n        "
        return pycaret.internal.persistence.load_model(model_name, platform, authentication, verbose)

    def get_logs(self, experiment_name: Optional[str]=None, save: bool=False) -> pd.DataFrame:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a table with experiment logs consisting\n        run details, parameter, metrics and tags.\n\n        Example\n        -------\n        >>> logs = get_logs()\n\n        This will return pandas dataframe.\n\n        Parameters\n        ----------\n        experiment_name : str, default = None\n            When set to None current active run is used.\n\n        save : bool, default = False\n            When set to True, csv file is saved in current directory.\n\n        Returns\n        -------\n        pandas.DataFrame\n\n        '
        import mlflow
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        if experiment_name is None:
            exp_id = self.exp_id
            experiment = client.get_experiment(exp_id)
            if experiment is None:
                raise ValueError('No active run found. Check logging parameter in setup or to get logs for inactive run pass experiment_name.')
            exp_name_log_ = experiment.name
        else:
            exp_name_log_ = experiment_name
            experiment = client.get_experiment_by_name(exp_name_log_)
            if experiment is None:
                raise ValueError('No active run found. Check logging parameter in setup or to get logs for inactive run pass experiment_name.')
            exp_id = client.get_experiment_by_name(exp_name_log_).experiment_id
        runs = mlflow.search_runs(exp_id)
        if save:
            file_name = f'{exp_name_log_}_logs.csv'
            runs.to_csv(file_name, index=False)
        return runs

    def get_config(self, variable: Optional[str]=None) -> Any:
        if False:
            return 10
        "\n        This function is used to access global environment variables.\n\n        Example\n        -------\n        >>> X_train = get_config('X_train')\n\n        This will return training features.\n\n\n        variable : str, default = None\n            Name of the variable to return the value of. If None,\n            will return a list of possible names.\n\n\n        Returns\n        -------\n        variable\n\n        "
        function_params_str = ', '.join([f'{k}={v}' for (k, v) in locals().items() if not k == 'globals_d'])
        self.logger.info('Initializing get_config()')
        self.logger.info(f'get_config({function_params_str})')
        variable_and_property_keys = self.variable_and_property_keys
        if not variable:
            return variable_and_property_keys
        if variable not in variable_and_property_keys:
            raise ValueError(f"Variable '{variable}' not found. Possible variables are: {list(variable_and_property_keys)}")
        if any((variable.endswith(attr) for attr in ('train', 'test', 'dataset'))):
            msg = f"Variable: '{variable}' used to return the transformed values in PyCaret 2.x. From PyCaret 3.x, this will return the raw values. If you need the transformed values, call get_config with '{variable}_transformed' instead."
            self.logger.info(msg)
            warnings.warn(msg)
        var = getattr(self, variable)
        self.logger.info(f'Variable: {variable[:-12]} returned as {var}')
        self.logger.info('get_config() successfully completed......................................')
        return var

    def set_config(self, variable: Optional[str]=None, value: Optional[Any]=None, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        This function is used to reset global environment variables.\n\n        Example\n        -------\n        >>> set_config('seed', 123)\n\n        This will set the global seed to '123'.\n\n        "
        function_params_str = ', '.join([f'{k}={v}' for (k, v) in locals().items() if not k == 'globals_d'])
        self.logger.info('Initializing set_config()')
        self.logger.info(f'set_config({function_params_str})')
        if kwargs and variable:
            raise ValueError('variable parameter cannot be used together with keyword arguments.')
        elif kwargs:
            variables = kwargs
        elif variable:
            variables = {variable: value}
        else:
            variables = {}
        for (k, v) in variables.items():
            if k.startswith('_'):
                raise ValueError(f"Variable {k} is read only ('_' prefix).")
            writeable_keys = [x for x in self._variable_keys.difference(self._property_keys) if not x.startswith('_')]
            if k not in writeable_keys:
                raise ValueError(f'Variable {k} not found or is not writeable. Possible writeable variables are: {writeable_keys}')
            setattr(self, k, v)
            self.logger.info(f'Global variable: {k} updated to {v}')
        self.logger.info('set_config() successfully completed......................................')
        return

    def __getstate__(self) -> dict:
        if False:
            print('Hello World!')
        state = self.__dict__.copy()
        for key in self._attributes_to_not_save:
            state.pop(key, None)
        if state['_setup_params']:
            state['_setup_params'] = state['_setup_params'].copy()
            for key in self._attributes_to_not_save:
                state['_setup_params'].pop(key, None)
        return state

    @classmethod
    def _load_experiment(cls, path_or_file: Union[str, os.PathLike, BinaryIO], cloudpickle_kwargs=None, preprocess_data: bool=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        cloudpickle_kwargs = cloudpickle_kwargs or {}
        try:
            loaded_exp: _PyCaretExperiment = cloudpickle.load(path_or_file, **cloudpickle_kwargs)
        except TypeError:
            with open(path_or_file, mode='rb') as f:
                loaded_exp: _PyCaretExperiment = cloudpickle.load(f, **cloudpickle_kwargs)
        original_state = loaded_exp.__dict__.copy()
        new_params = kwargs
        setup_params = loaded_exp._setup_params or {}
        setup_params = setup_params.copy()
        setup_params.update({k: v for (k, v) in new_params.items() if k in inspect.signature(cls.setup).parameters})
        if preprocess_data and (not setup_params.get('data_func', None)):
            loaded_exp.setup(**setup_params)
        else:
            data = new_params.get('data', None)
            data_func = new_params.get('data_func', None)
            if data is None and data_func is None or (data is not None and data_func is not None):
                raise ValueError('One and only one of data and data_func must be set')
            for (key, value) in new_params.items():
                setattr(loaded_exp, key, value)
            original_state['_setup_params'] = setup_params
        loaded_exp.__dict__.update(original_state)
        return loaded_exp

    @classmethod
    def load_experiment(cls, path_or_file: Union[str, os.PathLike, BinaryIO], data: Optional[DATAFRAME_LIKE]=None, data_func: Optional[Callable[[], DATAFRAME_LIKE]]=None, preprocess_data: bool=True, **cloudpickle_kwargs) -> '_PyCaretExperiment':
        if False:
            for i in range(10):
                print('nop')
        "\n        Load an experiment saved with ``save_experiment`` from path\n        or file.\n\n        The data (and test data) is NOT saved with the experiment\n        and will need to be specified again.\n\n\n        path_or_file: str or BinaryIO (file pointer)\n            The path/file pointer to load the experiment from.\n            The pickle file must be created through ``save_experiment``.\n\n\n        data: dataframe-like\n            Data set with shape (n_samples, n_features), where n_samples is the\n            number of samples and n_features is the number of features. If data\n            is not a pandas dataframe, it's converted to one using default column\n            names.\n\n\n        data_func: Callable[[], DATAFRAME_LIKE] = None\n            The function that generate ``data`` (the dataframe-like input). This\n            is useful when the dataset is large, and you need parallel operations\n            such as ``compare_models``. It can avoid broadcasting large dataset\n            from driver to workers. Notice one and only one of ``data`` and\n            ``data_func`` must be set.\n\n\n        preprocess_data: bool, default = True\n            If True, the data will be preprocessed again (through running ``setup``\n            internally). If False, the data will not be preprocessed. This means\n            you can save the value of the ``data`` attribute of an experiment\n            separately, and then load it separately and pass it here with\n            ``preprocess_data`` set to False. This is an advanced feature.\n            We recommend leaving it set to True and passing the same data\n            as passed to the initial ``setup`` call.\n\n\n        **cloudpickle_kwargs:\n            Kwargs to pass to the ``cloudpickle.load`` call.\n\n\n        Returns:\n            loaded experiment\n\n        "
        return cls._load_experiment(path_or_file, cloudpickle_kwargs=cloudpickle_kwargs, preprocess_data=preprocess_data, data=data, data_func=data_func)

    def save_experiment(self, path_or_file: Union[str, os.PathLike, BinaryIO], **cloudpickle_kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Saves the experiment to a pickle file.\n\n        The experiment is saved using cloudpickle to deal with lambda\n        functions. The data or test data is NOT saved with the experiment\n        and will need to be specified again when loading using\n        ``load_experiment``.\n\n\n        path_or_file: str or BinaryIO (file pointer)\n            The path/file pointer to save the experiment to.\n\n\n        **cloudpickle_kwargs:\n            Kwargs to pass to the ``cloudpickle.dump`` call.\n\n\n        Returns:\n            None\n\n        '
        try:
            cloudpickle.dump(self, path_or_file, **cloudpickle_kwargs)
        except TypeError:
            with open(path_or_file, mode='wb') as f:
                cloudpickle.dump(self, f, **cloudpickle_kwargs)

    def pull(self, pop=False) -> pd.DataFrame:
        if False:
            return 10
        '\n        Returns the latest displayed table.\n\n        Parameters\n        ----------\n        pop : bool, default = False\n            If true, will pop (remove) the returned dataframe from the\n            display container.\n\n        Returns\n        -------\n        pandas.DataFrame\n\n        '
        return self._display_container.pop(-1) if pop else self._display_container[-1]

    @property
    def dataset(self):
        if False:
            for i in range(10):
                print('nop')
        'Complete dataset without ignored columns.'
        return self.data[[c for c in self.data.columns if c not in self._fxs['Ignore']]]

    @property
    def X(self):
        if False:
            return 10
        'Feature set.'
        return self.dataset

    @property
    def dataset_transformed(self):
        if False:
            i = 10
            return i + 15
        'Transformed dataset.'
        return self.train_transformed

    @property
    def X_train(self):
        if False:
            while True:
                i = 10
        'Feature set of the training set.'
        return self.train

    @property
    def train(self):
        if False:
            for i in range(10):
                print('nop')
        'Training set.'
        return self.dataset

    @property
    def X_train_transformed(self):
        if False:
            i = 10
            return i + 15
        'Transformed feature set of the training set.'
        return self.pipeline.transform(self.X_train, filter_train_only=False)

    @property
    def train_transformed(self):
        if False:
            print('Hello World!')
        'Transformed training set.'
        return self.X_train_transformed

    @property
    def X_transformed(self):
        if False:
            while True:
                i = 10
        'Transformed feature set.'
        return self.X_train_transformed