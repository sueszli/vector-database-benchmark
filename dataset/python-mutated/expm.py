from urllib.parse import urlparse
import mlflow
from filelock import FileLock
from mlflow.exceptions import MlflowException, RESOURCE_ALREADY_EXISTS, ErrorCode
from mlflow.entities import ViewType
import os
from typing import Optional, Text
from .exp import MLflowExperiment, Experiment
from ..config import C
from .recorder import Recorder
from ..log import get_module_logger
from ..utils.exceptions import ExpAlreadyExistError
logger = get_module_logger('workflow')

class ExpManager:
    """
    This is the `ExpManager` class for managing experiments. The API is designed similar to mlflow.
    (The link: https://mlflow.org/docs/latest/python_api/mlflow.html)

    The `ExpManager` is expected to be a singleton (btw, we can have multiple `Experiment`s with different uri. user can get different experiments from different uri, and then compare records of them). Global Config (i.e. `C`)  is also a singleton.

    So we try to align them together.  They share the same variable, which is called **default uri**. Please refer to `ExpManager.default_uri` for details of variable sharing.

    When the user starts an experiment, the user may want to set the uri to a specific uri (it will override **default uri** during this period), and then unset the **specific uri** and fallback to the **default uri**.    `ExpManager._active_exp_uri` is that **specific uri**.
    """
    active_experiment: Optional[Experiment]

    def __init__(self, uri: Text, default_exp_name: Optional[Text]):
        if False:
            print('Hello World!')
        self.default_uri = uri
        self._active_exp_uri = None
        self._default_exp_name = default_exp_name
        self.active_experiment = None
        logger.debug(f'experiment manager uri is at {self.uri}')

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '{name}(uri={uri})'.format(name=self.__class__.__name__, uri=self.uri)

    def start_exp(self, *, experiment_id: Optional[Text]=None, experiment_name: Optional[Text]=None, recorder_id: Optional[Text]=None, recorder_name: Optional[Text]=None, uri: Optional[Text]=None, resume: bool=False, **kwargs) -> Experiment:
        if False:
            i = 10
            return i + 15
        '\n        Start an experiment. This method includes first get_or_create an experiment, and then\n        set it to be active.\n\n        Maintaining `_active_exp_uri` is included in start_exp, remaining implementation should be included in _end_exp in subclass\n\n        Parameters\n        ----------\n        experiment_id : str\n            id of the active experiment.\n        experiment_name : str\n            name of the active experiment.\n        recorder_id : str\n            id of the recorder to be started.\n        recorder_name : str\n            name of the recorder to be started.\n        uri : str\n            the current tracking URI.\n        resume : boolean\n            whether to resume the experiment and recorder.\n\n        Returns\n        -------\n        An active experiment.\n        '
        self._active_exp_uri = uri
        return self._start_exp(experiment_id=experiment_id, experiment_name=experiment_name, recorder_id=recorder_id, recorder_name=recorder_name, resume=resume, **kwargs)

    def _start_exp(self, *args, **kwargs) -> Experiment:
        if False:
            i = 10
            return i + 15
        'Please refer to the doc of `start_exp`'
        raise NotImplementedError(f'Please implement the `start_exp` method.')

    def end_exp(self, recorder_status: Text=Recorder.STATUS_S, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        End an active experiment.\n\n        Maintaining `_active_exp_uri` is included in end_exp, remaining implementation should be included in _end_exp in subclass\n\n        Parameters\n        ----------\n        experiment_name : str\n            name of the active experiment.\n        recorder_status : str\n            the status of the active recorder of the experiment.\n        '
        self._active_exp_uri = None
        self._end_exp(recorder_status=recorder_status, **kwargs)

    def _end_exp(self, recorder_status: Text=Recorder.STATUS_S, **kwargs):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError(f'Please implement the `end_exp` method.')

    def create_exp(self, experiment_name: Optional[Text]=None):
        if False:
            while True:
                i = 10
        '\n        Create an experiment.\n\n        Parameters\n        ----------\n        experiment_name : str\n            the experiment name, which must be unique.\n\n        Returns\n        -------\n        An experiment object.\n\n        Raise\n        -----\n        ExpAlreadyExistError\n        '
        raise NotImplementedError(f'Please implement the `create_exp` method.')

    def search_records(self, experiment_ids=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Get a pandas DataFrame of records that fit the search criteria of the experiment.\n        Inputs are the search criteria user want to apply.\n\n        Returns\n        -------\n        A pandas.DataFrame of records, where each metric, parameter, and tag\n        are expanded into their own columns named metrics.*, params.*, and tags.*\n        respectively. For records that don't have a particular metric, parameter, or tag, their\n        value will be (NumPy) Nan, None, or None respectively.\n        "
        raise NotImplementedError(f'Please implement the `search_records` method.')

    def get_exp(self, *, experiment_id=None, experiment_name=None, create: bool=True, start: bool=False):
        if False:
            i = 10
            return i + 15
        "\n        Retrieve an experiment. This method includes getting an active experiment, and get_or_create a specific experiment.\n\n        When user specify experiment id and name, the method will try to return the specific experiment.\n        When user does not provide recorder id or name, the method will try to return the current active experiment.\n        The `create` argument determines whether the method will automatically create a new experiment according\n        to user's specification if the experiment hasn't been created before.\n\n        * If `create` is True:\n\n            * If `active experiment` exists:\n\n                * no id or name specified, return the active experiment.\n                * if id or name is specified, return the specified experiment. If no such exp found, create a new experiment with given id or name. If `start` is set to be True, the experiment is set to be active.\n\n            * If `active experiment` not exists:\n\n                * no id or name specified, create a default experiment.\n                * if id or name is specified, return the specified experiment. If no such exp found, create a new experiment with given id or name. If `start` is set to be True, the experiment is set to be active.\n\n        * Else If `create` is False:\n\n            * If `active experiment` exists:\n\n                * no id or name specified, return the active experiment.\n                * if id or name is specified, return the specified experiment. If no such exp found, raise Error.\n\n            * If `active experiment` not exists:\n\n                *  no id or name specified. If the default experiment exists, return it, otherwise, raise Error.\n                * if id or name is specified, return the specified experiment. If no such exp found, raise Error.\n\n        Parameters\n        ----------\n        experiment_id : str\n            id of the experiment to return.\n        experiment_name : str\n            name of the experiment to return.\n        create : boolean\n            create the experiment it if hasn't been created before.\n        start : boolean\n            start the new experiment if one is created.\n\n        Returns\n        -------\n        An experiment object.\n        "
        if experiment_id is None and experiment_name is None:
            if self.active_experiment is not None:
                return self.active_experiment
            experiment_name = self._default_exp_name
        if create:
            (exp, _) = self._get_or_create_exp(experiment_id=experiment_id, experiment_name=experiment_name)
        else:
            exp = self._get_exp(experiment_id=experiment_id, experiment_name=experiment_name)
        if self.active_experiment is None and start:
            self.active_experiment = exp
            self.active_experiment.start()
        return exp

    def _get_or_create_exp(self, experiment_id=None, experiment_name=None) -> (object, bool):
        if False:
            return 10
        '\n        Method for getting or creating an experiment. It will try to first get a valid experiment, if exception occurs, it will\n        automatically create a new experiment based on the given id and name.\n        '
        try:
            return (self._get_exp(experiment_id=experiment_id, experiment_name=experiment_name), False)
        except ValueError:
            if experiment_name is None:
                experiment_name = self._default_exp_name
            logger.warning(f'No valid experiment found. Create a new experiment with name {experiment_name}.')
            pr = urlparse(self.uri)
            if pr.scheme == 'file':
                with FileLock(os.path.join(pr.netloc, pr.path, 'filelock')):
                    return (self.create_exp(experiment_name), True)
            try:
                return (self.create_exp(experiment_name), True)
            except ExpAlreadyExistError:
                return (self._get_exp(experiment_id=experiment_id, experiment_name=experiment_name), False)

    def _get_exp(self, experiment_id=None, experiment_name=None) -> Experiment:
        if False:
            return 10
        '\n        Get specific experiment by name or id. If it does not exist, raise ValueError.\n\n        Parameters\n        ----------\n        experiment_id :\n            The id of experiment\n        experiment_name :\n            The name of experiment\n\n        Returns\n        -------\n        Experiment:\n            The searched experiment\n\n        Raises\n        ------\n        ValueError\n        '
        raise NotImplementedError(f'Please implement the `_get_exp` method')

    def delete_exp(self, experiment_id=None, experiment_name=None):
        if False:
            i = 10
            return i + 15
        '\n        Delete an experiment.\n\n        Parameters\n        ----------\n        experiment_id  : str\n            the experiment id.\n        experiment_name  : str\n            the experiment name.\n        '
        raise NotImplementedError(f'Please implement the `delete_exp` method.')

    @property
    def default_uri(self):
        if False:
            print('Hello World!')
        '\n        Get the default tracking URI from qlib.config.C\n        '
        if 'kwargs' not in C.exp_manager or 'uri' not in C.exp_manager['kwargs']:
            raise ValueError('The default URI is not set in qlib.config.C')
        return C.exp_manager['kwargs']['uri']

    @default_uri.setter
    def default_uri(self, value):
        if False:
            return 10
        C.exp_manager.setdefault('kwargs', {})['uri'] = value

    @property
    def uri(self):
        if False:
            i = 10
            return i + 15
        '\n        Get the default tracking URI or current URI.\n\n        Returns\n        -------\n        The tracking URI string.\n        '
        return self._active_exp_uri or self.default_uri

    def list_experiments(self):
        if False:
            print('Hello World!')
        '\n        List all the existing experiments.\n\n        Returns\n        -------\n        A dictionary (name -> experiment) of experiments information that being stored.\n        '
        raise NotImplementedError(f'Please implement the `list_experiments` method.')

class MLflowExpManager(ExpManager):
    """
    Use mlflow to implement ExpManager.
    """

    @property
    def client(self):
        if False:
            while True:
                i = 10
        return mlflow.tracking.MlflowClient(tracking_uri=self.uri)

    def _start_exp(self, *, experiment_id: Optional[Text]=None, experiment_name: Optional[Text]=None, recorder_id: Optional[Text]=None, recorder_name: Optional[Text]=None, resume: bool=False):
        if False:
            print('Hello World!')
        if experiment_name is None:
            experiment_name = self._default_exp_name
        (experiment, _) = self._get_or_create_exp(experiment_id=experiment_id, experiment_name=experiment_name)
        self.active_experiment = experiment
        self.active_experiment.start(recorder_id=recorder_id, recorder_name=recorder_name, resume=resume)
        return self.active_experiment

    def _end_exp(self, recorder_status: Text=Recorder.STATUS_S):
        if False:
            return 10
        if self.active_experiment is not None:
            self.active_experiment.end(recorder_status)
            self.active_experiment = None

    def create_exp(self, experiment_name: Optional[Text]=None):
        if False:
            print('Hello World!')
        assert experiment_name is not None
        try:
            experiment_id = self.client.create_experiment(experiment_name)
        except MlflowException as e:
            if e.error_code == ErrorCode.Name(RESOURCE_ALREADY_EXISTS):
                raise ExpAlreadyExistError() from e
            raise e
        return MLflowExperiment(experiment_id, experiment_name, self.uri)

    def _get_exp(self, experiment_id=None, experiment_name=None):
        if False:
            print('Hello World!')
        '\n        Method for getting or creating an experiment. It will try to first get a valid experiment, if exception occurs, it will\n        raise errors.\n        '
        assert experiment_id is not None or experiment_name is not None, 'Please input at least one of experiment/recorder id or name before retrieving experiment/recorder.'
        if experiment_id is not None:
            try:
                exp = self.client.get_experiment(experiment_id)
                if exp.lifecycle_stage.upper() == 'DELETED':
                    raise MlflowException('No valid experiment has been found.')
                experiment = MLflowExperiment(exp.experiment_id, exp.name, self.uri)
                return experiment
            except MlflowException as e:
                raise ValueError('No valid experiment has been found, please make sure the input experiment id is correct.') from e
        elif experiment_name is not None:
            try:
                exp = self.client.get_experiment_by_name(experiment_name)
                if exp is None or exp.lifecycle_stage.upper() == 'DELETED':
                    raise MlflowException('No valid experiment has been found.')
                experiment = MLflowExperiment(exp.experiment_id, experiment_name, self.uri)
                return experiment
            except MlflowException as e:
                raise ValueError('No valid experiment has been found, please make sure the input experiment name is correct.') from e

    def search_records(self, experiment_ids=None, **kwargs):
        if False:
            while True:
                i = 10
        filter_string = '' if kwargs.get('filter_string') is None else kwargs.get('filter_string')
        run_view_type = 1 if kwargs.get('run_view_type') is None else kwargs.get('run_view_type')
        max_results = 100000 if kwargs.get('max_results') is None else kwargs.get('max_results')
        order_by = kwargs.get('order_by')
        return self.client.search_runs(experiment_ids, filter_string, run_view_type, max_results, order_by)

    def delete_exp(self, experiment_id=None, experiment_name=None):
        if False:
            return 10
        assert experiment_id is not None or experiment_name is not None, 'Please input a valid experiment id or name before deleting.'
        try:
            if experiment_id is not None:
                self.client.delete_experiment(experiment_id)
            else:
                experiment = self.client.get_experiment_by_name(experiment_name)
                if experiment is None:
                    raise MlflowException('No valid experiment has been found.')
                self.client.delete_experiment(experiment.experiment_id)
        except MlflowException as e:
            raise ValueError(f'Error: {e}. Something went wrong when deleting experiment. Please check if the name/id of the experiment is correct.') from e

    def list_experiments(self):
        if False:
            while True:
                i = 10
        exps = self.client.list_experiments(view_type=ViewType.ACTIVE_ONLY)
        experiments = dict()
        for exp in exps:
            experiment = MLflowExperiment(exp.experiment_id, exp.name, self.uri)
            experiments[exp.name] = experiment
        return experiments