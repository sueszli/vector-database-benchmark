"""``PickleDataSet`` loads/saves data from/to a Redis database. The underlying
functionality is supported by the redis library, so it supports all allowed
options for instantiating the redis app ``from_url`` and setting a value."""
import importlib
import os
from copy import deepcopy
from typing import Any, Dict
import redis
from kedro.io.core import AbstractDataset, DatasetError

class PickleDataSet(AbstractDataset[Any, Any]):
    """``PickleDataSet`` loads/saves data from/to a Redis database. The
    underlying functionality is supported by the redis library, so it supports
    all allowed options for instantiating the redis app ``from_url`` and setting
    a value.

    Example usage for the
    `YAML API <https://kedro.readthedocs.io/en/stable/data/    data_catalog_yaml_examples.html>`_:


    .. code-block:: yaml

        my_python_object: # simple example
          type: redis.PickleDataSet
          key: my_object
          from_url_args:
            url: redis://127.0.0.1:6379

        final_python_object: # example with save args
          type: redis.PickleDataSet
          key: my_final_object
          from_url_args:
            url: redis://127.0.0.1:6379
            db: 1
          save_args:
            ex: 10

    Example usage for the
    `Python API <https://kedro.readthedocs.io/en/stable/data/    advanced_data_catalog_usage.html>`_:
    ::

        >>> from kedro.extras.datasets.redis import PickleDataSet
        >>> import pandas as pd
        >>>
        >>> data = pd.DataFrame({'col1': [1, 2], 'col2': [4, 5],
        >>>                       'col3': [5, 6]})
        >>>
        >>> my_data = PickleDataSet(key="my_data")
        >>> my_data.save(data)
        >>> reloaded = my_data.load()
        >>> assert data.equals(reloaded)
    """
    DEFAULT_REDIS_URL = os.getenv('REDIS_URL', 'redis://127.0.0.1:6379')
    DEFAULT_LOAD_ARGS = {}
    DEFAULT_SAVE_ARGS = {}

    def __init__(self, key: str, backend: str='pickle', load_args: Dict[str, Any]=None, save_args: Dict[str, Any]=None, credentials: Dict[str, Any]=None, redis_args: Dict[str, Any]=None) -> None:
        if False:
            return 10
        'Creates a new instance of ``PickleDataSet``. This loads/saves data from/to\n        a Redis database while deserialising/serialising. Supports custom backends to\n        serialise/deserialise objects.\n\n        Example backends that are compatible (non-exhaustive):\n            * `pickle`\n            * `dill`\n            * `compress_pickle`\n\n        Example backends that are incompatible:\n            * `torch`\n\n        Args:\n            key: The key to use for saving/loading object to Redis.\n            backend: Backend to use, must be an import path to a module which satisfies the\n                ``pickle`` interface. That is, contains a `loads` and `dumps` function.\n                Defaults to \'pickle\'.\n            load_args: Pickle options for loading pickle files.\n                You can pass in arguments that the backend load function specified accepts, e.g:\n                pickle.loads: https://docs.python.org/3/library/pickle.html#pickle.loads\n                dill.loads: https://dill.readthedocs.io/en/latest/index.html#dill.loads\n                compress_pickle.loads:\n                https://lucianopaz.github.io/compress_pickle/html/api/compress_pickle.html#compress_pickle.compress_pickle.loads\n                All defaults are preserved.\n            save_args: Pickle options for saving pickle files.\n                You can pass in arguments that the backend dump function specified accepts, e.g:\n                pickle.dumps: https://docs.python.org/3/library/pickle.html#pickle.dump\n                dill.dumps: https://dill.readthedocs.io/en/latest/index.html#dill.dumps\n                compress_pickle.dumps:\n                https://lucianopaz.github.io/compress_pickle/html/api/compress_pickle.html#compress_pickle.compress_pickle.dumps\n                All defaults are preserved.\n            credentials: Credentials required to get access to the redis server.\n                E.g. `{"password": None}`.\n            redis_args: Extra arguments to pass into the redis client constructor\n                ``redis.StrictRedis.from_url``. (e.g. `{"socket_timeout": 10}`), as well as to pass\n                to the ``redis.StrictRedis.set`` through nested keys `from_url_args` and `set_args`.\n                Here you can find all available arguments for `from_url`:\n                https://redis-py.readthedocs.io/en/stable/connections.html?highlight=from_url#redis.Redis.from_url\n                All defaults are preserved, except `url`, which is set to `redis://127.0.0.1:6379`.\n                You could also specify the url through the env variable ``REDIS_URL``.\n\n        Raises:\n            ValueError: If ``backend`` does not satisfy the `pickle` interface.\n            ImportError: If the ``backend`` module could not be imported.\n        '
        try:
            imported_backend = importlib.import_module(backend)
        except ImportError as exc:
            raise ImportError(f"Selected backend '{backend}' could not be imported. Make sure it is installed and importable.") from exc
        if not (hasattr(imported_backend, 'loads') and hasattr(imported_backend, 'dumps')):
            raise ValueError(f"Selected backend '{backend}' should satisfy the pickle interface. Missing one of 'loads' and 'dumps' on the backend.")
        self._backend = backend
        self._key = key
        _redis_args = deepcopy(redis_args) or {}
        self._redis_from_url_args = _redis_args.pop('from_url_args', {})
        self._redis_from_url_args.setdefault('url', self.DEFAULT_REDIS_URL)
        self._redis_set_args = _redis_args.pop('set_args', {})
        _credentials = deepcopy(credentials) or {}
        self._load_args = deepcopy(self.DEFAULT_LOAD_ARGS)
        if load_args is not None:
            self._load_args.update(load_args)
        self._save_args = deepcopy(self.DEFAULT_SAVE_ARGS)
        if save_args is not None:
            self._save_args.update(save_args)
        self._redis_db = redis.Redis.from_url(**self._redis_from_url_args, **_credentials)

    def _describe(self) -> Dict[str, Any]:
        if False:
            return 10
        return {'key': self._key, **self._redis_from_url_args}

    def _load(self) -> Any:
        if False:
            i = 10
            return i + 15
        if not self.exists():
            raise DatasetError(f'The provided key {self._key} does not exists.')
        imported_backend = importlib.import_module(self._backend)
        return imported_backend.loads(self._redis_db.get(self._key), **self._load_args)

    def _save(self, data: Any) -> None:
        if False:
            print('Hello World!')
        try:
            imported_backend = importlib.import_module(self._backend)
            self._redis_db.set(self._key, imported_backend.dumps(data, **self._save_args), **self._redis_set_args)
        except Exception as exc:
            raise DatasetError(f'{data.__class__} was not serialised due to: {exc}') from exc

    def _exists(self) -> bool:
        if False:
            print('Hello World!')
        try:
            return bool(self._redis_db.exists(self._key))
        except Exception as exc:
            raise DatasetError(f'The existence of key {self._key} could not be established due to: {exc}') from exc