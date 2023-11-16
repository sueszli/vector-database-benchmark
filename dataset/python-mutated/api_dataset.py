"""``APIDataSet`` loads the data from HTTP(S) APIs.
It uses the python requests library: https://requests.readthedocs.io/en/latest/
"""
from typing import Any, Dict, Iterable, List, NoReturn, Union
import requests
from requests.auth import AuthBase
from kedro.io.core import AbstractDataset, DatasetError

class APIDataSet(AbstractDataset[None, requests.Response]):
    """``APIDataSet`` loads the data from HTTP(S) APIs.
    It uses the python requests library: https://requests.readthedocs.io/en/latest/

    Example usage for the
    `YAML API <https://kedro.readthedocs.io/en/stable/data/    data_catalog_yaml_examples.html>`_:


    .. code-block:: yaml

        usda:
          type: api.APIDataSet
          url: https://quickstats.nass.usda.gov
          params:
            key: SOME_TOKEN,
            format: JSON,
            commodity_desc: CORN,
            statisticcat_des: YIELD,
            agg_level_desc: STATE,
            year: 2000

    Example usage for the
    `Python API <https://kedro.readthedocs.io/en/stable/data/    advanced_data_catalog_usage.html>`_:
    ::

        >>> from kedro.extras.datasets.api import APIDataSet
        >>>
        >>>
        >>> data_set = APIDataSet(
        >>>     url="https://quickstats.nass.usda.gov",
        >>>     params={
        >>>         "key": "SOME_TOKEN",
        >>>         "format": "JSON",
        >>>         "commodity_desc": "CORN",
        >>>         "statisticcat_des": "YIELD",
        >>>         "agg_level_desc": "STATE",
        >>>         "year": 2000
        >>>     }
        >>> )
        >>> data = data_set.load()
    """

    def __init__(self, url: str, method: str='GET', data: Any=None, params: Dict[str, Any]=None, headers: Dict[str, Any]=None, auth: Union[Iterable[str], AuthBase]=None, json: Union[List, Dict[str, Any]]=None, timeout: int=60, credentials: Union[Iterable[str], AuthBase]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Creates a new instance of ``APIDataSet`` to fetch data from an API endpoint.\n\n        Args:\n            url: The API URL endpoint.\n            method: The Method of the request, GET, POST, PUT, DELETE, HEAD, etc...\n            data: The request payload, used for POST, PUT, etc requests\n                https://requests.readthedocs.io/en/latest/user/quickstart/#more-complicated-post-requests\n            params: The url parameters of the API.\n                https://requests.readthedocs.io/en/latest/user/quickstart/#passing-parameters-in-urls\n            headers: The HTTP headers.\n                https://requests.readthedocs.io/en/latest/user/quickstart/#custom-headers\n            auth: Anything ``requests`` accepts. Normally it's either ``('login', 'password')``,\n                or ``AuthBase``, ``HTTPBasicAuth`` instance for more complex cases. Any\n                iterable will be cast to a tuple.\n            json: The request payload, used for POST, PUT, etc requests, passed in\n                to the json kwarg in the requests object.\n                https://requests.readthedocs.io/en/latest/user/quickstart/#more-complicated-post-requests\n            timeout: The wait time in seconds for a response, defaults to 1 minute.\n                https://requests.readthedocs.io/en/latest/user/quickstart/#timeouts\n            credentials: same as ``auth``. Allows specifying ``auth`` secrets in\n                credentials.yml.\n\n        Raises:\n            ValueError: if both ``credentials`` and ``auth`` are specified.\n        "
        super().__init__()
        if credentials is not None and auth is not None:
            raise ValueError('Cannot specify both auth and credentials.')
        auth = credentials or auth
        if isinstance(auth, Iterable):
            auth = tuple(auth)
        self._request_args: Dict[str, Any] = {'url': url, 'method': method, 'data': data, 'params': params, 'headers': headers, 'auth': auth, 'json': json, 'timeout': timeout}

    def _describe(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        return {**self._request_args}

    def _execute_request(self) -> requests.Response:
        if False:
            for i in range(10):
                print('nop')
        try:
            response = requests.request(**self._request_args)
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            raise DatasetError('Failed to fetch data', exc) from exc
        except OSError as exc:
            raise DatasetError('Failed to connect to the remote server') from exc
        return response

    def _load(self) -> requests.Response:
        if False:
            i = 10
            return i + 15
        return self._execute_request()

    def _save(self, data: None) -> NoReturn:
        if False:
            return 10
        raise DatasetError(f'{self.__class__.__name__} is a read only data set type')

    def _exists(self) -> bool:
        if False:
            return 10
        response = self._execute_request()
        return response.ok