from io import IOBase
from typing import Any, Callable, Dict, IO, Iterable, List, Optional, TypeVar, Union, cast, overload
import urllib.parse
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ResourceExistsError, ResourceNotFoundError, ResourceNotModifiedError, map_error
from azure.core.paging import ItemPaged
from azure.core.pipeline import PipelineResponse
from azure.core.pipeline.transport import HttpResponse
from azure.core.polling import LROPoller, NoPolling, PollingMethod
from azure.core.polling.base_polling import LROBasePolling
from azure.core.rest import HttpRequest
from azure.core.tracing.decorator import distributed_trace
from azure.core.utils import case_insensitive_dict
from .. import models as _models
from .._serialization import Serializer
from .._vendor import AzureAppConfigurationMixinABC, _convert_request
T = TypeVar('T')
ClsType = Optional[Callable[[PipelineResponse[HttpRequest, HttpResponse], T, Dict[str, Any]], Any]]
_SERIALIZER = Serializer()
_SERIALIZER.client_side_validation = False

def build_get_keys_request(*, name: Optional[str]=None, after: Optional[str]=None, accept_datetime: Optional[str]=None, sync_token: Optional[str]=None, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-10-01'))
    accept = _headers.pop('Accept', 'application/vnd.microsoft.appconfig.keyset+json, application/problem+json')
    _url = kwargs.pop('template_url', '/keys')
    if name is not None:
        _params['name'] = _SERIALIZER.query('name', name, 'str')
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if after is not None:
        _params['After'] = _SERIALIZER.query('after', after, 'str')
    if sync_token is not None:
        _headers['Sync-Token'] = _SERIALIZER.header('sync_token', sync_token, 'str')
    if accept_datetime is not None:
        _headers['Accept-Datetime'] = _SERIALIZER.header('accept_datetime', accept_datetime, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_check_keys_request(*, name: Optional[str]=None, after: Optional[str]=None, accept_datetime: Optional[str]=None, sync_token: Optional[str]=None, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-10-01'))
    _url = kwargs.pop('template_url', '/keys')
    if name is not None:
        _params['name'] = _SERIALIZER.query('name', name, 'str')
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if after is not None:
        _params['After'] = _SERIALIZER.query('after', after, 'str')
    if sync_token is not None:
        _headers['Sync-Token'] = _SERIALIZER.header('sync_token', sync_token, 'str')
    if accept_datetime is not None:
        _headers['Accept-Datetime'] = _SERIALIZER.header('accept_datetime', accept_datetime, 'str')
    return HttpRequest(method='HEAD', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_key_values_request(*, key: Optional[str]=None, label: Optional[str]=None, after: Optional[str]=None, accept_datetime: Optional[str]=None, select: Optional[List[Union[str, _models.KeyValueFields]]]=None, snapshot: Optional[str]=None, if_match: Optional[str]=None, if_none_match: Optional[str]=None, sync_token: Optional[str]=None, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-10-01'))
    accept = _headers.pop('Accept', 'application/vnd.microsoft.appconfig.kvset+json, application/problem+json')
    _url = kwargs.pop('template_url', '/kv')
    if key is not None:
        _params['key'] = _SERIALIZER.query('key', key, 'str')
    if label is not None:
        _params['label'] = _SERIALIZER.query('label', label, 'str')
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if after is not None:
        _params['After'] = _SERIALIZER.query('after', after, 'str')
    if select is not None:
        _params['$Select'] = _SERIALIZER.query('select', select, '[str]', div=',')
    if snapshot is not None:
        _params['snapshot'] = _SERIALIZER.query('snapshot', snapshot, 'str')
    if sync_token is not None:
        _headers['Sync-Token'] = _SERIALIZER.header('sync_token', sync_token, 'str')
    if accept_datetime is not None:
        _headers['Accept-Datetime'] = _SERIALIZER.header('accept_datetime', accept_datetime, 'str')
    if if_match is not None:
        _headers['If-Match'] = _SERIALIZER.header('if_match', if_match, 'str')
    if if_none_match is not None:
        _headers['If-None-Match'] = _SERIALIZER.header('if_none_match', if_none_match, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_check_key_values_request(*, key: Optional[str]=None, label: Optional[str]=None, after: Optional[str]=None, accept_datetime: Optional[str]=None, select: Optional[List[Union[str, _models.KeyValueFields]]]=None, snapshot: Optional[str]=None, if_match: Optional[str]=None, if_none_match: Optional[str]=None, sync_token: Optional[str]=None, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-10-01'))
    _url = kwargs.pop('template_url', '/kv')
    if key is not None:
        _params['key'] = _SERIALIZER.query('key', key, 'str')
    if label is not None:
        _params['label'] = _SERIALIZER.query('label', label, 'str')
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if after is not None:
        _params['After'] = _SERIALIZER.query('after', after, 'str')
    if select is not None:
        _params['$Select'] = _SERIALIZER.query('select', select, '[str]', div=',')
    if snapshot is not None:
        _params['snapshot'] = _SERIALIZER.query('snapshot', snapshot, 'str')
    if sync_token is not None:
        _headers['Sync-Token'] = _SERIALIZER.header('sync_token', sync_token, 'str')
    if accept_datetime is not None:
        _headers['Accept-Datetime'] = _SERIALIZER.header('accept_datetime', accept_datetime, 'str')
    if if_match is not None:
        _headers['If-Match'] = _SERIALIZER.header('if_match', if_match, 'str')
    if if_none_match is not None:
        _headers['If-None-Match'] = _SERIALIZER.header('if_none_match', if_none_match, 'str')
    return HttpRequest(method='HEAD', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_key_value_request(key: str, *, label: Optional[str]=None, accept_datetime: Optional[str]=None, if_match: Optional[str]=None, if_none_match: Optional[str]=None, select: Optional[List[Union[str, _models.KeyValueFields]]]=None, sync_token: Optional[str]=None, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-10-01'))
    accept = _headers.pop('Accept', 'application/vnd.microsoft.appconfig.kv+json, application/problem+json')
    _url = kwargs.pop('template_url', '/kv/{key}')
    path_format_arguments = {'key': _SERIALIZER.url('key', key, 'str')}
    _url: str = _url.format(**path_format_arguments)
    if label is not None:
        _params['label'] = _SERIALIZER.query('label', label, 'str')
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if select is not None:
        _params['$Select'] = _SERIALIZER.query('select', select, '[str]', div=',')
    if sync_token is not None:
        _headers['Sync-Token'] = _SERIALIZER.header('sync_token', sync_token, 'str')
    if accept_datetime is not None:
        _headers['Accept-Datetime'] = _SERIALIZER.header('accept_datetime', accept_datetime, 'str')
    if if_match is not None:
        _headers['If-Match'] = _SERIALIZER.header('if_match', if_match, 'str')
    if if_none_match is not None:
        _headers['If-None-Match'] = _SERIALIZER.header('if_none_match', if_none_match, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_put_key_value_request(key: str, *, label: Optional[str]=None, if_match: Optional[str]=None, if_none_match: Optional[str]=None, sync_token: Optional[str]=None, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-10-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/vnd.microsoft.appconfig.kv+json, application/problem+json')
    _url = kwargs.pop('template_url', '/kv/{key}')
    path_format_arguments = {'key': _SERIALIZER.url('key', key, 'str')}
    _url: str = _url.format(**path_format_arguments)
    if label is not None:
        _params['label'] = _SERIALIZER.query('label', label, 'str')
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if sync_token is not None:
        _headers['Sync-Token'] = _SERIALIZER.header('sync_token', sync_token, 'str')
    if if_match is not None:
        _headers['If-Match'] = _SERIALIZER.header('if_match', if_match, 'str')
    if if_none_match is not None:
        _headers['If-None-Match'] = _SERIALIZER.header('if_none_match', if_none_match, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

def build_delete_key_value_request(key: str, *, label: Optional[str]=None, if_match: Optional[str]=None, sync_token: Optional[str]=None, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-10-01'))
    accept = _headers.pop('Accept', 'application/vnd.microsoft.appconfig.kv+json, application/problem+json')
    _url = kwargs.pop('template_url', '/kv/{key}')
    path_format_arguments = {'key': _SERIALIZER.url('key', key, 'str')}
    _url: str = _url.format(**path_format_arguments)
    if label is not None:
        _params['label'] = _SERIALIZER.query('label', label, 'str')
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if sync_token is not None:
        _headers['Sync-Token'] = _SERIALIZER.header('sync_token', sync_token, 'str')
    if if_match is not None:
        _headers['If-Match'] = _SERIALIZER.header('if_match', if_match, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='DELETE', url=_url, params=_params, headers=_headers, **kwargs)

def build_check_key_value_request(key: str, *, label: Optional[str]=None, accept_datetime: Optional[str]=None, if_match: Optional[str]=None, if_none_match: Optional[str]=None, select: Optional[List[Union[str, _models.KeyValueFields]]]=None, sync_token: Optional[str]=None, **kwargs: Any) -> HttpRequest:
    if False:
        return 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-10-01'))
    _url = kwargs.pop('template_url', '/kv/{key}')
    path_format_arguments = {'key': _SERIALIZER.url('key', key, 'str')}
    _url: str = _url.format(**path_format_arguments)
    if label is not None:
        _params['label'] = _SERIALIZER.query('label', label, 'str')
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if select is not None:
        _params['$Select'] = _SERIALIZER.query('select', select, '[str]', div=',')
    if sync_token is not None:
        _headers['Sync-Token'] = _SERIALIZER.header('sync_token', sync_token, 'str')
    if accept_datetime is not None:
        _headers['Accept-Datetime'] = _SERIALIZER.header('accept_datetime', accept_datetime, 'str')
    if if_match is not None:
        _headers['If-Match'] = _SERIALIZER.header('if_match', if_match, 'str')
    if if_none_match is not None:
        _headers['If-None-Match'] = _SERIALIZER.header('if_none_match', if_none_match, 'str')
    return HttpRequest(method='HEAD', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_snapshots_request(*, name: Optional[str]=None, after: Optional[str]=None, select: Optional[List[Union[str, _models.SnapshotFields]]]=None, status: Optional[List[Union[str, _models.SnapshotStatus]]]=None, sync_token: Optional[str]=None, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-10-01'))
    accept = _headers.pop('Accept', 'application/vnd.microsoft.appconfig.snapshotset+json, application/problem+json')
    _url = kwargs.pop('template_url', '/snapshots')
    if name is not None:
        _params['name'] = _SERIALIZER.query('name', name, 'str')
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if after is not None:
        _params['After'] = _SERIALIZER.query('after', after, 'str')
    if select is not None:
        _params['$Select'] = _SERIALIZER.query('select', select, '[str]', div=',')
    if status is not None:
        _params['status'] = _SERIALIZER.query('status', status, '[str]', div=',')
    if sync_token is not None:
        _headers['Sync-Token'] = _SERIALIZER.header('sync_token', sync_token, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_check_snapshots_request(*, after: Optional[str]=None, sync_token: Optional[str]=None, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-10-01'))
    _url = kwargs.pop('template_url', '/snapshots')
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if after is not None:
        _params['After'] = _SERIALIZER.query('after', after, 'str')
    if sync_token is not None:
        _headers['Sync-Token'] = _SERIALIZER.header('sync_token', sync_token, 'str')
    return HttpRequest(method='HEAD', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_snapshot_request(name: str, *, if_match: Optional[str]=None, if_none_match: Optional[str]=None, select: Optional[List[Union[str, _models.SnapshotFields]]]=None, sync_token: Optional[str]=None, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-10-01'))
    accept = _headers.pop('Accept', 'application/vnd.microsoft.appconfig.snapshot+json, application/problem+json')
    _url = kwargs.pop('template_url', '/snapshots/{name}')
    path_format_arguments = {'name': _SERIALIZER.url('name', name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if select is not None:
        _params['$Select'] = _SERIALIZER.query('select', select, '[str]', div=',')
    if sync_token is not None:
        _headers['Sync-Token'] = _SERIALIZER.header('sync_token', sync_token, 'str')
    if if_match is not None:
        _headers['If-Match'] = _SERIALIZER.header('if_match', if_match, 'str')
    if if_none_match is not None:
        _headers['If-None-Match'] = _SERIALIZER.header('if_none_match', if_none_match, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_create_snapshot_request(name: str, *, sync_token: Optional[str]=None, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-10-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/vnd.microsoft.appconfig.snapshot+json, application/problem+json')
    _url = kwargs.pop('template_url', '/snapshots/{name}')
    path_format_arguments = {'name': _SERIALIZER.url('name', name, 'str', max_length=256)}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if sync_token is not None:
        _headers['Sync-Token'] = _SERIALIZER.header('sync_token', sync_token, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

def build_update_snapshot_request(name: str, *, if_match: Optional[str]=None, if_none_match: Optional[str]=None, sync_token: Optional[str]=None, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-10-01'))
    content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
    accept = _headers.pop('Accept', 'application/vnd.microsoft.appconfig.snapshot+json, application/problem+json')
    _url = kwargs.pop('template_url', '/snapshots/{name}')
    path_format_arguments = {'name': _SERIALIZER.url('name', name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if sync_token is not None:
        _headers['Sync-Token'] = _SERIALIZER.header('sync_token', sync_token, 'str')
    if if_match is not None:
        _headers['If-Match'] = _SERIALIZER.header('if_match', if_match, 'str')
    if if_none_match is not None:
        _headers['If-None-Match'] = _SERIALIZER.header('if_none_match', if_none_match, 'str')
    if content_type is not None:
        _headers['Content-Type'] = _SERIALIZER.header('content_type', content_type, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PATCH', url=_url, params=_params, headers=_headers, **kwargs)

def build_check_snapshot_request(name: str, *, if_match: Optional[str]=None, if_none_match: Optional[str]=None, sync_token: Optional[str]=None, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-10-01'))
    _url = kwargs.pop('template_url', '/snapshots/{name}')
    path_format_arguments = {'name': _SERIALIZER.url('name', name, 'str')}
    _url: str = _url.format(**path_format_arguments)
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if sync_token is not None:
        _headers['Sync-Token'] = _SERIALIZER.header('sync_token', sync_token, 'str')
    if if_match is not None:
        _headers['If-Match'] = _SERIALIZER.header('if_match', if_match, 'str')
    if if_none_match is not None:
        _headers['If-None-Match'] = _SERIALIZER.header('if_none_match', if_none_match, 'str')
    return HttpRequest(method='HEAD', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_labels_request(*, name: Optional[str]=None, after: Optional[str]=None, accept_datetime: Optional[str]=None, select: Optional[List[Union[str, _models.LabelFields]]]=None, sync_token: Optional[str]=None, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-10-01'))
    accept = _headers.pop('Accept', 'application/vnd.microsoft.appconfig.labelset+json, application/problem+json')
    _url = kwargs.pop('template_url', '/labels')
    if name is not None:
        _params['name'] = _SERIALIZER.query('name', name, 'str')
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if after is not None:
        _params['After'] = _SERIALIZER.query('after', after, 'str')
    if select is not None:
        _params['$Select'] = _SERIALIZER.query('select', select, '[str]', div=',')
    if sync_token is not None:
        _headers['Sync-Token'] = _SERIALIZER.header('sync_token', sync_token, 'str')
    if accept_datetime is not None:
        _headers['Accept-Datetime'] = _SERIALIZER.header('accept_datetime', accept_datetime, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_check_labels_request(*, name: Optional[str]=None, after: Optional[str]=None, accept_datetime: Optional[str]=None, select: Optional[List[Union[str, _models.LabelFields]]]=None, sync_token: Optional[str]=None, **kwargs: Any) -> HttpRequest:
    if False:
        i = 10
        return i + 15
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-10-01'))
    _url = kwargs.pop('template_url', '/labels')
    if name is not None:
        _params['name'] = _SERIALIZER.query('name', name, 'str')
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if after is not None:
        _params['After'] = _SERIALIZER.query('after', after, 'str')
    if select is not None:
        _params['$Select'] = _SERIALIZER.query('select', select, '[str]', div=',')
    if sync_token is not None:
        _headers['Sync-Token'] = _SERIALIZER.header('sync_token', sync_token, 'str')
    if accept_datetime is not None:
        _headers['Accept-Datetime'] = _SERIALIZER.header('accept_datetime', accept_datetime, 'str')
    return HttpRequest(method='HEAD', url=_url, params=_params, headers=_headers, **kwargs)

def build_put_lock_request(key: str, *, label: Optional[str]=None, if_match: Optional[str]=None, if_none_match: Optional[str]=None, sync_token: Optional[str]=None, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-10-01'))
    accept = _headers.pop('Accept', 'application/vnd.microsoft.appconfig.kv+json, application/problem+json')
    _url = kwargs.pop('template_url', '/locks/{key}')
    path_format_arguments = {'key': _SERIALIZER.url('key', key, 'str')}
    _url: str = _url.format(**path_format_arguments)
    if label is not None:
        _params['label'] = _SERIALIZER.query('label', label, 'str')
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if sync_token is not None:
        _headers['Sync-Token'] = _SERIALIZER.header('sync_token', sync_token, 'str')
    if if_match is not None:
        _headers['If-Match'] = _SERIALIZER.header('if_match', if_match, 'str')
    if if_none_match is not None:
        _headers['If-None-Match'] = _SERIALIZER.header('if_none_match', if_none_match, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='PUT', url=_url, params=_params, headers=_headers, **kwargs)

def build_delete_lock_request(key: str, *, label: Optional[str]=None, if_match: Optional[str]=None, if_none_match: Optional[str]=None, sync_token: Optional[str]=None, **kwargs: Any) -> HttpRequest:
    if False:
        for i in range(10):
            print('nop')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-10-01'))
    accept = _headers.pop('Accept', 'application/vnd.microsoft.appconfig.kv+json, application/problem+json')
    _url = kwargs.pop('template_url', '/locks/{key}')
    path_format_arguments = {'key': _SERIALIZER.url('key', key, 'str')}
    _url: str = _url.format(**path_format_arguments)
    if label is not None:
        _params['label'] = _SERIALIZER.query('label', label, 'str')
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if sync_token is not None:
        _headers['Sync-Token'] = _SERIALIZER.header('sync_token', sync_token, 'str')
    if if_match is not None:
        _headers['If-Match'] = _SERIALIZER.header('if_match', if_match, 'str')
    if if_none_match is not None:
        _headers['If-None-Match'] = _SERIALIZER.header('if_none_match', if_none_match, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='DELETE', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_revisions_request(*, key: Optional[str]=None, label: Optional[str]=None, after: Optional[str]=None, accept_datetime: Optional[str]=None, select: Optional[List[Union[str, _models.KeyValueFields]]]=None, sync_token: Optional[str]=None, **kwargs: Any) -> HttpRequest:
    if False:
        while True:
            i = 10
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-10-01'))
    accept = _headers.pop('Accept', 'application/vnd.microsoft.appconfig.kvset+json, application/problem+json')
    _url = kwargs.pop('template_url', '/revisions')
    if key is not None:
        _params['key'] = _SERIALIZER.query('key', key, 'str')
    if label is not None:
        _params['label'] = _SERIALIZER.query('label', label, 'str')
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if after is not None:
        _params['After'] = _SERIALIZER.query('after', after, 'str')
    if select is not None:
        _params['$Select'] = _SERIALIZER.query('select', select, '[str]', div=',')
    if sync_token is not None:
        _headers['Sync-Token'] = _SERIALIZER.header('sync_token', sync_token, 'str')
    if accept_datetime is not None:
        _headers['Accept-Datetime'] = _SERIALIZER.header('accept_datetime', accept_datetime, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

def build_check_revisions_request(*, key: Optional[str]=None, label: Optional[str]=None, after: Optional[str]=None, accept_datetime: Optional[str]=None, select: Optional[List[Union[str, _models.KeyValueFields]]]=None, sync_token: Optional[str]=None, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-10-01'))
    _url = kwargs.pop('template_url', '/revisions')
    if key is not None:
        _params['key'] = _SERIALIZER.query('key', key, 'str')
    if label is not None:
        _params['label'] = _SERIALIZER.query('label', label, 'str')
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    if after is not None:
        _params['After'] = _SERIALIZER.query('after', after, 'str')
    if select is not None:
        _params['$Select'] = _SERIALIZER.query('select', select, '[str]', div=',')
    if sync_token is not None:
        _headers['Sync-Token'] = _SERIALIZER.header('sync_token', sync_token, 'str')
    if accept_datetime is not None:
        _headers['Accept-Datetime'] = _SERIALIZER.header('accept_datetime', accept_datetime, 'str')
    return HttpRequest(method='HEAD', url=_url, params=_params, headers=_headers, **kwargs)

def build_get_operation_details_request(*, snapshot: str, **kwargs: Any) -> HttpRequest:
    if False:
        print('Hello World!')
    _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
    _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
    api_version: str = kwargs.pop('api_version', _params.pop('api-version', '2023-10-01'))
    accept = _headers.pop('Accept', 'application/json')
    _url = kwargs.pop('template_url', '/operations')
    _params['api-version'] = _SERIALIZER.query('api_version', api_version, 'str')
    _params['snapshot'] = _SERIALIZER.query('snapshot', snapshot, 'str')
    _headers['Accept'] = _SERIALIZER.header('accept', accept, 'str')
    return HttpRequest(method='GET', url=_url, params=_params, headers=_headers, **kwargs)

class AzureAppConfigurationOperationsMixin(AzureAppConfigurationMixinABC):

    @distributed_trace
    def get_keys(self, name: Optional[str]=None, after: Optional[str]=None, accept_datetime: Optional[str]=None, **kwargs: Any) -> Iterable['_models.Key']:
        if False:
            i = 10
            return i + 15
        'Gets a list of keys.\n\n        Gets a list of keys.\n\n        :param name: A filter for the name of the returned keys. Default value is None.\n        :type name: str\n        :param after: Instructs the server to return elements that appear after the element referred to\n         by the specified token. Default value is None.\n        :type after: str\n        :param accept_datetime: Requests the server to respond with the state of the resource at the\n         specified time. Default value is None.\n        :type accept_datetime: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either Key or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.appconfiguration.models.Key]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.KeyListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                return 10
            if not next_link:
                _request = build_get_keys_request(name=name, after=after, accept_datetime=accept_datetime, sync_token=self._config.sync_token, api_version=api_version, headers=_headers, params=_params)
                _request = _convert_request(_request)
                path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
                _request.url = self._client.format_url(_request.url, **path_format_arguments)
            else:
                _parsed_next_link = urllib.parse.urlparse(next_link)
                _next_request_params = case_insensitive_dict({key: [urllib.parse.quote(v) for v in value] for (key, value) in urllib.parse.parse_qs(_parsed_next_link.query).items()})
                _next_request_params['api-version'] = self._config.api_version
                _request = HttpRequest('GET', urllib.parse.urljoin(next_link, _parsed_next_link.path), params=_next_request_params)
                _request = _convert_request(_request)
                path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
                _request.url = self._client.format_url(_request.url, **path_format_arguments)
                _request.method = 'GET'
            return _request

        def extract_data(pipeline_response):
            if False:
                while True:
                    i = 10
            deserialized = self._deserialize('KeyListResult', pipeline_response)
            list_of_elem = deserialized.items
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                while True:
                    i = 10
            _request = prepare_request(next_link)
            _stream = False
            pipeline_response: PipelineResponse = self._client._pipeline.run(_request, stream=_stream, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.Error, pipeline_response)
                raise HttpResponseError(response=response, model=error)
            return pipeline_response
        return ItemPaged(get_next, extract_data)

    @distributed_trace
    def check_keys(self, name: Optional[str]=None, after: Optional[str]=None, accept_datetime: Optional[str]=None, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Requests the headers and status of the given resource.\n\n        Requests the headers and status of the given resource.\n\n        :param name: A filter for the name of the returned keys. Default value is None.\n        :type name: str\n        :param after: Instructs the server to return elements that appear after the element referred to\n         by the specified token. Default value is None.\n        :type after: str\n        :param accept_datetime: Requests the server to respond with the state of the resource at the\n         specified time. Default value is None.\n        :type accept_datetime: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        _request = build_check_keys_request(name=name, after=after, accept_datetime=accept_datetime, sync_token=self._config.sync_token, api_version=api_version, headers=_headers, params=_params)
        _request = _convert_request(_request)
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
        _request.url = self._client.format_url(_request.url, **path_format_arguments)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(_request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response)
        response_headers = {}
        response_headers['Sync-Token'] = self._deserialize('str', response.headers.get('Sync-Token'))
        if cls:
            return cls(pipeline_response, None, response_headers)

    @distributed_trace
    def get_key_values(self, key: Optional[str]=None, label: Optional[str]=None, after: Optional[str]=None, accept_datetime: Optional[str]=None, select: Optional[List[Union[str, _models.KeyValueFields]]]=None, snapshot: Optional[str]=None, if_match: Optional[str]=None, if_none_match: Optional[str]=None, **kwargs: Any) -> Iterable['_models.KeyValue']:
        if False:
            for i in range(10):
                print('nop')
        "Gets a list of key-values.\n\n        Gets a list of key-values.\n\n        :param key: A filter used to match keys. Default value is None.\n        :type key: str\n        :param label: A filter used to match labels. Default value is None.\n        :type label: str\n        :param after: Instructs the server to return elements that appear after the element referred to\n         by the specified token. Default value is None.\n        :type after: str\n        :param accept_datetime: Requests the server to respond with the state of the resource at the\n         specified time. Default value is None.\n        :type accept_datetime: str\n        :param select: Used to select what fields are present in the returned resource(s). Default\n         value is None.\n        :type select: list[str or ~azure.appconfiguration.models.KeyValueFields]\n        :param snapshot: A filter used get key-values for a snapshot. The value should be the name of\n         the snapshot. Not valid when used with 'key' and 'label' filters. Default value is None.\n        :type snapshot: str\n        :param if_match: Used to perform an operation only if the targeted resource's etag matches the\n         value provided. Default value is None.\n        :type if_match: str\n        :param if_none_match: Used to perform an operation only if the targeted resource's etag does\n         not match the value provided. Default value is None.\n        :type if_none_match: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either KeyValue or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.appconfiguration.models.KeyValue]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.KeyValueListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                i = 10
                return i + 15
            if not next_link:
                _request = build_get_key_values_request(key=key, label=label, after=after, accept_datetime=accept_datetime, select=select, snapshot=snapshot, if_match=if_match, if_none_match=if_none_match, sync_token=self._config.sync_token, api_version=api_version, headers=_headers, params=_params)
                _request = _convert_request(_request)
                path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
                _request.url = self._client.format_url(_request.url, **path_format_arguments)
            else:
                _parsed_next_link = urllib.parse.urlparse(next_link)
                _next_request_params = case_insensitive_dict({key: [urllib.parse.quote(v) for v in value] for (key, value) in urllib.parse.parse_qs(_parsed_next_link.query).items()})
                _next_request_params['api-version'] = self._config.api_version
                _request = HttpRequest('GET', urllib.parse.urljoin(next_link, _parsed_next_link.path), params=_next_request_params)
                _request = _convert_request(_request)
                path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
                _request.url = self._client.format_url(_request.url, **path_format_arguments)
                _request.method = 'GET'
            return _request

        def extract_data(pipeline_response):
            if False:
                print('Hello World!')
            deserialized = self._deserialize('KeyValueListResult', pipeline_response)
            list_of_elem = deserialized.items
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                i = 10
                return i + 15
            _request = prepare_request(next_link)
            _stream = False
            pipeline_response: PipelineResponse = self._client._pipeline.run(_request, stream=_stream, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.Error, pipeline_response)
                raise HttpResponseError(response=response, model=error)
            return pipeline_response
        return ItemPaged(get_next, extract_data)

    @distributed_trace
    def check_key_values(self, key: Optional[str]=None, label: Optional[str]=None, after: Optional[str]=None, accept_datetime: Optional[str]=None, select: Optional[List[Union[str, _models.KeyValueFields]]]=None, snapshot: Optional[str]=None, if_match: Optional[str]=None, if_none_match: Optional[str]=None, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Requests the headers and status of the given resource.\n\n        Requests the headers and status of the given resource.\n\n        :param key: A filter used to match keys. Default value is None.\n        :type key: str\n        :param label: A filter used to match labels. Default value is None.\n        :type label: str\n        :param after: Instructs the server to return elements that appear after the element referred to\n         by the specified token. Default value is None.\n        :type after: str\n        :param accept_datetime: Requests the server to respond with the state of the resource at the\n         specified time. Default value is None.\n        :type accept_datetime: str\n        :param select: Used to select what fields are present in the returned resource(s). Default\n         value is None.\n        :type select: list[str or ~azure.appconfiguration.models.KeyValueFields]\n        :param snapshot: A filter used get key-values for a snapshot. Not valid when used with 'key'\n         and 'label' filters. Default value is None.\n        :type snapshot: str\n        :param if_match: Used to perform an operation only if the targeted resource's etag matches the\n         value provided. Default value is None.\n        :type if_match: str\n        :param if_none_match: Used to perform an operation only if the targeted resource's etag does\n         not match the value provided. Default value is None.\n        :type if_none_match: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        _request = build_check_key_values_request(key=key, label=label, after=after, accept_datetime=accept_datetime, select=select, snapshot=snapshot, if_match=if_match, if_none_match=if_none_match, sync_token=self._config.sync_token, api_version=api_version, headers=_headers, params=_params)
        _request = _convert_request(_request)
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
        _request.url = self._client.format_url(_request.url, **path_format_arguments)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(_request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response)
        response_headers = {}
        response_headers['Sync-Token'] = self._deserialize('str', response.headers.get('Sync-Token'))
        response_headers['ETag'] = self._deserialize('str', response.headers.get('ETag'))
        if cls:
            return cls(pipeline_response, None, response_headers)

    @distributed_trace
    def get_key_value(self, key: str, label: Optional[str]=None, accept_datetime: Optional[str]=None, if_match: Optional[str]=None, if_none_match: Optional[str]=None, select: Optional[List[Union[str, _models.KeyValueFields]]]=None, **kwargs: Any) -> _models.KeyValue:
        if False:
            return 10
        "Gets a single key-value.\n\n        Gets a single key-value.\n\n        :param key: The key of the key-value to retrieve. Required.\n        :type key: str\n        :param label: The label of the key-value to retrieve. Default value is None.\n        :type label: str\n        :param accept_datetime: Requests the server to respond with the state of the resource at the\n         specified time. Default value is None.\n        :type accept_datetime: str\n        :param if_match: Used to perform an operation only if the targeted resource's etag matches the\n         value provided. Default value is None.\n        :type if_match: str\n        :param if_none_match: Used to perform an operation only if the targeted resource's etag does\n         not match the value provided. Default value is None.\n        :type if_none_match: str\n        :param select: Used to select what fields are present in the returned resource(s). Default\n         value is None.\n        :type select: list[str or ~azure.appconfiguration.models.KeyValueFields]\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: KeyValue or the result of cls(response)\n        :rtype: ~azure.appconfiguration.models.KeyValue\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.KeyValue] = kwargs.pop('cls', None)
        _request = build_get_key_value_request(key=key, label=label, accept_datetime=accept_datetime, if_match=if_match, if_none_match=if_none_match, select=select, sync_token=self._config.sync_token, api_version=api_version, headers=_headers, params=_params)
        _request = _convert_request(_request)
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
        _request.url = self._client.format_url(_request.url, **path_format_arguments)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(_request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.Error, pipeline_response)
            raise HttpResponseError(response=response, model=error)
        response_headers = {}
        response_headers['Sync-Token'] = self._deserialize('str', response.headers.get('Sync-Token'))
        response_headers['ETag'] = self._deserialize('str', response.headers.get('ETag'))
        deserialized = self._deserialize('KeyValue', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized

    @overload
    def put_key_value(self, key: str, label: Optional[str]=None, if_match: Optional[str]=None, if_none_match: Optional[str]=None, entity: Optional[_models.KeyValue]=None, *, content_type: str='application/json', **kwargs: Any) -> _models.KeyValue:
        if False:
            i = 10
            return i + 15
        'Creates a key-value.\n\n        Creates a key-value.\n\n        :param key: The key of the key-value to create. Required.\n        :type key: str\n        :param label: The label of the key-value to create. Default value is None.\n        :type label: str\n        :param if_match: Used to perform an operation only if the targeted resource\'s etag matches the\n         value provided. Default value is None.\n        :type if_match: str\n        :param if_none_match: Used to perform an operation only if the targeted resource\'s etag does\n         not match the value provided. Default value is None.\n        :type if_none_match: str\n        :param entity: The key-value to create. Default value is None.\n        :type entity: ~azure.appconfiguration.models.KeyValue\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: KeyValue or the result of cls(response)\n        :rtype: ~azure.appconfiguration.models.KeyValue\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def put_key_value(self, key: str, label: Optional[str]=None, if_match: Optional[str]=None, if_none_match: Optional[str]=None, entity: Optional[IO]=None, *, content_type: str='application/json', **kwargs: Any) -> _models.KeyValue:
        if False:
            print('Hello World!')
        'Creates a key-value.\n\n        Creates a key-value.\n\n        :param key: The key of the key-value to create. Required.\n        :type key: str\n        :param label: The label of the key-value to create. Default value is None.\n        :type label: str\n        :param if_match: Used to perform an operation only if the targeted resource\'s etag matches the\n         value provided. Default value is None.\n        :type if_match: str\n        :param if_none_match: Used to perform an operation only if the targeted resource\'s etag does\n         not match the value provided. Default value is None.\n        :type if_none_match: str\n        :param entity: The key-value to create. Default value is None.\n        :type entity: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Known values are: \'application/*+json\', \'application/json\', \'application/json-patch+json\',\n         \'application/vnd.microsoft.appconfig.kv+json\',\n         \'application/vnd.microsoft.appconfig.kvset+json\', \'text/json\'. Default value is\n         "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: KeyValue or the result of cls(response)\n        :rtype: ~azure.appconfiguration.models.KeyValue\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def put_key_value(self, key: str, label: Optional[str]=None, if_match: Optional[str]=None, if_none_match: Optional[str]=None, entity: Optional[Union[_models.KeyValue, IO]]=None, **kwargs: Any) -> _models.KeyValue:
        if False:
            while True:
                i = 10
        "Creates a key-value.\n\n        Creates a key-value.\n\n        :param key: The key of the key-value to create. Required.\n        :type key: str\n        :param label: The label of the key-value to create. Default value is None.\n        :type label: str\n        :param if_match: Used to perform an operation only if the targeted resource's etag matches the\n         value provided. Default value is None.\n        :type if_match: str\n        :param if_none_match: Used to perform an operation only if the targeted resource's etag does\n         not match the value provided. Default value is None.\n        :type if_none_match: str\n        :param entity: The key-value to create. Is either a KeyValue type or a IO type. Default value\n         is None.\n        :type entity: ~azure.appconfiguration.models.KeyValue or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/*+json',\n         'application/json', 'application/json-patch+json',\n         'application/vnd.microsoft.appconfig.kv+json',\n         'application/vnd.microsoft.appconfig.kvset+json', 'text/json'. Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: KeyValue or the result of cls(response)\n        :rtype: ~azure.appconfiguration.models.KeyValue\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.KeyValue] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(entity, (IOBase, bytes)):
            _content = entity
        elif entity is not None:
            _json = self._serialize.body(entity, 'KeyValue')
        else:
            _json = None
        _request = build_put_key_value_request(key=key, label=label, if_match=if_match, if_none_match=if_none_match, sync_token=self._config.sync_token, api_version=api_version, content_type=content_type, json=_json, content=_content, headers=_headers, params=_params)
        _request = _convert_request(_request)
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
        _request.url = self._client.format_url(_request.url, **path_format_arguments)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(_request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.Error, pipeline_response)
            raise HttpResponseError(response=response, model=error)
        response_headers = {}
        response_headers['Sync-Token'] = self._deserialize('str', response.headers.get('Sync-Token'))
        response_headers['ETag'] = self._deserialize('str', response.headers.get('ETag'))
        deserialized = self._deserialize('KeyValue', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized

    @distributed_trace
    def delete_key_value(self, key: str, label: Optional[str]=None, if_match: Optional[str]=None, **kwargs: Any) -> Optional[_models.KeyValue]:
        if False:
            i = 10
            return i + 15
        "Deletes a key-value.\n\n        Deletes a key-value.\n\n        :param key: The key of the key-value to delete. Required.\n        :type key: str\n        :param label: The label of the key-value to delete. Default value is None.\n        :type label: str\n        :param if_match: Used to perform an operation only if the targeted resource's etag matches the\n         value provided. Default value is None.\n        :type if_match: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: KeyValue or None or the result of cls(response)\n        :rtype: ~azure.appconfiguration.models.KeyValue or None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[Optional[_models.KeyValue]] = kwargs.pop('cls', None)
        _request = build_delete_key_value_request(key=key, label=label, if_match=if_match, sync_token=self._config.sync_token, api_version=api_version, headers=_headers, params=_params)
        _request = _convert_request(_request)
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
        _request.url = self._client.format_url(_request.url, **path_format_arguments)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(_request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200, 204]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.Error, pipeline_response)
            raise HttpResponseError(response=response, model=error)
        deserialized = None
        response_headers = {}
        if response.status_code == 200:
            response_headers['Sync-Token'] = self._deserialize('str', response.headers.get('Sync-Token'))
            response_headers['ETag'] = self._deserialize('str', response.headers.get('ETag'))
            deserialized = self._deserialize('KeyValue', pipeline_response)
        if response.status_code == 204:
            response_headers['Sync-Token'] = self._deserialize('str', response.headers.get('Sync-Token'))
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized

    @distributed_trace
    def check_key_value(self, key: str, label: Optional[str]=None, accept_datetime: Optional[str]=None, if_match: Optional[str]=None, if_none_match: Optional[str]=None, select: Optional[List[Union[str, _models.KeyValueFields]]]=None, **kwargs: Any) -> None:
        if False:
            return 10
        "Requests the headers and status of the given resource.\n\n        Requests the headers and status of the given resource.\n\n        :param key: The key of the key-value to retrieve. Required.\n        :type key: str\n        :param label: The label of the key-value to retrieve. Default value is None.\n        :type label: str\n        :param accept_datetime: Requests the server to respond with the state of the resource at the\n         specified time. Default value is None.\n        :type accept_datetime: str\n        :param if_match: Used to perform an operation only if the targeted resource's etag matches the\n         value provided. Default value is None.\n        :type if_match: str\n        :param if_none_match: Used to perform an operation only if the targeted resource's etag does\n         not match the value provided. Default value is None.\n        :type if_none_match: str\n        :param select: Used to select what fields are present in the returned resource(s). Default\n         value is None.\n        :type select: list[str or ~azure.appconfiguration.models.KeyValueFields]\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        _request = build_check_key_value_request(key=key, label=label, accept_datetime=accept_datetime, if_match=if_match, if_none_match=if_none_match, select=select, sync_token=self._config.sync_token, api_version=api_version, headers=_headers, params=_params)
        _request = _convert_request(_request)
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
        _request.url = self._client.format_url(_request.url, **path_format_arguments)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(_request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response)
        response_headers = {}
        response_headers['Sync-Token'] = self._deserialize('str', response.headers.get('Sync-Token'))
        response_headers['ETag'] = self._deserialize('str', response.headers.get('ETag'))
        if cls:
            return cls(pipeline_response, None, response_headers)

    @distributed_trace
    def get_snapshots(self, name: Optional[str]=None, after: Optional[str]=None, select: Optional[List[Union[str, _models.SnapshotFields]]]=None, status: Optional[List[Union[str, _models.SnapshotStatus]]]=None, **kwargs: Any) -> Iterable['_models.Snapshot']:
        if False:
            while True:
                i = 10
        'Gets a list of key-value snapshots.\n\n        Gets a list of key-value snapshots.\n\n        :param name: A filter for the name of the returned snapshots. Default value is None.\n        :type name: str\n        :param after: Instructs the server to return elements that appear after the element referred to\n         by the specified token. Default value is None.\n        :type after: str\n        :param select: Used to select what fields are present in the returned resource(s). Default\n         value is None.\n        :type select: list[str or ~azure.appconfiguration.models.SnapshotFields]\n        :param status: Used to filter returned snapshots by their status property. Default value is\n         None.\n        :type status: list[str or ~azure.appconfiguration.models.SnapshotStatus]\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either Snapshot or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.appconfiguration.models.Snapshot]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.SnapshotListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                return 10
            if not next_link:
                _request = build_get_snapshots_request(name=name, after=after, select=select, status=status, sync_token=self._config.sync_token, api_version=api_version, headers=_headers, params=_params)
                _request = _convert_request(_request)
                path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
                _request.url = self._client.format_url(_request.url, **path_format_arguments)
            else:
                _parsed_next_link = urllib.parse.urlparse(next_link)
                _next_request_params = case_insensitive_dict({key: [urllib.parse.quote(v) for v in value] for (key, value) in urllib.parse.parse_qs(_parsed_next_link.query).items()})
                _next_request_params['api-version'] = self._config.api_version
                _request = HttpRequest('GET', urllib.parse.urljoin(next_link, _parsed_next_link.path), params=_next_request_params)
                _request = _convert_request(_request)
                path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
                _request.url = self._client.format_url(_request.url, **path_format_arguments)
                _request.method = 'GET'
            return _request

        def extract_data(pipeline_response):
            if False:
                i = 10
                return i + 15
            deserialized = self._deserialize('SnapshotListResult', pipeline_response)
            list_of_elem = deserialized.items
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                while True:
                    i = 10
            _request = prepare_request(next_link)
            _stream = False
            pipeline_response: PipelineResponse = self._client._pipeline.run(_request, stream=_stream, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.Error, pipeline_response)
                raise HttpResponseError(response=response, model=error)
            return pipeline_response
        return ItemPaged(get_next, extract_data)

    @distributed_trace
    def check_snapshots(self, after: Optional[str]=None, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        'Requests the headers and status of the given resource.\n\n        Requests the headers and status of the given resource.\n\n        :param after: Instructs the server to return elements that appear after the element referred to\n         by the specified token. Default value is None.\n        :type after: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        _request = build_check_snapshots_request(after=after, sync_token=self._config.sync_token, api_version=api_version, headers=_headers, params=_params)
        _request = _convert_request(_request)
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
        _request.url = self._client.format_url(_request.url, **path_format_arguments)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(_request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response)
        response_headers = {}
        response_headers['Sync-Token'] = self._deserialize('str', response.headers.get('Sync-Token'))
        if cls:
            return cls(pipeline_response, None, response_headers)

    @distributed_trace
    def get_snapshot(self, name: str, if_match: Optional[str]=None, if_none_match: Optional[str]=None, select: Optional[List[Union[str, _models.SnapshotFields]]]=None, **kwargs: Any) -> _models.Snapshot:
        if False:
            while True:
                i = 10
        "Gets a single key-value snapshot.\n\n        Gets a single key-value snapshot.\n\n        :param name: The name of the key-value snapshot to retrieve. Required.\n        :type name: str\n        :param if_match: Used to perform an operation only if the targeted resource's etag matches the\n         value provided. Default value is None.\n        :type if_match: str\n        :param if_none_match: Used to perform an operation only if the targeted resource's etag does\n         not match the value provided. Default value is None.\n        :type if_none_match: str\n        :param select: Used to select what fields are present in the returned resource(s). Default\n         value is None.\n        :type select: list[str or ~azure.appconfiguration.models.SnapshotFields]\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Snapshot or the result of cls(response)\n        :rtype: ~azure.appconfiguration.models.Snapshot\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.Snapshot] = kwargs.pop('cls', None)
        _request = build_get_snapshot_request(name=name, if_match=if_match, if_none_match=if_none_match, select=select, sync_token=self._config.sync_token, api_version=api_version, headers=_headers, params=_params)
        _request = _convert_request(_request)
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
        _request.url = self._client.format_url(_request.url, **path_format_arguments)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(_request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.Error, pipeline_response)
            raise HttpResponseError(response=response, model=error)
        response_headers = {}
        response_headers['Sync-Token'] = self._deserialize('str', response.headers.get('Sync-Token'))
        response_headers['ETag'] = self._deserialize('str', response.headers.get('ETag'))
        response_headers['Link'] = self._deserialize('str', response.headers.get('Link'))
        deserialized = self._deserialize('Snapshot', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized

    def _create_snapshot_initial(self, name: str, entity: Union[_models.Snapshot, IO], **kwargs: Any) -> _models.Snapshot:
        if False:
            return 10
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.Snapshot] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(entity, (IOBase, bytes)):
            _content = entity
        else:
            _json = self._serialize.body(entity, 'Snapshot')
        _request = build_create_snapshot_request(name=name, sync_token=self._config.sync_token, api_version=api_version, content_type=content_type, json=_json, content=_content, headers=_headers, params=_params)
        _request = _convert_request(_request)
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
        _request.url = self._client.format_url(_request.url, **path_format_arguments)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(_request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [201]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.Error, pipeline_response)
            raise HttpResponseError(response=response, model=error)
        response_headers = {}
        response_headers['Sync-Token'] = self._deserialize('str', response.headers.get('Sync-Token'))
        response_headers['ETag'] = self._deserialize('str', response.headers.get('ETag'))
        response_headers['Link'] = self._deserialize('str', response.headers.get('Link'))
        response_headers['Operation-Location'] = self._deserialize('str', response.headers.get('Operation-Location'))
        deserialized = self._deserialize('Snapshot', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized

    @overload
    def begin_create_snapshot(self, name: str, entity: _models.Snapshot, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.Snapshot]:
        if False:
            print('Hello World!')
        'Creates a key-value snapshot.\n\n        Creates a key-value snapshot.\n\n        :param name: The name of the key-value snapshot to create. Required.\n        :type name: str\n        :param entity: The key-value snapshot to create. Required.\n        :type entity: ~azure.appconfiguration.models.Snapshot\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be LROBasePolling. Pass in False for\n         this operation to not poll, or pass in your own initialized polling object for a personal\n         polling strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either Snapshot or the result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.appconfiguration.models.Snapshot]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def begin_create_snapshot(self, name: str, entity: IO, *, content_type: str='application/json', **kwargs: Any) -> LROPoller[_models.Snapshot]:
        if False:
            i = 10
            return i + 15
        'Creates a key-value snapshot.\n\n        Creates a key-value snapshot.\n\n        :param name: The name of the key-value snapshot to create. Required.\n        :type name: str\n        :param entity: The key-value snapshot to create. Required.\n        :type entity: IO\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Known values are: \'application/json\', \'application/vnd.microsoft.appconfig.snapshot+json\'.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be LROBasePolling. Pass in False for\n         this operation to not poll, or pass in your own initialized polling object for a personal\n         polling strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either Snapshot or the result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.appconfiguration.models.Snapshot]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def begin_create_snapshot(self, name: str, entity: Union[_models.Snapshot, IO], **kwargs: Any) -> LROPoller[_models.Snapshot]:
        if False:
            while True:
                i = 10
        "Creates a key-value snapshot.\n\n        Creates a key-value snapshot.\n\n        :param name: The name of the key-value snapshot to create. Required.\n        :type name: str\n        :param entity: The key-value snapshot to create. Is either a Snapshot type or a IO type.\n         Required.\n        :type entity: ~azure.appconfiguration.models.Snapshot or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json',\n         'application/vnd.microsoft.appconfig.snapshot+json'. Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: By default, your polling method will be LROBasePolling. Pass in False for\n         this operation to not poll, or pass in your own initialized polling object for a personal\n         polling strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time between two polls for LRO operations if no\n         Retry-After header is present.\n        :return: An instance of LROPoller that returns either Snapshot or the result of cls(response)\n        :rtype: ~azure.core.polling.LROPoller[~azure.appconfiguration.models.Snapshot]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.Snapshot] = kwargs.pop('cls', None)
        polling: Union[bool, PollingMethod] = kwargs.pop('polling', True)
        lro_delay = kwargs.pop('polling_interval', self._config.polling_interval)
        cont_token: Optional[str] = kwargs.pop('continuation_token', None)
        if cont_token is None:
            raw_result = self._create_snapshot_initial(name=name, entity=entity, api_version=api_version, content_type=content_type, cls=lambda x, y, z: x, headers=_headers, params=_params, **kwargs)
        kwargs.pop('error_map', None)

        def get_long_running_output(pipeline_response):
            if False:
                for i in range(10):
                    print('nop')
            response_headers = {}
            response = pipeline_response.http_response
            response_headers['Sync-Token'] = self._deserialize('str', response.headers.get('Sync-Token'))
            response_headers['ETag'] = self._deserialize('str', response.headers.get('ETag'))
            response_headers['Link'] = self._deserialize('str', response.headers.get('Link'))
            response_headers['Operation-Location'] = self._deserialize('str', response.headers.get('Operation-Location'))
            deserialized = self._deserialize('Snapshot', pipeline_response)
            if cls:
                return cls(pipeline_response, deserialized, response_headers)
            return deserialized
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
        if polling is True:
            polling_method: PollingMethod = cast(PollingMethod, LROBasePolling(lro_delay, path_format_arguments=path_format_arguments, **kwargs))
        elif polling is False:
            polling_method = cast(PollingMethod, NoPolling())
        else:
            polling_method = polling
        if cont_token:
            return LROPoller[_models.Snapshot].from_continuation_token(polling_method=polling_method, continuation_token=cont_token, client=self._client, deserialization_callback=get_long_running_output)
        return LROPoller[_models.Snapshot](self._client, raw_result, get_long_running_output, polling_method)

    @overload
    def update_snapshot(self, name: str, entity: _models.SnapshotUpdateParameters, if_match: Optional[str]=None, if_none_match: Optional[str]=None, *, content_type: str='application/json', **kwargs: Any) -> _models.Snapshot:
        if False:
            i = 10
            return i + 15
        'Updates the state of a key-value snapshot.\n\n        Updates the state of a key-value snapshot.\n\n        :param name: The name of the key-value snapshot to update. Required.\n        :type name: str\n        :param entity: The parameters used to update the snapshot. Required.\n        :type entity: ~azure.appconfiguration.models.SnapshotUpdateParameters\n        :param if_match: Used to perform an operation only if the targeted resource\'s etag matches the\n         value provided. Default value is None.\n        :type if_match: str\n        :param if_none_match: Used to perform an operation only if the targeted resource\'s etag does\n         not match the value provided. Default value is None.\n        :type if_none_match: str\n        :keyword content_type: Body Parameter content-type. Content type parameter for JSON body.\n         Default value is "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Snapshot or the result of cls(response)\n        :rtype: ~azure.appconfiguration.models.Snapshot\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @overload
    def update_snapshot(self, name: str, entity: IO, if_match: Optional[str]=None, if_none_match: Optional[str]=None, *, content_type: str='application/json', **kwargs: Any) -> _models.Snapshot:
        if False:
            i = 10
            return i + 15
        'Updates the state of a key-value snapshot.\n\n        Updates the state of a key-value snapshot.\n\n        :param name: The name of the key-value snapshot to update. Required.\n        :type name: str\n        :param entity: The parameters used to update the snapshot. Required.\n        :type entity: IO\n        :param if_match: Used to perform an operation only if the targeted resource\'s etag matches the\n         value provided. Default value is None.\n        :type if_match: str\n        :param if_none_match: Used to perform an operation only if the targeted resource\'s etag does\n         not match the value provided. Default value is None.\n        :type if_none_match: str\n        :keyword content_type: Body Parameter content-type. Content type parameter for binary body.\n         Known values are: \'application/json\', \'application/merge-patch+json\'. Default value is\n         "application/json".\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Snapshot or the result of cls(response)\n        :rtype: ~azure.appconfiguration.models.Snapshot\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '

    @distributed_trace
    def update_snapshot(self, name: str, entity: Union[_models.SnapshotUpdateParameters, IO], if_match: Optional[str]=None, if_none_match: Optional[str]=None, **kwargs: Any) -> _models.Snapshot:
        if False:
            return 10
        "Updates the state of a key-value snapshot.\n\n        Updates the state of a key-value snapshot.\n\n        :param name: The name of the key-value snapshot to update. Required.\n        :type name: str\n        :param entity: The parameters used to update the snapshot. Is either a SnapshotUpdateParameters\n         type or a IO type. Required.\n        :type entity: ~azure.appconfiguration.models.SnapshotUpdateParameters or IO\n        :param if_match: Used to perform an operation only if the targeted resource's etag matches the\n         value provided. Default value is None.\n        :type if_match: str\n        :param if_none_match: Used to perform an operation only if the targeted resource's etag does\n         not match the value provided. Default value is None.\n        :type if_none_match: str\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json',\n         'application/merge-patch+json'. Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: Snapshot or the result of cls(response)\n        :rtype: ~azure.appconfiguration.models.Snapshot\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = case_insensitive_dict(kwargs.pop('headers', {}) or {})
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        content_type: Optional[str] = kwargs.pop('content_type', _headers.pop('Content-Type', None))
        cls: ClsType[_models.Snapshot] = kwargs.pop('cls', None)
        content_type = content_type or 'application/json'
        _json = None
        _content = None
        if isinstance(entity, (IOBase, bytes)):
            _content = entity
        else:
            _json = self._serialize.body(entity, 'SnapshotUpdateParameters')
        _request = build_update_snapshot_request(name=name, if_match=if_match, if_none_match=if_none_match, sync_token=self._config.sync_token, api_version=api_version, content_type=content_type, json=_json, content=_content, headers=_headers, params=_params)
        _request = _convert_request(_request)
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
        _request.url = self._client.format_url(_request.url, **path_format_arguments)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(_request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.Error, pipeline_response)
            raise HttpResponseError(response=response, model=error)
        response_headers = {}
        response_headers['Sync-Token'] = self._deserialize('str', response.headers.get('Sync-Token'))
        response_headers['ETag'] = self._deserialize('str', response.headers.get('ETag'))
        response_headers['Link'] = self._deserialize('str', response.headers.get('Link'))
        deserialized = self._deserialize('Snapshot', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized

    @distributed_trace
    def check_snapshot(self, name: str, if_match: Optional[str]=None, if_none_match: Optional[str]=None, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        "Requests the headers and status of the given resource.\n\n        Requests the headers and status of the given resource.\n\n        :param name: The name of the key-value snapshot to check. Required.\n        :type name: str\n        :param if_match: Used to perform an operation only if the targeted resource's etag matches the\n         value provided. Default value is None.\n        :type if_match: str\n        :param if_none_match: Used to perform an operation only if the targeted resource's etag does\n         not match the value provided. Default value is None.\n        :type if_none_match: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        _request = build_check_snapshot_request(name=name, if_match=if_match, if_none_match=if_none_match, sync_token=self._config.sync_token, api_version=api_version, headers=_headers, params=_params)
        _request = _convert_request(_request)
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
        _request.url = self._client.format_url(_request.url, **path_format_arguments)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(_request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response)
        response_headers = {}
        response_headers['Sync-Token'] = self._deserialize('str', response.headers.get('Sync-Token'))
        response_headers['ETag'] = self._deserialize('str', response.headers.get('ETag'))
        response_headers['Link'] = self._deserialize('str', response.headers.get('Link'))
        if cls:
            return cls(pipeline_response, None, response_headers)

    @distributed_trace
    def get_labels(self, name: Optional[str]=None, after: Optional[str]=None, accept_datetime: Optional[str]=None, select: Optional[List[Union[str, _models.LabelFields]]]=None, **kwargs: Any) -> Iterable['_models.Label']:
        if False:
            for i in range(10):
                print('nop')
        'Gets a list of labels.\n\n        Gets a list of labels.\n\n        :param name: A filter for the name of the returned labels. Default value is None.\n        :type name: str\n        :param after: Instructs the server to return elements that appear after the element referred to\n         by the specified token. Default value is None.\n        :type after: str\n        :param accept_datetime: Requests the server to respond with the state of the resource at the\n         specified time. Default value is None.\n        :type accept_datetime: str\n        :param select: Used to select what fields are present in the returned resource(s). Default\n         value is None.\n        :type select: list[str or ~azure.appconfiguration.models.LabelFields]\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either Label or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.appconfiguration.models.Label]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.LabelListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                return 10
            if not next_link:
                _request = build_get_labels_request(name=name, after=after, accept_datetime=accept_datetime, select=select, sync_token=self._config.sync_token, api_version=api_version, headers=_headers, params=_params)
                _request = _convert_request(_request)
                path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
                _request.url = self._client.format_url(_request.url, **path_format_arguments)
            else:
                _parsed_next_link = urllib.parse.urlparse(next_link)
                _next_request_params = case_insensitive_dict({key: [urllib.parse.quote(v) for v in value] for (key, value) in urllib.parse.parse_qs(_parsed_next_link.query).items()})
                _next_request_params['api-version'] = self._config.api_version
                _request = HttpRequest('GET', urllib.parse.urljoin(next_link, _parsed_next_link.path), params=_next_request_params)
                _request = _convert_request(_request)
                path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
                _request.url = self._client.format_url(_request.url, **path_format_arguments)
                _request.method = 'GET'
            return _request

        def extract_data(pipeline_response):
            if False:
                i = 10
                return i + 15
            deserialized = self._deserialize('LabelListResult', pipeline_response)
            list_of_elem = deserialized.items
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                i = 10
                return i + 15
            _request = prepare_request(next_link)
            _stream = False
            pipeline_response: PipelineResponse = self._client._pipeline.run(_request, stream=_stream, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.Error, pipeline_response)
                raise HttpResponseError(response=response, model=error)
            return pipeline_response
        return ItemPaged(get_next, extract_data)

    @distributed_trace
    def check_labels(self, name: Optional[str]=None, after: Optional[str]=None, accept_datetime: Optional[str]=None, select: Optional[List[Union[str, _models.LabelFields]]]=None, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        'Requests the headers and status of the given resource.\n\n        Requests the headers and status of the given resource.\n\n        :param name: A filter for the name of the returned labels. Default value is None.\n        :type name: str\n        :param after: Instructs the server to return elements that appear after the element referred to\n         by the specified token. Default value is None.\n        :type after: str\n        :param accept_datetime: Requests the server to respond with the state of the resource at the\n         specified time. Default value is None.\n        :type accept_datetime: str\n        :param select: Used to select what fields are present in the returned resource(s). Default\n         value is None.\n        :type select: list[str or ~azure.appconfiguration.models.LabelFields]\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        _request = build_check_labels_request(name=name, after=after, accept_datetime=accept_datetime, select=select, sync_token=self._config.sync_token, api_version=api_version, headers=_headers, params=_params)
        _request = _convert_request(_request)
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
        _request.url = self._client.format_url(_request.url, **path_format_arguments)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(_request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response)
        response_headers = {}
        response_headers['Sync-Token'] = self._deserialize('str', response.headers.get('Sync-Token'))
        if cls:
            return cls(pipeline_response, None, response_headers)

    @distributed_trace
    def put_lock(self, key: str, label: Optional[str]=None, if_match: Optional[str]=None, if_none_match: Optional[str]=None, **kwargs: Any) -> _models.KeyValue:
        if False:
            while True:
                i = 10
        "Locks a key-value.\n\n        Locks a key-value.\n\n        :param key: The key of the key-value to lock. Required.\n        :type key: str\n        :param label: The label, if any, of the key-value to lock. Default value is None.\n        :type label: str\n        :param if_match: Used to perform an operation only if the targeted resource's etag matches the\n         value provided. Default value is None.\n        :type if_match: str\n        :param if_none_match: Used to perform an operation only if the targeted resource's etag does\n         not match the value provided. Default value is None.\n        :type if_none_match: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: KeyValue or the result of cls(response)\n        :rtype: ~azure.appconfiguration.models.KeyValue\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.KeyValue] = kwargs.pop('cls', None)
        _request = build_put_lock_request(key=key, label=label, if_match=if_match, if_none_match=if_none_match, sync_token=self._config.sync_token, api_version=api_version, headers=_headers, params=_params)
        _request = _convert_request(_request)
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
        _request.url = self._client.format_url(_request.url, **path_format_arguments)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(_request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.Error, pipeline_response)
            raise HttpResponseError(response=response, model=error)
        response_headers = {}
        response_headers['Sync-Token'] = self._deserialize('str', response.headers.get('Sync-Token'))
        response_headers['ETag'] = self._deserialize('str', response.headers.get('ETag'))
        deserialized = self._deserialize('KeyValue', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized

    @distributed_trace
    def delete_lock(self, key: str, label: Optional[str]=None, if_match: Optional[str]=None, if_none_match: Optional[str]=None, **kwargs: Any) -> _models.KeyValue:
        if False:
            return 10
        "Unlocks a key-value.\n\n        Unlocks a key-value.\n\n        :param key: The key of the key-value to unlock. Required.\n        :type key: str\n        :param label: The label, if any, of the key-value to unlock. Default value is None.\n        :type label: str\n        :param if_match: Used to perform an operation only if the targeted resource's etag matches the\n         value provided. Default value is None.\n        :type if_match: str\n        :param if_none_match: Used to perform an operation only if the targeted resource's etag does\n         not match the value provided. Default value is None.\n        :type if_none_match: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: KeyValue or the result of cls(response)\n        :rtype: ~azure.appconfiguration.models.KeyValue\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.KeyValue] = kwargs.pop('cls', None)
        _request = build_delete_lock_request(key=key, label=label, if_match=if_match, if_none_match=if_none_match, sync_token=self._config.sync_token, api_version=api_version, headers=_headers, params=_params)
        _request = _convert_request(_request)
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
        _request.url = self._client.format_url(_request.url, **path_format_arguments)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(_request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.Error, pipeline_response)
            raise HttpResponseError(response=response, model=error)
        response_headers = {}
        response_headers['Sync-Token'] = self._deserialize('str', response.headers.get('Sync-Token'))
        response_headers['ETag'] = self._deserialize('str', response.headers.get('ETag'))
        deserialized = self._deserialize('KeyValue', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, response_headers)
        return deserialized

    @distributed_trace
    def get_revisions(self, key: Optional[str]=None, label: Optional[str]=None, after: Optional[str]=None, accept_datetime: Optional[str]=None, select: Optional[List[Union[str, _models.KeyValueFields]]]=None, **kwargs: Any) -> Iterable['_models.KeyValue']:
        if False:
            print('Hello World!')
        'Gets a list of key-value revisions.\n\n        Gets a list of key-value revisions.\n\n        :param key: A filter used to match keys. Default value is None.\n        :type key: str\n        :param label: A filter used to match labels. Default value is None.\n        :type label: str\n        :param after: Instructs the server to return elements that appear after the element referred to\n         by the specified token. Default value is None.\n        :type after: str\n        :param accept_datetime: Requests the server to respond with the state of the resource at the\n         specified time. Default value is None.\n        :type accept_datetime: str\n        :param select: Used to select what fields are present in the returned resource(s). Default\n         value is None.\n        :type select: list[str or ~azure.appconfiguration.models.KeyValueFields]\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either KeyValue or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.appconfiguration.models.KeyValue]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.KeyValueListResult] = kwargs.pop('cls', None)
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})

        def prepare_request(next_link=None):
            if False:
                i = 10
                return i + 15
            if not next_link:
                _request = build_get_revisions_request(key=key, label=label, after=after, accept_datetime=accept_datetime, select=select, sync_token=self._config.sync_token, api_version=api_version, headers=_headers, params=_params)
                _request = _convert_request(_request)
                path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
                _request.url = self._client.format_url(_request.url, **path_format_arguments)
            else:
                _parsed_next_link = urllib.parse.urlparse(next_link)
                _next_request_params = case_insensitive_dict({key: [urllib.parse.quote(v) for v in value] for (key, value) in urllib.parse.parse_qs(_parsed_next_link.query).items()})
                _next_request_params['api-version'] = self._config.api_version
                _request = HttpRequest('GET', urllib.parse.urljoin(next_link, _parsed_next_link.path), params=_next_request_params)
                _request = _convert_request(_request)
                path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
                _request.url = self._client.format_url(_request.url, **path_format_arguments)
                _request.method = 'GET'
            return _request

        def extract_data(pipeline_response):
            if False:
                for i in range(10):
                    print('nop')
            deserialized = self._deserialize('KeyValueListResult', pipeline_response)
            list_of_elem = deserialized.items
            if cls:
                list_of_elem = cls(list_of_elem)
            return (deserialized.next_link or None, iter(list_of_elem))

        def get_next(next_link=None):
            if False:
                while True:
                    i = 10
            _request = prepare_request(next_link)
            _stream = False
            pipeline_response: PipelineResponse = self._client._pipeline.run(_request, stream=_stream, **kwargs)
            response = pipeline_response.http_response
            if response.status_code not in [200]:
                map_error(status_code=response.status_code, response=response, error_map=error_map)
                error = self._deserialize.failsafe_deserialize(_models.Error, pipeline_response)
                raise HttpResponseError(response=response, model=error)
            return pipeline_response
        return ItemPaged(get_next, extract_data)

    @distributed_trace
    def check_revisions(self, key: Optional[str]=None, label: Optional[str]=None, after: Optional[str]=None, accept_datetime: Optional[str]=None, select: Optional[List[Union[str, _models.KeyValueFields]]]=None, **kwargs: Any) -> None:
        if False:
            return 10
        'Requests the headers and status of the given resource.\n\n        Requests the headers and status of the given resource.\n\n        :param key: A filter used to match keys. Default value is None.\n        :type key: str\n        :param label: A filter used to match labels. Default value is None.\n        :type label: str\n        :param after: Instructs the server to return elements that appear after the element referred to\n         by the specified token. Default value is None.\n        :type after: str\n        :param accept_datetime: Requests the server to respond with the state of the resource at the\n         specified time. Default value is None.\n        :type accept_datetime: str\n        :param select: Used to select what fields are present in the returned resource(s). Default\n         value is None.\n        :type select: list[str or ~azure.appconfiguration.models.KeyValueFields]\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[None] = kwargs.pop('cls', None)
        _request = build_check_revisions_request(key=key, label=label, after=after, accept_datetime=accept_datetime, select=select, sync_token=self._config.sync_token, api_version=api_version, headers=_headers, params=_params)
        _request = _convert_request(_request)
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
        _request.url = self._client.format_url(_request.url, **path_format_arguments)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(_request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            raise HttpResponseError(response=response)
        response_headers = {}
        response_headers['Sync-Token'] = self._deserialize('str', response.headers.get('Sync-Token'))
        response_headers['ETag'] = self._deserialize('str', response.headers.get('ETag'))
        if cls:
            return cls(pipeline_response, None, response_headers)

    @distributed_trace
    def get_operation_details(self, snapshot: str, **kwargs: Any) -> _models.OperationDetails:
        if False:
            return 10
        'Gets the state of a long running operation.\n\n        Gets the state of a long running operation.\n\n        :param snapshot: Snapshot identifier for the long running operation. Required.\n        :type snapshot: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: OperationDetails or the result of cls(response)\n        :rtype: ~azure.appconfiguration.models.OperationDetails\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        error_map = {401: ClientAuthenticationError, 404: ResourceNotFoundError, 409: ResourceExistsError, 304: ResourceNotModifiedError}
        error_map.update(kwargs.pop('error_map', {}) or {})
        _headers = kwargs.pop('headers', {}) or {}
        _params = case_insensitive_dict(kwargs.pop('params', {}) or {})
        api_version: str = kwargs.pop('api_version', _params.pop('api-version', self._config.api_version))
        cls: ClsType[_models.OperationDetails] = kwargs.pop('cls', None)
        _request = build_get_operation_details_request(snapshot=snapshot, api_version=api_version, headers=_headers, params=_params)
        _request = _convert_request(_request)
        path_format_arguments = {'endpoint': self._serialize.url('self._config.endpoint', self._config.endpoint, 'str', skip_quote=True)}
        _request.url = self._client.format_url(_request.url, **path_format_arguments)
        _stream = False
        pipeline_response: PipelineResponse = self._client._pipeline.run(_request, stream=_stream, **kwargs)
        response = pipeline_response.http_response
        if response.status_code not in [200]:
            map_error(status_code=response.status_code, response=response, error_map=error_map)
            error = self._deserialize.failsafe_deserialize(_models.Error, pipeline_response)
            raise HttpResponseError(response=response, model=error)
        deserialized = self._deserialize('OperationDetails', pipeline_response)
        if cls:
            return cls(pipeline_response, deserialized, {})
        return deserialized