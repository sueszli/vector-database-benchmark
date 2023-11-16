"""Handler that integrates with Stackdriver."""
from __future__ import annotations
import logging
from functools import cached_property
from typing import TYPE_CHECKING, Collection
from urllib.parse import urlencode
from google.cloud import logging as gcp_logging
from google.cloud.logging import Resource
from google.cloud.logging.handlers.transports import BackgroundThreadTransport, Transport
from google.cloud.logging_v2.services.logging_service_v2 import LoggingServiceV2Client
from google.cloud.logging_v2.types import ListLogEntriesRequest, ListLogEntriesResponse
from airflow.providers.google.cloud.utils.credentials_provider import get_credentials_and_project_id
from airflow.providers.google.common.consts import CLIENT_INFO
if TYPE_CHECKING:
    from contextvars import ContextVar
    from google.auth.credentials import Credentials
    from airflow.models import TaskInstance
try:
    ctx_indiv_trigger: ContextVar | None
    from airflow.utils.log.trigger_handler import ctx_indiv_trigger
except ImportError:
    ctx_indiv_trigger = None
DEFAULT_LOGGER_NAME = 'airflow'
_GLOBAL_RESOURCE = Resource(type='global', labels={})
_DEFAULT_SCOPESS = frozenset(['https://www.googleapis.com/auth/logging.read', 'https://www.googleapis.com/auth/logging.write'])

class StackdriverTaskHandler(logging.Handler):
    """Handler that directly makes Stackdriver logging API calls.

    This is a Python standard ``logging`` handler using that can be used to
    route Python standard logging messages directly to the Stackdriver
    Logging API.

    It can also be used to save logs for executing tasks. To do this, you should set as a handler with
    the name "tasks". In this case, it will also be used to read the log for display in Web UI.

    This handler supports both an asynchronous and synchronous transport.

    :param gcp_key_path: Path to Google Cloud Credential JSON file.
        If omitted, authorization based on `the Application Default Credentials
        <https://cloud.google.com/docs/authentication/production#finding_credentials_automatically>`__ will
        be used.
    :param scopes: OAuth scopes for the credentials,
    :param name: the name of the custom log in Stackdriver Logging. Defaults
        to 'airflow'. The name of the Python logger will be represented
        in the ``python_logger`` field.
    :param transport: Class for creating new transport objects. It should
        extend from the base :class:`google.cloud.logging.handlers.Transport` type and
        implement :meth`google.cloud.logging.handlers.Transport.send`. Defaults to
        :class:`google.cloud.logging.handlers.BackgroundThreadTransport`. The other
        option is :class:`google.cloud.logging.handlers.SyncTransport`.
    :param resource: (Optional) Monitored resource of the entry, defaults
                     to the global resource type.
    :param labels: (Optional) Mapping of labels for the entry.
    """
    LABEL_TASK_ID = 'task_id'
    LABEL_DAG_ID = 'dag_id'
    LABEL_EXECUTION_DATE = 'execution_date'
    LABEL_TRY_NUMBER = 'try_number'
    LOG_VIEWER_BASE_URL = 'https://console.cloud.google.com/logs/viewer'
    LOG_NAME = 'Google Stackdriver'
    trigger_supported = True
    trigger_should_queue = False
    trigger_should_wrap = False
    trigger_send_end_marker = False

    def __init__(self, gcp_key_path: str | None=None, scopes: Collection[str] | None=_DEFAULT_SCOPESS, name: str=DEFAULT_LOGGER_NAME, transport: type[Transport]=BackgroundThreadTransport, resource: Resource=_GLOBAL_RESOURCE, labels: dict[str, str] | None=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.gcp_key_path: str | None = gcp_key_path
        self.scopes: Collection[str] | None = scopes
        self.name: str = name
        self.transport_type: type[Transport] = transport
        self.resource: Resource = resource
        self.labels: dict[str, str] | None = labels
        self.task_instance_labels: dict[str, str] | None = {}
        self.task_instance_hostname = 'default-hostname'

    @cached_property
    def _credentials_and_project(self) -> tuple[Credentials, str]:
        if False:
            for i in range(10):
                print('nop')
        (credentials, project) = get_credentials_and_project_id(key_path=self.gcp_key_path, scopes=self.scopes, disable_logging=True)
        return (credentials, project)

    @property
    def _client(self) -> gcp_logging.Client:
        if False:
            print('Hello World!')
        'The Cloud Library API client.'
        (credentials, project) = self._credentials_and_project
        client = gcp_logging.Client(credentials=credentials, project=project, client_info=CLIENT_INFO)
        return client

    @property
    def _logging_service_client(self) -> LoggingServiceV2Client:
        if False:
            while True:
                i = 10
        'The Cloud logging service v2 client.'
        (credentials, _) = self._credentials_and_project
        client = LoggingServiceV2Client(credentials=credentials, client_info=CLIENT_INFO)
        return client

    @cached_property
    def _transport(self) -> Transport:
        if False:
            return 10
        'Object responsible for sending data to Stackdriver.'
        return self.transport_type(self._client, self.name)

    def _get_labels(self, task_instance=None):
        if False:
            print('Hello World!')
        if task_instance:
            ti_labels = self._task_instance_to_labels(task_instance)
        else:
            ti_labels = self.task_instance_labels
        labels: dict[str, str] | None
        if self.labels and ti_labels:
            labels = {}
            labels.update(self.labels)
            labels.update(ti_labels)
        elif self.labels:
            labels = self.labels
        elif ti_labels:
            labels = ti_labels
        else:
            labels = None
        return labels or {}

    def emit(self, record: logging.LogRecord) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Actually log the specified logging record.\n\n        :param record: The record to be logged.\n        '
        message = self.format(record)
        ti = None
        if ctx_indiv_trigger is not None and getattr(record, ctx_indiv_trigger.name, None):
            ti = getattr(record, 'task_instance', None)
        labels = self._get_labels(ti)
        self._transport.send(record, message, resource=self.resource, labels=labels)

    def set_context(self, task_instance: TaskInstance) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Configures the logger to add information with information about the current task.\n\n        :param task_instance: Currently executed task\n        '
        self.task_instance_labels = self._task_instance_to_labels(task_instance)
        self.task_instance_hostname = task_instance.hostname

    def read(self, task_instance: TaskInstance, try_number: int | None=None, metadata: dict | None=None) -> tuple[list[tuple[tuple[str, str]]], list[dict[str, str | bool]]]:
        if False:
            i = 10
            return i + 15
        '\n        Read logs of given task instance from Stackdriver logging.\n\n        :param task_instance: task instance object\n        :param try_number: task instance try_number to read logs from. If None\n           it returns all logs\n        :param metadata: log metadata. It is used for steaming log reading and auto-tailing.\n        :return: a tuple of (\n            list of (one element tuple with two element tuple - hostname and logs)\n            and list of metadata)\n        '
        if try_number is not None and try_number < 1:
            logs = f'Error fetching the logs. Try number {try_number} is invalid.'
            return ([((self.task_instance_hostname, logs),)], [{'end_of_log': 'true'}])
        if not metadata:
            metadata = {}
        ti_labels = self._task_instance_to_labels(task_instance)
        if try_number is not None:
            ti_labels[self.LABEL_TRY_NUMBER] = str(try_number)
        else:
            del ti_labels[self.LABEL_TRY_NUMBER]
        log_filter = self._prepare_log_filter(ti_labels)
        next_page_token = metadata.get('next_page_token', None)
        all_pages = 'download_logs' in metadata and metadata['download_logs']
        (messages, end_of_log, next_page_token) = self._read_logs(log_filter, next_page_token, all_pages)
        new_metadata: dict[str, str | bool] = {'end_of_log': end_of_log}
        if next_page_token:
            new_metadata['next_page_token'] = next_page_token
        return ([((self.task_instance_hostname, messages),)], [new_metadata])

    def _prepare_log_filter(self, ti_labels: dict[str, str]) -> str:
        if False:
            while True:
                i = 10
        "\n        Prepares the filter that chooses which log entries to fetch.\n\n        More information:\n        https://cloud.google.com/logging/docs/reference/v2/rest/v2/entries/list#body.request_body.FIELDS.filter\n        https://cloud.google.com/logging/docs/view/advanced-queries\n\n        :param ti_labels: Task Instance's labels that will be used to search for logs\n        :return: logs filter\n        "

        def escape_label_key(key: str) -> str:
            if False:
                i = 10
                return i + 15
            return f'"{key}"' if '.' in key else key

        def escale_label_value(value: str) -> str:
            if False:
                i = 10
                return i + 15
            escaped_value = value.replace('\\', '\\\\').replace('"', '\\"')
            return f'"{escaped_value}"'
        (_, project) = self._credentials_and_project
        log_filters = [f'resource.type={escale_label_value(self.resource.type)}', f'logName="projects/{project}/logs/{self.name}"']
        for (key, value) in self.resource.labels.items():
            log_filters.append(f'resource.labels.{escape_label_key(key)}={escale_label_value(value)}')
        for (key, value) in ti_labels.items():
            log_filters.append(f'labels.{escape_label_key(key)}={escale_label_value(value)}')
        return '\n'.join(log_filters)

    def _read_logs(self, log_filter: str, next_page_token: str | None, all_pages: bool) -> tuple[str, bool, str | None]:
        if False:
            while True:
                i = 10
        '\n        Sends requests to the Stackdriver service and downloads logs.\n\n        :param log_filter: Filter specifying the logs to be downloaded.\n        :param next_page_token: The token of the page from which the log download will start.\n            If None is passed, it will start from the first page.\n        :param all_pages: If True is passed, all subpages will be downloaded. Otherwise, only the first\n            page will be downloaded\n        :return: A token that contains the following items:\n            * string with logs\n            * Boolean value describing whether there are more logs,\n            * token of the next page\n        '
        messages = []
        (new_messages, next_page_token) = self._read_single_logs_page(log_filter=log_filter, page_token=next_page_token)
        messages.append(new_messages)
        if all_pages:
            while next_page_token:
                (new_messages, next_page_token) = self._read_single_logs_page(log_filter=log_filter, page_token=next_page_token)
                messages.append(new_messages)
                if not messages:
                    break
            end_of_log = True
            next_page_token = None
        else:
            end_of_log = not bool(next_page_token)
        return ('\n'.join(messages), end_of_log, next_page_token)

    def _read_single_logs_page(self, log_filter: str, page_token: str | None=None) -> tuple[str, str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Sends requests to the Stackdriver service and downloads single pages with logs.\n\n        :param log_filter: Filter specifying the logs to be downloaded.\n        :param page_token: The token of the page to be downloaded. If None is passed, the first page will be\n            downloaded.\n        :return: Downloaded logs and next page token\n        '
        (_, project) = self._credentials_and_project
        request = ListLogEntriesRequest(resource_names=[f'projects/{project}'], filter=log_filter, page_token=page_token, order_by='timestamp asc', page_size=1000)
        response = self._logging_service_client.list_log_entries(request=request)
        page: ListLogEntriesResponse = next(response.pages)
        messages: list[str] = []
        for entry in page.entries:
            if 'message' in (entry.json_payload or {}):
                messages.append(entry.json_payload['message'])
            elif entry.text_payload:
                messages.append(entry.text_payload)
        return ('\n'.join(messages), page.next_page_token)

    @classmethod
    def _task_instance_to_labels(cls, ti: TaskInstance) -> dict[str, str]:
        if False:
            for i in range(10):
                print('nop')
        return {cls.LABEL_TASK_ID: ti.task_id, cls.LABEL_DAG_ID: ti.dag_id, cls.LABEL_EXECUTION_DATE: str(ti.execution_date.isoformat()), cls.LABEL_TRY_NUMBER: str(ti.try_number)}

    @property
    def log_name(self):
        if False:
            while True:
                i = 10
        'Return log name.'
        return self.LOG_NAME

    @cached_property
    def _resource_path(self):
        if False:
            for i in range(10):
                print('nop')
        segments = [self.resource.type]
        for (key, value) in self.resource.labels:
            segments += [key]
            segments += [value]
        return '/'.join(segments)

    def get_external_log_url(self, task_instance: TaskInstance, try_number: int) -> str:
        if False:
            print('Hello World!')
        '\n        Creates an address for an external log collecting service.\n\n        :param task_instance: task instance object\n        :param try_number: task instance try_number to read logs from\n        :return: URL to the external log collection service\n        '
        (_, project_id) = self._credentials_and_project
        ti_labels = self._task_instance_to_labels(task_instance)
        ti_labels[self.LABEL_TRY_NUMBER] = str(try_number)
        log_filter = self._prepare_log_filter(ti_labels)
        url_query_string = {'project': project_id, 'interval': 'NO_LIMIT', 'resource': self._resource_path, 'advancedFilter': log_filter}
        url = f'{self.LOG_VIEWER_BASE_URL}?{urlencode(url_query_string)}'
        return url

    def close(self) -> None:
        if False:
            return 10
        self._transport.flush()