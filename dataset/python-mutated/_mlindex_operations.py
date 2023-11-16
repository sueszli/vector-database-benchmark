import json
import os.path
from os import PathLike, path
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Optional, Union
from azure.core.tracing.decorator import distributed_trace
from azure.ai.ml import MLClient
from azure.ai.ml.entities import AccountKeyConfiguration, Data
from azure.ai.ml._utils._storage_utils import get_ds_name_and_path_prefix, get_storage_client
from azure.ai.resources.entities.mlindex import Index as MLIndex
from azure.ai.resources._telemetry import ActivityType, monitor_with_activity, OpsLogger
ops_logger = OpsLogger(__name__)
(logger, module_logger) = (ops_logger.package_logger, ops_logger.module_logger)

class MLIndexOperations:
    """MLIndexOperations.

    You should not instantiate this class directly. Instead, you should
    create an MLClient instance that instantiates it for you and
    attaches it as an attribute.
    """

    def __init__(self, ml_client: MLClient, **kwargs: Any):
        if False:
            i = 10
            return i + 15
        self._ml_client = ml_client
        ops_logger.update_info(kwargs)

    @distributed_trace
    @monitor_with_activity(logger, 'Index.List', ActivityType.PUBLICAPI)
    def list(self, **kwargs) -> Iterable[MLIndex]:
        if False:
            return 10
        return [MLIndex._from_data_asset(data) for data in self._ml_client.data.list()]

    @distributed_trace
    @monitor_with_activity(logger, 'Index.Get', ActivityType.PUBLICAPI)
    def get(self, name: str, version: Optional[str]=None, label: Optional[str]=None) -> MLIndex:
        if False:
            print('Hello World!')
        data = self._ml_client.data.get(name, version, label)
        if 'azureml.mlIndexAsset' not in data.properties:
            raise Exception(f'No Index with name {name} and version {version} found.')
        return MLIndex._from_data_asset(data)

    @distributed_trace
    @monitor_with_activity(logger, 'Index.CreateOrUpdate', ActivityType.PUBLICAPI)
    def create_or_update(self, mlindex: MLIndex, **kwargs) -> MLIndex:
        if False:
            print('Hello World!')
        try:
            path = mlindex.path
        except:
            try:
                path = mlindex['path']
            except Exception as e:
                raise e
        data = Data(name=mlindex.name, version=mlindex.version, path=path, properties=mlindex.properties)
        if 'name' in kwargs:
            data.name = kwargs['name']
        if 'version' in kwargs:
            data.version = kwargs['version']
        if 'description' in kwargs:
            data.description = kwargs['description']
        created = self._ml_client.data.create_or_update(data)
        return MLIndex(name=created.name, version=created.version, description=created.description, tags=created.tags, path=created.path)

    @distributed_trace
    @monitor_with_activity(logger, 'Index.Download', ActivityType.PUBLICAPI)
    def download(self, name: str, download_path: Union[str, PathLike], version: str=None, label: str=None) -> None:
        if False:
            print('Hello World!')
        model_uri = self.get(name=name, version=version, label=label).path
        (ds_name, path_prefix) = get_ds_name_and_path_prefix(model_uri)
        ds = self._ml_client.datastores.get(ds_name, include_secrets=True)
        acc_name = ds.account_name
        if isinstance(ds.credentials, AccountKeyConfiguration):
            credential = ds.credentials.account_key
        else:
            try:
                credential = ds.credentials.sas_token
            except Exception as e:
                if not hasattr(ds.credentials, 'sas_token'):
                    credential = self._ml_client._credential
                else:
                    raise e
        container = ds.container_name
        datastore_type = ds.type
        storage_client = get_storage_client(credential=credential, container_name=container, storage_account=acc_name, storage_type=datastore_type)
        path_file = '{}{}'.format(download_path, path.sep)
        storage_client.download(starts_with=path_prefix, destination=path_file)

    @distributed_trace
    @monitor_with_activity(logger, 'Index.restore', ActivityType.PUBLICAPI)
    def restore(self, name: str, version: Optional[str]=None, label: Optional[str]=None) -> None:
        if False:
            i = 10
            return i + 15
        'Restore an archived index.\n\n            :param name: index name.\n            :type name: str\n            :param version: index version.\n            :type version: str\n            :param label: Label of the index. (mutually exclusive with version)\n            :type label: str\n            :return: None\n\n            '
        self._ml_client.data.restore(name, version, label)

    @distributed_trace
    @monitor_with_activity(logger, 'Index.archive', ActivityType.PUBLICAPI)
    def archive(self, name: str, version: Optional[str]=None, label: Optional[str]=None) -> None:
        if False:
            print('Hello World!')
        'Archive an index.\n\n            :param name: index name.\n            :type name: str\n            :param version: index version.\n            :type version: str\n            :param label: Label of the index. (mutually exclusive with version)\n            :type label: str\n            :return: None\n\n            '
        self._ml_client.data.archive(name, version, label)