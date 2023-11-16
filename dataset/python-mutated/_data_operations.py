from typing import Optional, Union
from azure.ai.ml._telemetry import ActivityType, monitor_with_activity
from azure.ai.ml._utils._asset_utils import _validate_auto_delete_setting_in_data_output, _validate_workspace_managed_datastore
from azure.ai.ml._utils._experimental import experimental
from azure.ai.ml.constants._common import AssetTypes, AzureMLResourceType
from azure.ai.ml.entities import PipelineJob, PipelineJobSettings
from azure.ai.ml.entities._credentials import ManagedIdentityConfiguration, UserIdentityConfiguration
from azure.ai.ml.entities._inputs_outputs import Input
from azure.ai.ml.operations._data_operations import DataOperations, logger
from azure.ai.generative.index._dataindex.data_index import index_data as index_data_func
from azure.ai.generative.index._dataindex.entities.data_index import DataIndex

@monitor_with_activity(logger, 'Data.IndexData', ActivityType.PUBLICAPI)
@experimental
def index_data(self, data_index: DataIndex, identity: Optional[Union[ManagedIdentityConfiguration, UserIdentityConfiguration]]=None, compute: str='serverless', serverless_instance_type: Optional[str]=None, input_data_override: Optional[Input]=None, submit_job: bool=True, **kwargs) -> PipelineJob:
    if False:
        return 10
    '\n    Returns the data import job that is creating the data asset.\n\n    :param data_index: DataIndex object.\n    :type data_index: azure.ai.ml.entities._dataindex\n    :param identity: Identity configuration for the job.\n    :type identity: Optional[Union[ManagedIdentityConfiguration, UserIdentityConfiguration]]\n    :param compute: The compute target to use for the job. Default: "serverless".\n    :type compute: str\n    :param serverless_instance_type: The instance type to use for serverless compute.\n    :type serverless_instance_type: Optional[str]\n    :param input_data_override: Input data override for the job.\n        Used to pipe output of step into DataIndex Job in a pipeline.\n    :type input_data_override: Optional[Input]\n    :param submit_job: Whether to submit the job to the service. Default: True.\n    :type submit_job: bool\n    :return: data import job object.\n    :rtype: ~azure.ai.ml.entities.PipelineJob.\n    '
    from azure.ai.ml import MLClient
    default_name = 'data_index_' + data_index.name
    experiment_name = kwargs.pop('experiment_name', None) or default_name
    data_index.type = AssetTypes.URI_FOLDER
    _validate_auto_delete_setting_in_data_output(data_index.auto_delete_setting)
    data_index.path = _validate_workspace_managed_datastore(data_index.path)
    if '${{name}}' not in data_index.path and '{name}' not in data_index.path:
        data_index.path = data_index.path.rstrip('/') + '/${{name}}'
    index_job = index_data_func(description=data_index.description or kwargs.pop('description', None) or default_name, name=data_index.name or kwargs.pop('name', None), display_name=kwargs.pop('display_name', None) or default_name, experiment_name=experiment_name, compute=compute, serverless_instance_type=serverless_instance_type, data_index=data_index, ml_client=MLClient(subscription_id=self._subscription_id, resource_group_name=self._resource_group_name, workspace_name=self._workspace_name, credential=self._service_client._config.credential), identity=identity, input_data_override=input_data_override, **kwargs)
    index_pipeline = PipelineJob(description=index_job.description, tags=index_job.tags, name=index_job.name, display_name=index_job.display_name, experiment_name=experiment_name, properties=index_job.properties or {}, settings=PipelineJobSettings(force_rerun=True, default_compute=compute), jobs={default_name: index_job})
    index_pipeline.properties['azureml.mlIndexAssetName'] = data_index.name
    index_pipeline.properties['azureml.mlIndexAssetKind'] = data_index.index.type
    index_pipeline.properties['azureml.mlIndexAssetSource'] = kwargs.pop('mlindex_asset_source', 'Data Asset')
    if submit_job:
        return self._all_operations.all_operations[AzureMLResourceType.JOB].create_or_update(job=index_pipeline, skip_validation=True, **kwargs)
    return index_pipeline
DataOperations.index_data = index_data