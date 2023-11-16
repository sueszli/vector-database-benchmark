from typing import Any, Optional, TYPE_CHECKING
from azure.mgmt.core import ARMPipelineClient
from azure.profiles import KnownProfiles, ProfileDefinition
from azure.profiles.multiapiclient import MultiApiClientMixin
from ._configuration import ContainerServiceClientConfiguration
from ._serialization import Deserializer, Serializer
if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential

class _SDKClient(object):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'This is a fake class to support current implemetation of MultiApiClientMixin."\n        Will be removed in final version of multiapi azure-core based client\n        '
        pass

class ContainerServiceClient(MultiApiClientMixin, _SDKClient):
    """The Container Service Client.

    This ready contains multiple API versions, to help you deal with all of the Azure clouds
    (Azure Stack, Azure Government, Azure China, etc.).
    By default, it uses the latest API version available on public Azure.
    For production, you should stick to a particular api-version and/or profile.
    The profile sets a mapping between an operation group and its API version.
    The api-version parameter sets the default API version if the operation
    group is not described in the profile.

    :param credential: Credential needed for the client to connect to Azure. Required.
    :type credential: ~azure.core.credentials.TokenCredential
    :param subscription_id: The ID of the target subscription. Required.
    :type subscription_id: str
    :param api_version: API version to use if no profile is provided, or if missing in profile.
    :type api_version: str
    :param base_url: Service URL
    :type base_url: str
    :param profile: A profile definition, from KnownProfiles to dict.
    :type profile: azure.profiles.KnownProfiles
    :keyword int polling_interval: Default waiting time between two polls for LRO operations if no Retry-After header is present.
    """
    DEFAULT_API_VERSION = '2023-08-01'
    _PROFILE_TAG = 'azure.mgmt.containerservice.ContainerServiceClient'
    LATEST_PROFILE = ProfileDefinition({_PROFILE_TAG: {None: DEFAULT_API_VERSION, 'container_services': '2019-04-01', 'fleet_members': '2022-09-02-preview', 'fleets': '2022-09-02-preview', 'machines': '2023-07-02-preview', 'managed_cluster_snapshots': '2023-07-02-preview', 'open_shift_managed_clusters': '2019-04-30', 'trusted_access_role_bindings': '2023-07-02-preview', 'trusted_access_roles': '2023-07-02-preview'}}, _PROFILE_TAG + ' latest')

    def __init__(self, credential: 'TokenCredential', subscription_id: str, api_version: Optional[str]=None, base_url: str='https://management.azure.com', profile: KnownProfiles=KnownProfiles.default, **kwargs: Any):
        if False:
            while True:
                i = 10
        if api_version:
            kwargs.setdefault('api_version', api_version)
        self._config = ContainerServiceClientConfiguration(credential, subscription_id, **kwargs)
        self._client = ARMPipelineClient(base_url=base_url, config=self._config, **kwargs)
        super(ContainerServiceClient, self).__init__(api_version=api_version, profile=profile)

    @classmethod
    def _models_dict(cls, api_version):
        if False:
            return 10
        return {k: v for (k, v) in cls.models(api_version).__dict__.items() if isinstance(v, type)}

    @classmethod
    def models(cls, api_version=DEFAULT_API_VERSION):
        if False:
            i = 10
            return i + 15
        'Module depends on the API version:\n\n           * 2017-07-01: :mod:`v2017_07_01.models<azure.mgmt.containerservice.v2017_07_01.models>`\n           * 2018-03-31: :mod:`v2018_03_31.models<azure.mgmt.containerservice.v2018_03_31.models>`\n           * 2018-08-01-preview: :mod:`v2018_08_01_preview.models<azure.mgmt.containerservice.v2018_08_01_preview.models>`\n           * 2018-09-30-preview: :mod:`v2018_09_30_preview.models<azure.mgmt.containerservice.v2018_09_30_preview.models>`\n           * 2019-02-01: :mod:`v2019_02_01.models<azure.mgmt.containerservice.v2019_02_01.models>`\n           * 2019-04-01: :mod:`v2019_04_01.models<azure.mgmt.containerservice.v2019_04_01.models>`\n           * 2019-04-30: :mod:`v2019_04_30.models<azure.mgmt.containerservice.v2019_04_30.models>`\n           * 2019-06-01: :mod:`v2019_06_01.models<azure.mgmt.containerservice.v2019_06_01.models>`\n           * 2019-08-01: :mod:`v2019_08_01.models<azure.mgmt.containerservice.v2019_08_01.models>`\n           * 2019-09-30-preview: :mod:`v2019_09_30_preview.models<azure.mgmt.containerservice.v2019_09_30_preview.models>`\n           * 2019-10-01: :mod:`v2019_10_01.models<azure.mgmt.containerservice.v2019_10_01.models>`\n           * 2019-10-27-preview: :mod:`v2019_10_27_preview.models<azure.mgmt.containerservice.v2019_10_27_preview.models>`\n           * 2019-11-01: :mod:`v2019_11_01.models<azure.mgmt.containerservice.v2019_11_01.models>`\n           * 2020-01-01: :mod:`v2020_01_01.models<azure.mgmt.containerservice.v2020_01_01.models>`\n           * 2020-02-01: :mod:`v2020_02_01.models<azure.mgmt.containerservice.v2020_02_01.models>`\n           * 2020-03-01: :mod:`v2020_03_01.models<azure.mgmt.containerservice.v2020_03_01.models>`\n           * 2020-04-01: :mod:`v2020_04_01.models<azure.mgmt.containerservice.v2020_04_01.models>`\n           * 2020-06-01: :mod:`v2020_06_01.models<azure.mgmt.containerservice.v2020_06_01.models>`\n           * 2020-07-01: :mod:`v2020_07_01.models<azure.mgmt.containerservice.v2020_07_01.models>`\n           * 2020-09-01: :mod:`v2020_09_01.models<azure.mgmt.containerservice.v2020_09_01.models>`\n           * 2020-11-01: :mod:`v2020_11_01.models<azure.mgmt.containerservice.v2020_11_01.models>`\n           * 2020-12-01: :mod:`v2020_12_01.models<azure.mgmt.containerservice.v2020_12_01.models>`\n           * 2021-02-01: :mod:`v2021_02_01.models<azure.mgmt.containerservice.v2021_02_01.models>`\n           * 2021-03-01: :mod:`v2021_03_01.models<azure.mgmt.containerservice.v2021_03_01.models>`\n           * 2021-05-01: :mod:`v2021_05_01.models<azure.mgmt.containerservice.v2021_05_01.models>`\n           * 2021-07-01: :mod:`v2021_07_01.models<azure.mgmt.containerservice.v2021_07_01.models>`\n           * 2021-08-01: :mod:`v2021_08_01.models<azure.mgmt.containerservice.v2021_08_01.models>`\n           * 2021-09-01: :mod:`v2021_09_01.models<azure.mgmt.containerservice.v2021_09_01.models>`\n           * 2021-10-01: :mod:`v2021_10_01.models<azure.mgmt.containerservice.v2021_10_01.models>`\n           * 2021-11-01-preview: :mod:`v2021_11_01_preview.models<azure.mgmt.containerservice.v2021_11_01_preview.models>`\n           * 2022-01-01: :mod:`v2022_01_01.models<azure.mgmt.containerservice.v2022_01_01.models>`\n           * 2022-01-02-preview: :mod:`v2022_01_02_preview.models<azure.mgmt.containerservice.v2022_01_02_preview.models>`\n           * 2022-02-01: :mod:`v2022_02_01.models<azure.mgmt.containerservice.v2022_02_01.models>`\n           * 2022-02-02-preview: :mod:`v2022_02_02_preview.models<azure.mgmt.containerservice.v2022_02_02_preview.models>`\n           * 2022-03-01: :mod:`v2022_03_01.models<azure.mgmt.containerservice.v2022_03_01.models>`\n           * 2022-03-02-preview: :mod:`v2022_03_02_preview.models<azure.mgmt.containerservice.v2022_03_02_preview.models>`\n           * 2022-04-01: :mod:`v2022_04_01.models<azure.mgmt.containerservice.v2022_04_01.models>`\n           * 2022-04-02-preview: :mod:`v2022_04_02_preview.models<azure.mgmt.containerservice.v2022_04_02_preview.models>`\n           * 2022-05-02-preview: :mod:`v2022_05_02_preview.models<azure.mgmt.containerservice.v2022_05_02_preview.models>`\n           * 2022-06-01: :mod:`v2022_06_01.models<azure.mgmt.containerservice.v2022_06_01.models>`\n           * 2022-06-02-preview: :mod:`v2022_06_02_preview.models<azure.mgmt.containerservice.v2022_06_02_preview.models>`\n           * 2022-07-01: :mod:`v2022_07_01.models<azure.mgmt.containerservice.v2022_07_01.models>`\n           * 2022-07-02-preview: :mod:`v2022_07_02_preview.models<azure.mgmt.containerservice.v2022_07_02_preview.models>`\n           * 2022-08-02-preview: :mod:`v2022_08_02_preview.models<azure.mgmt.containerservice.v2022_08_02_preview.models>`\n           * 2022-08-03-preview: :mod:`v2022_08_03_preview.models<azure.mgmt.containerservice.v2022_08_03_preview.models>`\n           * 2022-09-01: :mod:`v2022_09_01.models<azure.mgmt.containerservice.v2022_09_01.models>`\n           * 2022-09-02-preview: :mod:`v2022_09_02_preview.models<azure.mgmt.containerservice.v2022_09_02_preview.models>`\n           * 2022-10-02-preview: :mod:`v2022_10_02_preview.models<azure.mgmt.containerservice.v2022_10_02_preview.models>`\n           * 2022-11-01: :mod:`v2022_11_01.models<azure.mgmt.containerservice.v2022_11_01.models>`\n           * 2022-11-02-preview: :mod:`v2022_11_02_preview.models<azure.mgmt.containerservice.v2022_11_02_preview.models>`\n           * 2023-01-01: :mod:`v2023_01_01.models<azure.mgmt.containerservice.v2023_01_01.models>`\n           * 2023-01-02-preview: :mod:`v2023_01_02_preview.models<azure.mgmt.containerservice.v2023_01_02_preview.models>`\n           * 2023-02-01: :mod:`v2023_02_01.models<azure.mgmt.containerservice.v2023_02_01.models>`\n           * 2023-02-02-preview: :mod:`v2023_02_02_preview.models<azure.mgmt.containerservice.v2023_02_02_preview.models>`\n           * 2023-03-01: :mod:`v2023_03_01.models<azure.mgmt.containerservice.v2023_03_01.models>`\n           * 2023-03-02-preview: :mod:`v2023_03_02_preview.models<azure.mgmt.containerservice.v2023_03_02_preview.models>`\n           * 2023-04-01: :mod:`v2023_04_01.models<azure.mgmt.containerservice.v2023_04_01.models>`\n           * 2023-04-02-preview: :mod:`v2023_04_02_preview.models<azure.mgmt.containerservice.v2023_04_02_preview.models>`\n           * 2023-05-01: :mod:`v2023_05_01.models<azure.mgmt.containerservice.v2023_05_01.models>`\n           * 2023-05-02-preview: :mod:`v2023_05_02_preview.models<azure.mgmt.containerservice.v2023_05_02_preview.models>`\n           * 2023-06-01: :mod:`v2023_06_01.models<azure.mgmt.containerservice.v2023_06_01.models>`\n           * 2023-06-02-preview: :mod:`v2023_06_02_preview.models<azure.mgmt.containerservice.v2023_06_02_preview.models>`\n           * 2023-07-01: :mod:`v2023_07_01.models<azure.mgmt.containerservice.v2023_07_01.models>`\n           * 2023-07-02-preview: :mod:`v2023_07_02_preview.models<azure.mgmt.containerservice.v2023_07_02_preview.models>`\n           * 2023-08-01: :mod:`v2023_08_01.models<azure.mgmt.containerservice.v2023_08_01.models>`\n        '
        if api_version == '2017-07-01':
            from .v2017_07_01 import models
            return models
        elif api_version == '2018-03-31':
            from .v2018_03_31 import models
            return models
        elif api_version == '2018-08-01-preview':
            from .v2018_08_01_preview import models
            return models
        elif api_version == '2018-09-30-preview':
            from .v2018_09_30_preview import models
            return models
        elif api_version == '2019-02-01':
            from .v2019_02_01 import models
            return models
        elif api_version == '2019-04-01':
            from .v2019_04_01 import models
            return models
        elif api_version == '2019-04-30':
            from .v2019_04_30 import models
            return models
        elif api_version == '2019-06-01':
            from .v2019_06_01 import models
            return models
        elif api_version == '2019-08-01':
            from .v2019_08_01 import models
            return models
        elif api_version == '2019-09-30-preview':
            from .v2019_09_30_preview import models
            return models
        elif api_version == '2019-10-01':
            from .v2019_10_01 import models
            return models
        elif api_version == '2019-10-27-preview':
            from .v2019_10_27_preview import models
            return models
        elif api_version == '2019-11-01':
            from .v2019_11_01 import models
            return models
        elif api_version == '2020-01-01':
            from .v2020_01_01 import models
            return models
        elif api_version == '2020-02-01':
            from .v2020_02_01 import models
            return models
        elif api_version == '2020-03-01':
            from .v2020_03_01 import models
            return models
        elif api_version == '2020-04-01':
            from .v2020_04_01 import models
            return models
        elif api_version == '2020-06-01':
            from .v2020_06_01 import models
            return models
        elif api_version == '2020-07-01':
            from .v2020_07_01 import models
            return models
        elif api_version == '2020-09-01':
            from .v2020_09_01 import models
            return models
        elif api_version == '2020-11-01':
            from .v2020_11_01 import models
            return models
        elif api_version == '2020-12-01':
            from .v2020_12_01 import models
            return models
        elif api_version == '2021-02-01':
            from .v2021_02_01 import models
            return models
        elif api_version == '2021-03-01':
            from .v2021_03_01 import models
            return models
        elif api_version == '2021-05-01':
            from .v2021_05_01 import models
            return models
        elif api_version == '2021-07-01':
            from .v2021_07_01 import models
            return models
        elif api_version == '2021-08-01':
            from .v2021_08_01 import models
            return models
        elif api_version == '2021-09-01':
            from .v2021_09_01 import models
            return models
        elif api_version == '2021-10-01':
            from .v2021_10_01 import models
            return models
        elif api_version == '2021-11-01-preview':
            from .v2021_11_01_preview import models
            return models
        elif api_version == '2022-01-01':
            from .v2022_01_01 import models
            return models
        elif api_version == '2022-01-02-preview':
            from .v2022_01_02_preview import models
            return models
        elif api_version == '2022-02-01':
            from .v2022_02_01 import models
            return models
        elif api_version == '2022-02-02-preview':
            from .v2022_02_02_preview import models
            return models
        elif api_version == '2022-03-01':
            from .v2022_03_01 import models
            return models
        elif api_version == '2022-03-02-preview':
            from .v2022_03_02_preview import models
            return models
        elif api_version == '2022-04-01':
            from .v2022_04_01 import models
            return models
        elif api_version == '2022-04-02-preview':
            from .v2022_04_02_preview import models
            return models
        elif api_version == '2022-05-02-preview':
            from .v2022_05_02_preview import models
            return models
        elif api_version == '2022-06-01':
            from .v2022_06_01 import models
            return models
        elif api_version == '2022-06-02-preview':
            from .v2022_06_02_preview import models
            return models
        elif api_version == '2022-07-01':
            from .v2022_07_01 import models
            return models
        elif api_version == '2022-07-02-preview':
            from .v2022_07_02_preview import models
            return models
        elif api_version == '2022-08-02-preview':
            from .v2022_08_02_preview import models
            return models
        elif api_version == '2022-08-03-preview':
            from .v2022_08_03_preview import models
            return models
        elif api_version == '2022-09-01':
            from .v2022_09_01 import models
            return models
        elif api_version == '2022-09-02-preview':
            from .v2022_09_02_preview import models
            return models
        elif api_version == '2022-10-02-preview':
            from .v2022_10_02_preview import models
            return models
        elif api_version == '2022-11-01':
            from .v2022_11_01 import models
            return models
        elif api_version == '2022-11-02-preview':
            from .v2022_11_02_preview import models
            return models
        elif api_version == '2023-01-01':
            from .v2023_01_01 import models
            return models
        elif api_version == '2023-01-02-preview':
            from .v2023_01_02_preview import models
            return models
        elif api_version == '2023-02-01':
            from .v2023_02_01 import models
            return models
        elif api_version == '2023-02-02-preview':
            from .v2023_02_02_preview import models
            return models
        elif api_version == '2023-03-01':
            from .v2023_03_01 import models
            return models
        elif api_version == '2023-03-02-preview':
            from .v2023_03_02_preview import models
            return models
        elif api_version == '2023-04-01':
            from .v2023_04_01 import models
            return models
        elif api_version == '2023-04-02-preview':
            from .v2023_04_02_preview import models
            return models
        elif api_version == '2023-05-01':
            from .v2023_05_01 import models
            return models
        elif api_version == '2023-05-02-preview':
            from .v2023_05_02_preview import models
            return models
        elif api_version == '2023-06-01':
            from .v2023_06_01 import models
            return models
        elif api_version == '2023-06-02-preview':
            from .v2023_06_02_preview import models
            return models
        elif api_version == '2023-07-01':
            from .v2023_07_01 import models
            return models
        elif api_version == '2023-07-02-preview':
            from .v2023_07_02_preview import models
            return models
        elif api_version == '2023-08-01':
            from .v2023_08_01 import models
            return models
        raise ValueError('API version {} is not available'.format(api_version))

    @property
    def agent_pools(self):
        if False:
            while True:
                i = 10
        'Instance depends on the API version:\n\n           * 2019-02-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2019_02_01.operations.AgentPoolsOperations>`\n           * 2019-04-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2019_04_01.operations.AgentPoolsOperations>`\n           * 2019-06-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2019_06_01.operations.AgentPoolsOperations>`\n           * 2019-08-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2019_08_01.operations.AgentPoolsOperations>`\n           * 2019-10-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2019_10_01.operations.AgentPoolsOperations>`\n           * 2019-11-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2019_11_01.operations.AgentPoolsOperations>`\n           * 2020-01-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2020_01_01.operations.AgentPoolsOperations>`\n           * 2020-02-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2020_02_01.operations.AgentPoolsOperations>`\n           * 2020-03-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2020_03_01.operations.AgentPoolsOperations>`\n           * 2020-04-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2020_04_01.operations.AgentPoolsOperations>`\n           * 2020-06-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2020_06_01.operations.AgentPoolsOperations>`\n           * 2020-07-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2020_07_01.operations.AgentPoolsOperations>`\n           * 2020-09-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2020_09_01.operations.AgentPoolsOperations>`\n           * 2020-11-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2020_11_01.operations.AgentPoolsOperations>`\n           * 2020-12-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2020_12_01.operations.AgentPoolsOperations>`\n           * 2021-02-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2021_02_01.operations.AgentPoolsOperations>`\n           * 2021-03-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2021_03_01.operations.AgentPoolsOperations>`\n           * 2021-05-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2021_05_01.operations.AgentPoolsOperations>`\n           * 2021-07-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2021_07_01.operations.AgentPoolsOperations>`\n           * 2021-08-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2021_08_01.operations.AgentPoolsOperations>`\n           * 2021-09-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2021_09_01.operations.AgentPoolsOperations>`\n           * 2021-10-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2021_10_01.operations.AgentPoolsOperations>`\n           * 2021-11-01-preview: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2021_11_01_preview.operations.AgentPoolsOperations>`\n           * 2022-01-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2022_01_01.operations.AgentPoolsOperations>`\n           * 2022-01-02-preview: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2022_01_02_preview.operations.AgentPoolsOperations>`\n           * 2022-02-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2022_02_01.operations.AgentPoolsOperations>`\n           * 2022-02-02-preview: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2022_02_02_preview.operations.AgentPoolsOperations>`\n           * 2022-03-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2022_03_01.operations.AgentPoolsOperations>`\n           * 2022-03-02-preview: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2022_03_02_preview.operations.AgentPoolsOperations>`\n           * 2022-04-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2022_04_01.operations.AgentPoolsOperations>`\n           * 2022-04-02-preview: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2022_04_02_preview.operations.AgentPoolsOperations>`\n           * 2022-05-02-preview: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2022_05_02_preview.operations.AgentPoolsOperations>`\n           * 2022-06-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2022_06_01.operations.AgentPoolsOperations>`\n           * 2022-06-02-preview: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2022_06_02_preview.operations.AgentPoolsOperations>`\n           * 2022-07-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2022_07_01.operations.AgentPoolsOperations>`\n           * 2022-07-02-preview: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2022_07_02_preview.operations.AgentPoolsOperations>`\n           * 2022-08-02-preview: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2022_08_02_preview.operations.AgentPoolsOperations>`\n           * 2022-08-03-preview: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2022_08_03_preview.operations.AgentPoolsOperations>`\n           * 2022-09-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2022_09_01.operations.AgentPoolsOperations>`\n           * 2022-09-02-preview: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2022_09_02_preview.operations.AgentPoolsOperations>`\n           * 2022-10-02-preview: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2022_10_02_preview.operations.AgentPoolsOperations>`\n           * 2022-11-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2022_11_01.operations.AgentPoolsOperations>`\n           * 2022-11-02-preview: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2022_11_02_preview.operations.AgentPoolsOperations>`\n           * 2023-01-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2023_01_01.operations.AgentPoolsOperations>`\n           * 2023-01-02-preview: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2023_01_02_preview.operations.AgentPoolsOperations>`\n           * 2023-02-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2023_02_01.operations.AgentPoolsOperations>`\n           * 2023-02-02-preview: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2023_02_02_preview.operations.AgentPoolsOperations>`\n           * 2023-03-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2023_03_01.operations.AgentPoolsOperations>`\n           * 2023-03-02-preview: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2023_03_02_preview.operations.AgentPoolsOperations>`\n           * 2023-04-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2023_04_01.operations.AgentPoolsOperations>`\n           * 2023-04-02-preview: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2023_04_02_preview.operations.AgentPoolsOperations>`\n           * 2023-05-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2023_05_01.operations.AgentPoolsOperations>`\n           * 2023-05-02-preview: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2023_05_02_preview.operations.AgentPoolsOperations>`\n           * 2023-06-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2023_06_01.operations.AgentPoolsOperations>`\n           * 2023-06-02-preview: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2023_06_02_preview.operations.AgentPoolsOperations>`\n           * 2023-07-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2023_07_01.operations.AgentPoolsOperations>`\n           * 2023-07-02-preview: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2023_07_02_preview.operations.AgentPoolsOperations>`\n           * 2023-08-01: :class:`AgentPoolsOperations<azure.mgmt.containerservice.v2023_08_01.operations.AgentPoolsOperations>`\n        '
        api_version = self._get_api_version('agent_pools')
        if api_version == '2019-02-01':
            from .v2019_02_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2019-04-01':
            from .v2019_04_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2019-06-01':
            from .v2019_06_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2019-10-01':
            from .v2019_10_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2019-11-01':
            from .v2019_11_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2020-01-01':
            from .v2020_01_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2020-02-01':
            from .v2020_02_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2020-03-01':
            from .v2020_03_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2020-04-01':
            from .v2020_04_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2020-07-01':
            from .v2020_07_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2020-11-01':
            from .v2020_11_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2021-02-01':
            from .v2021_02_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2021-03-01':
            from .v2021_03_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2021-05-01':
            from .v2021_05_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2021-07-01':
            from .v2021_07_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2021-08-01':
            from .v2021_08_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2021-09-01':
            from .v2021_09_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2021-10-01':
            from .v2021_10_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2021-11-01-preview':
            from .v2021_11_01_preview.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2022-01-01':
            from .v2022_01_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2022-01-02-preview':
            from .v2022_01_02_preview.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2022-02-01':
            from .v2022_02_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2022-02-02-preview':
            from .v2022_02_02_preview.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2022-03-01':
            from .v2022_03_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2022-03-02-preview':
            from .v2022_03_02_preview.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2022-04-01':
            from .v2022_04_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2022-04-02-preview':
            from .v2022_04_02_preview.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2022-05-02-preview':
            from .v2022_05_02_preview.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2022-06-01':
            from .v2022_06_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2022-06-02-preview':
            from .v2022_06_02_preview.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2022-07-01':
            from .v2022_07_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2022-07-02-preview':
            from .v2022_07_02_preview.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2022-08-02-preview':
            from .v2022_08_02_preview.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2022-08-03-preview':
            from .v2022_08_03_preview.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2022-09-02-preview':
            from .v2022_09_02_preview.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2022-10-02-preview':
            from .v2022_10_02_preview.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2022-11-01':
            from .v2022_11_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2022-11-02-preview':
            from .v2022_11_02_preview.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2023-01-01':
            from .v2023_01_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2023-01-02-preview':
            from .v2023_01_02_preview.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2023-02-01':
            from .v2023_02_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2023-02-02-preview':
            from .v2023_02_02_preview.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2023-03-01':
            from .v2023_03_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2023-03-02-preview':
            from .v2023_03_02_preview.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2023-04-01':
            from .v2023_04_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2023-04-02-preview':
            from .v2023_04_02_preview.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2023-05-01':
            from .v2023_05_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2023-05-02-preview':
            from .v2023_05_02_preview.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2023-06-01':
            from .v2023_06_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2023-06-02-preview':
            from .v2023_06_02_preview.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2023-07-01':
            from .v2023_07_01.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2023-07-02-preview':
            from .v2023_07_02_preview.operations import AgentPoolsOperations as OperationClass
        elif api_version == '2023-08-01':
            from .v2023_08_01.operations import AgentPoolsOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'agent_pools'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)), api_version)

    @property
    def container_services(self):
        if False:
            i = 10
            return i + 15
        'Instance depends on the API version:\n\n           * 2017-07-01: :class:`ContainerServicesOperations<azure.mgmt.containerservice.v2017_07_01.operations.ContainerServicesOperations>`\n           * 2019-04-01: :class:`ContainerServicesOperations<azure.mgmt.containerservice.v2019_04_01.operations.ContainerServicesOperations>`\n        '
        api_version = self._get_api_version('container_services')
        if api_version == '2017-07-01':
            from .v2017_07_01.operations import ContainerServicesOperations as OperationClass
        elif api_version == '2019-04-01':
            from .v2019_04_01.operations import ContainerServicesOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'container_services'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)), api_version)

    @property
    def fleet_members(self):
        if False:
            print('Hello World!')
        'Instance depends on the API version:\n\n           * 2022-06-02-preview: :class:`FleetMembersOperations<azure.mgmt.containerservice.v2022_06_02_preview.operations.FleetMembersOperations>`\n           * 2022-07-02-preview: :class:`FleetMembersOperations<azure.mgmt.containerservice.v2022_07_02_preview.operations.FleetMembersOperations>`\n           * 2022-09-02-preview: :class:`FleetMembersOperations<azure.mgmt.containerservice.v2022_09_02_preview.operations.FleetMembersOperations>`\n        '
        api_version = self._get_api_version('fleet_members')
        if api_version == '2022-06-02-preview':
            from .v2022_06_02_preview.operations import FleetMembersOperations as OperationClass
        elif api_version == '2022-07-02-preview':
            from .v2022_07_02_preview.operations import FleetMembersOperations as OperationClass
        elif api_version == '2022-09-02-preview':
            from .v2022_09_02_preview.operations import FleetMembersOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'fleet_members'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)), api_version)

    @property
    def fleets(self):
        if False:
            return 10
        'Instance depends on the API version:\n\n           * 2022-06-02-preview: :class:`FleetsOperations<azure.mgmt.containerservice.v2022_06_02_preview.operations.FleetsOperations>`\n           * 2022-07-02-preview: :class:`FleetsOperations<azure.mgmt.containerservice.v2022_07_02_preview.operations.FleetsOperations>`\n           * 2022-09-02-preview: :class:`FleetsOperations<azure.mgmt.containerservice.v2022_09_02_preview.operations.FleetsOperations>`\n        '
        api_version = self._get_api_version('fleets')
        if api_version == '2022-06-02-preview':
            from .v2022_06_02_preview.operations import FleetsOperations as OperationClass
        elif api_version == '2022-07-02-preview':
            from .v2022_07_02_preview.operations import FleetsOperations as OperationClass
        elif api_version == '2022-09-02-preview':
            from .v2022_09_02_preview.operations import FleetsOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'fleets'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)), api_version)

    @property
    def machines(self):
        if False:
            return 10
        'Instance depends on the API version:\n\n           * 2023-07-02-preview: :class:`MachinesOperations<azure.mgmt.containerservice.v2023_07_02_preview.operations.MachinesOperations>`\n        '
        api_version = self._get_api_version('machines')
        if api_version == '2023-07-02-preview':
            from .v2023_07_02_preview.operations import MachinesOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'machines'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)), api_version)

    @property
    def maintenance_configurations(self):
        if False:
            for i in range(10):
                print('nop')
        'Instance depends on the API version:\n\n           * 2020-12-01: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2020_12_01.operations.MaintenanceConfigurationsOperations>`\n           * 2021-02-01: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2021_02_01.operations.MaintenanceConfigurationsOperations>`\n           * 2021-03-01: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2021_03_01.operations.MaintenanceConfigurationsOperations>`\n           * 2021-05-01: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2021_05_01.operations.MaintenanceConfigurationsOperations>`\n           * 2021-07-01: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2021_07_01.operations.MaintenanceConfigurationsOperations>`\n           * 2021-08-01: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2021_08_01.operations.MaintenanceConfigurationsOperations>`\n           * 2021-09-01: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2021_09_01.operations.MaintenanceConfigurationsOperations>`\n           * 2021-10-01: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2021_10_01.operations.MaintenanceConfigurationsOperations>`\n           * 2021-11-01-preview: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2021_11_01_preview.operations.MaintenanceConfigurationsOperations>`\n           * 2022-01-01: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2022_01_01.operations.MaintenanceConfigurationsOperations>`\n           * 2022-01-02-preview: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2022_01_02_preview.operations.MaintenanceConfigurationsOperations>`\n           * 2022-02-01: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2022_02_01.operations.MaintenanceConfigurationsOperations>`\n           * 2022-02-02-preview: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2022_02_02_preview.operations.MaintenanceConfigurationsOperations>`\n           * 2022-03-01: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2022_03_01.operations.MaintenanceConfigurationsOperations>`\n           * 2022-03-02-preview: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2022_03_02_preview.operations.MaintenanceConfigurationsOperations>`\n           * 2022-04-01: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2022_04_01.operations.MaintenanceConfigurationsOperations>`\n           * 2022-04-02-preview: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2022_04_02_preview.operations.MaintenanceConfigurationsOperations>`\n           * 2022-05-02-preview: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2022_05_02_preview.operations.MaintenanceConfigurationsOperations>`\n           * 2022-06-01: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2022_06_01.operations.MaintenanceConfigurationsOperations>`\n           * 2022-06-02-preview: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2022_06_02_preview.operations.MaintenanceConfigurationsOperations>`\n           * 2022-07-01: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2022_07_01.operations.MaintenanceConfigurationsOperations>`\n           * 2022-07-02-preview: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2022_07_02_preview.operations.MaintenanceConfigurationsOperations>`\n           * 2022-08-02-preview: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2022_08_02_preview.operations.MaintenanceConfigurationsOperations>`\n           * 2022-08-03-preview: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2022_08_03_preview.operations.MaintenanceConfigurationsOperations>`\n           * 2022-09-01: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2022_09_01.operations.MaintenanceConfigurationsOperations>`\n           * 2022-09-02-preview: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2022_09_02_preview.operations.MaintenanceConfigurationsOperations>`\n           * 2022-10-02-preview: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2022_10_02_preview.operations.MaintenanceConfigurationsOperations>`\n           * 2022-11-01: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2022_11_01.operations.MaintenanceConfigurationsOperations>`\n           * 2022-11-02-preview: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2022_11_02_preview.operations.MaintenanceConfigurationsOperations>`\n           * 2023-01-01: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2023_01_01.operations.MaintenanceConfigurationsOperations>`\n           * 2023-01-02-preview: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2023_01_02_preview.operations.MaintenanceConfigurationsOperations>`\n           * 2023-02-01: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2023_02_01.operations.MaintenanceConfigurationsOperations>`\n           * 2023-02-02-preview: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2023_02_02_preview.operations.MaintenanceConfigurationsOperations>`\n           * 2023-03-01: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2023_03_01.operations.MaintenanceConfigurationsOperations>`\n           * 2023-03-02-preview: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2023_03_02_preview.operations.MaintenanceConfigurationsOperations>`\n           * 2023-04-01: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2023_04_01.operations.MaintenanceConfigurationsOperations>`\n           * 2023-04-02-preview: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2023_04_02_preview.operations.MaintenanceConfigurationsOperations>`\n           * 2023-05-01: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2023_05_01.operations.MaintenanceConfigurationsOperations>`\n           * 2023-05-02-preview: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2023_05_02_preview.operations.MaintenanceConfigurationsOperations>`\n           * 2023-06-01: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2023_06_01.operations.MaintenanceConfigurationsOperations>`\n           * 2023-06-02-preview: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2023_06_02_preview.operations.MaintenanceConfigurationsOperations>`\n           * 2023-07-01: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2023_07_01.operations.MaintenanceConfigurationsOperations>`\n           * 2023-07-02-preview: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2023_07_02_preview.operations.MaintenanceConfigurationsOperations>`\n           * 2023-08-01: :class:`MaintenanceConfigurationsOperations<azure.mgmt.containerservice.v2023_08_01.operations.MaintenanceConfigurationsOperations>`\n        '
        api_version = self._get_api_version('maintenance_configurations')
        if api_version == '2020-12-01':
            from .v2020_12_01.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2021-02-01':
            from .v2021_02_01.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2021-03-01':
            from .v2021_03_01.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2021-05-01':
            from .v2021_05_01.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2021-07-01':
            from .v2021_07_01.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2021-08-01':
            from .v2021_08_01.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2021-09-01':
            from .v2021_09_01.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2021-10-01':
            from .v2021_10_01.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2021-11-01-preview':
            from .v2021_11_01_preview.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2022-01-01':
            from .v2022_01_01.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2022-01-02-preview':
            from .v2022_01_02_preview.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2022-02-01':
            from .v2022_02_01.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2022-02-02-preview':
            from .v2022_02_02_preview.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2022-03-01':
            from .v2022_03_01.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2022-03-02-preview':
            from .v2022_03_02_preview.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2022-04-01':
            from .v2022_04_01.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2022-04-02-preview':
            from .v2022_04_02_preview.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2022-05-02-preview':
            from .v2022_05_02_preview.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2022-06-01':
            from .v2022_06_01.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2022-06-02-preview':
            from .v2022_06_02_preview.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2022-07-01':
            from .v2022_07_01.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2022-07-02-preview':
            from .v2022_07_02_preview.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2022-08-02-preview':
            from .v2022_08_02_preview.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2022-08-03-preview':
            from .v2022_08_03_preview.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2022-09-02-preview':
            from .v2022_09_02_preview.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2022-10-02-preview':
            from .v2022_10_02_preview.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2022-11-01':
            from .v2022_11_01.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2022-11-02-preview':
            from .v2022_11_02_preview.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2023-01-01':
            from .v2023_01_01.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2023-01-02-preview':
            from .v2023_01_02_preview.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2023-02-01':
            from .v2023_02_01.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2023-02-02-preview':
            from .v2023_02_02_preview.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2023-03-01':
            from .v2023_03_01.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2023-03-02-preview':
            from .v2023_03_02_preview.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2023-04-01':
            from .v2023_04_01.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2023-04-02-preview':
            from .v2023_04_02_preview.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2023-05-01':
            from .v2023_05_01.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2023-05-02-preview':
            from .v2023_05_02_preview.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2023-06-01':
            from .v2023_06_01.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2023-06-02-preview':
            from .v2023_06_02_preview.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2023-07-01':
            from .v2023_07_01.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2023-07-02-preview':
            from .v2023_07_02_preview.operations import MaintenanceConfigurationsOperations as OperationClass
        elif api_version == '2023-08-01':
            from .v2023_08_01.operations import MaintenanceConfigurationsOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'maintenance_configurations'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)), api_version)

    @property
    def managed_cluster_snapshots(self):
        if False:
            i = 10
            return i + 15
        'Instance depends on the API version:\n\n           * 2022-02-02-preview: :class:`ManagedClusterSnapshotsOperations<azure.mgmt.containerservice.v2022_02_02_preview.operations.ManagedClusterSnapshotsOperations>`\n           * 2022-03-02-preview: :class:`ManagedClusterSnapshotsOperations<azure.mgmt.containerservice.v2022_03_02_preview.operations.ManagedClusterSnapshotsOperations>`\n           * 2022-04-02-preview: :class:`ManagedClusterSnapshotsOperations<azure.mgmt.containerservice.v2022_04_02_preview.operations.ManagedClusterSnapshotsOperations>`\n           * 2022-05-02-preview: :class:`ManagedClusterSnapshotsOperations<azure.mgmt.containerservice.v2022_05_02_preview.operations.ManagedClusterSnapshotsOperations>`\n           * 2022-06-02-preview: :class:`ManagedClusterSnapshotsOperations<azure.mgmt.containerservice.v2022_06_02_preview.operations.ManagedClusterSnapshotsOperations>`\n           * 2022-07-02-preview: :class:`ManagedClusterSnapshotsOperations<azure.mgmt.containerservice.v2022_07_02_preview.operations.ManagedClusterSnapshotsOperations>`\n           * 2022-08-02-preview: :class:`ManagedClusterSnapshotsOperations<azure.mgmt.containerservice.v2022_08_02_preview.operations.ManagedClusterSnapshotsOperations>`\n           * 2022-08-03-preview: :class:`ManagedClusterSnapshotsOperations<azure.mgmt.containerservice.v2022_08_03_preview.operations.ManagedClusterSnapshotsOperations>`\n           * 2022-09-02-preview: :class:`ManagedClusterSnapshotsOperations<azure.mgmt.containerservice.v2022_09_02_preview.operations.ManagedClusterSnapshotsOperations>`\n           * 2022-10-02-preview: :class:`ManagedClusterSnapshotsOperations<azure.mgmt.containerservice.v2022_10_02_preview.operations.ManagedClusterSnapshotsOperations>`\n           * 2022-11-02-preview: :class:`ManagedClusterSnapshotsOperations<azure.mgmt.containerservice.v2022_11_02_preview.operations.ManagedClusterSnapshotsOperations>`\n           * 2023-01-02-preview: :class:`ManagedClusterSnapshotsOperations<azure.mgmt.containerservice.v2023_01_02_preview.operations.ManagedClusterSnapshotsOperations>`\n           * 2023-02-02-preview: :class:`ManagedClusterSnapshotsOperations<azure.mgmt.containerservice.v2023_02_02_preview.operations.ManagedClusterSnapshotsOperations>`\n           * 2023-03-02-preview: :class:`ManagedClusterSnapshotsOperations<azure.mgmt.containerservice.v2023_03_02_preview.operations.ManagedClusterSnapshotsOperations>`\n           * 2023-04-02-preview: :class:`ManagedClusterSnapshotsOperations<azure.mgmt.containerservice.v2023_04_02_preview.operations.ManagedClusterSnapshotsOperations>`\n           * 2023-05-02-preview: :class:`ManagedClusterSnapshotsOperations<azure.mgmt.containerservice.v2023_05_02_preview.operations.ManagedClusterSnapshotsOperations>`\n           * 2023-06-02-preview: :class:`ManagedClusterSnapshotsOperations<azure.mgmt.containerservice.v2023_06_02_preview.operations.ManagedClusterSnapshotsOperations>`\n           * 2023-07-02-preview: :class:`ManagedClusterSnapshotsOperations<azure.mgmt.containerservice.v2023_07_02_preview.operations.ManagedClusterSnapshotsOperations>`\n        '
        api_version = self._get_api_version('managed_cluster_snapshots')
        if api_version == '2022-02-02-preview':
            from .v2022_02_02_preview.operations import ManagedClusterSnapshotsOperations as OperationClass
        elif api_version == '2022-03-02-preview':
            from .v2022_03_02_preview.operations import ManagedClusterSnapshotsOperations as OperationClass
        elif api_version == '2022-04-02-preview':
            from .v2022_04_02_preview.operations import ManagedClusterSnapshotsOperations as OperationClass
        elif api_version == '2022-05-02-preview':
            from .v2022_05_02_preview.operations import ManagedClusterSnapshotsOperations as OperationClass
        elif api_version == '2022-06-02-preview':
            from .v2022_06_02_preview.operations import ManagedClusterSnapshotsOperations as OperationClass
        elif api_version == '2022-07-02-preview':
            from .v2022_07_02_preview.operations import ManagedClusterSnapshotsOperations as OperationClass
        elif api_version == '2022-08-02-preview':
            from .v2022_08_02_preview.operations import ManagedClusterSnapshotsOperations as OperationClass
        elif api_version == '2022-08-03-preview':
            from .v2022_08_03_preview.operations import ManagedClusterSnapshotsOperations as OperationClass
        elif api_version == '2022-09-02-preview':
            from .v2022_09_02_preview.operations import ManagedClusterSnapshotsOperations as OperationClass
        elif api_version == '2022-10-02-preview':
            from .v2022_10_02_preview.operations import ManagedClusterSnapshotsOperations as OperationClass
        elif api_version == '2022-11-02-preview':
            from .v2022_11_02_preview.operations import ManagedClusterSnapshotsOperations as OperationClass
        elif api_version == '2023-01-02-preview':
            from .v2023_01_02_preview.operations import ManagedClusterSnapshotsOperations as OperationClass
        elif api_version == '2023-02-02-preview':
            from .v2023_02_02_preview.operations import ManagedClusterSnapshotsOperations as OperationClass
        elif api_version == '2023-03-02-preview':
            from .v2023_03_02_preview.operations import ManagedClusterSnapshotsOperations as OperationClass
        elif api_version == '2023-04-02-preview':
            from .v2023_04_02_preview.operations import ManagedClusterSnapshotsOperations as OperationClass
        elif api_version == '2023-05-02-preview':
            from .v2023_05_02_preview.operations import ManagedClusterSnapshotsOperations as OperationClass
        elif api_version == '2023-06-02-preview':
            from .v2023_06_02_preview.operations import ManagedClusterSnapshotsOperations as OperationClass
        elif api_version == '2023-07-02-preview':
            from .v2023_07_02_preview.operations import ManagedClusterSnapshotsOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'managed_cluster_snapshots'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)), api_version)

    @property
    def managed_clusters(self):
        if False:
            print('Hello World!')
        'Instance depends on the API version:\n\n           * 2018-03-31: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2018_03_31.operations.ManagedClustersOperations>`\n           * 2018-08-01-preview: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2018_08_01_preview.operations.ManagedClustersOperations>`\n           * 2019-02-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2019_02_01.operations.ManagedClustersOperations>`\n           * 2019-04-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2019_04_01.operations.ManagedClustersOperations>`\n           * 2019-06-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2019_06_01.operations.ManagedClustersOperations>`\n           * 2019-08-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2019_08_01.operations.ManagedClustersOperations>`\n           * 2019-10-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2019_10_01.operations.ManagedClustersOperations>`\n           * 2019-11-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2019_11_01.operations.ManagedClustersOperations>`\n           * 2020-01-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2020_01_01.operations.ManagedClustersOperations>`\n           * 2020-02-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2020_02_01.operations.ManagedClustersOperations>`\n           * 2020-03-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2020_03_01.operations.ManagedClustersOperations>`\n           * 2020-04-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2020_04_01.operations.ManagedClustersOperations>`\n           * 2020-06-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2020_06_01.operations.ManagedClustersOperations>`\n           * 2020-07-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2020_07_01.operations.ManagedClustersOperations>`\n           * 2020-09-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2020_09_01.operations.ManagedClustersOperations>`\n           * 2020-11-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2020_11_01.operations.ManagedClustersOperations>`\n           * 2020-12-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2020_12_01.operations.ManagedClustersOperations>`\n           * 2021-02-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2021_02_01.operations.ManagedClustersOperations>`\n           * 2021-03-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2021_03_01.operations.ManagedClustersOperations>`\n           * 2021-05-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2021_05_01.operations.ManagedClustersOperations>`\n           * 2021-07-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2021_07_01.operations.ManagedClustersOperations>`\n           * 2021-08-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2021_08_01.operations.ManagedClustersOperations>`\n           * 2021-09-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2021_09_01.operations.ManagedClustersOperations>`\n           * 2021-10-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2021_10_01.operations.ManagedClustersOperations>`\n           * 2021-11-01-preview: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2021_11_01_preview.operations.ManagedClustersOperations>`\n           * 2022-01-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2022_01_01.operations.ManagedClustersOperations>`\n           * 2022-01-02-preview: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2022_01_02_preview.operations.ManagedClustersOperations>`\n           * 2022-02-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2022_02_01.operations.ManagedClustersOperations>`\n           * 2022-02-02-preview: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2022_02_02_preview.operations.ManagedClustersOperations>`\n           * 2022-03-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2022_03_01.operations.ManagedClustersOperations>`\n           * 2022-03-02-preview: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2022_03_02_preview.operations.ManagedClustersOperations>`\n           * 2022-04-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2022_04_01.operations.ManagedClustersOperations>`\n           * 2022-04-02-preview: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2022_04_02_preview.operations.ManagedClustersOperations>`\n           * 2022-05-02-preview: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2022_05_02_preview.operations.ManagedClustersOperations>`\n           * 2022-06-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2022_06_01.operations.ManagedClustersOperations>`\n           * 2022-06-02-preview: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2022_06_02_preview.operations.ManagedClustersOperations>`\n           * 2022-07-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2022_07_01.operations.ManagedClustersOperations>`\n           * 2022-07-02-preview: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2022_07_02_preview.operations.ManagedClustersOperations>`\n           * 2022-08-02-preview: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2022_08_02_preview.operations.ManagedClustersOperations>`\n           * 2022-08-03-preview: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2022_08_03_preview.operations.ManagedClustersOperations>`\n           * 2022-09-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2022_09_01.operations.ManagedClustersOperations>`\n           * 2022-09-02-preview: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2022_09_02_preview.operations.ManagedClustersOperations>`\n           * 2022-10-02-preview: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2022_10_02_preview.operations.ManagedClustersOperations>`\n           * 2022-11-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2022_11_01.operations.ManagedClustersOperations>`\n           * 2022-11-02-preview: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2022_11_02_preview.operations.ManagedClustersOperations>`\n           * 2023-01-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2023_01_01.operations.ManagedClustersOperations>`\n           * 2023-01-02-preview: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2023_01_02_preview.operations.ManagedClustersOperations>`\n           * 2023-02-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2023_02_01.operations.ManagedClustersOperations>`\n           * 2023-02-02-preview: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2023_02_02_preview.operations.ManagedClustersOperations>`\n           * 2023-03-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2023_03_01.operations.ManagedClustersOperations>`\n           * 2023-03-02-preview: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2023_03_02_preview.operations.ManagedClustersOperations>`\n           * 2023-04-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2023_04_01.operations.ManagedClustersOperations>`\n           * 2023-04-02-preview: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2023_04_02_preview.operations.ManagedClustersOperations>`\n           * 2023-05-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2023_05_01.operations.ManagedClustersOperations>`\n           * 2023-05-02-preview: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2023_05_02_preview.operations.ManagedClustersOperations>`\n           * 2023-06-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2023_06_01.operations.ManagedClustersOperations>`\n           * 2023-06-02-preview: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2023_06_02_preview.operations.ManagedClustersOperations>`\n           * 2023-07-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2023_07_01.operations.ManagedClustersOperations>`\n           * 2023-07-02-preview: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2023_07_02_preview.operations.ManagedClustersOperations>`\n           * 2023-08-01: :class:`ManagedClustersOperations<azure.mgmt.containerservice.v2023_08_01.operations.ManagedClustersOperations>`\n        '
        api_version = self._get_api_version('managed_clusters')
        if api_version == '2018-03-31':
            from .v2018_03_31.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2018-08-01-preview':
            from .v2018_08_01_preview.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2019-02-01':
            from .v2019_02_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2019-04-01':
            from .v2019_04_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2019-06-01':
            from .v2019_06_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2019-10-01':
            from .v2019_10_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2019-11-01':
            from .v2019_11_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2020-01-01':
            from .v2020_01_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2020-02-01':
            from .v2020_02_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2020-03-01':
            from .v2020_03_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2020-04-01':
            from .v2020_04_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2020-07-01':
            from .v2020_07_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2020-11-01':
            from .v2020_11_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2021-02-01':
            from .v2021_02_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2021-03-01':
            from .v2021_03_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2021-05-01':
            from .v2021_05_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2021-07-01':
            from .v2021_07_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2021-08-01':
            from .v2021_08_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2021-09-01':
            from .v2021_09_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2021-10-01':
            from .v2021_10_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2021-11-01-preview':
            from .v2021_11_01_preview.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2022-01-01':
            from .v2022_01_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2022-01-02-preview':
            from .v2022_01_02_preview.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2022-02-01':
            from .v2022_02_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2022-02-02-preview':
            from .v2022_02_02_preview.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2022-03-01':
            from .v2022_03_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2022-03-02-preview':
            from .v2022_03_02_preview.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2022-04-01':
            from .v2022_04_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2022-04-02-preview':
            from .v2022_04_02_preview.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2022-05-02-preview':
            from .v2022_05_02_preview.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2022-06-01':
            from .v2022_06_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2022-06-02-preview':
            from .v2022_06_02_preview.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2022-07-01':
            from .v2022_07_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2022-07-02-preview':
            from .v2022_07_02_preview.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2022-08-02-preview':
            from .v2022_08_02_preview.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2022-08-03-preview':
            from .v2022_08_03_preview.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2022-09-02-preview':
            from .v2022_09_02_preview.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2022-10-02-preview':
            from .v2022_10_02_preview.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2022-11-01':
            from .v2022_11_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2022-11-02-preview':
            from .v2022_11_02_preview.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2023-01-01':
            from .v2023_01_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2023-01-02-preview':
            from .v2023_01_02_preview.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2023-02-01':
            from .v2023_02_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2023-02-02-preview':
            from .v2023_02_02_preview.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2023-03-01':
            from .v2023_03_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2023-03-02-preview':
            from .v2023_03_02_preview.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2023-04-01':
            from .v2023_04_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2023-04-02-preview':
            from .v2023_04_02_preview.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2023-05-01':
            from .v2023_05_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2023-05-02-preview':
            from .v2023_05_02_preview.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2023-06-01':
            from .v2023_06_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2023-06-02-preview':
            from .v2023_06_02_preview.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2023-07-01':
            from .v2023_07_01.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2023-07-02-preview':
            from .v2023_07_02_preview.operations import ManagedClustersOperations as OperationClass
        elif api_version == '2023-08-01':
            from .v2023_08_01.operations import ManagedClustersOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'managed_clusters'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)), api_version)

    @property
    def open_shift_managed_clusters(self):
        if False:
            while True:
                i = 10
        'Instance depends on the API version:\n\n           * 2018-09-30-preview: :class:`OpenShiftManagedClustersOperations<azure.mgmt.containerservice.v2018_09_30_preview.operations.OpenShiftManagedClustersOperations>`\n           * 2019-04-30: :class:`OpenShiftManagedClustersOperations<azure.mgmt.containerservice.v2019_04_30.operations.OpenShiftManagedClustersOperations>`\n           * 2019-09-30-preview: :class:`OpenShiftManagedClustersOperations<azure.mgmt.containerservice.v2019_09_30_preview.operations.OpenShiftManagedClustersOperations>`\n           * 2019-10-27-preview: :class:`OpenShiftManagedClustersOperations<azure.mgmt.containerservice.v2019_10_27_preview.operations.OpenShiftManagedClustersOperations>`\n        '
        api_version = self._get_api_version('open_shift_managed_clusters')
        if api_version == '2018-09-30-preview':
            from .v2018_09_30_preview.operations import OpenShiftManagedClustersOperations as OperationClass
        elif api_version == '2019-04-30':
            from .v2019_04_30.operations import OpenShiftManagedClustersOperations as OperationClass
        elif api_version == '2019-09-30-preview':
            from .v2019_09_30_preview.operations import OpenShiftManagedClustersOperations as OperationClass
        elif api_version == '2019-10-27-preview':
            from .v2019_10_27_preview.operations import OpenShiftManagedClustersOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'open_shift_managed_clusters'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)), api_version)

    @property
    def operations(self):
        if False:
            i = 10
            return i + 15
        'Instance depends on the API version:\n\n           * 2018-03-31: :class:`Operations<azure.mgmt.containerservice.v2018_03_31.operations.Operations>`\n           * 2018-08-01-preview: :class:`Operations<azure.mgmt.containerservice.v2018_08_01_preview.operations.Operations>`\n           * 2019-02-01: :class:`Operations<azure.mgmt.containerservice.v2019_02_01.operations.Operations>`\n           * 2019-04-01: :class:`Operations<azure.mgmt.containerservice.v2019_04_01.operations.Operations>`\n           * 2019-06-01: :class:`Operations<azure.mgmt.containerservice.v2019_06_01.operations.Operations>`\n           * 2019-08-01: :class:`Operations<azure.mgmt.containerservice.v2019_08_01.operations.Operations>`\n           * 2019-10-01: :class:`Operations<azure.mgmt.containerservice.v2019_10_01.operations.Operations>`\n           * 2019-11-01: :class:`Operations<azure.mgmt.containerservice.v2019_11_01.operations.Operations>`\n           * 2020-01-01: :class:`Operations<azure.mgmt.containerservice.v2020_01_01.operations.Operations>`\n           * 2020-02-01: :class:`Operations<azure.mgmt.containerservice.v2020_02_01.operations.Operations>`\n           * 2020-03-01: :class:`Operations<azure.mgmt.containerservice.v2020_03_01.operations.Operations>`\n           * 2020-04-01: :class:`Operations<azure.mgmt.containerservice.v2020_04_01.operations.Operations>`\n           * 2020-06-01: :class:`Operations<azure.mgmt.containerservice.v2020_06_01.operations.Operations>`\n           * 2020-07-01: :class:`Operations<azure.mgmt.containerservice.v2020_07_01.operations.Operations>`\n           * 2020-09-01: :class:`Operations<azure.mgmt.containerservice.v2020_09_01.operations.Operations>`\n           * 2020-11-01: :class:`Operations<azure.mgmt.containerservice.v2020_11_01.operations.Operations>`\n           * 2020-12-01: :class:`Operations<azure.mgmt.containerservice.v2020_12_01.operations.Operations>`\n           * 2021-02-01: :class:`Operations<azure.mgmt.containerservice.v2021_02_01.operations.Operations>`\n           * 2021-03-01: :class:`Operations<azure.mgmt.containerservice.v2021_03_01.operations.Operations>`\n           * 2021-05-01: :class:`Operations<azure.mgmt.containerservice.v2021_05_01.operations.Operations>`\n           * 2021-07-01: :class:`Operations<azure.mgmt.containerservice.v2021_07_01.operations.Operations>`\n           * 2021-08-01: :class:`Operations<azure.mgmt.containerservice.v2021_08_01.operations.Operations>`\n           * 2021-09-01: :class:`Operations<azure.mgmt.containerservice.v2021_09_01.operations.Operations>`\n           * 2021-10-01: :class:`Operations<azure.mgmt.containerservice.v2021_10_01.operations.Operations>`\n           * 2021-11-01-preview: :class:`Operations<azure.mgmt.containerservice.v2021_11_01_preview.operations.Operations>`\n           * 2022-01-01: :class:`Operations<azure.mgmt.containerservice.v2022_01_01.operations.Operations>`\n           * 2022-01-02-preview: :class:`Operations<azure.mgmt.containerservice.v2022_01_02_preview.operations.Operations>`\n           * 2022-02-01: :class:`Operations<azure.mgmt.containerservice.v2022_02_01.operations.Operations>`\n           * 2022-02-02-preview: :class:`Operations<azure.mgmt.containerservice.v2022_02_02_preview.operations.Operations>`\n           * 2022-03-01: :class:`Operations<azure.mgmt.containerservice.v2022_03_01.operations.Operations>`\n           * 2022-03-02-preview: :class:`Operations<azure.mgmt.containerservice.v2022_03_02_preview.operations.Operations>`\n           * 2022-04-01: :class:`Operations<azure.mgmt.containerservice.v2022_04_01.operations.Operations>`\n           * 2022-04-02-preview: :class:`Operations<azure.mgmt.containerservice.v2022_04_02_preview.operations.Operations>`\n           * 2022-05-02-preview: :class:`Operations<azure.mgmt.containerservice.v2022_05_02_preview.operations.Operations>`\n           * 2022-06-01: :class:`Operations<azure.mgmt.containerservice.v2022_06_01.operations.Operations>`\n           * 2022-06-02-preview: :class:`Operations<azure.mgmt.containerservice.v2022_06_02_preview.operations.Operations>`\n           * 2022-07-01: :class:`Operations<azure.mgmt.containerservice.v2022_07_01.operations.Operations>`\n           * 2022-07-02-preview: :class:`Operations<azure.mgmt.containerservice.v2022_07_02_preview.operations.Operations>`\n           * 2022-08-02-preview: :class:`Operations<azure.mgmt.containerservice.v2022_08_02_preview.operations.Operations>`\n           * 2022-08-03-preview: :class:`Operations<azure.mgmt.containerservice.v2022_08_03_preview.operations.Operations>`\n           * 2022-09-01: :class:`Operations<azure.mgmt.containerservice.v2022_09_01.operations.Operations>`\n           * 2022-09-02-preview: :class:`Operations<azure.mgmt.containerservice.v2022_09_02_preview.operations.Operations>`\n           * 2022-10-02-preview: :class:`Operations<azure.mgmt.containerservice.v2022_10_02_preview.operations.Operations>`\n           * 2022-11-01: :class:`Operations<azure.mgmt.containerservice.v2022_11_01.operations.Operations>`\n           * 2022-11-02-preview: :class:`Operations<azure.mgmt.containerservice.v2022_11_02_preview.operations.Operations>`\n           * 2023-01-01: :class:`Operations<azure.mgmt.containerservice.v2023_01_01.operations.Operations>`\n           * 2023-01-02-preview: :class:`Operations<azure.mgmt.containerservice.v2023_01_02_preview.operations.Operations>`\n           * 2023-02-01: :class:`Operations<azure.mgmt.containerservice.v2023_02_01.operations.Operations>`\n           * 2023-02-02-preview: :class:`Operations<azure.mgmt.containerservice.v2023_02_02_preview.operations.Operations>`\n           * 2023-03-01: :class:`Operations<azure.mgmt.containerservice.v2023_03_01.operations.Operations>`\n           * 2023-03-02-preview: :class:`Operations<azure.mgmt.containerservice.v2023_03_02_preview.operations.Operations>`\n           * 2023-04-01: :class:`Operations<azure.mgmt.containerservice.v2023_04_01.operations.Operations>`\n           * 2023-04-02-preview: :class:`Operations<azure.mgmt.containerservice.v2023_04_02_preview.operations.Operations>`\n           * 2023-05-01: :class:`Operations<azure.mgmt.containerservice.v2023_05_01.operations.Operations>`\n           * 2023-05-02-preview: :class:`Operations<azure.mgmt.containerservice.v2023_05_02_preview.operations.Operations>`\n           * 2023-06-01: :class:`Operations<azure.mgmt.containerservice.v2023_06_01.operations.Operations>`\n           * 2023-06-02-preview: :class:`Operations<azure.mgmt.containerservice.v2023_06_02_preview.operations.Operations>`\n           * 2023-07-01: :class:`Operations<azure.mgmt.containerservice.v2023_07_01.operations.Operations>`\n           * 2023-07-02-preview: :class:`Operations<azure.mgmt.containerservice.v2023_07_02_preview.operations.Operations>`\n           * 2023-08-01: :class:`Operations<azure.mgmt.containerservice.v2023_08_01.operations.Operations>`\n        '
        api_version = self._get_api_version('operations')
        if api_version == '2018-03-31':
            from .v2018_03_31.operations import Operations as OperationClass
        elif api_version == '2018-08-01-preview':
            from .v2018_08_01_preview.operations import Operations as OperationClass
        elif api_version == '2019-02-01':
            from .v2019_02_01.operations import Operations as OperationClass
        elif api_version == '2019-04-01':
            from .v2019_04_01.operations import Operations as OperationClass
        elif api_version == '2019-06-01':
            from .v2019_06_01.operations import Operations as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import Operations as OperationClass
        elif api_version == '2019-10-01':
            from .v2019_10_01.operations import Operations as OperationClass
        elif api_version == '2019-11-01':
            from .v2019_11_01.operations import Operations as OperationClass
        elif api_version == '2020-01-01':
            from .v2020_01_01.operations import Operations as OperationClass
        elif api_version == '2020-02-01':
            from .v2020_02_01.operations import Operations as OperationClass
        elif api_version == '2020-03-01':
            from .v2020_03_01.operations import Operations as OperationClass
        elif api_version == '2020-04-01':
            from .v2020_04_01.operations import Operations as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import Operations as OperationClass
        elif api_version == '2020-07-01':
            from .v2020_07_01.operations import Operations as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import Operations as OperationClass
        elif api_version == '2020-11-01':
            from .v2020_11_01.operations import Operations as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import Operations as OperationClass
        elif api_version == '2021-02-01':
            from .v2021_02_01.operations import Operations as OperationClass
        elif api_version == '2021-03-01':
            from .v2021_03_01.operations import Operations as OperationClass
        elif api_version == '2021-05-01':
            from .v2021_05_01.operations import Operations as OperationClass
        elif api_version == '2021-07-01':
            from .v2021_07_01.operations import Operations as OperationClass
        elif api_version == '2021-08-01':
            from .v2021_08_01.operations import Operations as OperationClass
        elif api_version == '2021-09-01':
            from .v2021_09_01.operations import Operations as OperationClass
        elif api_version == '2021-10-01':
            from .v2021_10_01.operations import Operations as OperationClass
        elif api_version == '2021-11-01-preview':
            from .v2021_11_01_preview.operations import Operations as OperationClass
        elif api_version == '2022-01-01':
            from .v2022_01_01.operations import Operations as OperationClass
        elif api_version == '2022-01-02-preview':
            from .v2022_01_02_preview.operations import Operations as OperationClass
        elif api_version == '2022-02-01':
            from .v2022_02_01.operations import Operations as OperationClass
        elif api_version == '2022-02-02-preview':
            from .v2022_02_02_preview.operations import Operations as OperationClass
        elif api_version == '2022-03-01':
            from .v2022_03_01.operations import Operations as OperationClass
        elif api_version == '2022-03-02-preview':
            from .v2022_03_02_preview.operations import Operations as OperationClass
        elif api_version == '2022-04-01':
            from .v2022_04_01.operations import Operations as OperationClass
        elif api_version == '2022-04-02-preview':
            from .v2022_04_02_preview.operations import Operations as OperationClass
        elif api_version == '2022-05-02-preview':
            from .v2022_05_02_preview.operations import Operations as OperationClass
        elif api_version == '2022-06-01':
            from .v2022_06_01.operations import Operations as OperationClass
        elif api_version == '2022-06-02-preview':
            from .v2022_06_02_preview.operations import Operations as OperationClass
        elif api_version == '2022-07-01':
            from .v2022_07_01.operations import Operations as OperationClass
        elif api_version == '2022-07-02-preview':
            from .v2022_07_02_preview.operations import Operations as OperationClass
        elif api_version == '2022-08-02-preview':
            from .v2022_08_02_preview.operations import Operations as OperationClass
        elif api_version == '2022-08-03-preview':
            from .v2022_08_03_preview.operations import Operations as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import Operations as OperationClass
        elif api_version == '2022-09-02-preview':
            from .v2022_09_02_preview.operations import Operations as OperationClass
        elif api_version == '2022-10-02-preview':
            from .v2022_10_02_preview.operations import Operations as OperationClass
        elif api_version == '2022-11-01':
            from .v2022_11_01.operations import Operations as OperationClass
        elif api_version == '2022-11-02-preview':
            from .v2022_11_02_preview.operations import Operations as OperationClass
        elif api_version == '2023-01-01':
            from .v2023_01_01.operations import Operations as OperationClass
        elif api_version == '2023-01-02-preview':
            from .v2023_01_02_preview.operations import Operations as OperationClass
        elif api_version == '2023-02-01':
            from .v2023_02_01.operations import Operations as OperationClass
        elif api_version == '2023-02-02-preview':
            from .v2023_02_02_preview.operations import Operations as OperationClass
        elif api_version == '2023-03-01':
            from .v2023_03_01.operations import Operations as OperationClass
        elif api_version == '2023-03-02-preview':
            from .v2023_03_02_preview.operations import Operations as OperationClass
        elif api_version == '2023-04-01':
            from .v2023_04_01.operations import Operations as OperationClass
        elif api_version == '2023-04-02-preview':
            from .v2023_04_02_preview.operations import Operations as OperationClass
        elif api_version == '2023-05-01':
            from .v2023_05_01.operations import Operations as OperationClass
        elif api_version == '2023-05-02-preview':
            from .v2023_05_02_preview.operations import Operations as OperationClass
        elif api_version == '2023-06-01':
            from .v2023_06_01.operations import Operations as OperationClass
        elif api_version == '2023-06-02-preview':
            from .v2023_06_02_preview.operations import Operations as OperationClass
        elif api_version == '2023-07-01':
            from .v2023_07_01.operations import Operations as OperationClass
        elif api_version == '2023-07-02-preview':
            from .v2023_07_02_preview.operations import Operations as OperationClass
        elif api_version == '2023-08-01':
            from .v2023_08_01.operations import Operations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'operations'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)), api_version)

    @property
    def private_endpoint_connections(self):
        if False:
            return 10
        'Instance depends on the API version:\n\n           * 2020-06-01: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2020_06_01.operations.PrivateEndpointConnectionsOperations>`\n           * 2020-07-01: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2020_07_01.operations.PrivateEndpointConnectionsOperations>`\n           * 2020-09-01: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2020_09_01.operations.PrivateEndpointConnectionsOperations>`\n           * 2020-11-01: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2020_11_01.operations.PrivateEndpointConnectionsOperations>`\n           * 2020-12-01: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2020_12_01.operations.PrivateEndpointConnectionsOperations>`\n           * 2021-02-01: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2021_02_01.operations.PrivateEndpointConnectionsOperations>`\n           * 2021-03-01: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2021_03_01.operations.PrivateEndpointConnectionsOperations>`\n           * 2021-05-01: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2021_05_01.operations.PrivateEndpointConnectionsOperations>`\n           * 2021-07-01: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2021_07_01.operations.PrivateEndpointConnectionsOperations>`\n           * 2021-08-01: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2021_08_01.operations.PrivateEndpointConnectionsOperations>`\n           * 2021-09-01: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2021_09_01.operations.PrivateEndpointConnectionsOperations>`\n           * 2021-10-01: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2021_10_01.operations.PrivateEndpointConnectionsOperations>`\n           * 2021-11-01-preview: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2021_11_01_preview.operations.PrivateEndpointConnectionsOperations>`\n           * 2022-01-01: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2022_01_01.operations.PrivateEndpointConnectionsOperations>`\n           * 2022-01-02-preview: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2022_01_02_preview.operations.PrivateEndpointConnectionsOperations>`\n           * 2022-02-01: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2022_02_01.operations.PrivateEndpointConnectionsOperations>`\n           * 2022-02-02-preview: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2022_02_02_preview.operations.PrivateEndpointConnectionsOperations>`\n           * 2022-03-01: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2022_03_01.operations.PrivateEndpointConnectionsOperations>`\n           * 2022-03-02-preview: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2022_03_02_preview.operations.PrivateEndpointConnectionsOperations>`\n           * 2022-04-01: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2022_04_01.operations.PrivateEndpointConnectionsOperations>`\n           * 2022-04-02-preview: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2022_04_02_preview.operations.PrivateEndpointConnectionsOperations>`\n           * 2022-05-02-preview: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2022_05_02_preview.operations.PrivateEndpointConnectionsOperations>`\n           * 2022-06-01: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2022_06_01.operations.PrivateEndpointConnectionsOperations>`\n           * 2022-06-02-preview: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2022_06_02_preview.operations.PrivateEndpointConnectionsOperations>`\n           * 2022-07-01: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2022_07_01.operations.PrivateEndpointConnectionsOperations>`\n           * 2022-07-02-preview: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2022_07_02_preview.operations.PrivateEndpointConnectionsOperations>`\n           * 2022-08-02-preview: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2022_08_02_preview.operations.PrivateEndpointConnectionsOperations>`\n           * 2022-08-03-preview: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2022_08_03_preview.operations.PrivateEndpointConnectionsOperations>`\n           * 2022-09-01: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2022_09_01.operations.PrivateEndpointConnectionsOperations>`\n           * 2022-09-02-preview: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2022_09_02_preview.operations.PrivateEndpointConnectionsOperations>`\n           * 2022-10-02-preview: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2022_10_02_preview.operations.PrivateEndpointConnectionsOperations>`\n           * 2022-11-01: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2022_11_01.operations.PrivateEndpointConnectionsOperations>`\n           * 2022-11-02-preview: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2022_11_02_preview.operations.PrivateEndpointConnectionsOperations>`\n           * 2023-01-01: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2023_01_01.operations.PrivateEndpointConnectionsOperations>`\n           * 2023-01-02-preview: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2023_01_02_preview.operations.PrivateEndpointConnectionsOperations>`\n           * 2023-02-01: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2023_02_01.operations.PrivateEndpointConnectionsOperations>`\n           * 2023-02-02-preview: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2023_02_02_preview.operations.PrivateEndpointConnectionsOperations>`\n           * 2023-03-01: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2023_03_01.operations.PrivateEndpointConnectionsOperations>`\n           * 2023-03-02-preview: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2023_03_02_preview.operations.PrivateEndpointConnectionsOperations>`\n           * 2023-04-01: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2023_04_01.operations.PrivateEndpointConnectionsOperations>`\n           * 2023-04-02-preview: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2023_04_02_preview.operations.PrivateEndpointConnectionsOperations>`\n           * 2023-05-01: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2023_05_01.operations.PrivateEndpointConnectionsOperations>`\n           * 2023-05-02-preview: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2023_05_02_preview.operations.PrivateEndpointConnectionsOperations>`\n           * 2023-06-01: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2023_06_01.operations.PrivateEndpointConnectionsOperations>`\n           * 2023-06-02-preview: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2023_06_02_preview.operations.PrivateEndpointConnectionsOperations>`\n           * 2023-07-01: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2023_07_01.operations.PrivateEndpointConnectionsOperations>`\n           * 2023-07-02-preview: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2023_07_02_preview.operations.PrivateEndpointConnectionsOperations>`\n           * 2023-08-01: :class:`PrivateEndpointConnectionsOperations<azure.mgmt.containerservice.v2023_08_01.operations.PrivateEndpointConnectionsOperations>`\n        '
        api_version = self._get_api_version('private_endpoint_connections')
        if api_version == '2020-06-01':
            from .v2020_06_01.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2020-07-01':
            from .v2020_07_01.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2020-11-01':
            from .v2020_11_01.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2021-02-01':
            from .v2021_02_01.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2021-03-01':
            from .v2021_03_01.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2021-05-01':
            from .v2021_05_01.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2021-07-01':
            from .v2021_07_01.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2021-08-01':
            from .v2021_08_01.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2021-09-01':
            from .v2021_09_01.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2021-10-01':
            from .v2021_10_01.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2021-11-01-preview':
            from .v2021_11_01_preview.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2022-01-01':
            from .v2022_01_01.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2022-01-02-preview':
            from .v2022_01_02_preview.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2022-02-01':
            from .v2022_02_01.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2022-02-02-preview':
            from .v2022_02_02_preview.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2022-03-01':
            from .v2022_03_01.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2022-03-02-preview':
            from .v2022_03_02_preview.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2022-04-01':
            from .v2022_04_01.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2022-04-02-preview':
            from .v2022_04_02_preview.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2022-05-02-preview':
            from .v2022_05_02_preview.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2022-06-01':
            from .v2022_06_01.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2022-06-02-preview':
            from .v2022_06_02_preview.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2022-07-01':
            from .v2022_07_01.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2022-07-02-preview':
            from .v2022_07_02_preview.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2022-08-02-preview':
            from .v2022_08_02_preview.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2022-08-03-preview':
            from .v2022_08_03_preview.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2022-09-02-preview':
            from .v2022_09_02_preview.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2022-10-02-preview':
            from .v2022_10_02_preview.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2022-11-01':
            from .v2022_11_01.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2022-11-02-preview':
            from .v2022_11_02_preview.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2023-01-01':
            from .v2023_01_01.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2023-01-02-preview':
            from .v2023_01_02_preview.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2023-02-01':
            from .v2023_02_01.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2023-02-02-preview':
            from .v2023_02_02_preview.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2023-03-01':
            from .v2023_03_01.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2023-03-02-preview':
            from .v2023_03_02_preview.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2023-04-01':
            from .v2023_04_01.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2023-04-02-preview':
            from .v2023_04_02_preview.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2023-05-01':
            from .v2023_05_01.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2023-05-02-preview':
            from .v2023_05_02_preview.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2023-06-01':
            from .v2023_06_01.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2023-06-02-preview':
            from .v2023_06_02_preview.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2023-07-01':
            from .v2023_07_01.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2023-07-02-preview':
            from .v2023_07_02_preview.operations import PrivateEndpointConnectionsOperations as OperationClass
        elif api_version == '2023-08-01':
            from .v2023_08_01.operations import PrivateEndpointConnectionsOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'private_endpoint_connections'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)), api_version)

    @property
    def private_link_resources(self):
        if False:
            for i in range(10):
                print('nop')
        'Instance depends on the API version:\n\n           * 2020-09-01: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2020_09_01.operations.PrivateLinkResourcesOperations>`\n           * 2020-11-01: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2020_11_01.operations.PrivateLinkResourcesOperations>`\n           * 2020-12-01: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2020_12_01.operations.PrivateLinkResourcesOperations>`\n           * 2021-02-01: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2021_02_01.operations.PrivateLinkResourcesOperations>`\n           * 2021-03-01: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2021_03_01.operations.PrivateLinkResourcesOperations>`\n           * 2021-05-01: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2021_05_01.operations.PrivateLinkResourcesOperations>`\n           * 2021-07-01: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2021_07_01.operations.PrivateLinkResourcesOperations>`\n           * 2021-08-01: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2021_08_01.operations.PrivateLinkResourcesOperations>`\n           * 2021-09-01: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2021_09_01.operations.PrivateLinkResourcesOperations>`\n           * 2021-10-01: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2021_10_01.operations.PrivateLinkResourcesOperations>`\n           * 2021-11-01-preview: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2021_11_01_preview.operations.PrivateLinkResourcesOperations>`\n           * 2022-01-01: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2022_01_01.operations.PrivateLinkResourcesOperations>`\n           * 2022-01-02-preview: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2022_01_02_preview.operations.PrivateLinkResourcesOperations>`\n           * 2022-02-01: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2022_02_01.operations.PrivateLinkResourcesOperations>`\n           * 2022-02-02-preview: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2022_02_02_preview.operations.PrivateLinkResourcesOperations>`\n           * 2022-03-01: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2022_03_01.operations.PrivateLinkResourcesOperations>`\n           * 2022-03-02-preview: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2022_03_02_preview.operations.PrivateLinkResourcesOperations>`\n           * 2022-04-01: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2022_04_01.operations.PrivateLinkResourcesOperations>`\n           * 2022-04-02-preview: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2022_04_02_preview.operations.PrivateLinkResourcesOperations>`\n           * 2022-05-02-preview: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2022_05_02_preview.operations.PrivateLinkResourcesOperations>`\n           * 2022-06-01: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2022_06_01.operations.PrivateLinkResourcesOperations>`\n           * 2022-06-02-preview: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2022_06_02_preview.operations.PrivateLinkResourcesOperations>`\n           * 2022-07-01: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2022_07_01.operations.PrivateLinkResourcesOperations>`\n           * 2022-07-02-preview: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2022_07_02_preview.operations.PrivateLinkResourcesOperations>`\n           * 2022-08-02-preview: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2022_08_02_preview.operations.PrivateLinkResourcesOperations>`\n           * 2022-08-03-preview: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2022_08_03_preview.operations.PrivateLinkResourcesOperations>`\n           * 2022-09-01: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2022_09_01.operations.PrivateLinkResourcesOperations>`\n           * 2022-09-02-preview: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2022_09_02_preview.operations.PrivateLinkResourcesOperations>`\n           * 2022-10-02-preview: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2022_10_02_preview.operations.PrivateLinkResourcesOperations>`\n           * 2022-11-01: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2022_11_01.operations.PrivateLinkResourcesOperations>`\n           * 2022-11-02-preview: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2022_11_02_preview.operations.PrivateLinkResourcesOperations>`\n           * 2023-01-01: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2023_01_01.operations.PrivateLinkResourcesOperations>`\n           * 2023-01-02-preview: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2023_01_02_preview.operations.PrivateLinkResourcesOperations>`\n           * 2023-02-01: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2023_02_01.operations.PrivateLinkResourcesOperations>`\n           * 2023-02-02-preview: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2023_02_02_preview.operations.PrivateLinkResourcesOperations>`\n           * 2023-03-01: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2023_03_01.operations.PrivateLinkResourcesOperations>`\n           * 2023-03-02-preview: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2023_03_02_preview.operations.PrivateLinkResourcesOperations>`\n           * 2023-04-01: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2023_04_01.operations.PrivateLinkResourcesOperations>`\n           * 2023-04-02-preview: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2023_04_02_preview.operations.PrivateLinkResourcesOperations>`\n           * 2023-05-01: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2023_05_01.operations.PrivateLinkResourcesOperations>`\n           * 2023-05-02-preview: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2023_05_02_preview.operations.PrivateLinkResourcesOperations>`\n           * 2023-06-01: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2023_06_01.operations.PrivateLinkResourcesOperations>`\n           * 2023-06-02-preview: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2023_06_02_preview.operations.PrivateLinkResourcesOperations>`\n           * 2023-07-01: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2023_07_01.operations.PrivateLinkResourcesOperations>`\n           * 2023-07-02-preview: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2023_07_02_preview.operations.PrivateLinkResourcesOperations>`\n           * 2023-08-01: :class:`PrivateLinkResourcesOperations<azure.mgmt.containerservice.v2023_08_01.operations.PrivateLinkResourcesOperations>`\n        '
        api_version = self._get_api_version('private_link_resources')
        if api_version == '2020-09-01':
            from .v2020_09_01.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2020-11-01':
            from .v2020_11_01.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2021-02-01':
            from .v2021_02_01.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2021-03-01':
            from .v2021_03_01.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2021-05-01':
            from .v2021_05_01.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2021-07-01':
            from .v2021_07_01.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2021-08-01':
            from .v2021_08_01.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2021-09-01':
            from .v2021_09_01.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2021-10-01':
            from .v2021_10_01.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2021-11-01-preview':
            from .v2021_11_01_preview.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2022-01-01':
            from .v2022_01_01.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2022-01-02-preview':
            from .v2022_01_02_preview.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2022-02-01':
            from .v2022_02_01.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2022-02-02-preview':
            from .v2022_02_02_preview.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2022-03-01':
            from .v2022_03_01.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2022-03-02-preview':
            from .v2022_03_02_preview.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2022-04-01':
            from .v2022_04_01.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2022-04-02-preview':
            from .v2022_04_02_preview.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2022-05-02-preview':
            from .v2022_05_02_preview.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2022-06-01':
            from .v2022_06_01.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2022-06-02-preview':
            from .v2022_06_02_preview.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2022-07-01':
            from .v2022_07_01.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2022-07-02-preview':
            from .v2022_07_02_preview.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2022-08-02-preview':
            from .v2022_08_02_preview.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2022-08-03-preview':
            from .v2022_08_03_preview.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2022-09-02-preview':
            from .v2022_09_02_preview.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2022-10-02-preview':
            from .v2022_10_02_preview.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2022-11-01':
            from .v2022_11_01.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2022-11-02-preview':
            from .v2022_11_02_preview.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2023-01-01':
            from .v2023_01_01.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2023-01-02-preview':
            from .v2023_01_02_preview.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2023-02-01':
            from .v2023_02_01.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2023-02-02-preview':
            from .v2023_02_02_preview.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2023-03-01':
            from .v2023_03_01.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2023-03-02-preview':
            from .v2023_03_02_preview.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2023-04-01':
            from .v2023_04_01.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2023-04-02-preview':
            from .v2023_04_02_preview.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2023-05-01':
            from .v2023_05_01.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2023-05-02-preview':
            from .v2023_05_02_preview.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2023-06-01':
            from .v2023_06_01.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2023-06-02-preview':
            from .v2023_06_02_preview.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2023-07-01':
            from .v2023_07_01.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2023-07-02-preview':
            from .v2023_07_02_preview.operations import PrivateLinkResourcesOperations as OperationClass
        elif api_version == '2023-08-01':
            from .v2023_08_01.operations import PrivateLinkResourcesOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'private_link_resources'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)), api_version)

    @property
    def resolve_private_link_service_id(self):
        if False:
            return 10
        'Instance depends on the API version:\n\n           * 2020-09-01: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2020_09_01.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2020-11-01: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2020_11_01.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2020-12-01: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2020_12_01.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2021-02-01: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2021_02_01.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2021-03-01: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2021_03_01.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2021-05-01: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2021_05_01.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2021-07-01: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2021_07_01.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2021-08-01: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2021_08_01.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2021-09-01: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2021_09_01.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2021-10-01: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2021_10_01.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2021-11-01-preview: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2021_11_01_preview.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2022-01-01: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2022_01_01.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2022-01-02-preview: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2022_01_02_preview.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2022-02-01: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2022_02_01.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2022-02-02-preview: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2022_02_02_preview.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2022-03-01: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2022_03_01.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2022-03-02-preview: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2022_03_02_preview.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2022-04-01: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2022_04_01.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2022-04-02-preview: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2022_04_02_preview.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2022-05-02-preview: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2022_05_02_preview.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2022-06-01: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2022_06_01.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2022-06-02-preview: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2022_06_02_preview.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2022-07-01: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2022_07_01.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2022-07-02-preview: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2022_07_02_preview.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2022-08-02-preview: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2022_08_02_preview.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2022-08-03-preview: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2022_08_03_preview.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2022-09-01: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2022_09_01.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2022-09-02-preview: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2022_09_02_preview.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2022-10-02-preview: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2022_10_02_preview.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2022-11-01: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2022_11_01.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2022-11-02-preview: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2022_11_02_preview.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2023-01-01: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2023_01_01.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2023-01-02-preview: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2023_01_02_preview.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2023-02-01: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2023_02_01.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2023-02-02-preview: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2023_02_02_preview.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2023-03-01: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2023_03_01.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2023-03-02-preview: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2023_03_02_preview.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2023-04-01: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2023_04_01.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2023-04-02-preview: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2023_04_02_preview.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2023-05-01: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2023_05_01.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2023-05-02-preview: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2023_05_02_preview.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2023-06-01: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2023_06_01.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2023-06-02-preview: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2023_06_02_preview.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2023-07-01: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2023_07_01.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2023-07-02-preview: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2023_07_02_preview.operations.ResolvePrivateLinkServiceIdOperations>`\n           * 2023-08-01: :class:`ResolvePrivateLinkServiceIdOperations<azure.mgmt.containerservice.v2023_08_01.operations.ResolvePrivateLinkServiceIdOperations>`\n        '
        api_version = self._get_api_version('resolve_private_link_service_id')
        if api_version == '2020-09-01':
            from .v2020_09_01.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2020-11-01':
            from .v2020_11_01.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2021-02-01':
            from .v2021_02_01.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2021-03-01':
            from .v2021_03_01.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2021-05-01':
            from .v2021_05_01.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2021-07-01':
            from .v2021_07_01.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2021-08-01':
            from .v2021_08_01.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2021-09-01':
            from .v2021_09_01.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2021-10-01':
            from .v2021_10_01.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2021-11-01-preview':
            from .v2021_11_01_preview.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2022-01-01':
            from .v2022_01_01.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2022-01-02-preview':
            from .v2022_01_02_preview.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2022-02-01':
            from .v2022_02_01.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2022-02-02-preview':
            from .v2022_02_02_preview.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2022-03-01':
            from .v2022_03_01.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2022-03-02-preview':
            from .v2022_03_02_preview.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2022-04-01':
            from .v2022_04_01.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2022-04-02-preview':
            from .v2022_04_02_preview.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2022-05-02-preview':
            from .v2022_05_02_preview.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2022-06-01':
            from .v2022_06_01.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2022-06-02-preview':
            from .v2022_06_02_preview.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2022-07-01':
            from .v2022_07_01.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2022-07-02-preview':
            from .v2022_07_02_preview.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2022-08-02-preview':
            from .v2022_08_02_preview.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2022-08-03-preview':
            from .v2022_08_03_preview.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2022-09-02-preview':
            from .v2022_09_02_preview.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2022-10-02-preview':
            from .v2022_10_02_preview.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2022-11-01':
            from .v2022_11_01.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2022-11-02-preview':
            from .v2022_11_02_preview.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2023-01-01':
            from .v2023_01_01.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2023-01-02-preview':
            from .v2023_01_02_preview.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2023-02-01':
            from .v2023_02_01.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2023-02-02-preview':
            from .v2023_02_02_preview.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2023-03-01':
            from .v2023_03_01.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2023-03-02-preview':
            from .v2023_03_02_preview.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2023-04-01':
            from .v2023_04_01.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2023-04-02-preview':
            from .v2023_04_02_preview.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2023-05-01':
            from .v2023_05_01.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2023-05-02-preview':
            from .v2023_05_02_preview.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2023-06-01':
            from .v2023_06_01.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2023-06-02-preview':
            from .v2023_06_02_preview.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2023-07-01':
            from .v2023_07_01.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2023-07-02-preview':
            from .v2023_07_02_preview.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        elif api_version == '2023-08-01':
            from .v2023_08_01.operations import ResolvePrivateLinkServiceIdOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'resolve_private_link_service_id'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)), api_version)

    @property
    def snapshots(self):
        if False:
            return 10
        'Instance depends on the API version:\n\n           * 2021-08-01: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2021_08_01.operations.SnapshotsOperations>`\n           * 2021-09-01: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2021_09_01.operations.SnapshotsOperations>`\n           * 2021-10-01: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2021_10_01.operations.SnapshotsOperations>`\n           * 2021-11-01-preview: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2021_11_01_preview.operations.SnapshotsOperations>`\n           * 2022-01-01: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2022_01_01.operations.SnapshotsOperations>`\n           * 2022-01-02-preview: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2022_01_02_preview.operations.SnapshotsOperations>`\n           * 2022-02-01: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2022_02_01.operations.SnapshotsOperations>`\n           * 2022-02-02-preview: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2022_02_02_preview.operations.SnapshotsOperations>`\n           * 2022-03-01: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2022_03_01.operations.SnapshotsOperations>`\n           * 2022-03-02-preview: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2022_03_02_preview.operations.SnapshotsOperations>`\n           * 2022-04-01: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2022_04_01.operations.SnapshotsOperations>`\n           * 2022-04-02-preview: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2022_04_02_preview.operations.SnapshotsOperations>`\n           * 2022-05-02-preview: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2022_05_02_preview.operations.SnapshotsOperations>`\n           * 2022-06-01: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2022_06_01.operations.SnapshotsOperations>`\n           * 2022-06-02-preview: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2022_06_02_preview.operations.SnapshotsOperations>`\n           * 2022-07-01: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2022_07_01.operations.SnapshotsOperations>`\n           * 2022-07-02-preview: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2022_07_02_preview.operations.SnapshotsOperations>`\n           * 2022-08-02-preview: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2022_08_02_preview.operations.SnapshotsOperations>`\n           * 2022-08-03-preview: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2022_08_03_preview.operations.SnapshotsOperations>`\n           * 2022-09-01: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2022_09_01.operations.SnapshotsOperations>`\n           * 2022-09-02-preview: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2022_09_02_preview.operations.SnapshotsOperations>`\n           * 2022-10-02-preview: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2022_10_02_preview.operations.SnapshotsOperations>`\n           * 2022-11-01: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2022_11_01.operations.SnapshotsOperations>`\n           * 2022-11-02-preview: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2022_11_02_preview.operations.SnapshotsOperations>`\n           * 2023-01-01: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2023_01_01.operations.SnapshotsOperations>`\n           * 2023-01-02-preview: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2023_01_02_preview.operations.SnapshotsOperations>`\n           * 2023-02-01: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2023_02_01.operations.SnapshotsOperations>`\n           * 2023-02-02-preview: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2023_02_02_preview.operations.SnapshotsOperations>`\n           * 2023-03-01: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2023_03_01.operations.SnapshotsOperations>`\n           * 2023-03-02-preview: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2023_03_02_preview.operations.SnapshotsOperations>`\n           * 2023-04-01: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2023_04_01.operations.SnapshotsOperations>`\n           * 2023-04-02-preview: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2023_04_02_preview.operations.SnapshotsOperations>`\n           * 2023-05-01: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2023_05_01.operations.SnapshotsOperations>`\n           * 2023-05-02-preview: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2023_05_02_preview.operations.SnapshotsOperations>`\n           * 2023-06-01: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2023_06_01.operations.SnapshotsOperations>`\n           * 2023-06-02-preview: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2023_06_02_preview.operations.SnapshotsOperations>`\n           * 2023-07-01: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2023_07_01.operations.SnapshotsOperations>`\n           * 2023-07-02-preview: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2023_07_02_preview.operations.SnapshotsOperations>`\n           * 2023-08-01: :class:`SnapshotsOperations<azure.mgmt.containerservice.v2023_08_01.operations.SnapshotsOperations>`\n        '
        api_version = self._get_api_version('snapshots')
        if api_version == '2021-08-01':
            from .v2021_08_01.operations import SnapshotsOperations as OperationClass
        elif api_version == '2021-09-01':
            from .v2021_09_01.operations import SnapshotsOperations as OperationClass
        elif api_version == '2021-10-01':
            from .v2021_10_01.operations import SnapshotsOperations as OperationClass
        elif api_version == '2021-11-01-preview':
            from .v2021_11_01_preview.operations import SnapshotsOperations as OperationClass
        elif api_version == '2022-01-01':
            from .v2022_01_01.operations import SnapshotsOperations as OperationClass
        elif api_version == '2022-01-02-preview':
            from .v2022_01_02_preview.operations import SnapshotsOperations as OperationClass
        elif api_version == '2022-02-01':
            from .v2022_02_01.operations import SnapshotsOperations as OperationClass
        elif api_version == '2022-02-02-preview':
            from .v2022_02_02_preview.operations import SnapshotsOperations as OperationClass
        elif api_version == '2022-03-01':
            from .v2022_03_01.operations import SnapshotsOperations as OperationClass
        elif api_version == '2022-03-02-preview':
            from .v2022_03_02_preview.operations import SnapshotsOperations as OperationClass
        elif api_version == '2022-04-01':
            from .v2022_04_01.operations import SnapshotsOperations as OperationClass
        elif api_version == '2022-04-02-preview':
            from .v2022_04_02_preview.operations import SnapshotsOperations as OperationClass
        elif api_version == '2022-05-02-preview':
            from .v2022_05_02_preview.operations import SnapshotsOperations as OperationClass
        elif api_version == '2022-06-01':
            from .v2022_06_01.operations import SnapshotsOperations as OperationClass
        elif api_version == '2022-06-02-preview':
            from .v2022_06_02_preview.operations import SnapshotsOperations as OperationClass
        elif api_version == '2022-07-01':
            from .v2022_07_01.operations import SnapshotsOperations as OperationClass
        elif api_version == '2022-07-02-preview':
            from .v2022_07_02_preview.operations import SnapshotsOperations as OperationClass
        elif api_version == '2022-08-02-preview':
            from .v2022_08_02_preview.operations import SnapshotsOperations as OperationClass
        elif api_version == '2022-08-03-preview':
            from .v2022_08_03_preview.operations import SnapshotsOperations as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import SnapshotsOperations as OperationClass
        elif api_version == '2022-09-02-preview':
            from .v2022_09_02_preview.operations import SnapshotsOperations as OperationClass
        elif api_version == '2022-10-02-preview':
            from .v2022_10_02_preview.operations import SnapshotsOperations as OperationClass
        elif api_version == '2022-11-01':
            from .v2022_11_01.operations import SnapshotsOperations as OperationClass
        elif api_version == '2022-11-02-preview':
            from .v2022_11_02_preview.operations import SnapshotsOperations as OperationClass
        elif api_version == '2023-01-01':
            from .v2023_01_01.operations import SnapshotsOperations as OperationClass
        elif api_version == '2023-01-02-preview':
            from .v2023_01_02_preview.operations import SnapshotsOperations as OperationClass
        elif api_version == '2023-02-01':
            from .v2023_02_01.operations import SnapshotsOperations as OperationClass
        elif api_version == '2023-02-02-preview':
            from .v2023_02_02_preview.operations import SnapshotsOperations as OperationClass
        elif api_version == '2023-03-01':
            from .v2023_03_01.operations import SnapshotsOperations as OperationClass
        elif api_version == '2023-03-02-preview':
            from .v2023_03_02_preview.operations import SnapshotsOperations as OperationClass
        elif api_version == '2023-04-01':
            from .v2023_04_01.operations import SnapshotsOperations as OperationClass
        elif api_version == '2023-04-02-preview':
            from .v2023_04_02_preview.operations import SnapshotsOperations as OperationClass
        elif api_version == '2023-05-01':
            from .v2023_05_01.operations import SnapshotsOperations as OperationClass
        elif api_version == '2023-05-02-preview':
            from .v2023_05_02_preview.operations import SnapshotsOperations as OperationClass
        elif api_version == '2023-06-01':
            from .v2023_06_01.operations import SnapshotsOperations as OperationClass
        elif api_version == '2023-06-02-preview':
            from .v2023_06_02_preview.operations import SnapshotsOperations as OperationClass
        elif api_version == '2023-07-01':
            from .v2023_07_01.operations import SnapshotsOperations as OperationClass
        elif api_version == '2023-07-02-preview':
            from .v2023_07_02_preview.operations import SnapshotsOperations as OperationClass
        elif api_version == '2023-08-01':
            from .v2023_08_01.operations import SnapshotsOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'snapshots'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)), api_version)

    @property
    def trusted_access_role_bindings(self):
        if False:
            while True:
                i = 10
        'Instance depends on the API version:\n\n           * 2022-04-02-preview: :class:`TrustedAccessRoleBindingsOperations<azure.mgmt.containerservice.v2022_04_02_preview.operations.TrustedAccessRoleBindingsOperations>`\n           * 2022-05-02-preview: :class:`TrustedAccessRoleBindingsOperations<azure.mgmt.containerservice.v2022_05_02_preview.operations.TrustedAccessRoleBindingsOperations>`\n           * 2022-06-02-preview: :class:`TrustedAccessRoleBindingsOperations<azure.mgmt.containerservice.v2022_06_02_preview.operations.TrustedAccessRoleBindingsOperations>`\n           * 2022-07-02-preview: :class:`TrustedAccessRoleBindingsOperations<azure.mgmt.containerservice.v2022_07_02_preview.operations.TrustedAccessRoleBindingsOperations>`\n           * 2022-08-02-preview: :class:`TrustedAccessRoleBindingsOperations<azure.mgmt.containerservice.v2022_08_02_preview.operations.TrustedAccessRoleBindingsOperations>`\n           * 2022-08-03-preview: :class:`TrustedAccessRoleBindingsOperations<azure.mgmt.containerservice.v2022_08_03_preview.operations.TrustedAccessRoleBindingsOperations>`\n           * 2022-09-02-preview: :class:`TrustedAccessRoleBindingsOperations<azure.mgmt.containerservice.v2022_09_02_preview.operations.TrustedAccessRoleBindingsOperations>`\n           * 2022-10-02-preview: :class:`TrustedAccessRoleBindingsOperations<azure.mgmt.containerservice.v2022_10_02_preview.operations.TrustedAccessRoleBindingsOperations>`\n           * 2022-11-02-preview: :class:`TrustedAccessRoleBindingsOperations<azure.mgmt.containerservice.v2022_11_02_preview.operations.TrustedAccessRoleBindingsOperations>`\n           * 2023-01-02-preview: :class:`TrustedAccessRoleBindingsOperations<azure.mgmt.containerservice.v2023_01_02_preview.operations.TrustedAccessRoleBindingsOperations>`\n           * 2023-02-02-preview: :class:`TrustedAccessRoleBindingsOperations<azure.mgmt.containerservice.v2023_02_02_preview.operations.TrustedAccessRoleBindingsOperations>`\n           * 2023-03-02-preview: :class:`TrustedAccessRoleBindingsOperations<azure.mgmt.containerservice.v2023_03_02_preview.operations.TrustedAccessRoleBindingsOperations>`\n           * 2023-04-02-preview: :class:`TrustedAccessRoleBindingsOperations<azure.mgmt.containerservice.v2023_04_02_preview.operations.TrustedAccessRoleBindingsOperations>`\n           * 2023-05-02-preview: :class:`TrustedAccessRoleBindingsOperations<azure.mgmt.containerservice.v2023_05_02_preview.operations.TrustedAccessRoleBindingsOperations>`\n           * 2023-06-02-preview: :class:`TrustedAccessRoleBindingsOperations<azure.mgmt.containerservice.v2023_06_02_preview.operations.TrustedAccessRoleBindingsOperations>`\n           * 2023-07-02-preview: :class:`TrustedAccessRoleBindingsOperations<azure.mgmt.containerservice.v2023_07_02_preview.operations.TrustedAccessRoleBindingsOperations>`\n        '
        api_version = self._get_api_version('trusted_access_role_bindings')
        if api_version == '2022-04-02-preview':
            from .v2022_04_02_preview.operations import TrustedAccessRoleBindingsOperations as OperationClass
        elif api_version == '2022-05-02-preview':
            from .v2022_05_02_preview.operations import TrustedAccessRoleBindingsOperations as OperationClass
        elif api_version == '2022-06-02-preview':
            from .v2022_06_02_preview.operations import TrustedAccessRoleBindingsOperations as OperationClass
        elif api_version == '2022-07-02-preview':
            from .v2022_07_02_preview.operations import TrustedAccessRoleBindingsOperations as OperationClass
        elif api_version == '2022-08-02-preview':
            from .v2022_08_02_preview.operations import TrustedAccessRoleBindingsOperations as OperationClass
        elif api_version == '2022-08-03-preview':
            from .v2022_08_03_preview.operations import TrustedAccessRoleBindingsOperations as OperationClass
        elif api_version == '2022-09-02-preview':
            from .v2022_09_02_preview.operations import TrustedAccessRoleBindingsOperations as OperationClass
        elif api_version == '2022-10-02-preview':
            from .v2022_10_02_preview.operations import TrustedAccessRoleBindingsOperations as OperationClass
        elif api_version == '2022-11-02-preview':
            from .v2022_11_02_preview.operations import TrustedAccessRoleBindingsOperations as OperationClass
        elif api_version == '2023-01-02-preview':
            from .v2023_01_02_preview.operations import TrustedAccessRoleBindingsOperations as OperationClass
        elif api_version == '2023-02-02-preview':
            from .v2023_02_02_preview.operations import TrustedAccessRoleBindingsOperations as OperationClass
        elif api_version == '2023-03-02-preview':
            from .v2023_03_02_preview.operations import TrustedAccessRoleBindingsOperations as OperationClass
        elif api_version == '2023-04-02-preview':
            from .v2023_04_02_preview.operations import TrustedAccessRoleBindingsOperations as OperationClass
        elif api_version == '2023-05-02-preview':
            from .v2023_05_02_preview.operations import TrustedAccessRoleBindingsOperations as OperationClass
        elif api_version == '2023-06-02-preview':
            from .v2023_06_02_preview.operations import TrustedAccessRoleBindingsOperations as OperationClass
        elif api_version == '2023-07-02-preview':
            from .v2023_07_02_preview.operations import TrustedAccessRoleBindingsOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'trusted_access_role_bindings'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)), api_version)

    @property
    def trusted_access_roles(self):
        if False:
            print('Hello World!')
        'Instance depends on the API version:\n\n           * 2022-04-02-preview: :class:`TrustedAccessRolesOperations<azure.mgmt.containerservice.v2022_04_02_preview.operations.TrustedAccessRolesOperations>`\n           * 2022-05-02-preview: :class:`TrustedAccessRolesOperations<azure.mgmt.containerservice.v2022_05_02_preview.operations.TrustedAccessRolesOperations>`\n           * 2022-06-02-preview: :class:`TrustedAccessRolesOperations<azure.mgmt.containerservice.v2022_06_02_preview.operations.TrustedAccessRolesOperations>`\n           * 2022-07-02-preview: :class:`TrustedAccessRolesOperations<azure.mgmt.containerservice.v2022_07_02_preview.operations.TrustedAccessRolesOperations>`\n           * 2022-08-02-preview: :class:`TrustedAccessRolesOperations<azure.mgmt.containerservice.v2022_08_02_preview.operations.TrustedAccessRolesOperations>`\n           * 2022-08-03-preview: :class:`TrustedAccessRolesOperations<azure.mgmt.containerservice.v2022_08_03_preview.operations.TrustedAccessRolesOperations>`\n           * 2022-09-02-preview: :class:`TrustedAccessRolesOperations<azure.mgmt.containerservice.v2022_09_02_preview.operations.TrustedAccessRolesOperations>`\n           * 2022-10-02-preview: :class:`TrustedAccessRolesOperations<azure.mgmt.containerservice.v2022_10_02_preview.operations.TrustedAccessRolesOperations>`\n           * 2022-11-02-preview: :class:`TrustedAccessRolesOperations<azure.mgmt.containerservice.v2022_11_02_preview.operations.TrustedAccessRolesOperations>`\n           * 2023-01-02-preview: :class:`TrustedAccessRolesOperations<azure.mgmt.containerservice.v2023_01_02_preview.operations.TrustedAccessRolesOperations>`\n           * 2023-02-02-preview: :class:`TrustedAccessRolesOperations<azure.mgmt.containerservice.v2023_02_02_preview.operations.TrustedAccessRolesOperations>`\n           * 2023-03-02-preview: :class:`TrustedAccessRolesOperations<azure.mgmt.containerservice.v2023_03_02_preview.operations.TrustedAccessRolesOperations>`\n           * 2023-04-02-preview: :class:`TrustedAccessRolesOperations<azure.mgmt.containerservice.v2023_04_02_preview.operations.TrustedAccessRolesOperations>`\n           * 2023-05-02-preview: :class:`TrustedAccessRolesOperations<azure.mgmt.containerservice.v2023_05_02_preview.operations.TrustedAccessRolesOperations>`\n           * 2023-06-02-preview: :class:`TrustedAccessRolesOperations<azure.mgmt.containerservice.v2023_06_02_preview.operations.TrustedAccessRolesOperations>`\n           * 2023-07-02-preview: :class:`TrustedAccessRolesOperations<azure.mgmt.containerservice.v2023_07_02_preview.operations.TrustedAccessRolesOperations>`\n        '
        api_version = self._get_api_version('trusted_access_roles')
        if api_version == '2022-04-02-preview':
            from .v2022_04_02_preview.operations import TrustedAccessRolesOperations as OperationClass
        elif api_version == '2022-05-02-preview':
            from .v2022_05_02_preview.operations import TrustedAccessRolesOperations as OperationClass
        elif api_version == '2022-06-02-preview':
            from .v2022_06_02_preview.operations import TrustedAccessRolesOperations as OperationClass
        elif api_version == '2022-07-02-preview':
            from .v2022_07_02_preview.operations import TrustedAccessRolesOperations as OperationClass
        elif api_version == '2022-08-02-preview':
            from .v2022_08_02_preview.operations import TrustedAccessRolesOperations as OperationClass
        elif api_version == '2022-08-03-preview':
            from .v2022_08_03_preview.operations import TrustedAccessRolesOperations as OperationClass
        elif api_version == '2022-09-02-preview':
            from .v2022_09_02_preview.operations import TrustedAccessRolesOperations as OperationClass
        elif api_version == '2022-10-02-preview':
            from .v2022_10_02_preview.operations import TrustedAccessRolesOperations as OperationClass
        elif api_version == '2022-11-02-preview':
            from .v2022_11_02_preview.operations import TrustedAccessRolesOperations as OperationClass
        elif api_version == '2023-01-02-preview':
            from .v2023_01_02_preview.operations import TrustedAccessRolesOperations as OperationClass
        elif api_version == '2023-02-02-preview':
            from .v2023_02_02_preview.operations import TrustedAccessRolesOperations as OperationClass
        elif api_version == '2023-03-02-preview':
            from .v2023_03_02_preview.operations import TrustedAccessRolesOperations as OperationClass
        elif api_version == '2023-04-02-preview':
            from .v2023_04_02_preview.operations import TrustedAccessRolesOperations as OperationClass
        elif api_version == '2023-05-02-preview':
            from .v2023_05_02_preview.operations import TrustedAccessRolesOperations as OperationClass
        elif api_version == '2023-06-02-preview':
            from .v2023_06_02_preview.operations import TrustedAccessRolesOperations as OperationClass
        elif api_version == '2023-07-02-preview':
            from .v2023_07_02_preview.operations import TrustedAccessRolesOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'trusted_access_roles'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)), api_version)

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        self._client.close()

    def __enter__(self):
        if False:
            while True:
                i = 10
        self._client.__enter__()
        return self

    def __exit__(self, *exc_details):
        if False:
            while True:
                i = 10
        self._client.__exit__(*exc_details)