from typing import Any, Optional, TYPE_CHECKING
from azure.mgmt.core import ARMPipelineClient
from azure.profiles import KnownProfiles, ProfileDefinition
from azure.profiles.multiapiclient import MultiApiClientMixin
from ._configuration import WebSiteManagementClientConfiguration
from ._operations_mixin import WebSiteManagementClientOperationsMixin
from ._serialization import Deserializer, Serializer
if TYPE_CHECKING:
    from azure.core.credentials import TokenCredential

class _SDKClient(object):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        'This is a fake class to support current implemetation of MultiApiClientMixin."\n        Will be removed in final version of multiapi azure-core based client\n        '
        pass

class WebSiteManagementClient(WebSiteManagementClientOperationsMixin, MultiApiClientMixin, _SDKClient):
    """WebSite Management Client.

    This ready contains multiple API versions, to help you deal with all of the Azure clouds
    (Azure Stack, Azure Government, Azure China, etc.).
    By default, it uses the latest API version available on public Azure.
    For production, you should stick to a particular api-version and/or profile.
    The profile sets a mapping between an operation group and its API version.
    The api-version parameter sets the default API version if the operation
    group is not described in the profile.

    :param credential: Credential needed for the client to connect to Azure. Required.
    :type credential: ~azure.core.credentials.TokenCredential
    :param subscription_id: Your Azure subscription ID. This is a GUID-formatted string (e.g. 00000000-0000-0000-0000-000000000000). Required.
    :type subscription_id: str
    :param api_version: API version to use if no profile is provided, or if missing in profile.
    :type api_version: str
    :param base_url: Service URL
    :type base_url: str
    :param profile: A profile definition, from KnownProfiles to dict.
    :type profile: azure.profiles.KnownProfiles
    :keyword int polling_interval: Default waiting time between two polls for LRO operations if no Retry-After header is present.
    """
    DEFAULT_API_VERSION = '2022-09-01'
    _PROFILE_TAG = 'azure.mgmt.web.WebSiteManagementClient'
    LATEST_PROFILE = ProfileDefinition({_PROFILE_TAG: {None: DEFAULT_API_VERSION, 'billing_meters': '2016-03-01', 'validate_container_settings': '2018-02-01'}}, _PROFILE_TAG + ' latest')

    def __init__(self, credential: 'TokenCredential', subscription_id: str, api_version: Optional[str]=None, base_url: str='https://management.azure.com', profile: KnownProfiles=KnownProfiles.default, **kwargs: Any):
        if False:
            while True:
                i = 10
        self._config = WebSiteManagementClientConfiguration(credential, subscription_id, **kwargs)
        self._client = ARMPipelineClient(base_url=base_url, config=self._config, **kwargs)
        super(WebSiteManagementClient, self).__init__(api_version=api_version, profile=profile)

    @classmethod
    def _models_dict(cls, api_version):
        if False:
            print('Hello World!')
        return {k: v for (k, v) in cls.models(api_version).__dict__.items() if isinstance(v, type)}

    @classmethod
    def models(cls, api_version=DEFAULT_API_VERSION):
        if False:
            for i in range(10):
                print('nop')
        'Module depends on the API version:\n\n           * 2015-04-01: :mod:`v2015_04_01.models<azure.mgmt.web.v2015_04_01.models>`\n           * 2015-08-01: :mod:`v2015_08_01.models<azure.mgmt.web.v2015_08_01.models>`\n           * 2016-03-01: :mod:`v2016_03_01.models<azure.mgmt.web.v2016_03_01.models>`\n           * 2016-08-01: :mod:`v2016_08_01.models<azure.mgmt.web.v2016_08_01.models>`\n           * 2016-09-01: :mod:`v2016_09_01.models<azure.mgmt.web.v2016_09_01.models>`\n           * 2018-02-01: :mod:`v2018_02_01.models<azure.mgmt.web.v2018_02_01.models>`\n           * 2018-11-01: :mod:`v2018_11_01.models<azure.mgmt.web.v2018_11_01.models>`\n           * 2019-08-01: :mod:`v2019_08_01.models<azure.mgmt.web.v2019_08_01.models>`\n           * 2020-06-01: :mod:`v2020_06_01.models<azure.mgmt.web.v2020_06_01.models>`\n           * 2020-09-01: :mod:`v2020_09_01.models<azure.mgmt.web.v2020_09_01.models>`\n           * 2020-12-01: :mod:`v2020_12_01.models<azure.mgmt.web.v2020_12_01.models>`\n           * 2021-01-01: :mod:`v2021_01_01.models<azure.mgmt.web.v2021_01_01.models>`\n           * 2021-01-15: :mod:`v2021_01_15.models<azure.mgmt.web.v2021_01_15.models>`\n           * 2022-09-01: :mod:`v2022_09_01.models<azure.mgmt.web.v2022_09_01.models>`\n        '
        if api_version == '2015-04-01':
            from .v2015_04_01 import models
            return models
        elif api_version == '2015-08-01':
            from .v2015_08_01 import models
            return models
        elif api_version == '2016-03-01':
            from .v2016_03_01 import models
            return models
        elif api_version == '2016-08-01':
            from .v2016_08_01 import models
            return models
        elif api_version == '2016-09-01':
            from .v2016_09_01 import models
            return models
        elif api_version == '2018-02-01':
            from .v2018_02_01 import models
            return models
        elif api_version == '2018-11-01':
            from .v2018_11_01 import models
            return models
        elif api_version == '2019-08-01':
            from .v2019_08_01 import models
            return models
        elif api_version == '2020-06-01':
            from .v2020_06_01 import models
            return models
        elif api_version == '2020-09-01':
            from .v2020_09_01 import models
            return models
        elif api_version == '2020-12-01':
            from .v2020_12_01 import models
            return models
        elif api_version == '2021-01-01':
            from .v2021_01_01 import models
            return models
        elif api_version == '2021-01-15':
            from .v2021_01_15 import models
            return models
        elif api_version == '2022-09-01':
            from .v2022_09_01 import models
            return models
        raise ValueError('API version {} is not available'.format(api_version))

    @property
    def app_service_certificate_orders(self):
        if False:
            while True:
                i = 10
        'Instance depends on the API version:\n\n           * 2015-08-01: :class:`AppServiceCertificateOrdersOperations<azure.mgmt.web.v2015_08_01.operations.AppServiceCertificateOrdersOperations>`\n           * 2018-02-01: :class:`AppServiceCertificateOrdersOperations<azure.mgmt.web.v2018_02_01.operations.AppServiceCertificateOrdersOperations>`\n           * 2019-08-01: :class:`AppServiceCertificateOrdersOperations<azure.mgmt.web.v2019_08_01.operations.AppServiceCertificateOrdersOperations>`\n           * 2020-06-01: :class:`AppServiceCertificateOrdersOperations<azure.mgmt.web.v2020_06_01.operations.AppServiceCertificateOrdersOperations>`\n           * 2020-09-01: :class:`AppServiceCertificateOrdersOperations<azure.mgmt.web.v2020_09_01.operations.AppServiceCertificateOrdersOperations>`\n           * 2020-12-01: :class:`AppServiceCertificateOrdersOperations<azure.mgmt.web.v2020_12_01.operations.AppServiceCertificateOrdersOperations>`\n           * 2021-01-01: :class:`AppServiceCertificateOrdersOperations<azure.mgmt.web.v2021_01_01.operations.AppServiceCertificateOrdersOperations>`\n           * 2021-01-15: :class:`AppServiceCertificateOrdersOperations<azure.mgmt.web.v2021_01_15.operations.AppServiceCertificateOrdersOperations>`\n           * 2022-09-01: :class:`AppServiceCertificateOrdersOperations<azure.mgmt.web.v2022_09_01.operations.AppServiceCertificateOrdersOperations>`\n        '
        api_version = self._get_api_version('app_service_certificate_orders')
        if api_version == '2015-08-01':
            from .v2015_08_01.operations import AppServiceCertificateOrdersOperations as OperationClass
        elif api_version == '2018-02-01':
            from .v2018_02_01.operations import AppServiceCertificateOrdersOperations as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import AppServiceCertificateOrdersOperations as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import AppServiceCertificateOrdersOperations as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import AppServiceCertificateOrdersOperations as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import AppServiceCertificateOrdersOperations as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import AppServiceCertificateOrdersOperations as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import AppServiceCertificateOrdersOperations as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import AppServiceCertificateOrdersOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'app_service_certificate_orders'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    @property
    def app_service_environments(self):
        if False:
            i = 10
            return i + 15
        'Instance depends on the API version:\n\n           * 2016-09-01: :class:`AppServiceEnvironmentsOperations<azure.mgmt.web.v2016_09_01.operations.AppServiceEnvironmentsOperations>`\n           * 2018-02-01: :class:`AppServiceEnvironmentsOperations<azure.mgmt.web.v2018_02_01.operations.AppServiceEnvironmentsOperations>`\n           * 2019-08-01: :class:`AppServiceEnvironmentsOperations<azure.mgmt.web.v2019_08_01.operations.AppServiceEnvironmentsOperations>`\n           * 2020-06-01: :class:`AppServiceEnvironmentsOperations<azure.mgmt.web.v2020_06_01.operations.AppServiceEnvironmentsOperations>`\n           * 2020-09-01: :class:`AppServiceEnvironmentsOperations<azure.mgmt.web.v2020_09_01.operations.AppServiceEnvironmentsOperations>`\n           * 2020-12-01: :class:`AppServiceEnvironmentsOperations<azure.mgmt.web.v2020_12_01.operations.AppServiceEnvironmentsOperations>`\n           * 2021-01-01: :class:`AppServiceEnvironmentsOperations<azure.mgmt.web.v2021_01_01.operations.AppServiceEnvironmentsOperations>`\n           * 2021-01-15: :class:`AppServiceEnvironmentsOperations<azure.mgmt.web.v2021_01_15.operations.AppServiceEnvironmentsOperations>`\n           * 2022-09-01: :class:`AppServiceEnvironmentsOperations<azure.mgmt.web.v2022_09_01.operations.AppServiceEnvironmentsOperations>`\n        '
        api_version = self._get_api_version('app_service_environments')
        if api_version == '2016-09-01':
            from .v2016_09_01.operations import AppServiceEnvironmentsOperations as OperationClass
        elif api_version == '2018-02-01':
            from .v2018_02_01.operations import AppServiceEnvironmentsOperations as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import AppServiceEnvironmentsOperations as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import AppServiceEnvironmentsOperations as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import AppServiceEnvironmentsOperations as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import AppServiceEnvironmentsOperations as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import AppServiceEnvironmentsOperations as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import AppServiceEnvironmentsOperations as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import AppServiceEnvironmentsOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'app_service_environments'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    @property
    def app_service_plans(self):
        if False:
            return 10
        'Instance depends on the API version:\n\n           * 2016-09-01: :class:`AppServicePlansOperations<azure.mgmt.web.v2016_09_01.operations.AppServicePlansOperations>`\n           * 2018-02-01: :class:`AppServicePlansOperations<azure.mgmt.web.v2018_02_01.operations.AppServicePlansOperations>`\n           * 2019-08-01: :class:`AppServicePlansOperations<azure.mgmt.web.v2019_08_01.operations.AppServicePlansOperations>`\n           * 2020-06-01: :class:`AppServicePlansOperations<azure.mgmt.web.v2020_06_01.operations.AppServicePlansOperations>`\n           * 2020-09-01: :class:`AppServicePlansOperations<azure.mgmt.web.v2020_09_01.operations.AppServicePlansOperations>`\n           * 2020-12-01: :class:`AppServicePlansOperations<azure.mgmt.web.v2020_12_01.operations.AppServicePlansOperations>`\n           * 2021-01-01: :class:`AppServicePlansOperations<azure.mgmt.web.v2021_01_01.operations.AppServicePlansOperations>`\n           * 2021-01-15: :class:`AppServicePlansOperations<azure.mgmt.web.v2021_01_15.operations.AppServicePlansOperations>`\n           * 2022-09-01: :class:`AppServicePlansOperations<azure.mgmt.web.v2022_09_01.operations.AppServicePlansOperations>`\n        '
        api_version = self._get_api_version('app_service_plans')
        if api_version == '2016-09-01':
            from .v2016_09_01.operations import AppServicePlansOperations as OperationClass
        elif api_version == '2018-02-01':
            from .v2018_02_01.operations import AppServicePlansOperations as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import AppServicePlansOperations as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import AppServicePlansOperations as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import AppServicePlansOperations as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import AppServicePlansOperations as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import AppServicePlansOperations as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import AppServicePlansOperations as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import AppServicePlansOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'app_service_plans'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    @property
    def billing_meters(self):
        if False:
            for i in range(10):
                print('nop')
        'Instance depends on the API version:\n\n           * 2016-03-01: :class:`BillingMetersOperations<azure.mgmt.web.v2016_03_01.operations.BillingMetersOperations>`\n        '
        api_version = self._get_api_version('billing_meters')
        if api_version == '2016-03-01':
            from .v2016_03_01.operations import BillingMetersOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'billing_meters'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    @property
    def certificate_orders_diagnostics(self):
        if False:
            for i in range(10):
                print('nop')
        'Instance depends on the API version:\n\n           * 2020-12-01: :class:`CertificateOrdersDiagnosticsOperations<azure.mgmt.web.v2020_12_01.operations.CertificateOrdersDiagnosticsOperations>`\n           * 2021-01-01: :class:`CertificateOrdersDiagnosticsOperations<azure.mgmt.web.v2021_01_01.operations.CertificateOrdersDiagnosticsOperations>`\n           * 2021-01-15: :class:`CertificateOrdersDiagnosticsOperations<azure.mgmt.web.v2021_01_15.operations.CertificateOrdersDiagnosticsOperations>`\n           * 2022-09-01: :class:`CertificateOrdersDiagnosticsOperations<azure.mgmt.web.v2022_09_01.operations.CertificateOrdersDiagnosticsOperations>`\n        '
        api_version = self._get_api_version('certificate_orders_diagnostics')
        if api_version == '2020-12-01':
            from .v2020_12_01.operations import CertificateOrdersDiagnosticsOperations as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import CertificateOrdersDiagnosticsOperations as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import CertificateOrdersDiagnosticsOperations as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import CertificateOrdersDiagnosticsOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'certificate_orders_diagnostics'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    @property
    def certificate_registration_provider(self):
        if False:
            for i in range(10):
                print('nop')
        'Instance depends on the API version:\n\n           * 2015-08-01: :class:`CertificateRegistrationProviderOperations<azure.mgmt.web.v2015_08_01.operations.CertificateRegistrationProviderOperations>`\n           * 2018-02-01: :class:`CertificateRegistrationProviderOperations<azure.mgmt.web.v2018_02_01.operations.CertificateRegistrationProviderOperations>`\n           * 2019-08-01: :class:`CertificateRegistrationProviderOperations<azure.mgmt.web.v2019_08_01.operations.CertificateRegistrationProviderOperations>`\n           * 2020-06-01: :class:`CertificateRegistrationProviderOperations<azure.mgmt.web.v2020_06_01.operations.CertificateRegistrationProviderOperations>`\n           * 2020-09-01: :class:`CertificateRegistrationProviderOperations<azure.mgmt.web.v2020_09_01.operations.CertificateRegistrationProviderOperations>`\n           * 2020-12-01: :class:`CertificateRegistrationProviderOperations<azure.mgmt.web.v2020_12_01.operations.CertificateRegistrationProviderOperations>`\n           * 2021-01-01: :class:`CertificateRegistrationProviderOperations<azure.mgmt.web.v2021_01_01.operations.CertificateRegistrationProviderOperations>`\n           * 2021-01-15: :class:`CertificateRegistrationProviderOperations<azure.mgmt.web.v2021_01_15.operations.CertificateRegistrationProviderOperations>`\n           * 2022-09-01: :class:`CertificateRegistrationProviderOperations<azure.mgmt.web.v2022_09_01.operations.CertificateRegistrationProviderOperations>`\n        '
        api_version = self._get_api_version('certificate_registration_provider')
        if api_version == '2015-08-01':
            from .v2015_08_01.operations import CertificateRegistrationProviderOperations as OperationClass
        elif api_version == '2018-02-01':
            from .v2018_02_01.operations import CertificateRegistrationProviderOperations as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import CertificateRegistrationProviderOperations as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import CertificateRegistrationProviderOperations as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import CertificateRegistrationProviderOperations as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import CertificateRegistrationProviderOperations as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import CertificateRegistrationProviderOperations as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import CertificateRegistrationProviderOperations as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import CertificateRegistrationProviderOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'certificate_registration_provider'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    @property
    def certificates(self):
        if False:
            while True:
                i = 10
        'Instance depends on the API version:\n\n           * 2016-03-01: :class:`CertificatesOperations<azure.mgmt.web.v2016_03_01.operations.CertificatesOperations>`\n           * 2018-02-01: :class:`CertificatesOperations<azure.mgmt.web.v2018_02_01.operations.CertificatesOperations>`\n           * 2018-11-01: :class:`CertificatesOperations<azure.mgmt.web.v2018_11_01.operations.CertificatesOperations>`\n           * 2019-08-01: :class:`CertificatesOperations<azure.mgmt.web.v2019_08_01.operations.CertificatesOperations>`\n           * 2020-06-01: :class:`CertificatesOperations<azure.mgmt.web.v2020_06_01.operations.CertificatesOperations>`\n           * 2020-09-01: :class:`CertificatesOperations<azure.mgmt.web.v2020_09_01.operations.CertificatesOperations>`\n           * 2020-12-01: :class:`CertificatesOperations<azure.mgmt.web.v2020_12_01.operations.CertificatesOperations>`\n           * 2021-01-01: :class:`CertificatesOperations<azure.mgmt.web.v2021_01_01.operations.CertificatesOperations>`\n           * 2021-01-15: :class:`CertificatesOperations<azure.mgmt.web.v2021_01_15.operations.CertificatesOperations>`\n           * 2022-09-01: :class:`CertificatesOperations<azure.mgmt.web.v2022_09_01.operations.CertificatesOperations>`\n        '
        api_version = self._get_api_version('certificates')
        if api_version == '2016-03-01':
            from .v2016_03_01.operations import CertificatesOperations as OperationClass
        elif api_version == '2018-02-01':
            from .v2018_02_01.operations import CertificatesOperations as OperationClass
        elif api_version == '2018-11-01':
            from .v2018_11_01.operations import CertificatesOperations as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import CertificatesOperations as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import CertificatesOperations as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import CertificatesOperations as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import CertificatesOperations as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import CertificatesOperations as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import CertificatesOperations as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import CertificatesOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'certificates'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    @property
    def container_apps(self):
        if False:
            while True:
                i = 10
        'Instance depends on the API version:\n\n           * 2022-09-01: :class:`ContainerAppsOperations<azure.mgmt.web.v2022_09_01.operations.ContainerAppsOperations>`\n        '
        api_version = self._get_api_version('container_apps')
        if api_version == '2022-09-01':
            from .v2022_09_01.operations import ContainerAppsOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'container_apps'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    @property
    def container_apps_revisions(self):
        if False:
            i = 10
            return i + 15
        'Instance depends on the API version:\n\n           * 2022-09-01: :class:`ContainerAppsRevisionsOperations<azure.mgmt.web.v2022_09_01.operations.ContainerAppsRevisionsOperations>`\n        '
        api_version = self._get_api_version('container_apps_revisions')
        if api_version == '2022-09-01':
            from .v2022_09_01.operations import ContainerAppsRevisionsOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'container_apps_revisions'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    @property
    def deleted_web_apps(self):
        if False:
            return 10
        'Instance depends on the API version:\n\n           * 2016-03-01: :class:`DeletedWebAppsOperations<azure.mgmt.web.v2016_03_01.operations.DeletedWebAppsOperations>`\n           * 2018-02-01: :class:`DeletedWebAppsOperations<azure.mgmt.web.v2018_02_01.operations.DeletedWebAppsOperations>`\n           * 2019-08-01: :class:`DeletedWebAppsOperations<azure.mgmt.web.v2019_08_01.operations.DeletedWebAppsOperations>`\n           * 2020-06-01: :class:`DeletedWebAppsOperations<azure.mgmt.web.v2020_06_01.operations.DeletedWebAppsOperations>`\n           * 2020-09-01: :class:`DeletedWebAppsOperations<azure.mgmt.web.v2020_09_01.operations.DeletedWebAppsOperations>`\n           * 2020-12-01: :class:`DeletedWebAppsOperations<azure.mgmt.web.v2020_12_01.operations.DeletedWebAppsOperations>`\n           * 2021-01-01: :class:`DeletedWebAppsOperations<azure.mgmt.web.v2021_01_01.operations.DeletedWebAppsOperations>`\n           * 2021-01-15: :class:`DeletedWebAppsOperations<azure.mgmt.web.v2021_01_15.operations.DeletedWebAppsOperations>`\n           * 2022-09-01: :class:`DeletedWebAppsOperations<azure.mgmt.web.v2022_09_01.operations.DeletedWebAppsOperations>`\n        '
        api_version = self._get_api_version('deleted_web_apps')
        if api_version == '2016-03-01':
            from .v2016_03_01.operations import DeletedWebAppsOperations as OperationClass
        elif api_version == '2018-02-01':
            from .v2018_02_01.operations import DeletedWebAppsOperations as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import DeletedWebAppsOperations as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import DeletedWebAppsOperations as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import DeletedWebAppsOperations as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import DeletedWebAppsOperations as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import DeletedWebAppsOperations as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import DeletedWebAppsOperations as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import DeletedWebAppsOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'deleted_web_apps'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    @property
    def diagnostics(self):
        if False:
            for i in range(10):
                print('nop')
        'Instance depends on the API version:\n\n           * 2016-03-01: :class:`DiagnosticsOperations<azure.mgmt.web.v2016_03_01.operations.DiagnosticsOperations>`\n           * 2018-02-01: :class:`DiagnosticsOperations<azure.mgmt.web.v2018_02_01.operations.DiagnosticsOperations>`\n           * 2019-08-01: :class:`DiagnosticsOperations<azure.mgmt.web.v2019_08_01.operations.DiagnosticsOperations>`\n           * 2020-06-01: :class:`DiagnosticsOperations<azure.mgmt.web.v2020_06_01.operations.DiagnosticsOperations>`\n           * 2020-09-01: :class:`DiagnosticsOperations<azure.mgmt.web.v2020_09_01.operations.DiagnosticsOperations>`\n           * 2020-12-01: :class:`DiagnosticsOperations<azure.mgmt.web.v2020_12_01.operations.DiagnosticsOperations>`\n           * 2021-01-01: :class:`DiagnosticsOperations<azure.mgmt.web.v2021_01_01.operations.DiagnosticsOperations>`\n           * 2021-01-15: :class:`DiagnosticsOperations<azure.mgmt.web.v2021_01_15.operations.DiagnosticsOperations>`\n           * 2022-09-01: :class:`DiagnosticsOperations<azure.mgmt.web.v2022_09_01.operations.DiagnosticsOperations>`\n        '
        api_version = self._get_api_version('diagnostics')
        if api_version == '2016-03-01':
            from .v2016_03_01.operations import DiagnosticsOperations as OperationClass
        elif api_version == '2018-02-01':
            from .v2018_02_01.operations import DiagnosticsOperations as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import DiagnosticsOperations as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import DiagnosticsOperations as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import DiagnosticsOperations as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import DiagnosticsOperations as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import DiagnosticsOperations as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import DiagnosticsOperations as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import DiagnosticsOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'diagnostics'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    @property
    def domain_registration_provider(self):
        if False:
            while True:
                i = 10
        'Instance depends on the API version:\n\n           * 2015-04-01: :class:`DomainRegistrationProviderOperations<azure.mgmt.web.v2015_04_01.operations.DomainRegistrationProviderOperations>`\n           * 2018-02-01: :class:`DomainRegistrationProviderOperations<azure.mgmt.web.v2018_02_01.operations.DomainRegistrationProviderOperations>`\n           * 2019-08-01: :class:`DomainRegistrationProviderOperations<azure.mgmt.web.v2019_08_01.operations.DomainRegistrationProviderOperations>`\n           * 2020-06-01: :class:`DomainRegistrationProviderOperations<azure.mgmt.web.v2020_06_01.operations.DomainRegistrationProviderOperations>`\n           * 2020-09-01: :class:`DomainRegistrationProviderOperations<azure.mgmt.web.v2020_09_01.operations.DomainRegistrationProviderOperations>`\n           * 2020-12-01: :class:`DomainRegistrationProviderOperations<azure.mgmt.web.v2020_12_01.operations.DomainRegistrationProviderOperations>`\n           * 2021-01-01: :class:`DomainRegistrationProviderOperations<azure.mgmt.web.v2021_01_01.operations.DomainRegistrationProviderOperations>`\n           * 2021-01-15: :class:`DomainRegistrationProviderOperations<azure.mgmt.web.v2021_01_15.operations.DomainRegistrationProviderOperations>`\n           * 2022-09-01: :class:`DomainRegistrationProviderOperations<azure.mgmt.web.v2022_09_01.operations.DomainRegistrationProviderOperations>`\n        '
        api_version = self._get_api_version('domain_registration_provider')
        if api_version == '2015-04-01':
            from .v2015_04_01.operations import DomainRegistrationProviderOperations as OperationClass
        elif api_version == '2018-02-01':
            from .v2018_02_01.operations import DomainRegistrationProviderOperations as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import DomainRegistrationProviderOperations as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import DomainRegistrationProviderOperations as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import DomainRegistrationProviderOperations as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import DomainRegistrationProviderOperations as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import DomainRegistrationProviderOperations as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import DomainRegistrationProviderOperations as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import DomainRegistrationProviderOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'domain_registration_provider'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    @property
    def domains(self):
        if False:
            for i in range(10):
                print('nop')
        'Instance depends on the API version:\n\n           * 2015-04-01: :class:`DomainsOperations<azure.mgmt.web.v2015_04_01.operations.DomainsOperations>`\n           * 2018-02-01: :class:`DomainsOperations<azure.mgmt.web.v2018_02_01.operations.DomainsOperations>`\n           * 2019-08-01: :class:`DomainsOperations<azure.mgmt.web.v2019_08_01.operations.DomainsOperations>`\n           * 2020-06-01: :class:`DomainsOperations<azure.mgmt.web.v2020_06_01.operations.DomainsOperations>`\n           * 2020-09-01: :class:`DomainsOperations<azure.mgmt.web.v2020_09_01.operations.DomainsOperations>`\n           * 2020-12-01: :class:`DomainsOperations<azure.mgmt.web.v2020_12_01.operations.DomainsOperations>`\n           * 2021-01-01: :class:`DomainsOperations<azure.mgmt.web.v2021_01_01.operations.DomainsOperations>`\n           * 2021-01-15: :class:`DomainsOperations<azure.mgmt.web.v2021_01_15.operations.DomainsOperations>`\n           * 2022-09-01: :class:`DomainsOperations<azure.mgmt.web.v2022_09_01.operations.DomainsOperations>`\n        '
        api_version = self._get_api_version('domains')
        if api_version == '2015-04-01':
            from .v2015_04_01.operations import DomainsOperations as OperationClass
        elif api_version == '2018-02-01':
            from .v2018_02_01.operations import DomainsOperations as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import DomainsOperations as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import DomainsOperations as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import DomainsOperations as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import DomainsOperations as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import DomainsOperations as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import DomainsOperations as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import DomainsOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'domains'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    @property
    def global_operations(self):
        if False:
            i = 10
            return i + 15
        'Instance depends on the API version:\n\n           * 2020-12-01: :class:`GlobalOperations<azure.mgmt.web.v2020_12_01.operations.GlobalOperations>`\n           * 2021-01-01: :class:`GlobalOperations<azure.mgmt.web.v2021_01_01.operations.GlobalOperations>`\n           * 2021-01-15: :class:`GlobalOperations<azure.mgmt.web.v2021_01_15.operations.GlobalOperations>`\n           * 2022-09-01: :class:`GlobalOperations<azure.mgmt.web.v2022_09_01.operations.GlobalOperations>`\n        '
        api_version = self._get_api_version('global_operations')
        if api_version == '2020-12-01':
            from .v2020_12_01.operations import GlobalOperations as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import GlobalOperations as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import GlobalOperations as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import GlobalOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'global_operations'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    @property
    def kube_environments(self):
        if False:
            for i in range(10):
                print('nop')
        'Instance depends on the API version:\n\n           * 2021-01-01: :class:`KubeEnvironmentsOperations<azure.mgmt.web.v2021_01_01.operations.KubeEnvironmentsOperations>`\n           * 2021-01-15: :class:`KubeEnvironmentsOperations<azure.mgmt.web.v2021_01_15.operations.KubeEnvironmentsOperations>`\n           * 2022-09-01: :class:`KubeEnvironmentsOperations<azure.mgmt.web.v2022_09_01.operations.KubeEnvironmentsOperations>`\n        '
        api_version = self._get_api_version('kube_environments')
        if api_version == '2021-01-01':
            from .v2021_01_01.operations import KubeEnvironmentsOperations as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import KubeEnvironmentsOperations as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import KubeEnvironmentsOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'kube_environments'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    @property
    def provider(self):
        if False:
            for i in range(10):
                print('nop')
        'Instance depends on the API version:\n\n           * 2016-03-01: :class:`ProviderOperations<azure.mgmt.web.v2016_03_01.operations.ProviderOperations>`\n           * 2018-02-01: :class:`ProviderOperations<azure.mgmt.web.v2018_02_01.operations.ProviderOperations>`\n           * 2019-08-01: :class:`ProviderOperations<azure.mgmt.web.v2019_08_01.operations.ProviderOperations>`\n           * 2020-06-01: :class:`ProviderOperations<azure.mgmt.web.v2020_06_01.operations.ProviderOperations>`\n           * 2020-09-01: :class:`ProviderOperations<azure.mgmt.web.v2020_09_01.operations.ProviderOperations>`\n           * 2020-12-01: :class:`ProviderOperations<azure.mgmt.web.v2020_12_01.operations.ProviderOperations>`\n           * 2021-01-01: :class:`ProviderOperations<azure.mgmt.web.v2021_01_01.operations.ProviderOperations>`\n           * 2021-01-15: :class:`ProviderOperations<azure.mgmt.web.v2021_01_15.operations.ProviderOperations>`\n           * 2022-09-01: :class:`ProviderOperations<azure.mgmt.web.v2022_09_01.operations.ProviderOperations>`\n        '
        api_version = self._get_api_version('provider')
        if api_version == '2016-03-01':
            from .v2016_03_01.operations import ProviderOperations as OperationClass
        elif api_version == '2018-02-01':
            from .v2018_02_01.operations import ProviderOperations as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import ProviderOperations as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import ProviderOperations as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import ProviderOperations as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import ProviderOperations as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import ProviderOperations as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import ProviderOperations as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import ProviderOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'provider'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    @property
    def recommendations(self):
        if False:
            for i in range(10):
                print('nop')
        'Instance depends on the API version:\n\n           * 2016-03-01: :class:`RecommendationsOperations<azure.mgmt.web.v2016_03_01.operations.RecommendationsOperations>`\n           * 2018-02-01: :class:`RecommendationsOperations<azure.mgmt.web.v2018_02_01.operations.RecommendationsOperations>`\n           * 2019-08-01: :class:`RecommendationsOperations<azure.mgmt.web.v2019_08_01.operations.RecommendationsOperations>`\n           * 2020-06-01: :class:`RecommendationsOperations<azure.mgmt.web.v2020_06_01.operations.RecommendationsOperations>`\n           * 2020-09-01: :class:`RecommendationsOperations<azure.mgmt.web.v2020_09_01.operations.RecommendationsOperations>`\n           * 2020-12-01: :class:`RecommendationsOperations<azure.mgmt.web.v2020_12_01.operations.RecommendationsOperations>`\n           * 2021-01-01: :class:`RecommendationsOperations<azure.mgmt.web.v2021_01_01.operations.RecommendationsOperations>`\n           * 2021-01-15: :class:`RecommendationsOperations<azure.mgmt.web.v2021_01_15.operations.RecommendationsOperations>`\n           * 2022-09-01: :class:`RecommendationsOperations<azure.mgmt.web.v2022_09_01.operations.RecommendationsOperations>`\n        '
        api_version = self._get_api_version('recommendations')
        if api_version == '2016-03-01':
            from .v2016_03_01.operations import RecommendationsOperations as OperationClass
        elif api_version == '2018-02-01':
            from .v2018_02_01.operations import RecommendationsOperations as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import RecommendationsOperations as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import RecommendationsOperations as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import RecommendationsOperations as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import RecommendationsOperations as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import RecommendationsOperations as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import RecommendationsOperations as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import RecommendationsOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'recommendations'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    @property
    def resource_health_metadata(self):
        if False:
            for i in range(10):
                print('nop')
        'Instance depends on the API version:\n\n           * 2016-03-01: :class:`ResourceHealthMetadataOperations<azure.mgmt.web.v2016_03_01.operations.ResourceHealthMetadataOperations>`\n           * 2018-02-01: :class:`ResourceHealthMetadataOperations<azure.mgmt.web.v2018_02_01.operations.ResourceHealthMetadataOperations>`\n           * 2019-08-01: :class:`ResourceHealthMetadataOperations<azure.mgmt.web.v2019_08_01.operations.ResourceHealthMetadataOperations>`\n           * 2020-06-01: :class:`ResourceHealthMetadataOperations<azure.mgmt.web.v2020_06_01.operations.ResourceHealthMetadataOperations>`\n           * 2020-09-01: :class:`ResourceHealthMetadataOperations<azure.mgmt.web.v2020_09_01.operations.ResourceHealthMetadataOperations>`\n           * 2020-12-01: :class:`ResourceHealthMetadataOperations<azure.mgmt.web.v2020_12_01.operations.ResourceHealthMetadataOperations>`\n           * 2021-01-01: :class:`ResourceHealthMetadataOperations<azure.mgmt.web.v2021_01_01.operations.ResourceHealthMetadataOperations>`\n           * 2021-01-15: :class:`ResourceHealthMetadataOperations<azure.mgmt.web.v2021_01_15.operations.ResourceHealthMetadataOperations>`\n           * 2022-09-01: :class:`ResourceHealthMetadataOperations<azure.mgmt.web.v2022_09_01.operations.ResourceHealthMetadataOperations>`\n        '
        api_version = self._get_api_version('resource_health_metadata')
        if api_version == '2016-03-01':
            from .v2016_03_01.operations import ResourceHealthMetadataOperations as OperationClass
        elif api_version == '2018-02-01':
            from .v2018_02_01.operations import ResourceHealthMetadataOperations as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import ResourceHealthMetadataOperations as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import ResourceHealthMetadataOperations as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import ResourceHealthMetadataOperations as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import ResourceHealthMetadataOperations as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import ResourceHealthMetadataOperations as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import ResourceHealthMetadataOperations as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import ResourceHealthMetadataOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'resource_health_metadata'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    @property
    def static_sites(self):
        if False:
            while True:
                i = 10
        'Instance depends on the API version:\n\n           * 2019-08-01: :class:`StaticSitesOperations<azure.mgmt.web.v2019_08_01.operations.StaticSitesOperations>`\n           * 2020-06-01: :class:`StaticSitesOperations<azure.mgmt.web.v2020_06_01.operations.StaticSitesOperations>`\n           * 2020-09-01: :class:`StaticSitesOperations<azure.mgmt.web.v2020_09_01.operations.StaticSitesOperations>`\n           * 2020-12-01: :class:`StaticSitesOperations<azure.mgmt.web.v2020_12_01.operations.StaticSitesOperations>`\n           * 2021-01-01: :class:`StaticSitesOperations<azure.mgmt.web.v2021_01_01.operations.StaticSitesOperations>`\n           * 2021-01-15: :class:`StaticSitesOperations<azure.mgmt.web.v2021_01_15.operations.StaticSitesOperations>`\n           * 2022-09-01: :class:`StaticSitesOperations<azure.mgmt.web.v2022_09_01.operations.StaticSitesOperations>`\n        '
        api_version = self._get_api_version('static_sites')
        if api_version == '2019-08-01':
            from .v2019_08_01.operations import StaticSitesOperations as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import StaticSitesOperations as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import StaticSitesOperations as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import StaticSitesOperations as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import StaticSitesOperations as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import StaticSitesOperations as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import StaticSitesOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'static_sites'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    @property
    def top_level_domains(self):
        if False:
            for i in range(10):
                print('nop')
        'Instance depends on the API version:\n\n           * 2015-04-01: :class:`TopLevelDomainsOperations<azure.mgmt.web.v2015_04_01.operations.TopLevelDomainsOperations>`\n           * 2018-02-01: :class:`TopLevelDomainsOperations<azure.mgmt.web.v2018_02_01.operations.TopLevelDomainsOperations>`\n           * 2019-08-01: :class:`TopLevelDomainsOperations<azure.mgmt.web.v2019_08_01.operations.TopLevelDomainsOperations>`\n           * 2020-06-01: :class:`TopLevelDomainsOperations<azure.mgmt.web.v2020_06_01.operations.TopLevelDomainsOperations>`\n           * 2020-09-01: :class:`TopLevelDomainsOperations<azure.mgmt.web.v2020_09_01.operations.TopLevelDomainsOperations>`\n           * 2020-12-01: :class:`TopLevelDomainsOperations<azure.mgmt.web.v2020_12_01.operations.TopLevelDomainsOperations>`\n           * 2021-01-01: :class:`TopLevelDomainsOperations<azure.mgmt.web.v2021_01_01.operations.TopLevelDomainsOperations>`\n           * 2021-01-15: :class:`TopLevelDomainsOperations<azure.mgmt.web.v2021_01_15.operations.TopLevelDomainsOperations>`\n           * 2022-09-01: :class:`TopLevelDomainsOperations<azure.mgmt.web.v2022_09_01.operations.TopLevelDomainsOperations>`\n        '
        api_version = self._get_api_version('top_level_domains')
        if api_version == '2015-04-01':
            from .v2015_04_01.operations import TopLevelDomainsOperations as OperationClass
        elif api_version == '2018-02-01':
            from .v2018_02_01.operations import TopLevelDomainsOperations as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import TopLevelDomainsOperations as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import TopLevelDomainsOperations as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import TopLevelDomainsOperations as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import TopLevelDomainsOperations as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import TopLevelDomainsOperations as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import TopLevelDomainsOperations as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import TopLevelDomainsOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'top_level_domains'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    @property
    def web_apps(self):
        if False:
            while True:
                i = 10
        'Instance depends on the API version:\n\n           * 2016-08-01: :class:`WebAppsOperations<azure.mgmt.web.v2016_08_01.operations.WebAppsOperations>`\n           * 2018-02-01: :class:`WebAppsOperations<azure.mgmt.web.v2018_02_01.operations.WebAppsOperations>`\n           * 2019-08-01: :class:`WebAppsOperations<azure.mgmt.web.v2019_08_01.operations.WebAppsOperations>`\n           * 2020-06-01: :class:`WebAppsOperations<azure.mgmt.web.v2020_06_01.operations.WebAppsOperations>`\n           * 2020-09-01: :class:`WebAppsOperations<azure.mgmt.web.v2020_09_01.operations.WebAppsOperations>`\n           * 2020-12-01: :class:`WebAppsOperations<azure.mgmt.web.v2020_12_01.operations.WebAppsOperations>`\n           * 2021-01-01: :class:`WebAppsOperations<azure.mgmt.web.v2021_01_01.operations.WebAppsOperations>`\n           * 2021-01-15: :class:`WebAppsOperations<azure.mgmt.web.v2021_01_15.operations.WebAppsOperations>`\n           * 2022-09-01: :class:`WebAppsOperations<azure.mgmt.web.v2022_09_01.operations.WebAppsOperations>`\n        '
        api_version = self._get_api_version('web_apps')
        if api_version == '2016-08-01':
            from .v2016_08_01.operations import WebAppsOperations as OperationClass
        elif api_version == '2018-02-01':
            from .v2018_02_01.operations import WebAppsOperations as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import WebAppsOperations as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import WebAppsOperations as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import WebAppsOperations as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import WebAppsOperations as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import WebAppsOperations as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import WebAppsOperations as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import WebAppsOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'web_apps'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    @property
    def workflow_run_action_repetitions(self):
        if False:
            for i in range(10):
                print('nop')
        'Instance depends on the API version:\n\n           * 2022-09-01: :class:`WorkflowRunActionRepetitionsOperations<azure.mgmt.web.v2022_09_01.operations.WorkflowRunActionRepetitionsOperations>`\n        '
        api_version = self._get_api_version('workflow_run_action_repetitions')
        if api_version == '2022-09-01':
            from .v2022_09_01.operations import WorkflowRunActionRepetitionsOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'workflow_run_action_repetitions'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    @property
    def workflow_run_action_repetitions_request_histories(self):
        if False:
            while True:
                i = 10
        'Instance depends on the API version:\n\n           * 2022-09-01: :class:`WorkflowRunActionRepetitionsRequestHistoriesOperations<azure.mgmt.web.v2022_09_01.operations.WorkflowRunActionRepetitionsRequestHistoriesOperations>`\n        '
        api_version = self._get_api_version('workflow_run_action_repetitions_request_histories')
        if api_version == '2022-09-01':
            from .v2022_09_01.operations import WorkflowRunActionRepetitionsRequestHistoriesOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'workflow_run_action_repetitions_request_histories'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    @property
    def workflow_run_action_scope_repetitions(self):
        if False:
            return 10
        'Instance depends on the API version:\n\n           * 2022-09-01: :class:`WorkflowRunActionScopeRepetitionsOperations<azure.mgmt.web.v2022_09_01.operations.WorkflowRunActionScopeRepetitionsOperations>`\n        '
        api_version = self._get_api_version('workflow_run_action_scope_repetitions')
        if api_version == '2022-09-01':
            from .v2022_09_01.operations import WorkflowRunActionScopeRepetitionsOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'workflow_run_action_scope_repetitions'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    @property
    def workflow_run_actions(self):
        if False:
            i = 10
            return i + 15
        'Instance depends on the API version:\n\n           * 2022-09-01: :class:`WorkflowRunActionsOperations<azure.mgmt.web.v2022_09_01.operations.WorkflowRunActionsOperations>`\n        '
        api_version = self._get_api_version('workflow_run_actions')
        if api_version == '2022-09-01':
            from .v2022_09_01.operations import WorkflowRunActionsOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'workflow_run_actions'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    @property
    def workflow_runs(self):
        if False:
            for i in range(10):
                print('nop')
        'Instance depends on the API version:\n\n           * 2022-09-01: :class:`WorkflowRunsOperations<azure.mgmt.web.v2022_09_01.operations.WorkflowRunsOperations>`\n        '
        api_version = self._get_api_version('workflow_runs')
        if api_version == '2022-09-01':
            from .v2022_09_01.operations import WorkflowRunsOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'workflow_runs'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    @property
    def workflow_trigger_histories(self):
        if False:
            return 10
        'Instance depends on the API version:\n\n           * 2022-09-01: :class:`WorkflowTriggerHistoriesOperations<azure.mgmt.web.v2022_09_01.operations.WorkflowTriggerHistoriesOperations>`\n        '
        api_version = self._get_api_version('workflow_trigger_histories')
        if api_version == '2022-09-01':
            from .v2022_09_01.operations import WorkflowTriggerHistoriesOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'workflow_trigger_histories'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    @property
    def workflow_triggers(self):
        if False:
            while True:
                i = 10
        'Instance depends on the API version:\n\n           * 2022-09-01: :class:`WorkflowTriggersOperations<azure.mgmt.web.v2022_09_01.operations.WorkflowTriggersOperations>`\n        '
        api_version = self._get_api_version('workflow_triggers')
        if api_version == '2022-09-01':
            from .v2022_09_01.operations import WorkflowTriggersOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'workflow_triggers'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    @property
    def workflow_versions(self):
        if False:
            for i in range(10):
                print('nop')
        'Instance depends on the API version:\n\n           * 2022-09-01: :class:`WorkflowVersionsOperations<azure.mgmt.web.v2022_09_01.operations.WorkflowVersionsOperations>`\n        '
        api_version = self._get_api_version('workflow_versions')
        if api_version == '2022-09-01':
            from .v2022_09_01.operations import WorkflowVersionsOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'workflow_versions'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    @property
    def workflows(self):
        if False:
            i = 10
            return i + 15
        'Instance depends on the API version:\n\n           * 2022-09-01: :class:`WorkflowsOperations<azure.mgmt.web.v2022_09_01.operations.WorkflowsOperations>`\n        '
        api_version = self._get_api_version('workflows')
        if api_version == '2022-09-01':
            from .v2022_09_01.operations import WorkflowsOperations as OperationClass
        else:
            raise ValueError("API version {} does not have operation group 'workflows'".format(api_version))
        self._config.api_version = api_version
        return OperationClass(self._client, self._config, Serializer(self._models_dict(api_version)), Deserializer(self._models_dict(api_version)))

    def close(self):
        if False:
            while True:
                i = 10
        self._client.close()

    def __enter__(self):
        if False:
            return 10
        self._client.__enter__()
        return self

    def __exit__(self, *exc_details):
        if False:
            print('Hello World!')
        self._client.__exit__(*exc_details)