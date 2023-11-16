from ._serialization import Serializer, Deserializer
import sys
from typing import Any, IO, Iterable, Optional, Union
from azure.core.paging import ItemPaged
from . import models as _models
if sys.version_info >= (3, 9):
    from collections.abc import MutableMapping
else:
    from typing import MutableMapping
JSON = MutableMapping[str, Any]

class WebSiteManagementClientOperationsMixin(object):

    def check_name_availability(self, name: str, type: Union[str, _models.CheckNameResourceTypes], is_fqdn: Optional[bool]=None, **kwargs: Any) -> _models.ResourceNameAvailability:
        if False:
            for i in range(10):
                print('nop')
        'Check if a resource name is available.\n\n        Description for Check if a resource name is available.\n\n        :param name: Resource name to verify. Required.\n        :type name: str\n        :param type: Resource type used for verification. Known values are: "Site", "Slot",\n         "HostingEnvironment", "PublishingUser", "Microsoft.Web/sites", "Microsoft.Web/sites/slots",\n         "Microsoft.Web/hostingEnvironments", and "Microsoft.Web/publishingUsers". Required.\n        :type type: str or ~azure.mgmt.web.v2022_09_01.models.CheckNameResourceTypes\n        :param is_fqdn: Is fully qualified domain name. Default value is None.\n        :type is_fqdn: bool\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ResourceNameAvailability or the result of cls(response)\n        :rtype: ~azure.mgmt.web.v2022_09_01.models.ResourceNameAvailability\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        api_version = self._get_api_version('check_name_availability')
        if api_version == '2016-03-01':
            from .v2016_03_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2018-02-01':
            from .v2018_02_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        else:
            raise ValueError("API version {} does not have operation 'check_name_availability'".format(api_version))
        mixin_instance = OperationClass()
        mixin_instance._client = self._client
        mixin_instance._config = self._config
        mixin_instance._config.api_version = api_version
        mixin_instance._serialize = Serializer(self._models_dict(api_version))
        mixin_instance._serialize.client_side_validation = False
        mixin_instance._deserialize = Deserializer(self._models_dict(api_version))
        return mixin_instance.check_name_availability(name, type, is_fqdn, **kwargs)

    def get_publishing_user(self, **kwargs: Any) -> _models.User:
        if False:
            while True:
                i = 10
        'Gets publishing user.\n\n        Description for Gets publishing user.\n\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: User or the result of cls(response)\n        :rtype: ~azure.mgmt.web.v2022_09_01.models.User\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        api_version = self._get_api_version('get_publishing_user')
        if api_version == '2016-03-01':
            from .v2016_03_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2018-02-01':
            from .v2018_02_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        else:
            raise ValueError("API version {} does not have operation 'get_publishing_user'".format(api_version))
        mixin_instance = OperationClass()
        mixin_instance._client = self._client
        mixin_instance._config = self._config
        mixin_instance._config.api_version = api_version
        mixin_instance._serialize = Serializer(self._models_dict(api_version))
        mixin_instance._serialize.client_side_validation = False
        mixin_instance._deserialize = Deserializer(self._models_dict(api_version))
        return mixin_instance.get_publishing_user(**kwargs)

    def get_source_control(self, source_control_type: str, **kwargs: Any) -> _models.SourceControl:
        if False:
            for i in range(10):
                print('nop')
        'Gets source control token.\n\n        Description for Gets source control token.\n\n        :param source_control_type: Type of source control. Required.\n        :type source_control_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: SourceControl or the result of cls(response)\n        :rtype: ~azure.mgmt.web.v2022_09_01.models.SourceControl\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        api_version = self._get_api_version('get_source_control')
        if api_version == '2016-03-01':
            from .v2016_03_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2018-02-01':
            from .v2018_02_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        else:
            raise ValueError("API version {} does not have operation 'get_source_control'".format(api_version))
        mixin_instance = OperationClass()
        mixin_instance._client = self._client
        mixin_instance._config = self._config
        mixin_instance._config.api_version = api_version
        mixin_instance._serialize = Serializer(self._models_dict(api_version))
        mixin_instance._serialize.client_side_validation = False
        mixin_instance._deserialize = Deserializer(self._models_dict(api_version))
        return mixin_instance.get_source_control(source_control_type, **kwargs)

    def get_subscription_deployment_locations(self, **kwargs: Any) -> _models.DeploymentLocations:
        if False:
            for i in range(10):
                print('nop')
        'Gets list of available geo regions plus ministamps.\n\n        Description for Gets list of available geo regions plus ministamps.\n\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: DeploymentLocations or the result of cls(response)\n        :rtype: ~azure.mgmt.web.v2022_09_01.models.DeploymentLocations\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        api_version = self._get_api_version('get_subscription_deployment_locations')
        if api_version == '2016-03-01':
            from .v2016_03_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2018-02-01':
            from .v2018_02_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        else:
            raise ValueError("API version {} does not have operation 'get_subscription_deployment_locations'".format(api_version))
        mixin_instance = OperationClass()
        mixin_instance._client = self._client
        mixin_instance._config = self._config
        mixin_instance._config.api_version = api_version
        mixin_instance._serialize = Serializer(self._models_dict(api_version))
        mixin_instance._serialize.client_side_validation = False
        mixin_instance._deserialize = Deserializer(self._models_dict(api_version))
        return mixin_instance.get_subscription_deployment_locations(**kwargs)

    def list_billing_meters(self, billing_location: Optional[str]=None, os_type: Optional[str]=None, **kwargs: Any) -> Iterable['_models.BillingMeter']:
        if False:
            while True:
                i = 10
        'Gets a list of meters for a given location.\n\n        Description for Gets a list of meters for a given location.\n\n        :param billing_location: Azure Location of billable resource. Default value is None.\n        :type billing_location: str\n        :param os_type: App Service OS type meters used for. Default value is None.\n        :type os_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either BillingMeter or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.web.v2022_09_01.models.BillingMeter]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        api_version = self._get_api_version('list_billing_meters')
        if api_version == '2018-02-01':
            from .v2018_02_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        else:
            raise ValueError("API version {} does not have operation 'list_billing_meters'".format(api_version))
        mixin_instance = OperationClass()
        mixin_instance._client = self._client
        mixin_instance._config = self._config
        mixin_instance._config.api_version = api_version
        mixin_instance._serialize = Serializer(self._models_dict(api_version))
        mixin_instance._serialize.client_side_validation = False
        mixin_instance._deserialize = Deserializer(self._models_dict(api_version))
        return mixin_instance.list_billing_meters(billing_location, os_type, **kwargs)

    def list_custom_host_name_sites(self, hostname: Optional[str]=None, **kwargs: Any) -> Iterable['_models.CustomHostnameSites']:
        if False:
            while True:
                i = 10
        'Get custom hostnames under this subscription.\n\n        Get custom hostnames under this subscription.\n\n        :param hostname: Specific hostname. Default value is None.\n        :type hostname: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either CustomHostnameSites or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.web.v2022_09_01.models.CustomHostnameSites]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        api_version = self._get_api_version('list_custom_host_name_sites')
        if api_version == '2022-09-01':
            from .v2022_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        else:
            raise ValueError("API version {} does not have operation 'list_custom_host_name_sites'".format(api_version))
        mixin_instance = OperationClass()
        mixin_instance._client = self._client
        mixin_instance._config = self._config
        mixin_instance._config.api_version = api_version
        mixin_instance._serialize = Serializer(self._models_dict(api_version))
        mixin_instance._serialize.client_side_validation = False
        mixin_instance._deserialize = Deserializer(self._models_dict(api_version))
        return mixin_instance.list_custom_host_name_sites(hostname, **kwargs)

    def list_geo_regions(self, sku: Optional[Union[str, _models.SkuName]]=None, linux_workers_enabled: Optional[bool]=None, xenon_workers_enabled: Optional[bool]=None, linux_dynamic_workers_enabled: Optional[bool]=None, **kwargs: Any) -> Iterable['_models.GeoRegion']:
        if False:
            for i in range(10):
                print('nop')
        'Get a list of available geographical regions.\n\n        Description for Get a list of available geographical regions.\n\n        :param sku: Name of SKU used to filter the regions. Known values are: "Free", "Shared",\n         "Basic", "Standard", "Premium", "Dynamic", "Isolated", "IsolatedV2", "PremiumV2", "PremiumV3",\n         "PremiumContainer", "ElasticPremium", and "ElasticIsolated". Default value is None.\n        :type sku: str or ~azure.mgmt.web.v2022_09_01.models.SkuName\n        :param linux_workers_enabled: Specify :code:`<code>true</code>` if you want to filter to only\n         regions that support Linux workers. Default value is None.\n        :type linux_workers_enabled: bool\n        :param xenon_workers_enabled: Specify :code:`<code>true</code>` if you want to filter to only\n         regions that support Xenon workers. Default value is None.\n        :type xenon_workers_enabled: bool\n        :param linux_dynamic_workers_enabled: Specify :code:`<code>true</code>` if you want to filter\n         to only regions that support Linux Consumption Workers. Default value is None.\n        :type linux_dynamic_workers_enabled: bool\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either GeoRegion or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.web.v2022_09_01.models.GeoRegion]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        api_version = self._get_api_version('list_geo_regions')
        if api_version == '2016-03-01':
            from .v2016_03_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2018-02-01':
            from .v2018_02_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        else:
            raise ValueError("API version {} does not have operation 'list_geo_regions'".format(api_version))
        mixin_instance = OperationClass()
        mixin_instance._client = self._client
        mixin_instance._config = self._config
        mixin_instance._config.api_version = api_version
        mixin_instance._serialize = Serializer(self._models_dict(api_version))
        mixin_instance._serialize.client_side_validation = False
        mixin_instance._deserialize = Deserializer(self._models_dict(api_version))
        return mixin_instance.list_geo_regions(sku, linux_workers_enabled, xenon_workers_enabled, linux_dynamic_workers_enabled, **kwargs)

    def list_premier_add_on_offers(self, **kwargs: Any) -> Iterable['_models.PremierAddOnOffer']:
        if False:
            while True:
                i = 10
        'List all premier add-on offers.\n\n        Description for List all premier add-on offers.\n\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either PremierAddOnOffer or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.web.v2022_09_01.models.PremierAddOnOffer]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        api_version = self._get_api_version('list_premier_add_on_offers')
        if api_version == '2016-03-01':
            from .v2016_03_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2018-02-01':
            from .v2018_02_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        else:
            raise ValueError("API version {} does not have operation 'list_premier_add_on_offers'".format(api_version))
        mixin_instance = OperationClass()
        mixin_instance._client = self._client
        mixin_instance._config = self._config
        mixin_instance._config.api_version = api_version
        mixin_instance._serialize = Serializer(self._models_dict(api_version))
        mixin_instance._serialize.client_side_validation = False
        mixin_instance._deserialize = Deserializer(self._models_dict(api_version))
        return mixin_instance.list_premier_add_on_offers(**kwargs)

    def list_site_identifiers_assigned_to_host_name(self, name_identifier: Union[_models.NameIdentifier, IO], **kwargs: Any) -> Iterable['_models.Identifier']:
        if False:
            while True:
                i = 10
        "List all apps that are assigned to a hostname.\n\n        Description for List all apps that are assigned to a hostname.\n\n        :param name_identifier: Hostname information. Is either a NameIdentifier type or a IO type.\n         Required.\n        :type name_identifier: ~azure.mgmt.web.v2022_09_01.models.NameIdentifier or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either Identifier or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.web.v2022_09_01.models.Identifier]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        api_version = self._get_api_version('list_site_identifiers_assigned_to_host_name')
        if api_version == '2016-03-01':
            from .v2016_03_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2018-02-01':
            from .v2018_02_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        else:
            raise ValueError("API version {} does not have operation 'list_site_identifiers_assigned_to_host_name'".format(api_version))
        mixin_instance = OperationClass()
        mixin_instance._client = self._client
        mixin_instance._config = self._config
        mixin_instance._config.api_version = api_version
        mixin_instance._serialize = Serializer(self._models_dict(api_version))
        mixin_instance._serialize.client_side_validation = False
        mixin_instance._deserialize = Deserializer(self._models_dict(api_version))
        return mixin_instance.list_site_identifiers_assigned_to_host_name(name_identifier, **kwargs)

    def list_skus(self, **kwargs: Any) -> _models.SkuInfos:
        if False:
            i = 10
            return i + 15
        'List all SKUs.\n\n        Description for List all SKUs.\n\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: SkuInfos or the result of cls(response)\n        :rtype: ~azure.mgmt.web.v2022_09_01.models.SkuInfos\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        api_version = self._get_api_version('list_skus')
        if api_version == '2016-03-01':
            from .v2016_03_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2018-02-01':
            from .v2018_02_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        else:
            raise ValueError("API version {} does not have operation 'list_skus'".format(api_version))
        mixin_instance = OperationClass()
        mixin_instance._client = self._client
        mixin_instance._config = self._config
        mixin_instance._config.api_version = api_version
        mixin_instance._serialize = Serializer(self._models_dict(api_version))
        mixin_instance._serialize.client_side_validation = False
        mixin_instance._deserialize = Deserializer(self._models_dict(api_version))
        return mixin_instance.list_skus(**kwargs)

    def list_source_controls(self, **kwargs: Any) -> Iterable['_models.SourceControl']:
        if False:
            for i in range(10):
                print('nop')
        'Gets the source controls available for Azure websites.\n\n        Description for Gets the source controls available for Azure websites.\n\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: An iterator like instance of either SourceControl or the result of cls(response)\n        :rtype: ~azure.core.paging.ItemPaged[~azure.mgmt.web.v2022_09_01.models.SourceControl]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        api_version = self._get_api_version('list_source_controls')
        if api_version == '2016-03-01':
            from .v2016_03_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2018-02-01':
            from .v2018_02_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        else:
            raise ValueError("API version {} does not have operation 'list_source_controls'".format(api_version))
        mixin_instance = OperationClass()
        mixin_instance._client = self._client
        mixin_instance._config = self._config
        mixin_instance._config.api_version = api_version
        mixin_instance._serialize = Serializer(self._models_dict(api_version))
        mixin_instance._serialize.client_side_validation = False
        mixin_instance._deserialize = Deserializer(self._models_dict(api_version))
        return mixin_instance.list_source_controls(**kwargs)

    def move(self, resource_group_name: str, move_resource_envelope: Union[_models.CsmMoveResourceEnvelope, IO], **kwargs: Any) -> None:
        if False:
            return 10
        "Move resources between resource groups.\n\n        Description for Move resources between resource groups.\n\n        :param resource_group_name: Name of the resource group to which the resource belongs. Required.\n        :type resource_group_name: str\n        :param move_resource_envelope: Object that represents the resource to move. Is either a\n         CsmMoveResourceEnvelope type or a IO type. Required.\n        :type move_resource_envelope: ~azure.mgmt.web.v2022_09_01.models.CsmMoveResourceEnvelope or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        api_version = self._get_api_version('move')
        if api_version == '2016-03-01':
            from .v2016_03_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2018-02-01':
            from .v2018_02_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        else:
            raise ValueError("API version {} does not have operation 'move'".format(api_version))
        mixin_instance = OperationClass()
        mixin_instance._client = self._client
        mixin_instance._config = self._config
        mixin_instance._config.api_version = api_version
        mixin_instance._serialize = Serializer(self._models_dict(api_version))
        mixin_instance._serialize.client_side_validation = False
        mixin_instance._deserialize = Deserializer(self._models_dict(api_version))
        return mixin_instance.move(resource_group_name, move_resource_envelope, **kwargs)

    def update_publishing_user(self, user_details: Union[_models.User, IO], **kwargs: Any) -> _models.User:
        if False:
            return 10
        "Updates publishing user.\n\n        Description for Updates publishing user.\n\n        :param user_details: Details of publishing user. Is either a User type or a IO type. Required.\n        :type user_details: ~azure.mgmt.web.v2022_09_01.models.User or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: User or the result of cls(response)\n        :rtype: ~azure.mgmt.web.v2022_09_01.models.User\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        api_version = self._get_api_version('update_publishing_user')
        if api_version == '2016-03-01':
            from .v2016_03_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2018-02-01':
            from .v2018_02_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        else:
            raise ValueError("API version {} does not have operation 'update_publishing_user'".format(api_version))
        mixin_instance = OperationClass()
        mixin_instance._client = self._client
        mixin_instance._config = self._config
        mixin_instance._config.api_version = api_version
        mixin_instance._serialize = Serializer(self._models_dict(api_version))
        mixin_instance._serialize.client_side_validation = False
        mixin_instance._deserialize = Deserializer(self._models_dict(api_version))
        return mixin_instance.update_publishing_user(user_details, **kwargs)

    def update_source_control(self, source_control_type: str, request_message: Union[_models.SourceControl, IO], **kwargs: Any) -> _models.SourceControl:
        if False:
            while True:
                i = 10
        "Updates source control token.\n\n        Description for Updates source control token.\n\n        :param source_control_type: Type of source control. Required.\n        :type source_control_type: str\n        :param request_message: Source control token information. Is either a SourceControl type or a\n         IO type. Required.\n        :type request_message: ~azure.mgmt.web.v2022_09_01.models.SourceControl or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: SourceControl or the result of cls(response)\n        :rtype: ~azure.mgmt.web.v2022_09_01.models.SourceControl\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        api_version = self._get_api_version('update_source_control')
        if api_version == '2016-03-01':
            from .v2016_03_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2018-02-01':
            from .v2018_02_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        else:
            raise ValueError("API version {} does not have operation 'update_source_control'".format(api_version))
        mixin_instance = OperationClass()
        mixin_instance._client = self._client
        mixin_instance._config = self._config
        mixin_instance._config.api_version = api_version
        mixin_instance._serialize = Serializer(self._models_dict(api_version))
        mixin_instance._serialize.client_side_validation = False
        mixin_instance._deserialize = Deserializer(self._models_dict(api_version))
        return mixin_instance.update_source_control(source_control_type, request_message, **kwargs)

    def validate(self, resource_group_name: str, validate_request: Union[_models.ValidateRequest, IO], **kwargs: Any) -> _models.ValidateResponse:
        if False:
            for i in range(10):
                print('nop')
        "Validate if a resource can be created.\n\n        Description for Validate if a resource can be created.\n\n        :param resource_group_name: Name of the resource group to which the resource belongs. Required.\n        :type resource_group_name: str\n        :param validate_request: Request with the resources to validate. Is either a ValidateRequest\n         type or a IO type. Required.\n        :type validate_request: ~azure.mgmt.web.v2022_09_01.models.ValidateRequest or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: ValidateResponse or the result of cls(response)\n        :rtype: ~azure.mgmt.web.v2022_09_01.models.ValidateResponse\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        api_version = self._get_api_version('validate')
        if api_version == '2016-03-01':
            from .v2016_03_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2018-02-01':
            from .v2018_02_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        else:
            raise ValueError("API version {} does not have operation 'validate'".format(api_version))
        mixin_instance = OperationClass()
        mixin_instance._client = self._client
        mixin_instance._config = self._config
        mixin_instance._config.api_version = api_version
        mixin_instance._serialize = Serializer(self._models_dict(api_version))
        mixin_instance._serialize.client_side_validation = False
        mixin_instance._deserialize = Deserializer(self._models_dict(api_version))
        return mixin_instance.validate(resource_group_name, validate_request, **kwargs)

    def validate_container_settings(self, resource_group_name: str, validate_container_settings_request: Union[_models.ValidateContainerSettingsRequest, IO], **kwargs: Any) -> JSON:
        if False:
            i = 10
            return i + 15
        "Validate if the container settings are correct.\n\n        Validate if the container settings are correct.\n\n        :param resource_group_name: Name of the resource group to which the resource belongs. Required.\n        :type resource_group_name: str\n        :param validate_container_settings_request: Is either a ValidateContainerSettingsRequest type\n         or a IO type. Required.\n        :type validate_container_settings_request:\n         ~azure.mgmt.web.v2018_02_01.models.ValidateContainerSettingsRequest or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: JSON or the result of cls(response)\n        :rtype: JSON\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        api_version = self._get_api_version('validate_container_settings')
        if api_version == '2018-02-01':
            from .v2018_02_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        else:
            raise ValueError("API version {} does not have operation 'validate_container_settings'".format(api_version))
        mixin_instance = OperationClass()
        mixin_instance._client = self._client
        mixin_instance._config = self._config
        mixin_instance._config.api_version = api_version
        mixin_instance._serialize = Serializer(self._models_dict(api_version))
        mixin_instance._serialize.client_side_validation = False
        mixin_instance._deserialize = Deserializer(self._models_dict(api_version))
        return mixin_instance.validate_container_settings(resource_group_name, validate_container_settings_request, **kwargs)

    def validate_move(self, resource_group_name: str, move_resource_envelope: Union[_models.CsmMoveResourceEnvelope, IO], **kwargs: Any) -> None:
        if False:
            return 10
        "Validate whether a resource can be moved.\n\n        Description for Validate whether a resource can be moved.\n\n        :param resource_group_name: Name of the resource group to which the resource belongs. Required.\n        :type resource_group_name: str\n        :param move_resource_envelope: Object that represents the resource to move. Is either a\n         CsmMoveResourceEnvelope type or a IO type. Required.\n        :type move_resource_envelope: ~azure.mgmt.web.v2022_09_01.models.CsmMoveResourceEnvelope or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: None or the result of cls(response)\n        :rtype: None\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        api_version = self._get_api_version('validate_move')
        if api_version == '2016-03-01':
            from .v2016_03_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2018-02-01':
            from .v2018_02_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        else:
            raise ValueError("API version {} does not have operation 'validate_move'".format(api_version))
        mixin_instance = OperationClass()
        mixin_instance._client = self._client
        mixin_instance._config = self._config
        mixin_instance._config.api_version = api_version
        mixin_instance._serialize = Serializer(self._models_dict(api_version))
        mixin_instance._serialize.client_side_validation = False
        mixin_instance._deserialize = Deserializer(self._models_dict(api_version))
        return mixin_instance.validate_move(resource_group_name, move_resource_envelope, **kwargs)

    def verify_hosting_environment_vnet(self, parameters: Union[_models.VnetParameters, IO], **kwargs: Any) -> _models.VnetValidationFailureDetails:
        if False:
            i = 10
            return i + 15
        "Verifies if this VNET is compatible with an App Service Environment by analyzing the Network\n        Security Group rules.\n\n        Description for Verifies if this VNET is compatible with an App Service Environment by\n        analyzing the Network Security Group rules.\n\n        :param parameters: VNET information. Is either a VnetParameters type or a IO type. Required.\n        :type parameters: ~azure.mgmt.web.v2022_09_01.models.VnetParameters or IO\n        :keyword content_type: Body Parameter content-type. Known values are: 'application/json'.\n         Default value is None.\n        :paramtype content_type: str\n        :keyword callable cls: A custom type or function that will be passed the direct response\n        :return: VnetValidationFailureDetails or the result of cls(response)\n        :rtype: ~azure.mgmt.web.v2022_09_01.models.VnetValidationFailureDetails\n        :raises ~azure.core.exceptions.HttpResponseError:\n        "
        api_version = self._get_api_version('verify_hosting_environment_vnet')
        if api_version == '2016-03-01':
            from .v2016_03_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2018-02-01':
            from .v2018_02_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2019-08-01':
            from .v2019_08_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-06-01':
            from .v2020_06_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-09-01':
            from .v2020_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2020-12-01':
            from .v2020_12_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-01':
            from .v2021_01_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2021-01-15':
            from .v2021_01_15.operations import WebSiteManagementClientOperationsMixin as OperationClass
        elif api_version == '2022-09-01':
            from .v2022_09_01.operations import WebSiteManagementClientOperationsMixin as OperationClass
        else:
            raise ValueError("API version {} does not have operation 'verify_hosting_environment_vnet'".format(api_version))
        mixin_instance = OperationClass()
        mixin_instance._client = self._client
        mixin_instance._config = self._config
        mixin_instance._config.api_version = api_version
        mixin_instance._serialize = Serializer(self._models_dict(api_version))
        mixin_instance._serialize.client_side_validation = False
        mixin_instance._deserialize = Deserializer(self._models_dict(api_version))
        return mixin_instance.verify_hosting_environment_vnet(parameters, **kwargs)