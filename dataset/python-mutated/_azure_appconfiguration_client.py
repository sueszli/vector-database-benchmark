import binascii
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Union, cast, overload
from typing_extensions import Literal
from azure.core import MatchConditions
from azure.core.paging import ItemPaged
from azure.core.credentials import TokenCredential, AzureKeyCredential
from azure.core.pipeline.policies import BearerTokenCredentialPolicy
from azure.core.polling import LROPoller
from azure.core.tracing.decorator import distributed_trace
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError, ResourceModifiedError, ResourceNotModifiedError
from azure.core.utils import CaseInsensitiveDict
from ._azure_appconfiguration_error import ResourceReadOnlyError
from ._azure_appconfiguration_requests import AppConfigRequestsCredentialsPolicy
from ._generated import AzureAppConfiguration
from ._generated.models import SnapshotUpdateParameters, SnapshotStatus
from ._models import ConfigurationSetting, ConfigurationSettingsFilter, ConfigurationSnapshot
from ._utils import prep_if_match, prep_if_none_match, get_key_filter, get_label_filter, parse_connection_string
from ._sync_token import SyncTokenPolicy

class AzureAppConfigurationClient:
    """Represents a client that calls restful API of Azure App Configuration service.

    :param str base_url: Base url of the service.
    :param credential: An object which can provide secrets for the app configuration service
    :type credential: ~azure.core.credentials.TokenCredential
    :keyword api_version: Api Version. Default value is "2023-10-01". Note that overriding this default
        value may result in unsupported behavior.
    :paramtype api_version: str

    """

    def __init__(self, base_url: str, credential: TokenCredential, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        try:
            if not base_url.lower().startswith('http'):
                base_url = f'https://{base_url}'
        except AttributeError as exc:
            raise ValueError('Base URL must be a string.') from exc
        if not credential:
            raise ValueError('Missing credential')
        credential_scopes = [f"{base_url.strip('/')}/.default"]
        self._sync_token_policy = SyncTokenPolicy()
        if isinstance(credential, AzureKeyCredential):
            id_credential = kwargs.pop('id_credential')
            kwargs.update({'authentication_policy': AppConfigRequestsCredentialsPolicy(credential, base_url, id_credential)})
        elif isinstance(credential, TokenCredential):
            kwargs.update({'authentication_policy': BearerTokenCredentialPolicy(credential, *credential_scopes, **kwargs)})
        else:
            raise TypeError(f'Unsupported credential: {type(credential)}. Use an instance of token credential from azure.identity')
        self._impl = AzureAppConfiguration(credential, base_url, per_call_policies=self._sync_token_policy, **kwargs)

    @classmethod
    def from_connection_string(cls, connection_string: str, **kwargs: Any) -> 'AzureAppConfigurationClient':
        if False:
            i = 10
            return i + 15
        'Create AzureAppConfigurationClient from a Connection String.\n\n        :param str connection_string: Connection String\n            (one of the access keys of the Azure App Configuration resource)\n            used to access the Azure App Configuration.\n        :return: An AzureAppConfigurationClient authenticated with the connection string\n        :rtype: ~azure.appconfiguration.AzureAppConfigurationClient\n\n        Example\n\n        .. code-block:: python\n\n            from azure.appconfiguration import AzureAppConfigurationClient\n            connection_str = "<my connection string>"\n            client = AzureAppConfigurationClient.from_connection_string(connection_str)\n        '
        (endpoint, id_credential, secret) = parse_connection_string(connection_string)
        return cls(credential=AzureKeyCredential(secret), base_url=endpoint, id_credential=id_credential, **kwargs)

    @overload
    def list_configuration_settings(self, *, key_filter: Optional[str]=None, label_filter: Optional[str]=None, accept_datetime: Optional[Union[datetime, str]]=None, fields: Optional[List[str]]=None, **kwargs: Any) -> ItemPaged[ConfigurationSetting]:
        if False:
            for i in range(10):
                print('nop')
        'List the configuration settings stored in the configuration service, optionally filtered by\n        key, label and accept_datetime.\n\n        :keyword key_filter: filter results based on their keys. \'*\' can be\n            used as wildcard in the beginning or end of the filter\n        :paramtype key_filter: str or None\n        :keyword label_filter: filter results based on their label. \'*\' can be\n            used as wildcard in the beginning or end of the filter\n        :paramtype label_filter: str or None\n        :keyword accept_datetime: retrieve ConfigurationSetting existed at this datetime\n        :paramtype accept_datetime: ~datetime.datetime or str or None\n        :keyword list[str] fields: specify which fields to include in the results. Leave None to include all fields\n        :return: An iterator of :class:`~azure.appconfiguration.ConfigurationSetting`\n        :rtype: ~azure.core.paging.ItemPaged[~azure.appconfiguration.ConfigurationSetting]\n        :raises: :class:`~azure.core.exceptions.HttpResponseError`,             :class:`~azure.core.exceptions.ClientAuthenticationError`\n\n        Example\n\n        .. code-block:: python\n\n            from datetime import datetime, timedelta\n\n            accept_datetime = datetime.utcnow() + timedelta(days=-1)\n\n            all_listed = client.list_configuration_settings()\n            for item in all_listed:\n                pass  # do something\n\n            filtered_listed = client.list_configuration_settings(\n                label_filter="Labe*", key_filter="Ke*", accept_datetime=str(accept_datetime)\n            )\n            for item in filtered_listed:\n                pass  # do something\n        '

    @overload
    def list_configuration_settings(self, *, snapshot_name: str, fields: Optional[List[str]]=None, **kwargs: Any) -> ItemPaged[ConfigurationSetting]:
        if False:
            while True:
                i = 10
        'List the configuration settings stored under a snapshot in the configuration service, optionally filtered by\n        fields to present in return.\n\n        :keyword str snapshot_name: The snapshot name.\n        :keyword fields: Specify which fields to include in the results. Leave None to include all fields.\n        :type fields: list[str] or None\n        :return: An iterator of :class:`~azure.appconfiguration.ConfigurationSetting`\n        :rtype: ~azure.core.paging.ItemPaged[~azure.appconfiguration.ConfigurationSetting]\n        :raises: :class:`~azure.core.exceptions.HttpResponseError`\n        '

    @distributed_trace
    def list_configuration_settings(self, *args, **kwargs) -> ItemPaged[ConfigurationSetting]:
        if False:
            print('Hello World!')
        accept_datetime = kwargs.pop('accept_datetime', None)
        if isinstance(accept_datetime, datetime):
            accept_datetime = str(accept_datetime)
        select = kwargs.pop('fields', None)
        if select:
            select = ['locked' if x == 'read_only' else x for x in select]
        snapshot_name = kwargs.pop('snapshot_name', None)
        try:
            if snapshot_name is not None:
                return self._impl.get_key_values(snapshot=snapshot_name, accept_datetime=accept_datetime, select=select, cls=lambda objs: [ConfigurationSetting._from_generated(x) for x in objs], **kwargs)
            (key_filter, kwargs) = get_key_filter(*args, **kwargs)
            (label_filter, kwargs) = get_label_filter(*args, **kwargs)
            return self._impl.get_key_values(key=key_filter, label=label_filter, accept_datetime=accept_datetime, select=select, cls=lambda objs: [ConfigurationSetting._from_generated(x) for x in objs], **kwargs)
        except binascii.Error as exc:
            raise binascii.Error('Connection string secret has incorrect padding') from exc

    @distributed_trace
    def get_configuration_setting(self, key: str, label: Optional[str]=None, etag: Optional[str]='*', match_condition: MatchConditions=MatchConditions.Unconditionally, **kwargs) -> Union[None, ConfigurationSetting]:
        if False:
            i = 10
            return i + 15
        'Get the matched ConfigurationSetting from Azure App Configuration service\n\n        :param key: key of the ConfigurationSetting\n        :type key: str\n        :param label: label used to identify the ConfigurationSetting. Default is `None`.\n        :type label: str or None\n        :param etag: check if the ConfigurationSetting is changed. Set None to skip checking etag\n        :type etag: str or None\n        :param match_condition: The match condition to use upon the etag\n        :type match_condition: ~azure.core.MatchConditions\n        :keyword accept_datetime: retrieve ConfigurationSetting existed at this datetime\n        :paramtype accept_datetime: ~datetime.datetime or str or None\n        :return: The matched ConfigurationSetting object\n        :rtype: ~azure.appconfiguration.ConfigurationSetting or None\n        :raises: :class:`~azure.core.exceptions.HttpResponseError`,             :class:`~azure.core.exceptions.ClientAuthenticationError`,             :class:`~azure.core.exceptions.ResourceNotFoundError`,             :class:`~azure.core.exceptions.ResourceModifiedError`,             :class:`~azure.core.exceptions.ResourceExistsError`\n\n        Example\n\n        .. code-block:: python\n\n            fetched_config_setting = client.get_configuration_setting(\n                key="MyKey", label="MyLabel"\n            )\n        '
        accept_datetime = kwargs.pop('accept_datetime', None)
        if isinstance(accept_datetime, datetime):
            accept_datetime = str(accept_datetime)
        error_map: Dict[int, Any] = {}
        if match_condition == MatchConditions.IfNotModified:
            error_map.update({412: ResourceModifiedError})
        if match_condition == MatchConditions.IfPresent:
            error_map.update({412: ResourceNotFoundError})
        if match_condition == MatchConditions.IfMissing:
            error_map.update({412: ResourceExistsError})
        try:
            key_value = self._impl.get_key_value(key=key, label=label, accept_datetime=accept_datetime, if_match=prep_if_match(etag, match_condition), if_none_match=prep_if_none_match(etag, match_condition), error_map=error_map, **kwargs)
            return ConfigurationSetting._from_generated(key_value)
        except ResourceNotModifiedError:
            return None
        except binascii.Error as exc:
            raise binascii.Error('Connection string secret has incorrect padding') from exc

    @distributed_trace
    def add_configuration_setting(self, configuration_setting: ConfigurationSetting, **kwargs) -> ConfigurationSetting:
        if False:
            print('Hello World!')
        'Add a ConfigurationSetting instance into the Azure App Configuration service.\n\n        :param configuration_setting: the ConfigurationSetting object to be added\n        :type configuration_setting: ~azure.appconfiguration.ConfigurationSetting\n        :return: The ConfigurationSetting object returned from the App Configuration service\n        :rtype: ~azure.appconfiguration.ConfigurationSetting\n        :raises: :class:`~azure.core.exceptions.HttpResponseError`,             :class:`~azure.core.exceptions.ClientAuthenticationError`,             :class:`~azure.core.exceptions.ResourceExistsError`\n\n        Example\n\n        .. code-block:: python\n\n            config_setting = ConfigurationSetting(\n                key="MyKey",\n                label="MyLabel",\n                value="my value",\n                content_type="my content type",\n                tags={"my tag": "my tag value"}\n            )\n            added_config_setting = client.add_configuration_setting(config_setting)\n        '
        key_value = configuration_setting._to_generated()
        custom_headers: Mapping[str, Any] = CaseInsensitiveDict(kwargs.get('headers'))
        error_map = {412: ResourceExistsError}
        try:
            key_value_added = self._impl.put_key_value(entity=key_value, key=key_value.key, label=key_value.label, if_none_match='*', headers=custom_headers, error_map=error_map)
            return ConfigurationSetting._from_generated(key_value_added)
        except binascii.Error as exc:
            raise binascii.Error('Connection string secret has incorrect padding') from exc

    @distributed_trace
    def set_configuration_setting(self, configuration_setting: ConfigurationSetting, match_condition: MatchConditions=MatchConditions.Unconditionally, **kwargs) -> ConfigurationSetting:
        if False:
            return 10
        'Add or update a ConfigurationSetting.\n        If the configuration setting identified by key and label does not exist, this is a create.\n        Otherwise this is an update.\n\n        :param configuration_setting: the ConfigurationSetting to be added (if not exists)             or updated (if exists) to the service\n        :type configuration_setting: ~azure.appconfiguration.ConfigurationSetting\n        :param match_condition: The match condition to use upon the etag\n        :type match_condition: ~azure.core.MatchConditions\n        :keyword str etag: check if the ConfigurationSetting is changed. Set None to skip checking etag\n        :return: The ConfigurationSetting returned from the service\n        :rtype: ~azure.appconfiguration.ConfigurationSetting\n        :raises: :class:`~azure.core.exceptions.HttpResponseError`,             :class:`~azure.core.exceptions.ClientAuthenticationError`,             :class:`~azure.core.exceptions.ResourceReadOnlyError`,             :class:`~azure.core.exceptions.ResourceModifiedError`,             :class:`~azure.core.exceptions.ResourceNotModifiedError`,             :class:`~azure.core.exceptions.ResourceNotFoundError`,             :class:`~azure.core.exceptions.ResourceExistsError`\n\n        Example\n\n        .. code-block:: python\n\n            config_setting = ConfigurationSetting(\n                key="MyKey",\n                label="MyLabel",\n                value="my set value",\n                content_type="my set content type",\n                tags={"my set tag": "my set tag value"}\n            )\n            returned_config_setting = client.set_configuration_setting(config_setting)\n        '
        key_value = configuration_setting._to_generated()
        custom_headers: Mapping[str, Any] = CaseInsensitiveDict(kwargs.get('headers'))
        error_map: Dict[int, Any] = {409: ResourceReadOnlyError}
        if match_condition == MatchConditions.IfNotModified:
            error_map.update({412: ResourceModifiedError})
        if match_condition == MatchConditions.IfModified:
            error_map.update({412: ResourceNotModifiedError})
        if match_condition == MatchConditions.IfPresent:
            error_map.update({412: ResourceNotFoundError})
        if match_condition == MatchConditions.IfMissing:
            error_map.update({412: ResourceExistsError})
        try:
            key_value_set = self._impl.put_key_value(entity=key_value, key=key_value.key, label=key_value.label, if_match=prep_if_match(configuration_setting.etag, match_condition), if_none_match=prep_if_none_match(configuration_setting.etag, match_condition), headers=custom_headers, error_map=error_map)
            return ConfigurationSetting._from_generated(key_value_set)
        except binascii.Error as exc:
            raise binascii.Error('Connection string secret has incorrect padding') from exc

    @distributed_trace
    def delete_configuration_setting(self, key: str, label: Optional[str]=None, **kwargs) -> ConfigurationSetting:
        if False:
            i = 10
            return i + 15
        'Delete a ConfigurationSetting if it exists\n\n        :param key: key used to identify the ConfigurationSetting\n        :type key: str\n        :param label: label used to identify the ConfigurationSetting. Default is `None`.\n        :type label: str\n        :keyword str etag: check if the ConfigurationSetting is changed. Set None to skip checking etag\n        :keyword match_condition: The match condition to use upon the etag\n        :paramtype match_condition: ~azure.core.MatchConditions\n        :return: The deleted ConfigurationSetting returned from the service, or None if it doesn\'t exist.\n        :rtype: ~azure.appconfiguration.ConfigurationSetting\n        :raises: :class:`~azure.core.exceptions.HttpResponseError`,             :class:`~azure.core.exceptions.ClientAuthenticationError`,             :class:`~azure.core.exceptions.ResourceReadOnlyError`,             :class:`~azure.core.exceptions.ResourceModifiedError`,             :class:`~azure.core.exceptions.ResourceNotModifiedError`,             :class:`~azure.core.exceptions.ResourceNotFoundError`,             :class:`~azure.core.exceptions.ResourceExistsError`\n\n        Example\n\n        .. code-block:: python\n\n            deleted_config_setting = client.delete_configuration_setting(\n                key="MyKey", label="MyLabel"\n            )\n        '
        etag = kwargs.pop('etag', None)
        match_condition = kwargs.pop('match_condition', MatchConditions.Unconditionally)
        custom_headers: Mapping[str, Any] = CaseInsensitiveDict(kwargs.get('headers'))
        error_map: Dict[int, Any] = {409: ResourceReadOnlyError}
        if match_condition == MatchConditions.IfNotModified:
            error_map.update({412: ResourceModifiedError})
        if match_condition == MatchConditions.IfModified:
            error_map.update({412: ResourceNotModifiedError})
        if match_condition == MatchConditions.IfPresent:
            error_map.update({412: ResourceNotFoundError})
        if match_condition == MatchConditions.IfMissing:
            error_map.update({412: ResourceExistsError})
        try:
            key_value_deleted = self._impl.delete_key_value(key=key, label=label, if_match=prep_if_match(etag, match_condition), headers=custom_headers, error_map=error_map)
            return ConfigurationSetting._from_generated(key_value_deleted)
        except binascii.Error as exc:
            raise binascii.Error('Connection string secret has incorrect padding') from exc

    @distributed_trace
    def list_revisions(self, key_filter: Optional[str]=None, label_filter: Optional[str]=None, **kwargs) -> ItemPaged[ConfigurationSetting]:
        if False:
            i = 10
            return i + 15
        '\n        Find the ConfigurationSetting revision history, optionally filtered by key, label and accept_datetime.\n\n        :param key_filter: filter results based on their keys. \'*\' can be\n            used as wildcard in the beginning or end of the filter\n        :type key_filter: str or None\n        :param label_filter: filter results based on their label. \'*\' can be\n            used as wildcard in the beginning or end of the filter\n        :type label_filter: str or None\n        :keyword accept_datetime: retrieve ConfigurationSetting existed at this datetime\n        :paramtype accept_datetime: ~datetime.datetime or str or None\n        :keyword list[str] fields: specify which fields to include in the results. Leave None to include all fields\n        :return: An iterator of :class:`~azure.appconfiguration.ConfigurationSetting`\n        :rtype: ~azure.core.paging.ItemPaged[~azure.appconfiguration.ConfigurationSetting]\n        :raises: :class:`~azure.core.exceptions.HttpResponseError`,             :class:`~azure.core.exceptions.ClientAuthenticationError`\n\n        Example\n\n        .. code-block:: python\n\n            from datetime import datetime, timedelta\n\n            accept_datetime = datetime.utcnow() + timedelta(days=-1)\n\n            all_revisions = client.list_revisions()\n            for item in all_revisions:\n                pass  # do something\n\n            filtered_revisions = client.list_revisions(\n                label_filter="Labe*", key_filter="Ke*", accept_datetime=str(accept_datetime)\n            )\n            for item in filtered_revisions:\n                pass  # do something\n        '
        accept_datetime = kwargs.pop('accept_datetime', None)
        if isinstance(accept_datetime, datetime):
            accept_datetime = str(accept_datetime)
        select = kwargs.pop('fields', None)
        if select:
            select = ['locked' if x == 'read_only' else x for x in select]
        try:
            return self._impl.get_revisions(label=label_filter, key=key_filter, accept_datetime=accept_datetime, select=select, cls=lambda objs: [ConfigurationSetting._from_generated(x) for x in objs], **kwargs)
        except binascii.Error as exc:
            raise binascii.Error('Connection string secret has incorrect padding') from exc

    @distributed_trace
    def set_read_only(self, configuration_setting: ConfigurationSetting, read_only: bool=True, **kwargs) -> ConfigurationSetting:
        if False:
            for i in range(10):
                print('nop')
        'Set a configuration setting read only\n\n        :param configuration_setting: the ConfigurationSetting to be set read only\n        :type configuration_setting: ~azure.appconfiguration.ConfigurationSetting\n        :param read_only: set the read only setting if true, else clear the read only setting\n        :type read_only: bool\n        :keyword match_condition: The match condition to use upon the etag\n        :paramtype match_condition: ~azure.core.MatchConditions\n        :return: The ConfigurationSetting returned from the service\n        :rtype: ~azure.appconfiguration.ConfigurationSetting\n        :raises: :class:`~azure.core.exceptions.HttpResponseError`,             :class:`~azure.core.exceptions.ClientAuthenticationError`,             :class:`~azure.core.exceptions.ResourceNotFoundError`\n\n        Example\n\n        .. code-block:: python\n\n            config_setting = client.get_configuration_setting(\n                key="MyKey", label="MyLabel"\n            )\n\n            read_only_config_setting = client.set_read_only(config_setting)\n            read_only_config_setting = client.set_read_only(config_setting, read_only=False)\n        '
        error_map: Dict[int, Any] = {}
        match_condition = kwargs.pop('match_condition', MatchConditions.Unconditionally)
        if match_condition == MatchConditions.IfNotModified:
            error_map.update({412: ResourceModifiedError})
        if match_condition == MatchConditions.IfModified:
            error_map.update({412: ResourceNotModifiedError})
        if match_condition == MatchConditions.IfPresent:
            error_map.update({412: ResourceNotFoundError})
        if match_condition == MatchConditions.IfMissing:
            error_map.update({412: ResourceExistsError})
        try:
            if read_only:
                key_value = self._impl.put_lock(key=configuration_setting.key, label=configuration_setting.label, if_match=prep_if_match(configuration_setting.etag, match_condition), if_none_match=prep_if_none_match(configuration_setting.etag, match_condition), error_map=error_map, **kwargs)
            else:
                key_value = self._impl.delete_lock(key=configuration_setting.key, label=configuration_setting.label, if_match=prep_if_match(configuration_setting.etag, match_condition), if_none_match=prep_if_none_match(configuration_setting.etag, match_condition), error_map=error_map, **kwargs)
            return ConfigurationSetting._from_generated(key_value)
        except binascii.Error as exc:
            raise binascii.Error('Connection string secret has incorrect padding') from exc

    @distributed_trace
    def begin_create_snapshot(self, name: str, filters: List[ConfigurationSettingsFilter], *, composition_type: Optional[Literal['key', 'key_label']]=None, retention_period: Optional[int]=None, tags: Optional[Dict[str, str]]=None, **kwargs) -> LROPoller[ConfigurationSnapshot]:
        if False:
            i = 10
            return i + 15
        'Create a snapshot of the configuration settings.\n\n        :param name: The name of the configuration snapshot to create.\n        :type name: str\n        :param filters: A list of filters used to filter the configuration settings by key field and label field\n            included in the configuration snapshot.\n        :type filters: list[~azure.appconfiguration.ConfigurationSettingsFilter]\n        :keyword composition_type: The composition type describes how the key-values within the configuration\n            snapshot are composed. Known values are: "key" and "key_label". The "key" composition type\n            ensures there are no two key-values containing the same key. The \'key_label\' composition type ensures\n            there are no two key-values containing the same key and label.\n        :type composition_type: str or None\n        :keyword retention_period: The amount of time, in seconds, that a configuration snapshot will remain in the\n            archived state before expiring. This property is only writable during the creation of a configuration\n            snapshot. If not specified, will set to 2592000(30 days). If specified, should be\n            in range 3600(1 hour) to 7776000(90 days).\n        :type retention_period: int or None\n        :keyword tags: The tags of the configuration snapshot.\n        :type tags: dict[str, str] or None\n        :return: A poller for create configuration snapshot operation. Call `result()` on this object to wait for the\n            operation to complete and get the created snapshot.\n        :rtype: ~azure.core.polling.LROPoller[~azure.appconfiguration.ConfigurationSnapshot]\n        :raises: :class:`~azure.core.exceptions.HttpResponseError`\n        '
        snapshot = ConfigurationSnapshot(filters=filters, composition_type=composition_type, retention_period=retention_period, tags=tags)
        try:
            return cast(LROPoller[ConfigurationSnapshot], self._impl.begin_create_snapshot(name=name, entity=snapshot._to_generated(), cls=ConfigurationSnapshot._from_deserialized, **kwargs))
        except binascii.Error:
            raise binascii.Error('Connection string secret has incorrect padding')

    @distributed_trace
    def archive_snapshot(self, name: str, *, match_condition: MatchConditions=MatchConditions.Unconditionally, etag: Optional[str]=None, **kwargs) -> ConfigurationSnapshot:
        if False:
            for i in range(10):
                print('nop')
        'Archive a configuration setting snapshot. It will update the status of a snapshot from "ready" to "archived".\n        The retention period will start to count, the snapshot will expire when the entire retention period elapses.\n\n        :param name: The name of the configuration setting snapshot to archive.\n        :type name: str\n        :keyword match_condition: The match condition to use upon the etag.\n        :type match_condition: ~azure.core.MatchConditions\n        :keyword etag: Check if the ConfigurationSnapshot is changed. Set None to skip checking etag.\n        :type etag: str or None\n        :return: The ConfigurationSnapshot returned from the service.\n        :rtype: ~azure.appconfiguration.ConfigurationSnapshot\n        :raises: :class:`~azure.core.exceptions.HttpResponseError`\n        '
        error_map: Dict[int, Any] = {}
        if match_condition == MatchConditions.IfNotModified:
            error_map.update({412: ResourceModifiedError})
        if match_condition == MatchConditions.IfModified:
            error_map.update({412: ResourceNotModifiedError})
        if match_condition == MatchConditions.IfPresent:
            error_map.update({412: ResourceNotFoundError})
        if match_condition == MatchConditions.IfMissing:
            error_map.update({412: ResourceExistsError})
        try:
            generated_snapshot = self._impl.update_snapshot(name=name, entity=SnapshotUpdateParameters(status=SnapshotStatus.ARCHIVED), if_match=prep_if_match(etag, match_condition), if_none_match=prep_if_none_match(etag, match_condition), error_map=error_map, **kwargs)
            return ConfigurationSnapshot._from_generated(generated_snapshot)
        except binascii.Error:
            raise binascii.Error('Connection string secret has incorrect padding')

    @distributed_trace
    def recover_snapshot(self, name: str, *, match_condition: MatchConditions=MatchConditions.Unconditionally, etag: Optional[str]=None, **kwargs) -> ConfigurationSnapshot:
        if False:
            print('Hello World!')
        'Recover a configuration setting snapshot. It will update the status of a snapshot from "archived" to "ready".\n\n        :param name: The name of the configuration setting snapshot to recover.\n        :type name: str\n        :keyword match_condition: The match condition to use upon the etag.\n        :type match_condition: ~azure.core.MatchConditions\n        :keyword etag: Check if the ConfigurationSnapshot is changed. Set None to skip checking etag.\n        :type etag: str or None\n        :return: The ConfigurationSnapshot returned from the service.\n        :rtype: ~azure.appconfiguration.ConfigurationSnapshot\n        :raises: :class:`~azure.core.exceptions.HttpResponseError`\n        '
        error_map: Dict[int, Any] = {}
        if match_condition == MatchConditions.IfNotModified:
            error_map.update({412: ResourceModifiedError})
        if match_condition == MatchConditions.IfModified:
            error_map.update({412: ResourceNotModifiedError})
        if match_condition == MatchConditions.IfPresent:
            error_map.update({412: ResourceNotFoundError})
        if match_condition == MatchConditions.IfMissing:
            error_map.update({412: ResourceExistsError})
        try:
            generated_snapshot = self._impl.update_snapshot(name=name, entity=SnapshotUpdateParameters(status=SnapshotStatus.READY), if_match=prep_if_match(etag, match_condition), if_none_match=prep_if_none_match(etag, match_condition), error_map=error_map, **kwargs)
            return ConfigurationSnapshot._from_generated(generated_snapshot)
        except binascii.Error:
            raise binascii.Error('Connection string secret has incorrect padding')

    @distributed_trace
    def get_snapshot(self, name: str, *, fields: Optional[List[str]]=None, **kwargs) -> ConfigurationSnapshot:
        if False:
            return 10
        'Get a configuration setting snapshot.\n\n        :param name: The name of the configuration setting snapshot to retrieve.\n        :type name: str\n        :keyword fields: Specify which fields to include in the results. Leave None to include all fields.\n        :type fields: list[str] or None\n        :return: The ConfigurationSnapshot returned from the service.\n        :rtype: ~azure.appconfiguration.ConfigurationSnapshot\n        :raises: :class:`~azure.core.exceptions.HttpResponseError`\n        '
        try:
            generated_snapshot = self._impl.get_snapshot(name=name, if_match=None, if_none_match=None, select=fields, **kwargs)
            return ConfigurationSnapshot._from_generated(generated_snapshot)
        except binascii.Error:
            raise binascii.Error('Connection string secret has incorrect padding')

    @distributed_trace
    def list_snapshots(self, *, name: Optional[str]=None, fields: Optional[List[str]]=None, status: Optional[List[Union[str, SnapshotStatus]]]=None, **kwargs) -> ItemPaged[ConfigurationSnapshot]:
        if False:
            while True:
                i = 10
        'List the configuration setting snapshots stored in the configuration service, optionally filtered by\n        snapshot name, snapshot status and fields to present in return.\n\n        :keyword name: Filter results based on snapshot name.\n        :type name: str or None\n        :keyword fields: Specify which fields to include in the results. Leave None to include all fields.\n        :type fields: list[str] or None\n        :keyword status: Filter results based on snapshot keys.\n        :type status: list[str] or list[~azure.appconfiguration.SnapshotStatus] or None\n        :return: An iterator of :class:`~azure.appconfiguration.ConfigurationSnapshot`\n        :rtype: ~azure.core.paging.ItemPaged[~azure.appconfiguration.ConfigurationSnapshot]\n        :raises: :class:`~azure.core.exceptions.HttpResponseError`\n        '
        try:
            return self._impl.get_snapshots(name=name, select=fields, status=status, cls=lambda objs: [ConfigurationSnapshot._from_generated(x) for x in objs], **kwargs)
        except binascii.Error:
            raise binascii.Error('Connection string secret has incorrect padding')

    def update_sync_token(self, token: str) -> None:
        if False:
            i = 10
            return i + 15
        'Add a sync token to the internal list of tokens.\n\n        :param str token: The sync token to be added to the internal list of tokens\n        '
        if not self._sync_token_policy:
            raise AttributeError('Client has no sync token policy, possibly because it was not provided during instantiation.')
        self._sync_token_policy.add_token(token)

    def close(self) -> None:
        if False:
            return 10
        'Close all connections made by the client'
        self._impl._client.close()

    def __enter__(self) -> 'AzureAppConfigurationClient':
        if False:
            i = 10
            return i + 15
        self._impl.__enter__()
        return self

    def __exit__(self, *args: Any) -> None:
        if False:
            return 10
        self._impl.__exit__(*args)