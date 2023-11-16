import io
import json
import os
import re
import sys
import warnings
try:
    from inspect import getfullargspec as get_arg_spec
except ImportError:
    from inspect import getargspec as get_arg_spec
from .credentials import get_azure_cli_credentials
from .cloud import get_cli_active_cloud

def _instantiate_client(client_class, **kwargs):
    if False:
        print('Hello World!')
    'Instantiate a client from kwargs, filtering kwargs to match client signature.\n    '
    args = get_arg_spec(client_class.__init__).args
    for key in ['subscription_id', 'tenant_id', 'base_url', 'credential', 'credentials']:
        if key not in kwargs:
            continue
        if key not in args:
            del kwargs[key]
        elif sys.version_info < (3, 0) and isinstance(kwargs[key], unicode):
            kwargs[key] = kwargs[key].encode('utf-8')
    return client_class(**kwargs)

def _client_resource(client_class, cloud):
    if False:
        i = 10
        return i + 15
    'Return a tuple of the resource (used to get the right access token), and the base URL for the client.\n    Either or both can be None to signify that the default value should be used.\n    '
    if client_class.__name__ == 'GraphRbacManagementClient':
        return (cloud.endpoints.active_directory_graph_resource_id, cloud.endpoints.active_directory_graph_resource_id)
    if client_class.__name__ == 'ApplicationInsightsDataClient':
        return (cloud.endpoints.app_insights_resource_id, cloud.endpoints.app_insights_resource_id + '/v1')
    if client_class.__name__ == 'KeyVaultClient':
        vault_host = cloud.suffixes.keyvault_dns[1:]
        vault_url = 'https://{}'.format(vault_host)
        return (vault_url, None)
    return (None, None)

def get_client_from_cli_profile(client_class, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Return a SDK client initialized with current CLI credentials, CLI default subscription and CLI default cloud.\n\n    *Disclaimer*: This is NOT the recommended approach to authenticate with CLI login, this method is deprecated.\n    use https://pypi.org/project/azure-identity/ and AzureCliCredential instead. See example code below:\n\n    .. code:: python\n\n        from azure.identity import AzureCliCredential\n        from azure.mgmt.compute import ComputeManagementClient\n        client = ComputeManagementClient(AzureCliCredential(), subscription_id)\n\n    This method is not working for azure-cli-core>=2.21.0 (released in March 2021).\n\n    For compatible azure-cli-core (< 2.20.0), This method will automatically fill the following client parameters:\n    - credentials/credential\n    - subscription_id\n    - base_url\n\n    Parameters provided in kwargs will override CLI parameters and be passed directly to the client.\n\n    :Example:\n\n    .. code:: python\n\n        from azure.common.client_factory import get_client_from_cli_profile\n        from azure.mgmt.compute import ComputeManagementClient\n        client = get_client_from_cli_profile(ComputeManagementClient)\n\n    .. versionadded:: 1.1.6\n\n    .. deprecated:: 1.1.28\n\n    .. seealso:: https://aka.ms/azsdk/python/identity/migration\n\n    :param client_class: A SDK client class\n    :return: An instantiated client\n    :raises: ImportError if azure-cli-core package is not available\n    '
    warnings.warn('get_client_from_cli_profile is deprecated, please use azure-identity and AzureCliCredential instead. ' + 'See also https://aka.ms/azsdk/python/identity/migration.', DeprecationWarning)
    cloud = get_cli_active_cloud()
    parameters = {}
    no_credential_sentinel = object()
    kwarg_cred = kwargs.pop('credentials', no_credential_sentinel)
    if kwarg_cred is no_credential_sentinel:
        kwarg_cred = kwargs.pop('credential', no_credential_sentinel)
    if kwarg_cred is no_credential_sentinel or 'subscription_id' not in kwargs:
        (resource, _) = _client_resource(client_class, cloud)
        (credentials, subscription_id, tenant_id) = get_azure_cli_credentials(resource=resource, with_tenant=True)
        credential_to_pass = credentials if kwarg_cred is no_credential_sentinel else kwarg_cred
        parameters.update({'credentials': credential_to_pass, 'credential': credential_to_pass, 'subscription_id': kwargs.get('subscription_id', subscription_id)})
    args = get_arg_spec(client_class.__init__).args
    if 'adla_job_dns_suffix' in args and 'adla_job_dns_suffix' not in kwargs:
        parameters['adla_job_dns_suffix'] = cloud.suffixes.azure_datalake_analytics_catalog_and_job_endpoint
    elif 'base_url' in args and 'base_url' not in kwargs:
        (_, base_url) = _client_resource(client_class, cloud)
        if base_url:
            parameters['base_url'] = base_url
        else:
            parameters['base_url'] = cloud.endpoints.resource_manager
    if 'tenant_id' in args and 'tenant_id' not in kwargs:
        parameters['tenant_id'] = tenant_id
    parameters.update(kwargs)
    return _instantiate_client(client_class, **parameters)

def _is_autorest_v3(client_class):
    if False:
        i = 10
        return i + 15
    'Is this client a autorestv3/track2 one?.\n    Could be refined later if necessary.\n    '
    args = get_arg_spec(client_class.__init__).args
    return 'credential' in args

def get_client_from_json_dict(client_class, config_dict, **kwargs):
    if False:
        i = 10
        return i + 15
    'Return a SDK client initialized with a JSON auth dict.\n\n    *Disclaimer*: This is NOT the recommended approach, see https://aka.ms/azsdk/python/identity/migration for guidance.\n\n    This method will fill automatically the following client parameters:\n    - credentials\n    - subscription_id\n    - base_url\n    - tenant_id\n\n    Parameters provided in kwargs will override parameters and be passed directly to the client.\n\n    :Example:\n\n    .. code:: python\n\n        from azure.common.client_factory import get_client_from_auth_file\n        from azure.mgmt.compute import ComputeManagementClient\n        config_dict = {\n            "clientId": "ad735158-65ca-11e7-ba4d-ecb1d756380e",\n            "clientSecret": "b70bb224-65ca-11e7-810c-ecb1d756380e",\n            "subscriptionId": "bfc42d3a-65ca-11e7-95cf-ecb1d756380e",\n            "tenantId": "c81da1d8-65ca-11e7-b1d1-ecb1d756380e",\n            "activeDirectoryEndpointUrl": "https://login.microsoftonline.com",\n            "resourceManagerEndpointUrl": "https://management.azure.com/",\n            "activeDirectoryGraphResourceId": "https://graph.windows.net/",\n            "sqlManagementEndpointUrl": "https://management.core.windows.net:8443/",\n            "galleryEndpointUrl": "https://gallery.azure.com/",\n            "managementEndpointUrl": "https://management.core.windows.net/"\n        }\n        client = get_client_from_json_dict(ComputeManagementClient, config_dict)\n\n    .. versionadded:: 1.1.7\n\n    .. deprecated:: 1.1.28\n\n    .. seealso:: https://aka.ms/azsdk/python/identity/migration\n\n    :param client_class: A SDK client class\n    :param dict config_dict: A config dict.\n    :return: An instantiated client\n    '
    if _is_autorest_v3(client_class):
        raise ValueError('Auth file or JSON dict are deprecated auth approach and are not supported anymore. See also https://aka.ms/azsdk/python/identity/migration.')
    import adal
    from msrestazure.azure_active_directory import AdalAuthentication
    is_graphrbac = client_class.__name__ == 'GraphRbacManagementClient'
    is_keyvault = client_class.__name__ == 'KeyVaultClient'
    parameters = {'subscription_id': config_dict.get('subscriptionId'), 'base_url': config_dict.get('resourceManagerEndpointUrl'), 'tenant_id': config_dict.get('tenantId')}
    if is_graphrbac:
        parameters['base_url'] = config_dict['activeDirectoryGraphResourceId']
    if 'credentials' not in kwargs:
        if is_graphrbac:
            resource = config_dict['activeDirectoryGraphResourceId']
        elif is_keyvault:
            resource = 'https://vault.azure.net'
        else:
            if 'activeDirectoryResourceId' not in config_dict and 'resourceManagerEndpointUrl' not in config_dict:
                raise ValueError('Need activeDirectoryResourceId or resourceManagerEndpointUrl key')
            resource = config_dict.get('activeDirectoryResourceId', config_dict['resourceManagerEndpointUrl'])
        authority_url = config_dict['activeDirectoryEndpointUrl']
        is_adfs = bool(re.match('.+(/adfs|/adfs/)$', authority_url, re.I))
        if is_adfs:
            authority_url = authority_url.rstrip('/')
        else:
            authority_url = authority_url + '/' + config_dict['tenantId']
        context = adal.AuthenticationContext(authority_url, api_version=None, validate_authority=not is_adfs)
        parameters['credentials'] = AdalAuthentication(context.acquire_token_with_client_credentials, resource, config_dict['clientId'], config_dict['clientSecret'])
    parameters.update(kwargs)
    return _instantiate_client(client_class, **parameters)

def get_client_from_auth_file(client_class, auth_path=None, **kwargs):
    if False:
        i = 10
        return i + 15
    'Return a SDK client initialized with auth file.\n\n    *Disclaimer*: This is NOT the recommended approach, see https://aka.ms/azsdk/python/identity/migration for guidance.\n\n    You can specific the file path directly, or fill the environment variable AZURE_AUTH_LOCATION.\n    File must be UTF-8.\n\n    This method will fill automatically the following client parameters:\n    - credentials\n    - subscription_id\n    - base_url\n\n    Parameters provided in kwargs will override parameters and be passed directly to the client.\n\n    :Example:\n\n    .. code:: python\n\n        from azure.common.client_factory import get_client_from_auth_file\n        from azure.mgmt.compute import ComputeManagementClient\n        client = get_client_from_auth_file(ComputeManagementClient)\n\n    Example of file:\n\n    .. code:: json\n\n        {\n            "clientId": "ad735158-65ca-11e7-ba4d-ecb1d756380e",\n            "clientSecret": "b70bb224-65ca-11e7-810c-ecb1d756380e",\n            "subscriptionId": "bfc42d3a-65ca-11e7-95cf-ecb1d756380e",\n            "tenantId": "c81da1d8-65ca-11e7-b1d1-ecb1d756380e",\n            "activeDirectoryEndpointUrl": "https://login.microsoftonline.com",\n            "resourceManagerEndpointUrl": "https://management.azure.com/",\n            "activeDirectoryGraphResourceId": "https://graph.windows.net/",\n            "sqlManagementEndpointUrl": "https://management.core.windows.net:8443/",\n            "galleryEndpointUrl": "https://gallery.azure.com/",\n            "managementEndpointUrl": "https://management.core.windows.net/"\n        }\n\n    .. versionadded:: 1.1.7\n\n    .. deprecated:: 1.1.28\n\n    .. seealso:: https://aka.ms/azsdk/python/identity/migration\n\n    :param client_class: A SDK client class\n    :param str auth_path: Path to the file.\n    :return: An instantiated client\n    :raises: KeyError if AZURE_AUTH_LOCATION is not an environment variable and no path is provided\n    :raises: FileNotFoundError if provided file path does not exists\n    :raises: json.JSONDecodeError if provided file is not JSON valid\n    :raises: UnicodeDecodeError if file is not UTF8 compliant\n    '
    auth_path = auth_path or os.environ['AZURE_AUTH_LOCATION']
    with io.open(auth_path, 'r', encoding='utf-8-sig') as auth_fd:
        config_dict = json.load(auth_fd)
    return get_client_from_json_dict(client_class, config_dict, **kwargs)