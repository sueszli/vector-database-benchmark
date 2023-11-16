"""
This module contains utility functions for boto3 library
"""
from typing import Any, Optional
from boto3 import Session
from botocore.config import Config
from botocore.exceptions import ClientError
from typing_extensions import Protocol
from samcli import __version__
from samcli.cli.global_config import GlobalConfig

def get_boto_config_with_user_agent(**kwargs) -> Config:
    if False:
        i = 10
        return i + 15
    '\n    Automatically add user agent string to boto configs.\n\n    Parameters\n    ----------\n    kwargs :\n        key=value params which will be added to the Config object\n\n    Returns\n    -------\n    Config\n        Returns config instance which contains given parameters in it\n    '
    gc = GlobalConfig()
    return Config(user_agent_extra=f'aws-sam-cli/{__version__}/{gc.installation_id}' if gc.telemetry_enabled else f'aws-sam-cli/{__version__}', **kwargs)

class BotoProviderType(Protocol):

    def __call__(self, service_name: str) -> Any:
        if False:
            for i in range(10):
                print('nop')
        ...

def get_boto_client_provider_from_session_with_config(session: Session, **kwargs) -> BotoProviderType:
    if False:
        print('Hello World!')
    '\n    Returns a wrapper function for boto client with given configuration. It can be used like;\n\n    client_provider = get_boto_client_wrapper_with_config(session=session)\n    lambda_client = client_provider("lambda")\n\n    Parameters\n    ----------\n    session: Session\n        Boto3 session object\n    kwargs :\n        Key-value params that will be passed to get_boto_config_with_user_agent\n\n    Returns\n    -------\n        A callable function which will return a boto client\n    '
    return lambda client_name: session.client(client_name, config=get_boto_config_with_user_agent(**kwargs))

def get_boto_client_provider_with_config(region: Optional[str]=None, profile: Optional[str]=None, **kwargs) -> BotoProviderType:
    if False:
        while True:
            i = 10
    '\n    Returns a wrapper function for boto client with given configuration. It can be used like;\n\n    client_provider = get_boto_client_wrapper_with_config(region_name=region)\n    lambda_client = client_provider("lambda")\n\n    Parameters\n    ----------\n    region: Optional[str]\n        AWS region name\n    profile: Optional[str]\n        Profile name from credentials\n    kwargs :\n        Key-value params that will be passed to get_boto_config_with_user_agent\n\n    Returns\n    -------\n        A callable function which will return a boto client\n    '
    return get_boto_client_provider_from_session_with_config(Session(region_name=region, profile_name=profile), **kwargs)

def get_boto_resource_provider_from_session_with_config(session: Session, **kwargs) -> BotoProviderType:
    if False:
        return 10
    '\n    Returns a wrapper function for boto resource with given configuration. It can be used like;\n\n    resource_provider = get_boto_resource_wrapper_with_config(session=session)\n    cloudformation_resource = resource_provider("cloudformation")\n\n    Parameters\n    ----------\n    session: Session\n        Boto3 session object\n    kwargs :\n        Key-value params that will be passed to get_boto_config_with_user_agent\n\n    Returns\n    -------\n        A callable function which will return a boto resource\n    '
    return lambda resource_name: session.resource(resource_name, config=get_boto_config_with_user_agent(**kwargs))

def get_boto_resource_provider_with_config(region: Optional[str]=None, profile: Optional[str]=None, **kwargs) -> BotoProviderType:
    if False:
        while True:
            i = 10
    '\n    Returns a wrapper function for boto resource with given configuration. It can be used like;\n\n    resource_provider = get_boto_resource_wrapper_with_config(region_name=region)\n    cloudformation_resource = resource_provider("cloudformation")\n\n    Parameters\n    ----------\n    region: Optional[str]\n        AWS region name\n    profile: Optional[str]\n        Profile name from credentials\n    kwargs :\n        Key-value params that will be passed to get_boto_config_with_user_agent\n\n    Returns\n    -------\n        A callable function which will return a boto resource\n    '
    return get_boto_resource_provider_from_session_with_config(Session(region_name=region, profile_name=profile), **kwargs)

def get_client_error_code(client_error: ClientError) -> Optional[str]:
    if False:
        for i in range(10):
            print('nop')
    'Extracts error code from boto ClientError'
    return client_error.response.get('Error', {}).get('Code')