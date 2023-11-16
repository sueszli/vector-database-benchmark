from typing import Union
from azure.core.credentials import TokenCredential, AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential
from azure.core.pipeline.policies import AsyncBearerTokenCredentialPolicy, BearerTokenCredentialPolicy
from .._shared.policy import HMACCredentialsPolicy

def get_authentication_policy(endpoint: str, credential: Union[TokenCredential, AsyncTokenCredential, AzureKeyCredential, str], decode_url: bool=False, is_async: bool=False):
    if False:
        print('Hello World!')
    'Returns the correct authentication policy based on which credential is being passed.\n\n    :param endpoint: The endpoint to which we are authenticating to.\n    :type endpoint: str\n    :param credential: The credential we use to authenticate to the service\n    :type credential: Union[TokenCredential, AsyncTokenCredential, AzureKeyCredential, str]\n    :param bool decode_url: `True` if there is a need to decode the url. Default value is `False`\n    :param bool is_async: For async clients there is a need to decode the url\n\n    :return: Either AsyncBearerTokenCredentialPolicy or BearerTokenCredentialPolicy or HMACCredentialsPolicy\n    :rtype: ~azure.core.pipeline.policies.AsyncBearerTokenCredentialPolicy or\n    ~azure.core.pipeline.policies.BearerTokenCredentialPolicy or\n    ~azure.communication.callautomation.shared.policy.HMACCredentialsPolicy\n    '
    if credential is None:
        raise ValueError("Parameter 'credential' must not be None.")
    if hasattr(credential, 'get_token'):
        if is_async:
            return AsyncBearerTokenCredentialPolicy(credential, 'https://communication.azure.com//.default')
        return BearerTokenCredentialPolicy(credential, 'https://communication.azure.com//.default')
    if isinstance(credential, (AzureKeyCredential, str)):
        return HMACCredentialsPolicy(endpoint, credential, decode_url=decode_url)
    raise TypeError(f'Unsupported credential: {format(type(credential))}. Use an access token string to use HMACCredentialsPolicyor a token credential from azure.identity')