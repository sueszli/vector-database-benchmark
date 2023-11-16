import sys
from typing import Any, Union, IO
from azure.core.credentials import AzureKeyCredential
from azure.core.credentials import TokenCredential
from azure.core.polling import LROPoller
from azure.core.tracing.decorator import distributed_trace
from ._shared.auth_policy_utils import get_authentication_policy
from ._shared.utils import parse_connection_str
from ._generated._client import AzureCommunicationEmailService
from ._version import SDK_MONIKER
from ._api_versions import DEFAULT_VERSION
if sys.version_info >= (3, 9):
    from collections.abc import MutableMapping
else:
    from typing import MutableMapping
JSON = MutableMapping[str, Any]

class EmailClient(object):
    """A client to interact with the AzureCommunicationService Email gateway.

    This client provides operations to send an email and monitor its status.

    :param str endpoint:
        The endpoint url for Azure Communication Service resource.
    :param Union[TokenCredential, AzureKeyCredential] credential:
        The credential we use to authenticate against the service.
    :keyword api_version: Azure Communication Email API version.
        Default value is "2023-03-31".
        Note that overriding this default value may result in unsupported behavior.
    :paramtype api_version: str
    """

    def __init__(self, endpoint: str, credential: Union[TokenCredential, AzureKeyCredential], **kwargs) -> None:
        if False:
            return 10
        try:
            if not endpoint.lower().startswith('http'):
                endpoint = 'https://' + endpoint
        except AttributeError:
            raise ValueError('Account URL must be a string.')
        if endpoint.endswith('/'):
            endpoint = endpoint[:-1]
        self._api_version = kwargs.pop('api_version', DEFAULT_VERSION)
        authentication_policy = get_authentication_policy(endpoint, credential)
        self._generated_client = AzureCommunicationEmailService(endpoint, authentication_policy=authentication_policy, sdk_moniker=SDK_MONIKER, **kwargs)

    @classmethod
    def from_connection_string(cls, conn_str: str, **kwargs) -> 'EmailClient':
        if False:
            i = 10
            return i + 15
        'Create EmailClient from a Connection String.\n\n        :param str conn_str:\n            A connection string to an Azure Communication Service resource.\n        :returns: Instance of EmailClient.\n        :rtype: ~azure.communication.EmailClient\n        '
        (endpoint, access_key) = parse_connection_str(conn_str)
        return cls(endpoint, AzureKeyCredential(access_key), **kwargs)

    @distributed_trace
    def begin_send(self, message: Union[JSON, IO], **kwargs: Any) -> LROPoller[JSON]:
        if False:
            i = 10
            return i + 15
        'Queues an email message to be sent to one or more recipients.\n\n        Queues an email message to be sent to one or more recipients.\n\n        :param message: Message payload for sending an email. Required.\n        :type message: JSON\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :return: An instance of LROPoller that returns JSON object\n        :rtype: ~azure.core.polling.LROPoller[JSON]\n        :raises ~azure.core.exceptions.HttpResponseError:\n\n         Example:\n            .. code-block:: python\n\n                # JSON input template you can fill out and use as your body input.\n                message = {\n                    "content": {\n                        "subject": "str",  # Subject of the email message. Required.\n                        "html": "str",  # Optional. Html version of the email message.\n                        "plainText": "str"  # Optional. Plain text version of the email\n                          message.\n                    },\n                    "recipients": {\n                        "to": [\n                            {\n                                "address": "str",  # Email address. Required.\n                                "displayName": "str"  # Optional. Email display name.\n                            }\n                        ],\n                        "bcc": [\n                            {\n                                "address": "str",  # Email address. Required.\n                                "displayName": "str"  # Optional. Email display name.\n                            }\n                        ],\n                        "cc": [\n                            {\n                                "address": "str",  # Email address. Required.\n                                "displayName": "str"  # Optional. Email display name.\n                            }\n                        ]\n                    },\n                    "senderAddress": "str",  # Sender email address from a verified domain.\n                      Required.\n                    "attachments": [\n                        {\n                            "contentInBase64": "str",  # Base64 encoded contents of the\n                              attachment. Required.\n                            "contentType": "str",  # MIME type of the content being\n                              attached. Required.\n                            "name": "str"  # Name of the attachment. Required.\n                        }\n                    ],\n                    "userEngagementTrackingDisabled": bool,  # Optional. Indicates whether user\n                      engagement tracking should be disabled for this request if the resource-level\n                      user engagement tracking setting was already enabled in the control plane.\n                    "headers": {\n                        "str": "str"  # Optional. Custom email headers to be passed.\n                    },\n                    "replyTo": [\n                        {\n                            "address": "str",  # Email address. Required.\n                            "displayName": "str"  # Optional. Email display name.\n                        }\n                    ]\n                }\n\n                # response body for status code(s): 202\n                response == {\n                    "id": "str",  # The unique id of the operation. Use a UUID. Required.\n                    "status": "str",  # Status of operation. Required. Known values are:\n                      "NotStarted", "Running", "Succeeded", "Failed", and "Canceled".\n                    "error": {\n                        "additionalInfo": [\n                            {\n                                "info": {},  # Optional. The additional info.\n                                "type": "str"  # Optional. The additional info type.\n                            }\n                        ],\n                        "code": "str",  # Optional. The error code.\n                        "details": [\n                            ...\n                        ],\n                        "message": "str",  # Optional. The error message.\n                        "target": "str"  # Optional. The error target.\n                    }\n                }\n        '
        return self._generated_client.email.begin_send(message=message, **kwargs)

    def __enter__(self) -> 'EmailClient':
        if False:
            print('Hello World!')
        self._generated_client.__enter__()
        return self

    def __exit__(self, *args) -> None:
        if False:
            i = 10
            return i + 15
        self._generated_client.__exit__(*args)