from typing import Union
from uuid import uuid4
from azure.core.tracing.decorator import distributed_trace
from azure.communication.sms._generated.models import SendMessageRequest, SmsRecipient, SmsSendOptions
from azure.communication.sms._models import SmsSendResult
from azure.core.credentials import TokenCredential, AzureKeyCredential
from ._generated._azure_communication_sms_service import AzureCommunicationSMSService
from ._shared.auth_policy_utils import get_authentication_policy
from ._shared.utils import parse_connection_str, get_current_utc_time
from ._version import SDK_MONIKER

class SmsClient(object):
    """A client to interact with the AzureCommunicationService Sms gateway.

    This client provides operations to send an SMS via a phone number.

    :param str endpoint:
        The endpoint url for Azure Communication Service resource.
    :param Union[TokenCredential, AzureKeyCredential] credential:
        The credential we use to authenticate against the service.
    """

    def __init__(self, endpoint, credential, **kwargs):
        if False:
            print('Hello World!')
        try:
            if not endpoint.lower().startswith('http'):
                endpoint = 'https://' + endpoint
        except AttributeError:
            raise ValueError('Account URL must be a string.')
        if not credential:
            raise ValueError('invalid credential from connection string.')
        self._endpoint = endpoint
        self._authentication_policy = get_authentication_policy(endpoint, credential)
        self._sms_service_client = AzureCommunicationSMSService(self._endpoint, authentication_policy=self._authentication_policy, sdk_moniker=SDK_MONIKER, **kwargs)

    @classmethod
    def from_connection_string(cls, conn_str, **kwargs):
        if False:
            print('Hello World!')
        'Create SmsClient from a Connection String.\n\n        :param str conn_str:\n            A connection string to an Azure Communication Service resource.\n        :returns: Instance of SmsClient.\n        :rtype: ~azure.communication.SmsClient\n\n        .. admonition:: Example:\n\n            .. literalinclude:: ../samples/sms_sample.py\n                :start-after: [START auth_from_connection_string]\n                :end-before: [END auth_from_connection_string]\n                :language: python\n                :dedent: 8\n                :caption: Creating the SmsClient from a connection string.\n        '
        (endpoint, access_key) = parse_connection_str(conn_str)
        return cls(endpoint, access_key, **kwargs)

    @distributed_trace
    def send(self, from_, to, message, **kwargs):
        if False:
            return 10
        'Sends SMSs to phone numbers.\n\n        :param str from_: The sender of the SMS.\n        :param to: The single recipient or the list of recipients of the SMS.\n        :type to: Union[str, List[str]]\n        :param str message: The message in the SMS\n        :keyword bool enable_delivery_report: Enable this flag to receive a delivery report for this\n         message on the Azure Resource EventGrid.\n        :keyword str tag: Use this field to provide metadata that will then be sent back in the corresponding\n         Delivery Report.\n        :return: A list of SmsSendResult.\n        :rtype: [~azure.communication.sms.models.SmsSendResult]\n        '
        if isinstance(to, str):
            to = [to]
        enable_delivery_report = kwargs.pop('enable_delivery_report', False)
        tag = kwargs.pop('tag', None)
        sms_send_options = SmsSendOptions(enable_delivery_report=enable_delivery_report, tag=tag)
        request = SendMessageRequest(from_property=from_, sms_recipients=[SmsRecipient(to=p, repeatability_request_id=str(uuid4()), repeatability_first_sent=get_current_utc_time()) for p in to], message=message, sms_send_options=sms_send_options, **kwargs)
        return self._sms_service_client.sms.send(request, cls=lambda pr, r, e: [SmsSendResult(to=item.to, message_id=item.message_id, http_status_code=item.http_status_code, successful=item.successful, error_message=item.error_message) for item in r.value], **kwargs)