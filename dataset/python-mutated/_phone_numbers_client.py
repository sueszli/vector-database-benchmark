from typing import TYPE_CHECKING, Union
from azure.core.tracing.decorator import distributed_trace
from azure.core.exceptions import HttpResponseError
from ._generated._client import PhoneNumbersClient as PhoneNumbersClientGen
from ._generated.models import PhoneNumberSearchRequest, PhoneNumberCapabilitiesRequest, PhoneNumberPurchaseRequest, PhoneNumberType
from ._shared.auth_policy_utils import get_authentication_policy
from ._shared.utils import parse_connection_str
from ._version import SDK_MONIKER
from ._api_versions import DEFAULT_VERSION
_DEFAULT_POLLING_INTERVAL_IN_SECONDS = 2
if TYPE_CHECKING:
    from typing import Any
    from azure.core.credentials import TokenCredential, AzureKeyCredential
    from azure.core.paging import ItemPaged
    from azure.core.polling import LROPoller
    from ._generated.models import PhoneNumberCapabilities, PhoneNumberCapabilityType, PhoneNumberCountry, PhoneNumberOffering, PhoneNumberLocality, PhoneNumberSearchResult, PurchasedPhoneNumber

class PhoneNumbersClient(object):
    """A client to interact with the AzureCommunicationService Phone Numbers gateway.

    This client provides operations to interact with the phone numbers service
    :param str endpoint:
        The endpoint url for Azure Communication Service resource.
    :param Union[TokenCredential, AzureKeyCredential] credential:
        The credential we use to authenticate against the service.
    :keyword api_version: Azure Communication Phone Number API version.
        The default value is "2022-01-11-preview2".
        Note that overriding this default value may result in unsupported behavior.
    :paramtype api_version: str
    """

    def __init__(self, endpoint, credential, **kwargs):
        if False:
            return 10
        try:
            if not endpoint.lower().startswith('http'):
                endpoint = 'https://' + endpoint
        except AttributeError:
            raise ValueError('Account URL must be a string.')
        if not credential:
            raise ValueError('You need to provide account shared key to authenticate.')
        self._endpoint = endpoint
        self._accepted_language = kwargs.pop('accepted_language', None)
        self._api_version = kwargs.pop('api_version', DEFAULT_VERSION.value)
        self._phone_number_client = PhoneNumbersClientGen(self._endpoint, api_version=self._api_version, authentication_policy=get_authentication_policy(endpoint, credential), sdk_moniker=SDK_MONIKER, **kwargs)

    @classmethod
    def from_connection_string(cls, conn_str, **kwargs):
        if False:
            while True:
                i = 10
        'Create PhoneNumbersClient from a Connection String.\n        :param str conn_str:\n            A connection string to an Azure Communication Service resource.\n        :returns: Instance of PhoneNumbersClient.\n        :rtype: ~azure.communication.phonenumbers.PhoneNumbersClient\n        '
        (endpoint, access_key) = parse_connection_str(conn_str)
        return cls(endpoint, access_key, **kwargs)

    @distributed_trace
    def begin_purchase_phone_numbers(self, search_id, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Purchases phone numbers.\n\n        :param search_id: The search id.\n        :type search_id: str\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: Pass in True if you'd like the LROBasePolling polling method,\n            False for no polling, or your own initialized polling object for a personal polling strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time (seconds) between two polls\n            for LRO operations if no Retry-After header is present.\n        :rtype: ~azure.core.polling.LROPoller[None]\n        "
        purchase_request = PhoneNumberPurchaseRequest(search_id=search_id)
        polling_interval = kwargs.pop('polling_interval', _DEFAULT_POLLING_INTERVAL_IN_SECONDS)
        return self._phone_number_client.phone_numbers.begin_purchase_phone_numbers(body=purchase_request, polling_interval=polling_interval, **kwargs)

    @distributed_trace
    def begin_release_phone_number(self, phone_number, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Releases an purchased phone number.\n\n        :param phone_number: Phone number to be released, e.g. +55534567890.\n        :type phone_number: str\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: Pass in True if you'd like the LROBasePolling polling method,\n            False for no polling, or your own initialized polling object for a personal polling strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time (seconds) between two polls\n            for LRO operations if no Retry-After header is present.\n        :rtype: ~azure.core.polling.LROPoller[None]\n        "
        polling_interval = kwargs.pop('polling_interval', _DEFAULT_POLLING_INTERVAL_IN_SECONDS)
        return self._phone_number_client.phone_numbers.begin_release_phone_number(phone_number, polling_interval=polling_interval, **kwargs)

    @distributed_trace
    def begin_search_available_phone_numbers(self, country_code, phone_number_type, assignment_type, capabilities, **kwargs):
        if False:
            i = 10
            return i + 15
        'Search for available phone numbers to purchase.\n\n        :param country_code: The ISO 3166-2 country code, e.g. US.\n        :type country_code: str\n        :param phone_number_type: Required. The type of phone numbers to search for, e.g. geographic,\n            or tollFree. Possible values include: "geographic", "tollFree".\n        :type phone_number_type: str or ~azure.communication.phonenumbers.models.PhoneNumberType\n        :param assignment_type: Required. The assignment type of the phone numbers to search for. A\n            phone number can be assigned to a person, or to an application. Possible values include:\n            "user", "application".\n        :type assignment_type: str or\n            ~azure.communication.phonenumbers.models.PhoneNumberAssignmentType\n        :param capabilities: Required. Capabilities of a phone number.\n        :type capabilities: ~azure.communication.phonenumbers.models.PhoneNumberCapabilities\n        :keyword str area_code: The area code of the desired phone number, e.g. 425. If not set,\n            any area code could be used in the final search.\n        :keyword int quantity: The quantity of phone numbers in the search. Default is 1.\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: Pass in True if you\'d like the LROBasePolling polling method,\n         False for no polling, or your own initialized polling object for a personal polling strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time (seconds) between two polls\n            for LRO operations if no Retry-After header is present.\n        :rtype: ~azure.core.polling.LROPoller[~azure.communication.phonenumbers.models.PhoneNumberSearchResult]\n        '
        search_request = PhoneNumberSearchRequest(phone_number_type=phone_number_type, assignment_type=assignment_type, capabilities=capabilities, quantity=kwargs.pop('quantity', None), area_code=kwargs.pop('area_code', None))
        polling_interval = kwargs.pop('polling_interval', _DEFAULT_POLLING_INTERVAL_IN_SECONDS)
        return self._phone_number_client.phone_numbers.begin_search_available_phone_numbers(country_code, search_request, polling_interval=polling_interval, **kwargs)

    @distributed_trace
    def begin_update_phone_number_capabilities(self, phone_number, sms=None, calling=None, **kwargs):
        if False:
            return 10
        "Updates the capabilities of a phone number.\n\n        :param phone_number: The phone number id in E.164 format. The leading plus can be either + or\n            encoded as %2B, e.g. +55534567890.\n        :type phone_number: str\n        :param calling: Capability value for calling.\n        :type calling: str or ~azure.communication.phonenumbers.models.PhoneNumberCapabilityType\n        :param sms: Capability value for SMS.\n        :type sms: str or ~azure.communication.phonenumbers.models.PhoneNumberCapabilityType\n        :keyword str continuation_token: A continuation token to restart a poller from a saved state.\n        :keyword polling: Pass in True if you'd like the LROBasePolling polling method,\n            False for no polling, or your own initialized polling object for a personal polling strategy.\n        :paramtype polling: bool or ~azure.core.polling.PollingMethod\n        :keyword int polling_interval: Default waiting time (seconds) between two polls\n            for LRO operations if no Retry-After header is present.\n        :rtype: ~azure.core.polling.LROPoller[~azure.communication.phonenumbers.models.PurchasedPhoneNumber]\n        "
        capabilities_request = PhoneNumberCapabilitiesRequest(calling=calling, sms=sms)
        polling_interval = kwargs.pop('polling_interval', _DEFAULT_POLLING_INTERVAL_IN_SECONDS)
        if not phone_number:
            raise ValueError("phone_number can't be empty")
        poller = self._phone_number_client.phone_numbers.begin_update_capabilities(phone_number, body=capabilities_request, polling_interval=polling_interval, **kwargs)
        result_properties = poller.result().additional_properties
        if 'status' in result_properties and result_properties['status'].lower() == 'failed':
            raise HttpResponseError(message=result_properties['error']['message'])
        return poller

    @distributed_trace
    def get_purchased_phone_number(self, phone_number, **kwargs):
        if False:
            print('Hello World!')
        'Gets the details of the given purchased phone number.\n\n        :param phone_number: The purchased phone number whose details are to be fetched in E.164 format,\n         e.g. +11234567890.\n        :type phone_number: str\n        :return: The details of the given purchased phone number.\n        :rtype: ~azure.communication.phonenumbers.models.PurchasedPhoneNumber\n        '
        return self._phone_number_client.phone_numbers.get_by_number(phone_number, **kwargs)

    @distributed_trace
    def list_purchased_phone_numbers(self, **kwargs):
        if False:
            return 10
        'Gets the list of all purchased phone numbers.\n\n        :keyword skip: An optional parameter for how many entries to skip, for pagination purposes. The\n         default value is 0. Default value is 0.\n        :paramtype skip: int\n        :keyword top: An optional parameter for how many entries to return, for pagination purposes.\n         The default value is 100. Default value is 100.\n        :paramtype top: int\n        :rtype: ~azure.core.paging.ItemPaged[~azure.communication.phonenumbers.models.PurchasedPhoneNumber]\n        '
        return self._phone_number_client.phone_numbers.list_phone_numbers(**kwargs)

    @distributed_trace
    def list_available_countries(self, **kwargs):
        if False:
            return 10
        'Gets the list of supported countries.\n\n        Gets the list of supported countries.\n\n        :keyword skip: An optional parameter for how many entries to skip, for pagination purposes. The\n         default value is 0. Default value is 0.\n        :paramtype skip: int\n        :return: An iterator like instance of PhoneNumberCountry\n        :rtype:\n         ~azure.core.paging.ItemPaged[~azure.communication.phonenumbers.models.PhoneNumberCountry]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        return self._phone_number_client.phone_numbers.list_available_countries(accept_language=self._accepted_language, **kwargs)

    @distributed_trace
    def list_available_localities(self, country_code, **kwargs):
        if False:
            print('Hello World!')
        'Gets the list of cities or towns with available phone numbers.\n\n        Gets the list of cities or towns with available phone numbers.\n\n        :param country_code: The ISO 3166-2 country/region two letter code, e.g. US. Required.\n        :type country_code: str\n        :param administrative_division: An optional parameter for the name of the state or province\n         in which to search for the area code. e.g. California. Default value is None.\n        :type administrative_division: str\n        :keyword skip: An optional parameter for how many entries to skip, for pagination purposes. The\n         default value is 0. Default value is 0.\n        :paramtype skip: int\n        :return: An iterator like instance of PhoneNumberLocality\n        :rtype:\n         ~azure.core.paging.ItemPaged[~azure.communication.phonenumbers.models.PhoneNumberLocality]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        return self._phone_number_client.phone_numbers.list_available_localities(country_code, administrative_division=kwargs.pop('administrative_division', None), accept_language=self._accepted_language, **kwargs)

    @distributed_trace
    def list_available_offerings(self, country_code, **kwargs):
        if False:
            i = 10
            return i + 15
        'List available offerings of capabilities with rates for the given country/region.\n\n        List available offerings of capabilities with rates for the given country/region.\n\n        :param country_code: The ISO 3166-2 country/region two letter code, e.g. US. Required.\n        :type country_code: str\n        :param phone_number_type: Filter by phoneNumberType, e.g. Geographic, TollFree. Known values\n         are: "geographic" and "tollFree". Default value is None.\n        :type phone_number_type: ~azure.communication.phonenumbers.models.PhoneNumberType\n        :param assignment_type: Filter by assignmentType, e.g. User, Application. Known values are:\n         "person" and "application". Default value is None.\n        :type assignment_type: ~azure.communication.phonenumbers.models.PhoneNumberAssignmentType\n        :keyword skip: An optional parameter for how many entries to skip, for pagination purposes. The\n         default value is 0. Default value is 0.\n        :paramtype skip: int\n        :return: An iterator like instance of PhoneNumberOffering\n        :rtype:\n         ~azure.core.paging.ItemPaged[~azure.communication.phonenumbers.models.PhoneNumberOffering]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        return self._phone_number_client.phone_numbers.list_offerings(country_code, phone_number_type=kwargs.pop('phone_number_type', None), assignment_type=kwargs.pop('assignment_type', None), **kwargs)

    @distributed_trace
    def list_available_area_codes(self, country_code, phone_number_type, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Gets the list of available area codes.\n\n        :param country_code: The ISO 3166-2 country/region two letter code, e.g. US. Required.\n        :type country_code: str\n        :param phone_number_type: Filter by phone number type, e.g. Geographic, TollFree. Known values are:\n        "geographic" and "tollFree". Required.\n        :type phone_number_type: ~azure.communication.phonenumbers.models.PhoneNumberType\n        :param assignment_type: Filter by assignmentType, e.g. User, Application. Known values are:\n        "person" and "application". Default value is None.\n        :type assignment_type: ~azure.communication.phonenumbers.models.PhoneNumberAssignmentType\n        :param locality: The name of locality in which to search for the area code. e.g. Seattle.\n        This is required if the phone number type is Geographic. Default value is None.\n        :type locality: str\n        :keyword administrative_division: The name of the state or province in which to search for the\n        area code. e.g. California. Default value is None.\n        :type administrative_division: str\n        :keyword skip: An optional parameter for how many entries to skip, for pagination purposes. The\n        default value is 0. Default value is 0.\n        :paramtype skip: int\n        :return: An iterator like instance of PhoneNumberAreaCode\n        :rtype: ~azure.core.paging.ItemPaged[~azure.communication.phonenumbers.models.PhoneNumberAreaCode]\n        :raises ~azure.core.exceptions.HttpResponseError:\n        '
        return self._phone_number_client.phone_numbers.list_area_codes(country_code, phone_number_type=phone_number_type, assignment_type=kwargs.pop('assignment_type', None), locality=kwargs.pop('locality', None), administrative_division=kwargs.pop('administrative_division', None), **kwargs)