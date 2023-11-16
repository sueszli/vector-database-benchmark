import unittest
from unittest.mock import MagicMock, Mock, patch
from tap_zendesk import discover, http
import tap_zendesk
import requests
import zenpy
ACCSESS_TOKEN_ERROR = '{"error": "Forbidden", "description": "You are missing the following required scopes: read"}'
API_TOKEN_ERROR = '{"error": {"title": "Forbidden","message": "You do not have access to this page. Please contact the account owner of this help desk for further help."}}'
AUTH_ERROR = '{"error": "Could not authenticate you"}'
START_DATE = '2021-10-30T00:00:00Z'

def mocked_get(*args, **kwargs):
    if False:
        print('Hello World!')
    fake_response = requests.models.Response()
    fake_response.headers.update(kwargs.get('headers', {}))
    fake_response.status_code = kwargs['status_code']
    fake_response.json = Mock()
    fake_response.json.side_effect = lambda : kwargs.get('json', {})
    return fake_response

class TestDiscovery(unittest.TestCase):
    """
    Test that we can call api for each stream in discovey mode and handle forbidden error.
    """

    @patch('tap_zendesk.discover.LOGGER.warning')
    @patch('tap_zendesk.streams.Organizations.check_access', side_effect=zenpy.lib.exception.APIException(ACCSESS_TOKEN_ERROR))
    @patch('tap_zendesk.streams.Users.check_access', side_effect=zenpy.lib.exception.APIException(ACCSESS_TOKEN_ERROR))
    @patch('tap_zendesk.streams.TicketForms.check_access', side_effect=zenpy.lib.exception.APIException(ACCSESS_TOKEN_ERROR))
    @patch('tap_zendesk.streams.SLAPolicies.check_access', side_effect=[mocked_get(status_code=200, json={'key1': 'val1'})])
    @patch('tap_zendesk.discover.load_shared_schema_refs', return_value={})
    @patch('tap_zendesk.streams.Stream.load_metadata', return_value={})
    @patch('tap_zendesk.streams.Stream.load_schema', return_value={})
    @patch('singer.resolve_schema_references', return_value={})
    @patch('requests.get', side_effect=[mocked_get(status_code=200, json={'tickets': [{'id': 't1'}]}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'})])
    def test_discovery_handles_403__raise_tap_zendesk_forbidden_error(self, mock_get, mock_resolve_schema_references, mock_load_metadata, mock_load_schema, mock_load_shared_schema_refs, mocked_sla_policies, mocked_ticket_forms, mock_users, mock_organizations, mock_logger):
        if False:
            return 10
        '\n        Test that we handle forbidden error for child streams. discover_streams calls check_access for each stream to\n        check the read perission. discover_streams call many other methods including load_shared_schema_refs, load_metadata,\n        load_schema, resolve_schema_references also which we mock to test forbidden error. We mock check_access method of\n        some of stream method which call request of zenpy module and also mock get method of requests module with 200, 403 error.\n\n        '
        discover.discover_streams('dummy_client', {'subdomain': 'arp', 'access_token': 'dummy_token', 'start_date': START_DATE})
        expected_call_count = 10
        actual_call_count = mock_get.call_count
        self.assertEqual(expected_call_count, actual_call_count)
        mock_logger.assert_called_with("The account credentials supplied do not have 'read' access to the following stream(s): groups, users, organizations, ticket_audits, ticket_comments, ticket_fields, ticket_forms, group_memberships, macros, satisfaction_ratings, tags, ticket_metrics. The data for these streams would not be collected due to lack of required permission.")

    @patch('tap_zendesk.discover.LOGGER.warning')
    @patch('tap_zendesk.streams.Organizations.check_access', side_effect=zenpy.lib.exception.APIException(ACCSESS_TOKEN_ERROR))
    @patch('tap_zendesk.streams.Users.check_access', side_effect=zenpy.lib.exception.APIException(ACCSESS_TOKEN_ERROR))
    @patch('tap_zendesk.streams.TicketForms.check_access', side_effect=zenpy.lib.exception.APIException(ACCSESS_TOKEN_ERROR))
    @patch('tap_zendesk.streams.SLAPolicies.check_access', side_effect=zenpy.lib.exception.APIException(ACCSESS_TOKEN_ERROR))
    @patch('tap_zendesk.discover.load_shared_schema_refs', return_value={})
    @patch('tap_zendesk.streams.Stream.load_metadata', return_value={})
    @patch('tap_zendesk.streams.Stream.load_schema', return_value={})
    @patch('singer.resolve_schema_references', return_value={})
    @patch('requests.get', side_effect=[mocked_get(status_code=200, json={'tickets': [{'id': 't1'}]}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'})])
    def test_discovery_handles_403_raise_zenpy_forbidden_error_for_access_token(self, mock_get, mock_resolve_schema_references, mock_load_metadata, mock_load_schema, mock_load_shared_schema_refs, mocked_sla_policies, mocked_ticket_forms, mock_users, mock_organizations, mock_logger):
        if False:
            return 10
        '\n        Test that we handle forbidden error received from last failed request which we called from zenpy module and\n        log proper warning message. discover_streams calls check_access for each stream to check the\n        read perission. discover_streams call many other methods including load_shared_schema_refs, load_metadata,\n        load_schema, resolve_schema_references also which we mock to test forbidden error. We mock check_access method of\n        some of stream method which call request of zenpy module and also mock get method of requests module with 200, 403 error.\n        '
        discover.discover_streams('dummy_client', {'subdomain': 'arp', 'access_token': 'dummy_token', 'start_date': START_DATE})
        expected_call_count = 10
        actual_call_count = mock_get.call_count
        self.assertEqual(expected_call_count, actual_call_count)
        mock_logger.assert_called_with("The account credentials supplied do not have 'read' access to the following stream(s): groups, users, organizations, ticket_audits, ticket_comments, ticket_fields, ticket_forms, group_memberships, macros, satisfaction_ratings, tags, ticket_metrics, sla_policies. The data for these streams would not be collected due to lack of required permission.")

    @patch('tap_zendesk.discover.LOGGER.warning')
    @patch('tap_zendesk.streams.Organizations.check_access', side_effect=zenpy.lib.exception.APIException(API_TOKEN_ERROR))
    @patch('tap_zendesk.streams.Users.check_access', side_effect=zenpy.lib.exception.APIException(API_TOKEN_ERROR))
    @patch('tap_zendesk.streams.TicketForms.check_access', side_effect=zenpy.lib.exception.APIException(API_TOKEN_ERROR))
    @patch('tap_zendesk.streams.SLAPolicies.check_access', side_effect=[mocked_get(status_code=200, json={'key1': 'val1'})])
    @patch('tap_zendesk.discover.load_shared_schema_refs', return_value={})
    @patch('tap_zendesk.streams.Stream.load_metadata', return_value={})
    @patch('tap_zendesk.streams.Stream.load_schema', return_value={})
    @patch('singer.resolve_schema_references', return_value={})
    @patch('requests.get', side_effect=[mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=404, json={'key1': 'val1'}), mocked_get(status_code=404, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=404, json={'key1': 'val1'})])
    def test_discovery_handles_403_raise_zenpy_forbidden_error_for_api_token(self, mock_get, mock_resolve_schema_references, mock_load_metadata, mock_load_schema, mock_load_shared_schema_refs, mocked_sla_policies, mocked_ticket_forms, mock_users, mock_organizations, mock_logger):
        if False:
            print('Hello World!')
        '\n        Test that we handle forbidden error received from last failed request which we called from zenpy module and\n        log proper warning message. discover_streams calls check_access for each stream to check the \n        read perission. discover_streams call many other methods including load_shared_schema_refs, load_metadata, \n        load_schema, resolve_schema_references also which we mock to test forbidden error. We mock check_access method of \n        some of stream method which call request of zenpy module and also mock get method of requests module with 200, 403 error.\n        '
        responses = discover.discover_streams('dummy_client', {'subdomain': 'arp', 'access_token': 'dummy_token', 'start_date': START_DATE})
        expected_call_count = 10
        actual_call_count = mock_get.call_count
        self.assertEqual(expected_call_count, actual_call_count)
        mock_logger.assert_called_with("The account credentials supplied do not have 'read' access to the following stream(s): tickets, groups, users, organizations, ticket_fields, ticket_forms, group_memberships, macros, satisfaction_ratings, tags. The data for these streams would not be collected due to lack of required permission.")

    @patch('tap_zendesk.streams.Organizations.check_access', side_effect=zenpy.lib.exception.APIException(ACCSESS_TOKEN_ERROR))
    @patch('tap_zendesk.streams.Users.check_access', side_effect=zenpy.lib.exception.APIException(ACCSESS_TOKEN_ERROR))
    @patch('tap_zendesk.streams.TicketForms.check_access', side_effect=zenpy.lib.exception.APIException(ACCSESS_TOKEN_ERROR))
    @patch('tap_zendesk.streams.SLAPolicies.check_access', side_effect=[mocked_get(status_code=200, json={'key1': 'val1'})])
    @patch('tap_zendesk.discover.load_shared_schema_refs', return_value={})
    @patch('tap_zendesk.streams.Stream.load_metadata', return_value={})
    @patch('tap_zendesk.streams.Stream.load_schema', return_value={})
    @patch('singer.resolve_schema_references', return_value={})
    @patch('requests.get', side_effect=[mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=400, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'})])
    def test_discovery_handles_except_403_error_requests_module(self, mock_get, mock_resolve_schema_references, mock_load_metadata, mock_load_schema, mock_load_shared_schema_refs, mocked_sla_policies, mocked_ticket_forms, mock_users, mock_organizations):
        if False:
            i = 10
            return i + 15
        '\n        Test that function raises error directly if error code is other than 403. discover_streams calls check_access for each \n        stream to check the read perission. discover_streams call many other methods including load_shared_schema_refs, load_metadata, \n        load_schema, resolve_schema_references also which we mock to test forbidden error. We mock check_access method of \n        some of stream method which call request of zenpy module and also mock get method of requests module with 200, 403 error.\n        '
        try:
            responses = discover.discover_streams('dummy_client', {'subdomain': 'arp', 'access_token': 'dummy_token', 'start_date': START_DATE})
        except http.ZendeskBadRequestError as e:
            expected_error_message = 'HTTP-error-code: 400, Error: A validation exception has occurred.'
            self.assertEqual(str(e), expected_error_message)
        expected_call_count = 4
        actual_call_count = mock_get.call_count
        self.assertEqual(expected_call_count, actual_call_count)

    @patch('tap_zendesk.streams.Organizations.check_access', side_effect=zenpy.lib.exception.APIException(AUTH_ERROR))
    @patch('tap_zendesk.streams.Users.check_access', side_effect=zenpy.lib.exception.APIException(AUTH_ERROR))
    @patch('tap_zendesk.streams.TicketForms.check_access', side_effect=zenpy.lib.exception.APIException(AUTH_ERROR))
    @patch('tap_zendesk.streams.SLAPolicies.check_access', side_effect=[mocked_get(status_code=200, json={'key1': 'val1'})])
    @patch('tap_zendesk.discover.load_shared_schema_refs', return_value={})
    @patch('tap_zendesk.streams.Stream.load_metadata', return_value={})
    @patch('tap_zendesk.streams.Stream.load_schema', return_value={})
    @patch('singer.resolve_schema_references', return_value={})
    @patch('requests.get', side_effect=[mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=400, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'})])
    def test_discovery_handles_except_403_error_zenpy_module(self, mock_get, mock_resolve_schema_references, mock_load_metadata, mock_load_schema, mock_load_shared_schema_refs, mocked_sla_policies, mocked_ticket_forms, mock_users, mock_organizations):
        if False:
            i = 10
            return i + 15
        '\n        Test that discovery mode raise error direclty if it is rather than 403 for request zenpy module. discover_streams call \n        many other methods including load_shared_schema_refs, load_metadata, load_schema, resolve_schema_references\n        also which we mock to test forbidden error. We mock check_access method of some of stream method which\n        call request of zenpy module and also mock get method of requests module with 400, 403 error.\n        '
        try:
            responses = discover.discover_streams('dummy_client', {'subdomain': 'arp', 'access_token': 'dummy_token', 'start_date': START_DATE})
        except zenpy.lib.exception.APIException as e:
            expected_error_message = AUTH_ERROR
            self.assertEqual(str(e), expected_error_message)
        expected_call_count = 2
        actual_call_count = mock_get.call_count
        self.assertEqual(expected_call_count, actual_call_count)

    @patch('tap_zendesk.streams.Organizations.check_access', side_effect=[mocked_get(status_code=200, json={'key1': 'val1'})])
    @patch('tap_zendesk.streams.Users.check_access', side_effect=[mocked_get(status_code=200, json={'key1': 'val1'})])
    @patch('tap_zendesk.streams.TicketForms.check_access', side_effect=[mocked_get(status_code=200, json={'key1': 'val1'})])
    @patch('tap_zendesk.streams.SLAPolicies.check_access', side_effect=[mocked_get(status_code=200, json={'key1': 'val1'})])
    @patch('tap_zendesk.discover.load_shared_schema_refs', return_value={})
    @patch('tap_zendesk.streams.Stream.load_metadata', return_value={})
    @patch('tap_zendesk.streams.Stream.load_schema', return_value={})
    @patch('singer.resolve_schema_references', return_value={})
    @patch('requests.get', side_effect=[mocked_get(status_code=200, json={'tickets': [{'id': 't1'}]}), mocked_get(status_code=200, json={'key1': 'val1'}), mocked_get(status_code=200, json={'key1': 'val1'}), mocked_get(status_code=200, json={'key1': 'val1'}), mocked_get(status_code=200, json={'key1': 'val1'}), mocked_get(status_code=200, json={'key1': 'val1'}), mocked_get(status_code=200, json={'key1': 'val1'}), mocked_get(status_code=200, json={'key1': 'val1'}), mocked_get(status_code=200, json={'key1': 'val1'}), mocked_get(status_code=200, json={'key1': 'val1'})])
    def test_discovery_handles_200_response(self, mock_get, mock_resolve_schema_references, mock_load_metadata, mock_load_schema, mock_load_shared_schema_refs, mocked_sla_policies, mocked_ticket_forms, mock_users, mock_organizations):
        if False:
            while True:
                i = 10
        '\n        Test that discovery mode does not raise any error in case of all streams have read permission\n        '
        discover.discover_streams('dummy_client', {'subdomain': 'arp', 'access_token': 'dummy_token', 'start_date': START_DATE})
        expected_call_count = 10
        actual_call_count = mock_get.call_count
        self.assertEqual(expected_call_count, actual_call_count)

    @patch('tap_zendesk.discover.LOGGER.warning')
    @patch('tap_zendesk.streams.Organizations.check_access', side_effect=zenpy.lib.exception.APIException(API_TOKEN_ERROR))
    @patch('tap_zendesk.streams.Users.check_access', side_effect=zenpy.lib.exception.APIException(API_TOKEN_ERROR))
    @patch('tap_zendesk.streams.TicketForms.check_access', side_effect=zenpy.lib.exception.APIException(API_TOKEN_ERROR))
    @patch('tap_zendesk.streams.SLAPolicies.check_access', side_effect=zenpy.lib.exception.APIException(API_TOKEN_ERROR))
    @patch('tap_zendesk.discover.load_shared_schema_refs', return_value={})
    @patch('tap_zendesk.streams.Stream.load_metadata', return_value={})
    @patch('tap_zendesk.streams.Stream.load_schema', return_value={})
    @patch('singer.resolve_schema_references', return_value={})
    @patch('requests.get', side_effect=[mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'}), mocked_get(status_code=403, json={'key1': 'val1'})])
    def test_discovery_handles_403_for_all_streams_api_token(self, mock_get, mock_resolve_schema_references, mock_load_metadata, mock_load_schema, mock_load_shared_schema_refs, mocked_sla_policies, mocked_ticket_forms, mock_users, mock_organizations, mock_logger):
        if False:
            while True:
                i = 10
        '\n        Test that we handle forbidden error received from all streams and raise the ZendeskForbiddenError\n        with proper error message. discover_streams calls check_access for each stream to check the \n        read perission. discover_streams call many other methods including load_shared_schema_refs, load_metadata, \n        load_schema, resolve_schema_references also which we mock to test forbidden error. We mock check_access method of \n        some of stream method which call request of zenpy module and also mock get method of requests module with 200, 403 error.\n        '
        try:
            responses = discover.discover_streams('dummy_client', {'subdomain': 'arp', 'access_token': 'dummy_token', 'start_date': START_DATE})
        except http.ZendeskForbiddenError as e:
            expected_message = "HTTP-error-code: 403, Error: The account credentials supplied do not have 'read' access to any of streams supported by the tap. Data collection cannot be initiated due to lack of permissions."
            self.assertEqual(str(e), expected_message)