from datetime import datetime
from unittest import TestCase, mock
import pytest
from _pytest.monkeypatch import MonkeyPatch
from chargebee import APIError
from pytz import UTC
from organisations.chargebee import add_single_seat, extract_subscription_metadata, get_customer_id_from_subscription_id, get_hosted_page_url_for_subscription_upgrade, get_max_api_calls_for_plan, get_max_seats_for_plan, get_plan_meta_data, get_portal_url, get_subscription_data_from_hosted_page, get_subscription_metadata_from_id
from organisations.chargebee.chargebee import cancel_subscription
from organisations.chargebee.constants import ADDITIONAL_SEAT_ADDON_ID
from organisations.chargebee.metadata import ChargebeeObjMetadata
from organisations.subscriptions.exceptions import CannotCancelChargebeeSubscription, UpgradeSeatsError

class MockChargeBeePlanResponse:

    def __init__(self, max_seats=0, max_api_calls=50000):
        if False:
            print('Hello World!')
        self.max_seats = max_seats
        self.max_api_calls = 50000
        self.plan = MockChargeBeePlan(max_seats, max_api_calls)

class MockChargeBeePlan:

    def __init__(self, max_seats=0, max_api_calls=50000):
        if False:
            i = 10
            return i + 15
        self.meta_data = {'seats': max_seats, 'api_calls': max_api_calls}

class MockChargeBeeHostedPageResponse:

    def __init__(self, subscription_id='subscription-id', plan_id='plan-id', created_at=datetime.utcnow(), customer_id='customer-id', customer_email='test@example.com'):
        if False:
            return 10
        self.hosted_page = MockChargeBeeHostedPage(subscription_id=subscription_id, plan_id=plan_id, created_at=created_at, customer_id=customer_id, customer_email=customer_email)

class MockChargeBeeHostedPage:

    def __init__(self, subscription_id, plan_id, created_at, customer_id, customer_email, hosted_page_id='some-id'):
        if False:
            for i in range(10):
                print('nop')
        self.id = hosted_page_id
        self.content = MockChargeBeeHostedPageContent(subscription_id=subscription_id, plan_id=plan_id, created_at=created_at, customer_id=customer_id, customer_email=customer_email)

class MockChargeBeeHostedPageContent:

    def __init__(self, subscription_id, plan_id, created_at, customer_id, customer_email):
        if False:
            print('Hello World!')
        self.subscription = MockChargeBeeSubscription(subscription_id=subscription_id, plan_id=plan_id, created_at=created_at)
        self.customer = MockChargeBeeCustomer(customer_id, customer_email)

class MockChargeBeeAddOn:

    def __init__(self, addon_id: str, quantity: int):
        if False:
            for i in range(10):
                print('nop')
        self.id = addon_id
        self.quantity = quantity

class MockChargeBeeSubscriptionResponse:

    def __init__(self, subscription_id: str='subscription-id', plan_id: str='plan-id', created_at: datetime=None, customer_id: str='customer-id', customer_email: str='test@example.com', addons: list[MockChargeBeeAddOn]=None):
        if False:
            return 10
        self.subscription = MockChargeBeeSubscription(subscription_id, plan_id, created_at or datetime.now(), addons)
        self.customer = MockChargeBeeCustomer(customer_id, customer_email)

class MockChargeBeeSubscription:

    def __init__(self, subscription_id: str, plan_id: str, created_at: datetime, addons: list[MockChargeBeeAddOn]=None):
        if False:
            i = 10
            return i + 15
        self.id = subscription_id
        self.plan_id = plan_id
        self.created_at = datetime.timestamp(created_at)
        self.addons = addons or []

class MockChargeBeeCustomer:

    def __init__(self, customer_id, customer_email):
        if False:
            print('Hello World!')
        self.id = customer_id
        self.email = customer_email

class MockChargeBeePortalSessionResponse:

    def __init__(self, access_url='https://test.portal.url'):
        if False:
            for i in range(10):
                print('nop')
        self.portal_session = MockChargeBeePortalSession(access_url)

class MockChargeBeePortalSession:

    def __init__(self, access_url):
        if False:
            return 10
        self.access_url = access_url

class ChargeBeeTestCase(TestCase):

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        monkeypatch = MonkeyPatch()
        self.mock_cb = mock.MagicMock()
        monkeypatch.setattr('organisations.chargebee.chargebee.chargebee', self.mock_cb)

    def test_get_max_seats_for_plan_returns_max_seats_for_plan(self):
        if False:
            for i in range(10):
                print('nop')
        meta_data = {'seats': 3, 'api_calls': 50000}
        max_seats = get_max_seats_for_plan(meta_data)
        assert max_seats == meta_data['seats']

    def test_get_max_api_calls_for_plan_returns_max_api_calls_for_plan(self):
        if False:
            while True:
                i = 10
        meta_data = {'seats': 3, 'api_calls': 50000}
        max_api_calls = get_max_api_calls_for_plan(meta_data)
        assert max_api_calls == meta_data['api_calls']

    def test_get_plan_meta_data_returns_correct_metadata(self):
        if False:
            print('Hello World!')
        plan_id = 'startup'
        expected_max_seats = 3
        expected_max_api_calls = 50
        self.mock_cb.Plan.retrieve.return_value = MockChargeBeePlanResponse(expected_max_seats, expected_max_api_calls)
        plan_meta_data = get_plan_meta_data(plan_id)
        assert plan_meta_data == {'api_calls': expected_max_api_calls, 'seats': expected_max_seats}
        self.mock_cb.Plan.retrieve.assert_called_with(plan_id)

    def test_get_subscription_data_from_hosted_page_returns_expected_values(self):
        if False:
            i = 10
            return i + 15
        subscription_id = 'abc123'
        plan_id = 'startup'
        expected_max_seats = 3
        created_at = datetime.now(tz=UTC)
        customer_id = 'customer-id'
        self.mock_cb.HostedPage.retrieve.return_value = MockChargeBeeHostedPageResponse(subscription_id=subscription_id, plan_id=plan_id, created_at=created_at, customer_id=customer_id)
        self.mock_cb.Plan.retrieve.return_value = MockChargeBeePlanResponse(expected_max_seats)
        subscription_data = get_subscription_data_from_hosted_page('hosted_page_id')
        assert subscription_data['subscription_id'] == subscription_id
        assert subscription_data['plan'] == plan_id
        assert subscription_data['max_seats'] == expected_max_seats
        assert subscription_data['subscription_date'] == created_at
        assert subscription_data['customer_id'] == customer_id

    def test_get_portal_url(self):
        if False:
            return 10
        access_url = 'https://test.url.com'
        self.mock_cb.PortalSession.create.return_value = MockChargeBeePortalSessionResponse(access_url)
        portal_url = get_portal_url('some-customer-id', 'https://redirect.url.com')
        assert portal_url == access_url

    def test_get_customer_id_from_subscription(self):
        if False:
            print('Hello World!')
        expected_customer_id = 'customer-id'
        self.mock_cb.Subscription.retrieve.return_value = MockChargeBeeSubscriptionResponse(customer_id=expected_customer_id)
        customer_id = get_customer_id_from_subscription_id('subscription-id')
        assert customer_id == expected_customer_id

    def test_get_hosted_page_url_for_subscription_upgrade(self):
        if False:
            return 10
        subscription_id = 'test-id'
        plan_id = 'plan-id'
        url = 'https://some.url.com/some/page/'
        self.mock_cb.HostedPage.checkout_existing.return_value = mock.MagicMock(hosted_page=mock.MagicMock(url=url))
        response = get_hosted_page_url_for_subscription_upgrade(subscription_id, plan_id)
        assert response == url
        self.mock_cb.HostedPage.checkout_existing.assert_called_once_with({'subscription': {'id': subscription_id, 'plan_id': plan_id}})

def test_extract_subscription_metadata(mock_subscription_response_with_addons: MockChargeBeeSubscriptionResponse, chargebee_object_metadata: ChargebeeObjMetadata):
    if False:
        while True:
            i = 10
    status = 'status'
    plan_id = 'plan-id'
    addon_id = 'addon-id'
    subscription_id = 'subscription-id'
    customer_email = 'test@example.com'
    subscription = {'status': status, 'id': subscription_id, 'plan_id': plan_id, 'addons': [{'id': addon_id, 'quantity': 2, 'unit_price': 0, 'amount': 0}]}
    subscription_metadata = extract_subscription_metadata(subscription, customer_email)
    assert subscription_metadata.seats == chargebee_object_metadata.seats * 3
    assert subscription_metadata.api_calls == chargebee_object_metadata.api_calls * 3
    assert subscription_metadata.projects == chargebee_object_metadata.projects * 3
    assert subscription_metadata.chargebee_email == customer_email

def test_extract_subscription_metadata_when_addon_list_is_empty(mock_subscription_response_with_addons: MockChargeBeeSubscriptionResponse, chargebee_object_metadata: ChargebeeObjMetadata):
    if False:
        for i in range(10):
            print('nop')
    status = 'status'
    plan_id = 'plan-id'
    subscription_id = 'subscription-id'
    customer_email = 'test@example.com'
    subscription = {'status': status, 'id': subscription_id, 'plan_id': plan_id, 'addons': []}
    subscription_metadata = extract_subscription_metadata(subscription, customer_email)
    assert subscription_metadata.seats == chargebee_object_metadata.seats
    assert subscription_metadata.api_calls == chargebee_object_metadata.api_calls
    assert subscription_metadata.projects == chargebee_object_metadata.projects
    assert subscription_metadata.chargebee_email == customer_email

def test_get_subscription_metadata_from_id(mock_subscription_response_with_addons: MockChargeBeeSubscriptionResponse, chargebee_object_metadata: ChargebeeObjMetadata):
    if False:
        print('Hello World!')
    customer_email = 'test@example.com'
    subscription_id = mock_subscription_response_with_addons.subscription.id
    subscription_metadata = get_subscription_metadata_from_id(subscription_id)
    assert subscription_metadata.seats == chargebee_object_metadata.seats * 2
    assert subscription_metadata.api_calls == chargebee_object_metadata.api_calls * 2
    assert subscription_metadata.projects == chargebee_object_metadata.projects * 2
    assert subscription_metadata.chargebee_email == customer_email

def test_cancel_subscription(mocker):
    if False:
        while True:
            i = 10
    mocked_chargebee = mocker.patch('organisations.chargebee.chargebee.chargebee')
    subscription_id = 'sub-id'
    cancel_subscription(subscription_id)
    mocked_chargebee.Subscription.cancel.assert_called_once_with(subscription_id, {'end_of_term': True})

def test_cancel_subscription_throws_cannot_cancel_error_if_api_error(mocker, caplog):
    if False:
        print('Hello World!')
    mocked_chargebee = mocker.patch('organisations.chargebee.chargebee.chargebee')
    subscription_id = 'sub-id'

    class MockException(Exception):
        pass
    mocker.patch('organisations.chargebee.chargebee.ChargebeeAPIError', MockException)
    mocked_chargebee.Subscription.cancel.side_effect = MockException
    with pytest.raises(CannotCancelChargebeeSubscription):
        cancel_subscription(subscription_id)
    mocked_chargebee.Subscription.cancel.assert_called_once_with(subscription_id, {'end_of_term': True})
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == 'ERROR'
    assert caplog.records[0].message == 'Cannot cancel CB subscription for subscription id: %s' % subscription_id

def test_get_subscription_metadata_from_id_returns_null_if_chargebee_error(mocker, chargebee_object_metadata):
    if False:
        i = 10
        return i + 15
    mocked_chargebee = mocker.patch('organisations.chargebee.chargebee.chargebee')
    mocked_chargebee.Subscription.retrieve.side_effect = APIError(http_code=200, json_obj=mocker.MagicMock())
    subscription_id = 'foo'
    subscription_metadata = get_subscription_metadata_from_id(subscription_id)
    assert subscription_metadata is None

@pytest.mark.parametrize('subscription_id', [None, '', ' '])
def test_get_subscription_metadata_from_id_returns_none_for_invalid_subscription_id(mocker, chargebee_object_metadata, subscription_id):
    if False:
        for i in range(10):
            print('nop')
    mocked_chargebee = mocker.patch('organisations.chargebee.chargebee.chargebee')
    subscription_metadata = get_subscription_metadata_from_id(subscription_id)
    mocked_chargebee.Subscription.retrieve.assert_not_called()
    assert subscription_metadata is None

def test_get_subscription_metadata_from_id_returns_valid_metadata_if_addons_is_none(mock_subscription_response: MockChargeBeeSubscriptionResponse, chargebee_object_metadata: ChargebeeObjMetadata) -> None:
    if False:
        for i in range(10):
            print('nop')
    mock_subscription_response.addons = None
    subscription_id = mock_subscription_response.subscription.id
    subscription_metadata = get_subscription_metadata_from_id(subscription_id)
    assert subscription_metadata.seats == chargebee_object_metadata.seats
    assert subscription_metadata.api_calls == chargebee_object_metadata.api_calls
    assert subscription_metadata.projects == chargebee_object_metadata.projects

def test_add_single_seat_with_existing_addon(mocker):
    if False:
        for i in range(10):
            print('nop')
    plan_id = 'plan-id'
    addon_id = ADDITIONAL_SEAT_ADDON_ID
    subscription_id = 'subscription-id'
    addon_quantity = 1
    mocked_subscription = mocker.MagicMock(id=subscription_id, plan_id=plan_id, addons=[mocker.MagicMock(id=addon_id, quantity=addon_quantity)])
    mocked_chargebee = mocker.patch('organisations.chargebee.chargebee.chargebee')
    mocked_chargebee.Subscription.retrieve.return_value.subscription = mocked_subscription
    add_single_seat(subscription_id)
    mocked_chargebee.Subscription.update.assert_called_once_with(subscription_id, {'addons': [{'id': ADDITIONAL_SEAT_ADDON_ID, 'quantity': addon_quantity + 1}], 'prorate': True, 'invoice_immediately': True})

def test_add_single_seat_without_existing_addon(mocker):
    if False:
        print('Hello World!')
    subscription_id = 'subscription-id'
    mocked_subscription = mocker.MagicMock(id=subscription_id, plan_id='plan_id', addons=[])
    mocked_chargebee = mocker.patch('organisations.chargebee.chargebee.chargebee')
    mocked_chargebee.Subscription.retrieve.return_value.subscription = mocked_subscription
    add_single_seat(subscription_id)
    mocked_chargebee.Subscription.update.assert_called_once_with(subscription_id, {'addons': [{'id': ADDITIONAL_SEAT_ADDON_ID, 'quantity': 1}], 'prorate': True, 'invoice_immediately': True})

def test_add_single_seat_throws_upgrade_seats_error_error_if_api_error(mocker, caplog):
    if False:
        i = 10
        return i + 15
    mocked_chargebee = mocker.patch('organisations.chargebee.chargebee.chargebee')
    chargebee_response_data = {'message': '82sa2Sqa5 not found', 'type': 'invalid_request', 'api_error_code': 'resource_not_found', 'param': 'item_id', 'error_code': 'DeprecatedField'}
    mocked_chargebee.Subscription.update.side_effect = APIError(http_code=404, json_obj=chargebee_response_data)
    subscription_id = 'sub-id'
    mocked_subscription = mocker.MagicMock(id=subscription_id, plan_id='plan-id', addons=[])
    mocked_chargebee.Subscription.retrieve.return_value.subscription = mocked_subscription
    with pytest.raises(UpgradeSeatsError):
        add_single_seat(subscription_id)
    mocked_chargebee.Subscription.update.assert_called_once_with(subscription_id, {'addons': [{'id': ADDITIONAL_SEAT_ADDON_ID, 'quantity': 1}], 'prorate': True, 'invoice_immediately': True})
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == 'ERROR'
    assert caplog.records[0].message == 'Failed to add additional seat to CB subscription for subscription id: %s' % subscription_id