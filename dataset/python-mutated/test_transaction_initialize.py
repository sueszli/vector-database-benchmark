from decimal import Decimal
from unittest import mock
import pytest
from .....channel import TransactionFlowStrategy
from .....checkout import CheckoutAuthorizeStatus, CheckoutChargeStatus
from .....checkout.calculations import fetch_checkout_data
from .....checkout.fetch import fetch_checkout_info, fetch_checkout_lines
from .....core.prices import quantize_price
from .....payment import TransactionEventType
from .....payment.interface import PaymentGatewayData, TransactionProcessActionData, TransactionSessionData, TransactionSessionResult
from ....channel.enums import TransactionFlowStrategyEnum
from ....core.enums import TransactionInitializeErrorCode
from ....core.utils import to_global_id_or_none
from ....tests.utils import assert_no_permission, get_graphql_content
TRANSACTION_INITIALIZE = '\nmutation TransactionInitialize(\n  $action: TransactionFlowStrategyEnum,\n  $amount: PositiveDecimal,\n  $id: ID!,\n  $paymentGateway: PaymentGatewayToInitialize!\n  $customerIpAddress: String\n) {\n  transactionInitialize(\n    action: $action\n    amount: $amount\n    id: $id\n    paymentGateway: $paymentGateway\n    customerIpAddress: $customerIpAddress\n  ) {\n    data\n    transaction {\n      id\n      authorizedAmount {\n        currency\n        amount\n      }\n      chargedAmount {\n        currency\n        amount\n      }\n      chargePendingAmount {\n        amount\n        currency\n      }\n      authorizePendingAmount {\n        amount\n        currency\n      }\n    }\n    transactionEvent {\n      amount {\n        currency\n        amount\n      }\n      type\n      createdBy {\n        ... on App {\n          id\n        }\n      }\n      pspReference\n      message\n      externalUrl\n    }\n    errors{\n      field\n      message\n      code\n    }\n  }\n}\n'

def _assert_fields(content, source_object, expected_amount, expected_psp_reference, response_event_type, app_identifier, mocked_initialize, request_event_type=TransactionEventType.CHARGE_REQUEST, action_type=TransactionFlowStrategy.CHARGE, request_event_include_in_calculations=False, authorized_value=Decimal(0), charged_value=Decimal(0), charge_pending_value=Decimal(0), authorize_pending_value=Decimal(0), returned_data=None):
    if False:
        return 10
    assert not content['data']['transactionInitialize']['errors']
    response_data = content['data']['transactionInitialize']
    assert response_data['data'] == returned_data
    transaction_data = response_data['transaction']
    transaction = source_object.payment_transactions.last()
    assert transaction
    assert quantize_price(Decimal(transaction_data['authorizePendingAmount']['amount']), source_object.currency) == authorize_pending_value == transaction.authorize_pending_value
    assert quantize_price(Decimal(transaction_data['authorizedAmount']['amount']), source_object.currency) == authorized_value == transaction.authorized_value
    assert quantize_price(Decimal(transaction_data['chargePendingAmount']['amount']), source_object.currency) == charge_pending_value == transaction.charge_pending_value
    assert quantize_price(Decimal(transaction_data['chargedAmount']['amount']), source_object.currency) == charged_value
    assert charged_value == transaction.charged_value
    assert response_data['transactionEvent']
    assert response_data['transactionEvent']['type'] == response_event_type.upper()
    assert quantize_price(Decimal(response_data['transactionEvent']['amount']['amount']), source_object.currency) == expected_amount
    request_event = transaction.events.filter(type=request_event_type).first()
    assert request_event
    assert request_event.amount_value == expected_amount
    assert request_event.include_in_calculations == request_event_include_in_calculations
    assert request_event.psp_reference == expected_psp_reference
    response_event = transaction.events.filter(type=response_event_type).first()
    assert response_event
    assert response_event.amount_value == expected_amount
    assert response_event.include_in_calculations
    assert response_event.psp_reference == expected_psp_reference
    mocked_initialize.assert_called_with(TransactionSessionData(transaction=transaction, source_object=source_object, action=TransactionProcessActionData(action_type=action_type, amount=expected_amount, currency=source_object.currency), customer_ip_address='127.0.0.1', payment_gateway_data=PaymentGatewayData(app_identifier=app_identifier, data=None, error=None)))

@mock.patch('saleor.plugins.manager.PluginsManager.transaction_initialize_session')
def test_for_checkout_without_payment_gateway_data(mocked_initialize, user_api_client, checkout_with_prices, webhook_app, transaction_session_response):
    if False:
        return 10
    checkout = checkout_with_prices
    expected_app_identifier = 'webhook.app.identifier'
    webhook_app.identifier = expected_app_identifier
    webhook_app.save()
    expected_amount = Decimal('10.00')
    expected_psp_reference = 'ppp-123'
    expected_response = transaction_session_response.copy()
    expected_response['result'] = 'CHARGE_SUCCESS'
    expected_response['pspReference'] = expected_psp_reference
    mocked_initialize.return_value = TransactionSessionResult(app_identifier=expected_app_identifier, response=expected_response)
    variables = {'action': None, 'amount': expected_amount, 'id': to_global_id_or_none(checkout), 'paymentGateway': {'id': expected_app_identifier, 'data': None}}
    response = user_api_client.post_graphql(TRANSACTION_INITIALIZE, variables)
    content = get_graphql_content(response)
    checkout.refresh_from_db()
    _assert_fields(content=content, source_object=checkout, expected_amount=expected_amount, expected_psp_reference=expected_psp_reference, response_event_type=TransactionEventType.CHARGE_SUCCESS, app_identifier=webhook_app.identifier, mocked_initialize=mocked_initialize, charged_value=expected_amount, returned_data=expected_response['data'])
    assert checkout.charge_status == CheckoutChargeStatus.PARTIAL
    assert checkout.authorize_status == CheckoutAuthorizeStatus.PARTIAL

@mock.patch('saleor.plugins.manager.PluginsManager.transaction_initialize_session')
def test_for_order_without_payment_gateway_data(mocked_initialize, user_api_client, order_with_lines, webhook_app, transaction_session_response):
    if False:
        return 10
    order = order_with_lines
    expected_app_identifier = 'webhook.app.identifier'
    webhook_app.identifier = expected_app_identifier
    webhook_app.save()
    expected_amount = Decimal('10.00')
    expected_psp_reference = 'ppp-123'
    expected_response = transaction_session_response.copy()
    expected_response['result'] = 'CHARGE_SUCCESS'
    expected_response['pspReference'] = expected_psp_reference
    mocked_initialize.return_value = TransactionSessionResult(app_identifier=expected_app_identifier, response=expected_response)
    variables = {'action': None, 'amount': expected_amount, 'id': to_global_id_or_none(order), 'paymentGateway': {'id': expected_app_identifier, 'data': None}}
    response = user_api_client.post_graphql(TRANSACTION_INITIALIZE, variables)
    content = get_graphql_content(response)
    _assert_fields(content=content, source_object=order, expected_amount=expected_amount, expected_psp_reference=expected_psp_reference, response_event_type=TransactionEventType.CHARGE_SUCCESS, app_identifier=webhook_app.identifier, mocked_initialize=mocked_initialize, charged_value=expected_amount, returned_data=expected_response['data'])
    order.refresh_from_db()
    assert order.total_authorized_amount == Decimal(0)
    assert order.total_charged_amount == expected_amount

@mock.patch('saleor.plugins.manager.PluginsManager.transaction_initialize_session')
def test_checkout_with_pending_amount(mocked_initialize, user_api_client, checkout_with_prices, webhook_app, transaction_session_response):
    if False:
        for i in range(10):
            print('nop')
    checkout = checkout_with_prices
    expected_app_identifier = 'webhook.app.identifier'
    webhook_app.identifier = expected_app_identifier
    webhook_app.save()
    expected_amount = Decimal('10.00')
    expected_psp_reference = 'ppp-123'
    expected_response = transaction_session_response.copy()
    expected_response['result'] = TransactionEventType.CHARGE_REQUEST.upper()
    expected_response['pspReference'] = expected_psp_reference
    mocked_initialize.return_value = TransactionSessionResult(app_identifier=expected_app_identifier, response=expected_response)
    variables = {'action': None, 'amount': expected_amount, 'id': to_global_id_or_none(checkout), 'paymentGateway': {'id': expected_app_identifier, 'data': None}}
    response = user_api_client.post_graphql(TRANSACTION_INITIALIZE, variables)
    content = get_graphql_content(response)
    checkout.refresh_from_db()
    _assert_fields(content=content, source_object=checkout, expected_amount=expected_amount, expected_psp_reference=expected_psp_reference, response_event_type=TransactionEventType.CHARGE_REQUEST, app_identifier=webhook_app.identifier, mocked_initialize=mocked_initialize, charge_pending_value=expected_amount, request_event_include_in_calculations=True, returned_data=expected_response['data'])
    assert checkout.charge_status == CheckoutChargeStatus.PARTIAL
    assert checkout.authorize_status == CheckoutAuthorizeStatus.PARTIAL

@mock.patch('saleor.plugins.manager.PluginsManager.transaction_initialize_session')
def test_order_with_pending_amount(mocked_initialize, user_api_client, order_with_lines, webhook_app, transaction_session_response):
    if False:
        for i in range(10):
            print('nop')
    order = order_with_lines
    expected_app_identifier = 'webhook.app.identifier'
    webhook_app.identifier = expected_app_identifier
    webhook_app.save()
    expected_amount = Decimal('10.00')
    expected_psp_reference = 'ppp-123'
    expected_response = transaction_session_response.copy()
    expected_response['result'] = TransactionEventType.CHARGE_REQUEST.upper()
    expected_response['pspReference'] = expected_psp_reference
    mocked_initialize.return_value = TransactionSessionResult(app_identifier=expected_app_identifier, response=expected_response)
    variables = {'action': None, 'amount': expected_amount, 'id': to_global_id_or_none(order), 'paymentGateway': {'id': expected_app_identifier, 'data': None}}
    response = user_api_client.post_graphql(TRANSACTION_INITIALIZE, variables)
    content = get_graphql_content(response)
    _assert_fields(content=content, source_object=order, expected_amount=expected_amount, expected_psp_reference=expected_psp_reference, response_event_type=TransactionEventType.CHARGE_REQUEST, app_identifier=webhook_app.identifier, mocked_initialize=mocked_initialize, charge_pending_value=expected_amount, request_event_include_in_calculations=True, returned_data=expected_response['data'])
    order.refresh_from_db()
    assert order.total_authorized_amount == Decimal(0)
    assert order.total_charged_amount == Decimal(0)

@mock.patch('saleor.plugins.manager.PluginsManager.transaction_initialize_session')
def test_checkout_with_action_required_response(mocked_initialize, user_api_client, checkout_with_prices, webhook_app, transaction_session_response):
    if False:
        print('Hello World!')
    checkout = checkout_with_prices
    expected_app_identifier = 'webhook.app.identifier'
    webhook_app.identifier = expected_app_identifier
    webhook_app.save()
    expected_amount = Decimal('10.00')
    expected_psp_reference = 'ppp-123'
    expected_response = transaction_session_response.copy()
    expected_response['result'] = TransactionEventType.CHARGE_ACTION_REQUIRED.upper()
    expected_response['pspReference'] = expected_psp_reference
    mocked_initialize.return_value = TransactionSessionResult(app_identifier=expected_app_identifier, response=expected_response)
    variables = {'action': None, 'amount': expected_amount, 'id': to_global_id_or_none(checkout), 'paymentGateway': {'id': expected_app_identifier, 'data': None}}
    response = user_api_client.post_graphql(TRANSACTION_INITIALIZE, variables)
    content = get_graphql_content(response)
    checkout.refresh_from_db()
    _assert_fields(content=content, source_object=checkout, expected_amount=expected_amount, expected_psp_reference=expected_psp_reference, response_event_type=TransactionEventType.CHARGE_ACTION_REQUIRED, app_identifier=webhook_app.identifier, mocked_initialize=mocked_initialize, returned_data=expected_response['data'])
    assert checkout.charge_status == CheckoutChargeStatus.NONE
    assert checkout.authorize_status == CheckoutAuthorizeStatus.NONE

@mock.patch('saleor.plugins.manager.PluginsManager.transaction_initialize_session')
def test_order_with_action_required_response(mocked_initialize, user_api_client, order_with_lines, webhook_app, transaction_session_response):
    if False:
        while True:
            i = 10
    order = order_with_lines
    expected_app_identifier = 'webhook.app.identifier'
    webhook_app.identifier = expected_app_identifier
    webhook_app.save()
    expected_amount = Decimal('10.00')
    expected_psp_reference = 'ppp-123'
    expected_response = transaction_session_response.copy()
    expected_response['result'] = TransactionEventType.CHARGE_ACTION_REQUIRED.upper()
    expected_response['pspReference'] = expected_psp_reference
    mocked_initialize.return_value = TransactionSessionResult(app_identifier=expected_app_identifier, response=expected_response)
    variables = {'action': None, 'amount': expected_amount, 'id': to_global_id_or_none(order), 'paymentGateway': {'id': expected_app_identifier, 'data': None}}
    response = user_api_client.post_graphql(TRANSACTION_INITIALIZE, variables)
    content = get_graphql_content(response)
    _assert_fields(content=content, source_object=order, expected_amount=expected_amount, expected_psp_reference=expected_psp_reference, response_event_type=TransactionEventType.CHARGE_ACTION_REQUIRED, app_identifier=webhook_app.identifier, mocked_initialize=mocked_initialize, returned_data=expected_response['data'])
    order.refresh_from_db()
    assert order.total_authorized_amount == Decimal(0)
    assert order.total_charged_amount == Decimal(0)

@mock.patch('saleor.plugins.manager.PluginsManager.transaction_initialize_session')
def test_checkout_with_action_required_response_and_missing_psp_reference(mocked_initialize, user_api_client, checkout_with_prices, webhook_app, transaction_session_response):
    if False:
        return 10
    checkout = checkout_with_prices
    expected_app_identifier = 'webhook.app.identifier'
    webhook_app.identifier = expected_app_identifier
    webhook_app.save()
    expected_amount = Decimal('10.00')
    expected_response = transaction_session_response.copy()
    expected_response['result'] = TransactionEventType.CHARGE_ACTION_REQUIRED.upper()
    del expected_response['pspReference']
    mocked_initialize.return_value = TransactionSessionResult(app_identifier=expected_app_identifier, response=expected_response)
    variables = {'action': None, 'amount': expected_amount, 'id': to_global_id_or_none(checkout), 'paymentGateway': {'id': expected_app_identifier, 'data': None}}
    response = user_api_client.post_graphql(TRANSACTION_INITIALIZE, variables)
    content = get_graphql_content(response)
    checkout.refresh_from_db()
    _assert_fields(content=content, source_object=checkout, expected_amount=expected_amount, expected_psp_reference=None, response_event_type=TransactionEventType.CHARGE_ACTION_REQUIRED, app_identifier=webhook_app.identifier, mocked_initialize=mocked_initialize, returned_data=expected_response['data'])
    assert checkout.charge_status == CheckoutChargeStatus.NONE
    assert checkout.authorize_status == CheckoutAuthorizeStatus.NONE

@mock.patch('saleor.plugins.manager.PluginsManager.transaction_initialize_session')
def test_order_with_action_required_response_and_missing_psp_reference(mocked_initialize, user_api_client, order_with_lines, webhook_app, transaction_session_response):
    if False:
        for i in range(10):
            print('nop')
    order = order_with_lines
    expected_app_identifier = 'webhook.app.identifier'
    webhook_app.identifier = expected_app_identifier
    webhook_app.save()
    expected_amount = Decimal('10.00')
    expected_response = transaction_session_response.copy()
    expected_response['result'] = TransactionEventType.CHARGE_ACTION_REQUIRED.upper()
    del expected_response['pspReference']
    mocked_initialize.return_value = TransactionSessionResult(app_identifier=expected_app_identifier, response=expected_response)
    variables = {'action': None, 'amount': expected_amount, 'id': to_global_id_or_none(order), 'paymentGateway': {'id': expected_app_identifier, 'data': None}}
    response = user_api_client.post_graphql(TRANSACTION_INITIALIZE, variables)
    content = get_graphql_content(response)
    _assert_fields(content=content, source_object=order, expected_amount=expected_amount, expected_psp_reference=None, response_event_type=TransactionEventType.CHARGE_ACTION_REQUIRED, app_identifier=webhook_app.identifier, mocked_initialize=mocked_initialize, returned_data=expected_response['data'])
    order.refresh_from_db()
    assert order.total_authorized_amount == Decimal(0)
    assert order.total_charged_amount == Decimal(0)

@mock.patch('saleor.plugins.manager.PluginsManager.transaction_initialize_session')
def test_checkout_when_amount_is_not_provided(mocked_initialize, user_api_client, checkout_with_prices, webhook_app, transaction_session_response, plugins_manager):
    if False:
        for i in range(10):
            print('nop')
    checkout = checkout_with_prices
    (lines, _) = fetch_checkout_lines(checkout)
    checkout_info = fetch_checkout_info(checkout, lines, plugins_manager)
    (checkout_info, _) = fetch_checkout_data(checkout_info, plugins_manager, lines)
    expected_app_identifier = 'webhook.app.identifier'
    webhook_app.identifier = expected_app_identifier
    webhook_app.save()
    expected_psp_reference = 'ppp-123'
    expected_response = transaction_session_response.copy()
    expected_response['amount'] = str(checkout_info.checkout.total_gross_amount)
    expected_response['result'] = TransactionEventType.CHARGE_SUCCESS.upper()
    expected_response['pspReference'] = expected_psp_reference
    mocked_initialize.return_value = TransactionSessionResult(app_identifier=expected_app_identifier, response=expected_response)
    variables = {'action': None, 'amount': None, 'id': to_global_id_or_none(checkout), 'paymentGateway': {'id': expected_app_identifier, 'data': None}}
    response = user_api_client.post_graphql(TRANSACTION_INITIALIZE, variables)
    content = get_graphql_content(response)
    checkout.refresh_from_db()
    _assert_fields(content=content, source_object=checkout, expected_amount=checkout.total_gross_amount, expected_psp_reference=expected_psp_reference, response_event_type=TransactionEventType.CHARGE_SUCCESS, app_identifier=webhook_app.identifier, mocked_initialize=mocked_initialize, charged_value=checkout.total_gross_amount, returned_data=expected_response['data'])
    assert checkout.charge_status == CheckoutChargeStatus.FULL
    assert checkout.authorize_status == CheckoutAuthorizeStatus.FULL

@mock.patch('saleor.plugins.manager.PluginsManager.transaction_initialize_session')
def test_order_when_amount_is_not_provided(mocked_initialize, user_api_client, order_with_lines, webhook_app, transaction_session_response):
    if False:
        i = 10
        return i + 15
    order = order_with_lines
    expected_app_identifier = 'webhook.app.identifier'
    webhook_app.identifier = expected_app_identifier
    webhook_app.save()
    expected_psp_reference = 'ppp-123'
    expected_response = transaction_session_response.copy()
    expected_response['result'] = TransactionEventType.CHARGE_SUCCESS.upper()
    expected_response['amount'] = str(order.total_gross_amount)
    expected_response['pspReference'] = expected_psp_reference
    mocked_initialize.return_value = TransactionSessionResult(app_identifier=expected_app_identifier, response=expected_response)
    variables = {'action': None, 'amount': None, 'id': to_global_id_or_none(order), 'paymentGateway': {'id': expected_app_identifier, 'data': None}}
    response = user_api_client.post_graphql(TRANSACTION_INITIALIZE, variables)
    content = get_graphql_content(response)
    _assert_fields(content=content, source_object=order, expected_amount=order.total_gross_amount, expected_psp_reference=expected_psp_reference, response_event_type=TransactionEventType.CHARGE_SUCCESS, app_identifier=webhook_app.identifier, mocked_initialize=mocked_initialize, charged_value=order.total_gross_amount, returned_data=expected_response['data'])
    order.refresh_from_db()
    assert order.total_authorized_amount == Decimal(0)
    assert order.total_charged_amount == order.total_gross_amount

@mock.patch('saleor.plugins.manager.PluginsManager.transaction_initialize_session')
def test_order_with_transaction_when_amount_is_not_provided(mocked_initialize, user_api_client, order_with_lines, webhook_app, transaction_session_response, transaction_item_generator):
    if False:
        i = 10
        return i + 15
    order = order_with_lines
    expected_charged_amount = Decimal('10')
    expected_authorized_amount = Decimal('3')
    transaction_item_generator(order_id=order.pk, charged_value=expected_charged_amount, authorized_value=expected_authorized_amount)
    expected_app_identifier = 'webhook.app.identifier'
    webhook_app.identifier = expected_app_identifier
    webhook_app.save()
    expected_psp_reference = 'ppp-123'
    expected_response = transaction_session_response.copy()
    expected_response['result'] = TransactionEventType.CHARGE_SUCCESS.upper()
    expected_response['amount'] = str(order.total_gross_amount - expected_charged_amount)
    expected_response['pspReference'] = expected_psp_reference
    mocked_initialize.return_value = TransactionSessionResult(app_identifier=expected_app_identifier, response=expected_response)
    variables = {'action': None, 'amount': None, 'id': to_global_id_or_none(order), 'paymentGateway': {'id': expected_app_identifier, 'data': None}}
    response = user_api_client.post_graphql(TRANSACTION_INITIALIZE, variables)
    content = get_graphql_content(response)
    _assert_fields(content=content, source_object=order, expected_amount=order.total_gross_amount - expected_charged_amount, expected_psp_reference=expected_psp_reference, response_event_type=TransactionEventType.CHARGE_SUCCESS, app_identifier=webhook_app.identifier, mocked_initialize=mocked_initialize, charged_value=order.total_gross_amount - expected_charged_amount, returned_data=expected_response['data'])
    order.refresh_from_db()
    assert order.total_authorized_amount == expected_authorized_amount
    assert order.total_charged_amount == order.total_gross_amount

@mock.patch('saleor.plugins.manager.PluginsManager.transaction_initialize_session')
def test_checkout_with_transaction_when_amount_is_not_provided(mocked_initialize, user_api_client, checkout_with_prices, webhook_app, transaction_session_response, transaction_item_generator, plugins_manager):
    if False:
        return 10
    checkout = checkout_with_prices
    (lines, _) = fetch_checkout_lines(checkout)
    checkout_info = fetch_checkout_info(checkout, lines, plugins_manager)
    (checkout_info, _) = fetch_checkout_data(checkout_info, plugins_manager, lines)
    checkout = checkout_info.checkout
    expected_charged_amount = Decimal('10')
    expected_authorized_amount = Decimal('3')
    transaction_item_generator(checkout_id=checkout.pk, charged_value=expected_charged_amount, authorized_value=expected_authorized_amount)
    expected_app_identifier = 'webhook.app.identifier'
    webhook_app.identifier = expected_app_identifier
    webhook_app.save()
    expected_psp_reference = 'ppp-123'
    expected_response = transaction_session_response.copy()
    expected_response['result'] = TransactionEventType.CHARGE_SUCCESS.upper()
    expected_response['amount'] = str(checkout.total_gross_amount - expected_charged_amount)
    expected_response['pspReference'] = expected_psp_reference
    mocked_initialize.return_value = TransactionSessionResult(app_identifier=expected_app_identifier, response=expected_response)
    variables = {'action': None, 'amount': None, 'id': to_global_id_or_none(checkout), 'paymentGateway': {'id': expected_app_identifier, 'data': None}}
    response = user_api_client.post_graphql(TRANSACTION_INITIALIZE, variables)
    content = get_graphql_content(response)
    checkout.refresh_from_db()
    _assert_fields(content=content, source_object=checkout, expected_amount=checkout.total_gross_amount - expected_charged_amount, expected_psp_reference=expected_psp_reference, response_event_type=TransactionEventType.CHARGE_SUCCESS, app_identifier=webhook_app.identifier, mocked_initialize=mocked_initialize, charged_value=checkout.total_gross_amount - expected_charged_amount, returned_data=expected_response['data'])
    assert checkout.charge_status == CheckoutChargeStatus.FULL
    assert checkout.authorize_status == CheckoutAuthorizeStatus.FULL

@mock.patch('saleor.plugins.manager.PluginsManager.transaction_initialize_session')
def test_app_with_action_field_and_handle_payments(mocked_initialize, app_api_client, checkout_with_prices, webhook_app, transaction_session_response, permission_manage_payments, plugins_manager):
    if False:
        return 10
    checkout = checkout_with_prices
    (lines, _) = fetch_checkout_lines(checkout)
    checkout_info = fetch_checkout_info(checkout, lines, plugins_manager)
    (checkout_info, _) = fetch_checkout_data(checkout_info, plugins_manager, lines)
    expected_app_identifier = 'webhook.app.identifier'
    webhook_app.identifier = expected_app_identifier
    webhook_app.save()
    app_api_client.app.permissions.set([permission_manage_payments])
    expected_psp_reference = 'ppp-123'
    expected_response = transaction_session_response.copy()
    expected_response['result'] = TransactionEventType.AUTHORIZATION_SUCCESS.upper()
    expected_response['amount'] = str(checkout.total_gross_amount)
    expected_response['pspReference'] = expected_psp_reference
    del expected_response['data']
    mocked_initialize.return_value = TransactionSessionResult(app_identifier=expected_app_identifier, response=expected_response)
    variables = {'action': TransactionFlowStrategyEnum.AUTHORIZATION.name, 'amount': None, 'id': to_global_id_or_none(checkout), 'paymentGateway': {'id': expected_app_identifier, 'data': None}}
    response = app_api_client.post_graphql(TRANSACTION_INITIALIZE, variables)
    content = get_graphql_content(response)
    checkout.refresh_from_db()
    _assert_fields(content=content, source_object=checkout, expected_amount=checkout.total_gross_amount, expected_psp_reference=expected_psp_reference, response_event_type=TransactionEventType.AUTHORIZATION_SUCCESS, app_identifier=webhook_app.identifier, mocked_initialize=mocked_initialize, request_event_type=TransactionEventType.AUTHORIZATION_REQUEST, authorized_value=checkout.total_gross_amount, action_type=TransactionFlowStrategy.AUTHORIZATION, returned_data=None)
    assert checkout.charge_status == CheckoutChargeStatus.NONE
    assert checkout.authorize_status == CheckoutAuthorizeStatus.FULL

@mock.patch('saleor.plugins.manager.PluginsManager.transaction_initialize_session')
def test_uses_default_channel_action(mocked_initialize, user_api_client, checkout_with_prices, webhook_app, transaction_session_response):
    if False:
        return 10
    checkout = checkout_with_prices
    expected_app_identifier = 'webhook.app.identifier'
    webhook_app.identifier = expected_app_identifier
    webhook_app.save()
    channel = checkout.channel
    channel.default_transaction_flow_strategy = TransactionFlowStrategy.AUTHORIZATION
    channel.save()
    expected_amount = Decimal('10.00')
    expected_psp_reference = 'ppp-123'
    expected_response = transaction_session_response.copy()
    expected_response['result'] = 'CHARGE_SUCCESS'
    expected_response['pspReference'] = expected_psp_reference
    mocked_initialize.return_value = TransactionSessionResult(app_identifier=expected_app_identifier, response=expected_response)
    variables = {'action': None, 'amount': expected_amount, 'id': to_global_id_or_none(checkout), 'paymentGateway': {'id': expected_app_identifier, 'data': None}}
    response = user_api_client.post_graphql(TRANSACTION_INITIALIZE, variables)
    content = get_graphql_content(response)
    _assert_fields(content=content, source_object=checkout, expected_amount=expected_amount, expected_psp_reference=expected_psp_reference, response_event_type=TransactionEventType.CHARGE_SUCCESS, app_identifier=webhook_app.identifier, mocked_initialize=mocked_initialize, charged_value=expected_amount, action_type=TransactionFlowStrategy.AUTHORIZATION, request_event_type=TransactionEventType.AUTHORIZATION_REQUEST, returned_data=expected_response['data'])

@mock.patch('saleor.plugins.manager.PluginsManager.transaction_initialize_session')
def test_app_with_action_field(mocked_initialize, app_api_client, checkout_with_prices, webhook_app, transaction_session_response, transaction_item_generator, permission_manage_payments):
    if False:
        return 10
    checkout = checkout_with_prices
    expected_charged_amount = Decimal('10')
    expected_authorized_amount = Decimal('3')
    transaction_item_generator(checkout_id=checkout.pk, charged_value=expected_charged_amount, authorized_value=expected_authorized_amount)
    expected_app_identifier = 'webhook.app.identifier'
    webhook_app.identifier = expected_app_identifier
    webhook_app.save()
    expected_psp_reference = 'ppp-123'
    expected_response = transaction_session_response.copy()
    expected_response['result'] = TransactionEventType.AUTHORIZATION_SUCCESS.upper()
    expected_response['amount'] = str(checkout.total_gross_amount - expected_charged_amount)
    expected_response['pspReference'] = expected_psp_reference
    mocked_initialize.return_value = TransactionSessionResult(app_identifier=expected_app_identifier, response=expected_response)
    variables = {'action': TransactionFlowStrategyEnum.AUTHORIZATION.name, 'amount': None, 'id': to_global_id_or_none(checkout), 'paymentGateway': {'id': expected_app_identifier, 'data': None}}
    response = app_api_client.post_graphql(TRANSACTION_INITIALIZE, variables)
    assert_no_permission(response)

def test_customer_with_action_field(app_api_client, checkout_with_prices, webhook_app, transaction_session_response, transaction_item_generator, permission_manage_payments):
    if False:
        return 10
    checkout = checkout_with_prices
    expected_charged_amount = Decimal('10')
    expected_authorized_amount = Decimal('3')
    transaction_item_generator(checkout_id=checkout.pk, charged_value=expected_charged_amount, authorized_value=expected_authorized_amount)
    expected_app_identifier = 'webhook.app.identifier'
    webhook_app.identifier = expected_app_identifier
    webhook_app.save()
    variables = {'action': TransactionFlowStrategyEnum.AUTHORIZATION.name, 'amount': None, 'id': to_global_id_or_none(checkout), 'paymentGateway': {'id': expected_app_identifier, 'data': None}}
    response = app_api_client.post_graphql(TRANSACTION_INITIALIZE, variables)
    assert_no_permission(response)

def test_incorrect_source_object_id(user_api_client, checkout_with_prices, webhook_app, transaction_session_response, transaction_item_generator, permission_manage_payments, product):
    if False:
        return 10
    checkout = checkout_with_prices
    expected_charged_amount = Decimal('10')
    expected_authorized_amount = Decimal('3')
    transaction_item_generator(checkout_id=checkout.pk, charged_value=expected_charged_amount, authorized_value=expected_authorized_amount)
    expected_app_identifier = 'webhook.app.identifier'
    webhook_app.identifier = expected_app_identifier
    webhook_app.save()
    variables = {'action': None, 'amount': None, 'id': to_global_id_or_none(product), 'paymentGateway': {'id': expected_app_identifier, 'data': None}}
    response = user_api_client.post_graphql(TRANSACTION_INITIALIZE, variables)
    content = get_graphql_content(response, ignore_errors=True)
    errors = content['data']['transactionInitialize']['errors']
    assert errors
    assert len(errors) == 1
    assert errors[0]['field'] == 'id'
    assert errors[0]['code'] == TransactionInitializeErrorCode.INVALID.name

def test_checkout_doesnt_exist(user_api_client, checkout_with_prices, webhook_app, transaction_session_response, permission_manage_payments):
    if False:
        return 10
    checkout = checkout_with_prices
    expected_app_identifier = 'webhook.app.identifier'
    webhook_app.identifier = expected_app_identifier
    webhook_app.save()
    variables = {'action': None, 'amount': None, 'id': to_global_id_or_none(checkout), 'paymentGateway': {'id': expected_app_identifier, 'data': None}}
    checkout.delete()
    response = user_api_client.post_graphql(TRANSACTION_INITIALIZE, variables)
    content = get_graphql_content(response, ignore_errors=True)
    errors = content['data']['transactionInitialize']['errors']
    assert errors
    assert len(errors) == 1
    assert errors[0]['field'] == 'id'
    assert errors[0]['code'] == TransactionInitializeErrorCode.NOT_FOUND.name

def test_order_doesnt_exists(user_api_client, order_with_lines, webhook_app, transaction_session_response, permission_manage_payments):
    if False:
        i = 10
        return i + 15
    order = order_with_lines
    expected_app_identifier = 'webhook.app.identifier'
    webhook_app.identifier = expected_app_identifier
    webhook_app.save()
    variables = {'action': None, 'amount': None, 'id': to_global_id_or_none(order), 'paymentGateway': {'id': expected_app_identifier, 'data': None}}
    order.delete()
    response = user_api_client.post_graphql(TRANSACTION_INITIALIZE, variables)
    content = get_graphql_content(response, ignore_errors=True)
    errors = content['data']['transactionInitialize']['errors']
    assert errors
    assert len(errors) == 1
    assert errors[0]['field'] == 'id'
    assert errors[0]['code'] == TransactionInitializeErrorCode.NOT_FOUND.name

@pytest.mark.parametrize('result', [TransactionEventType.CHARGE_REQUEST, TransactionEventType.CHARGE_SUCCESS])
@mock.patch('saleor.plugins.manager.PluginsManager.checkout_fully_paid')
@mock.patch('saleor.plugins.manager.PluginsManager.transaction_initialize_session')
def test_checkout_fully_paid(mocked_initialize, mocked_fully_paid, result, user_api_client, checkout_with_prices, webhook_app, transaction_session_response, plugins_manager):
    if False:
        while True:
            i = 10
    checkout = checkout_with_prices
    (lines, _) = fetch_checkout_lines(checkout)
    checkout_info = fetch_checkout_info(checkout, lines, plugins_manager)
    (checkout_info, _) = fetch_checkout_data(checkout_info, plugins_manager, lines)
    expected_app_identifier = 'webhook.app.identifier'
    webhook_app.identifier = expected_app_identifier
    webhook_app.save()
    expected_psp_reference = 'ppp-123'
    expected_response = transaction_session_response.copy()
    expected_response['amount'] = str(checkout_info.checkout.total_gross_amount)
    expected_response['result'] = result.upper()
    expected_response['pspReference'] = expected_psp_reference
    mocked_initialize.return_value = TransactionSessionResult(app_identifier=expected_app_identifier, response=expected_response)
    variables = {'action': None, 'amount': None, 'id': to_global_id_or_none(checkout), 'paymentGateway': {'id': expected_app_identifier, 'data': None}}
    response = user_api_client.post_graphql(TRANSACTION_INITIALIZE, variables)
    content = get_graphql_content(response)
    assert not content['data']['transactionInitialize']['errors']
    checkout.refresh_from_db()
    mocked_fully_paid.assert_called_once_with(checkout)
    assert checkout.charge_status == CheckoutChargeStatus.FULL
    assert checkout.authorize_status == CheckoutAuthorizeStatus.FULL

def test_user_missing_permission_for_customer_ip_address(user_api_client, order_with_lines, webhook_app, transaction_session_response):
    if False:
        print('Hello World!')
    order = order_with_lines
    expected_app_identifier = 'webhook.app.identifier'
    webhook_app.identifier = expected_app_identifier
    webhook_app.save()
    variables = {'action': None, 'amount': None, 'id': to_global_id_or_none(order), 'paymentGateway': {'id': expected_app_identifier, 'data': None}, 'customerIpAddress': '127.0.0.1'}
    response = user_api_client.post_graphql(TRANSACTION_INITIALIZE, variables)
    assert_no_permission(response)

def test_app_missing_permission_for_customer_ip_address(app_api_client, order_with_lines, webhook_app, transaction_session_response):
    if False:
        print('Hello World!')
    order = order_with_lines
    expected_app_identifier = 'webhook.app.identifier'
    webhook_app.identifier = expected_app_identifier
    webhook_app.save()
    variables = {'action': None, 'amount': None, 'id': to_global_id_or_none(order), 'paymentGateway': {'id': expected_app_identifier, 'data': None}, 'customerIpAddress': '127.0.0.1'}
    response = app_api_client.post_graphql(TRANSACTION_INITIALIZE, variables)
    assert_no_permission(response)

@mock.patch('saleor.plugins.manager.PluginsManager.transaction_initialize_session')
def test_with_customer_ip_address(mocked_initialize, app_api_client, order_with_lines, webhook_app, transaction_session_response, permission_manage_payments):
    if False:
        while True:
            i = 10
    order = order_with_lines
    expected_app_identifier = 'webhook.app.identifier'
    webhook_app.identifier = expected_app_identifier
    webhook_app.save()
    expected_amount = Decimal('10.00')
    expected_psp_reference = 'ppp-123'
    expected_response = transaction_session_response.copy()
    expected_response['result'] = TransactionEventType.CHARGE_REQUEST.upper()
    expected_response['pspReference'] = expected_psp_reference
    mocked_initialize.return_value = TransactionSessionResult(app_identifier=expected_app_identifier, response=expected_response)
    app_api_client.app.permissions.set([permission_manage_payments])
    variables = {'action': None, 'amount': expected_amount, 'id': to_global_id_or_none(order), 'paymentGateway': {'id': expected_app_identifier, 'data': None}, 'customerIpAddress': '127.0.0.2'}
    response = app_api_client.post_graphql(TRANSACTION_INITIALIZE, variables)
    get_graphql_content(response)
    transaction = order.payment_transactions.last()
    mocked_initialize.assert_called_with(TransactionSessionData(transaction=transaction, source_object=order, action=TransactionProcessActionData(action_type=TransactionFlowStrategy.CHARGE, amount=expected_amount, currency=order.currency), customer_ip_address='127.0.0.2', payment_gateway_data=PaymentGatewayData(app_identifier=webhook_app.identifier, data=None, error=None)))

@mock.patch('saleor.plugins.manager.PluginsManager.transaction_initialize_session')
def test_sets_customer_ip_address_when_not_provided(mocked_initialize, app_api_client, order_with_lines, webhook_app, transaction_session_response, permission_manage_payments):
    if False:
        return 10
    order = order_with_lines
    expected_app_identifier = 'webhook.app.identifier'
    webhook_app.identifier = expected_app_identifier
    webhook_app.save()
    expected_amount = Decimal('10.00')
    expected_psp_reference = 'ppp-123'
    expected_response = transaction_session_response.copy()
    expected_response['result'] = TransactionEventType.CHARGE_REQUEST.upper()
    expected_response['pspReference'] = expected_psp_reference
    mocked_initialize.return_value = TransactionSessionResult(app_identifier=expected_app_identifier, response=expected_response)
    app_api_client.app.permissions.set([permission_manage_payments])
    variables = {'action': None, 'amount': expected_amount, 'id': to_global_id_or_none(order), 'paymentGateway': {'id': expected_app_identifier, 'data': None}}
    response = app_api_client.post_graphql(TRANSACTION_INITIALIZE, variables, REMOTE_ADDR='127.0.0.2')
    get_graphql_content(response)
    transaction = order.payment_transactions.last()
    mocked_initialize.assert_called_with(TransactionSessionData(transaction=transaction, source_object=order, action=TransactionProcessActionData(action_type=TransactionFlowStrategy.CHARGE, amount=expected_amount, currency=order.currency), customer_ip_address='127.0.0.2', payment_gateway_data=PaymentGatewayData(app_identifier=webhook_app.identifier, data=None, error=None)))

def test_customer_ip_address_wrong_format(app_api_client, order_with_lines, webhook_app, transaction_session_response, permission_manage_payments):
    if False:
        for i in range(10):
            print('nop')
    order = order_with_lines
    expected_app_identifier = 'webhook.app.identifier'
    webhook_app.identifier = expected_app_identifier
    webhook_app.save()
    expected_amount = Decimal('10.00')
    app_api_client.app.permissions.set([permission_manage_payments])
    variables = {'action': None, 'amount': expected_amount, 'id': to_global_id_or_none(order), 'paymentGateway': {'id': expected_app_identifier, 'data': None}, 'customerIpAddress': '127.0.02'}
    response = app_api_client.post_graphql(TRANSACTION_INITIALIZE, variables)
    content = get_graphql_content(response)
    errors = content['data']['transactionInitialize']['errors']
    assert len(errors) == 1
    assert errors[0]['field'] == 'customerIpAddress'
    assert errors[0]['code'] == TransactionInitializeErrorCode.INVALID.name

@mock.patch('saleor.plugins.manager.PluginsManager.transaction_initialize_session')
def test_customer_ip_address_ipv6(mocked_initialize, app_api_client, order_with_lines, webhook_app, transaction_session_response, permission_manage_payments):
    if False:
        i = 10
        return i + 15
    order = order_with_lines
    expected_app_identifier = 'webhook.app.identifier'
    webhook_app.identifier = expected_app_identifier
    webhook_app.save()
    expected_amount = Decimal('10.00')
    expected_psp_reference = 'ppp-123'
    expected_response = transaction_session_response.copy()
    expected_response['result'] = TransactionEventType.CHARGE_REQUEST.upper()
    expected_response['pspReference'] = expected_psp_reference
    mocked_initialize.return_value = TransactionSessionResult(app_identifier=expected_app_identifier, response=expected_response)
    app_api_client.app.permissions.set([permission_manage_payments])
    variables = {'action': None, 'amount': expected_amount, 'id': to_global_id_or_none(order), 'paymentGateway': {'id': expected_app_identifier, 'data': None}, 'customerIpAddress': '::1'}
    response = app_api_client.post_graphql(TRANSACTION_INITIALIZE, variables)
    get_graphql_content(response)
    transaction = order.payment_transactions.last()
    mocked_initialize.assert_called_with(TransactionSessionData(transaction=transaction, source_object=order, action=TransactionProcessActionData(action_type=TransactionFlowStrategy.CHARGE, amount=expected_amount, currency=order.currency), customer_ip_address='::1', payment_gateway_data=PaymentGatewayData(app_identifier=webhook_app.identifier, data=None, error=None)))