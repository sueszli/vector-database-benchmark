from unittest.mock import ANY
import graphene
import pytest
from .....payment.interface import CustomerSource, PaymentMethodInfo, TokenConfig
from .....payment.utils import fetch_customer_id, store_customer_id
from ....tests.utils import assert_no_permission, get_graphql_content
DUMMY_GATEWAY = 'mirumee.payments.dummy'

@pytest.fixture
def braintree_customer_id():
    if False:
        return 10
    return '1234'

@pytest.fixture
def dummy_customer_id():
    if False:
        for i in range(10):
            print('nop')
    return '4321'

def test_store_payment_gateway_meta(customer_user, braintree_customer_id):
    if False:
        return 10
    gateway_name = 'braintree'
    meta_key = 'BRAINTREE.customer_id'
    store_customer_id(customer_user, gateway_name, braintree_customer_id)
    assert customer_user.private_metadata[meta_key] == braintree_customer_id
    customer_user.refresh_from_db()
    assert fetch_customer_id(customer_user, gateway_name) == braintree_customer_id

@pytest.fixture
def token_config_with_customer(braintree_customer_id):
    if False:
        i = 10
        return i + 15
    return TokenConfig(customer_id=braintree_customer_id)

@pytest.fixture
def set_braintree_customer_id(customer_user, braintree_customer_id):
    if False:
        i = 10
        return i + 15
    gateway_name = 'braintree'
    store_customer_id(customer_user, gateway_name, braintree_customer_id)
    return customer_user

@pytest.fixture
def set_dummy_customer_id(customer_user, dummy_customer_id):
    if False:
        while True:
            i = 10
    gateway_name = DUMMY_GATEWAY
    store_customer_id(customer_user, gateway_name, dummy_customer_id)
    return customer_user

def test_list_payment_sources(mocker, dummy_customer_id, set_dummy_customer_id, user_api_client, channel_USD):
    if False:
        i = 10
        return i + 15
    metadata = {f'key_{i}': f'value_{i}' for i in range(5)}
    gateway = DUMMY_GATEWAY
    query = '\n    {\n        me {\n            storedPaymentSources {\n                gateway\n                paymentMethodId\n                creditCardInfo {\n                    lastDigits\n                    brand\n                    firstDigits\n                }\n                metadata {\n                    key\n                    value\n                }\n            }\n        }\n    }\n    '
    card = PaymentMethodInfo(last_4='5678', first_4='1234', exp_year=2020, exp_month=12, name='JohnDoe', brand='cardBrand')
    source = CustomerSource(id='payment-method-id', gateway=gateway, credit_card_info=card, metadata=metadata)
    mock_get_source_list = mocker.patch('saleor.graphql.account.resolvers.gateway.list_payment_sources', return_value=[source], autospec=True)
    response = user_api_client.post_graphql(query)
    mock_get_source_list.assert_called_once_with(gateway, dummy_customer_id, ANY, None)
    content = get_graphql_content(response)['data']['me']['storedPaymentSources']
    assert content is not None
    assert len(content) == 1
    assert content[0] == {'gateway': gateway, 'paymentMethodId': 'payment-method-id', 'creditCardInfo': {'firstDigits': '1234', 'lastDigits': '5678', 'brand': 'cardBrand'}, 'metadata': [{'key': key, 'value': value} for (key, value) in metadata.items()]}

def test_stored_payment_sources_restriction(mocker, staff_api_client, customer_user, permission_manage_users):
    if False:
        print('Hello World!')
    card = PaymentMethodInfo(last_4='5678', exp_year=2020, exp_month=12, name='JohnDoe')
    source = CustomerSource(id='test1', gateway='dummy', credit_card_info=card)
    mocker.patch('saleor.graphql.account.resolvers.gateway.list_payment_sources', return_value=[source], autospec=True)
    customer_user_id = graphene.Node.to_global_id('User', customer_user.pk)
    query = '\n        query PaymentSources($id: ID!) {\n            user(id: $id) {\n                storedPaymentSources {\n                    creditCardInfo {\n                        firstDigits\n                    }\n                }\n            }\n        }\n    '
    variables = {'id': customer_user_id}
    response = staff_api_client.post_graphql(query, variables, permissions=[permission_manage_users])
    assert_no_permission(response)