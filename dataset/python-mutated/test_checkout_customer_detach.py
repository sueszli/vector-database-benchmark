from .....account.models import User
from ....core.utils import to_global_id_or_none
from ....tests.utils import assert_no_permission, get_graphql_content
MUTATION_CHECKOUT_CUSTOMER_DETACH = '\n    mutation checkoutCustomerDetach($id: ID) {\n        checkoutCustomerDetach(id: $id) {\n            checkout {\n                token\n            }\n            errors {\n                field\n                message\n            }\n        }\n    }\n    '

def test_checkout_customer_detach(user_api_client, checkout_with_item, customer_user):
    if False:
        i = 10
        return i + 15
    checkout = checkout_with_item
    checkout.user = customer_user
    checkout.save(update_fields=['user'])
    previous_last_change = checkout.last_change
    variables = {'id': to_global_id_or_none(checkout)}
    response = user_api_client.post_graphql(MUTATION_CHECKOUT_CUSTOMER_DETACH, variables)
    content = get_graphql_content(response)
    data = content['data']['checkoutCustomerDetach']
    assert not data['errors']
    checkout.refresh_from_db()
    assert checkout.user is None
    assert checkout.last_change != previous_last_change
    other_user = User.objects.create_user('othercustomer@example.com', 'password')
    checkout.user = other_user
    checkout.save()
    response = user_api_client.post_graphql(MUTATION_CHECKOUT_CUSTOMER_DETACH, variables)
    assert_no_permission(response)

def test_checkout_customer_detach_by_app(app_api_client, checkout_with_item, customer_user, permission_impersonate_user):
    if False:
        while True:
            i = 10
    checkout = checkout_with_item
    checkout.user = customer_user
    checkout.save(update_fields=['user'])
    previous_last_change = checkout.last_change
    variables = {'id': to_global_id_or_none(checkout)}
    response = app_api_client.post_graphql(MUTATION_CHECKOUT_CUSTOMER_DETACH, variables, permissions=[permission_impersonate_user])
    content = get_graphql_content(response)
    data = content['data']['checkoutCustomerDetach']
    assert not data['errors']
    checkout.refresh_from_db()
    assert checkout.user is None
    assert checkout.last_change != previous_last_change

def test_checkout_customer_detach_by_app_without_permissions(app_api_client, checkout_with_item, customer_user):
    if False:
        while True:
            i = 10
    checkout = checkout_with_item
    checkout.user = customer_user
    checkout.save(update_fields=['user'])
    previous_last_change = checkout.last_change
    variables = {'id': to_global_id_or_none(checkout)}
    response = app_api_client.post_graphql(MUTATION_CHECKOUT_CUSTOMER_DETACH, variables)
    assert_no_permission(response)
    checkout.refresh_from_db()
    assert checkout.last_change == previous_last_change

def test_with_active_problems_flow(user_api_client, checkout_with_problems):
    if False:
        print('Hello World!')
    checkout_with_problems.user = user_api_client.user
    checkout_with_problems.save(update_fields=['user'])
    channel = checkout_with_problems.channel
    channel.use_legacy_error_flow_for_checkout = False
    channel.save(update_fields=['use_legacy_error_flow_for_checkout'])
    variables = {'id': to_global_id_or_none(checkout_with_problems)}
    response = user_api_client.post_graphql(MUTATION_CHECKOUT_CUSTOMER_DETACH, variables)
    content = get_graphql_content(response)
    assert not content['data']['checkoutCustomerDetach']['errors']