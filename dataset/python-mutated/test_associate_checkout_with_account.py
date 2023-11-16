import pytest
from saleor.channel import MarkAsPaidStrategy
from saleor.checkout.complete_checkout import complete_checkout
from ...plugins.manager import get_plugins_manager
from ..fetch import fetch_checkout_info, fetch_checkout_lines

@pytest.mark.django_db
@pytest.mark.parametrize('paid_strategy', [MarkAsPaidStrategy.TRANSACTION_FLOW, MarkAsPaidStrategy.PAYMENT_FLOW])
def test_associate_guest_checkout_with_account_if_exists(paid_strategy, app, address, checkout, customer_user):
    if False:
        return 10
    checkout.email = 'test@example.com'
    checkout.billing_address = address
    checkout.save()
    user = None
    manager = get_plugins_manager()
    (lines, _) = fetch_checkout_lines(checkout)
    checkout_info = fetch_checkout_info(checkout, lines, manager)
    checkout_info.channel.order_mark_as_paid_strategy == paid_strategy
    (order, _, _) = complete_checkout(checkout_info=checkout_info, manager=manager, lines=lines, payment_data={}, store_source=False, user=user, app=app)
    assert order.user == customer_user

@pytest.mark.django_db
@pytest.mark.parametrize('paid_strategy', [MarkAsPaidStrategy.TRANSACTION_FLOW, MarkAsPaidStrategy.PAYMENT_FLOW])
def test_associate_guest_checkout_with_account_if_exists_with_guest_user(paid_strategy, app, address, checkout):
    if False:
        print('Hello World!')
    checkout.email = 'guest@email.com'
    checkout.billing_address = address
    checkout.save()
    user = None
    manager = get_plugins_manager()
    (lines, _) = fetch_checkout_lines(checkout)
    checkout_info = fetch_checkout_info(checkout, lines, manager)
    checkout_info.channel.order_mark_as_paid_strategy == paid_strategy
    (order, _, _) = complete_checkout(checkout_info=checkout_info, manager=manager, lines=lines, payment_data={}, store_source=False, user=user, app=app)
    assert order.user is None

@pytest.mark.django_db
@pytest.mark.parametrize('paid_strategy', [MarkAsPaidStrategy.TRANSACTION_FLOW, MarkAsPaidStrategy.PAYMENT_FLOW])
def test_associate_guest_checkout_with_account_if_exists_with_inactive_user(paid_strategy, app, address, checkout, customer_user):
    if False:
        while True:
            i = 10
    checkout.email = 'test@example.com'
    checkout.billing_address = address
    checkout.save()
    customer_user.is_active = False
    customer_user.save()
    user = None
    manager = get_plugins_manager()
    (lines, _) = fetch_checkout_lines(checkout)
    checkout_info = fetch_checkout_info(checkout, lines, manager)
    checkout_info.channel.order_mark_as_paid_strategy == paid_strategy
    (order, _, _) = complete_checkout(checkout_info=checkout_info, manager=manager, lines=lines, payment_data={}, store_source=False, user=user, app=app)
    assert order.user is None