"""
This file defines several toy services that interact to form a shop of the
famous ACME Corporation. The AcmeShopService relies on the StockService,
InvoiceService and PaymentService to fulfil its orders. They are not best
practice examples! They're minimal services provided for the test at the
bottom of the file.

``test_shop_integration`` is a full integration test of the ACME shop
"checkout flow". It demonstrates how to test the multiple ACME services in
combination with each other, including limiting service interactions by
replacing certain entrypoints and dependencies.
"""
from collections import defaultdict
import pytest
from nameko.extensions import DependencyProvider
from nameko.events import EventDispatcher, event_handler
from nameko.exceptions import RemoteError
from nameko.rpc import rpc, RpcProxy
from nameko.standalone.rpc import ServiceRpcProxy
from nameko.testing.services import replace_dependencies, restrict_entrypoints
from nameko.testing.utils import get_container
from nameko.timer import timer

class NotLoggedInError(Exception):
    pass

class ItemOutOfStockError(Exception):
    pass

class ItemDoesNotExistError(Exception):
    pass

class ShoppingBasket(DependencyProvider):
    """ A shopping basket tied to the current ``user_id``.
    """

    def __init__(self):
        if False:
            return 10
        self.baskets = defaultdict(list)

    def get_dependency(self, worker_ctx):
        if False:
            for i in range(10):
                print('nop')

        class Basket(object):

            def __init__(self, basket):
                if False:
                    i = 10
                    return i + 15
                self._basket = basket
                self.worker_ctx = worker_ctx

            def add(self, item):
                if False:
                    print('Hello World!')
                self._basket.append(item)

            def __iter__(self):
                if False:
                    i = 10
                    return i + 15
                for item in self._basket:
                    yield item
        try:
            user_id = worker_ctx.data['user_id']
        except KeyError:
            raise NotLoggedInError()
        return Basket(self.baskets[user_id])

class AcmeShopService:
    name = 'acmeshopservice'
    user_basket = ShoppingBasket()
    stock_rpc = RpcProxy('stockservice')
    invoice_rpc = RpcProxy('invoiceservice')
    payment_rpc = RpcProxy('paymentservice')
    fire_event = EventDispatcher()

    @rpc
    def add_to_basket(self, item_code):
        if False:
            return 10
        ' Add item identified by ``item_code`` to the shopping basket.\n\n        This is a toy example! Ignore the obvious race condition.\n        '
        stock_level = self.stock_rpc.check_stock(item_code)
        if stock_level > 0:
            self.user_basket.add(item_code)
            self.fire_event('item_added_to_basket', item_code)
            return item_code
        raise ItemOutOfStockError(item_code)

    @rpc
    def checkout(self):
        if False:
            while True:
                i = 10
        ' Take payment for all items in the shopping basket.\n        '
        total_price = sum((self.stock_rpc.check_price(item) for item in self.user_basket))
        invoice = self.invoice_rpc.prepare_invoice(total_price)
        self.payment_rpc.take_payment(invoice)
        checkout_event_data = {'invoice': invoice, 'items': list(self.user_basket)}
        self.fire_event('checkout_complete', checkout_event_data)
        return total_price

class Warehouse(DependencyProvider):
    """ A database of items in the warehouse.

    This is a toy example! A dictionary is not a database.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.database = {'anvil': {'price': 100, 'stock': 3}, 'dehydrated_boulders': {'price': 999, 'stock': 12}, 'invisible_paint': {'price': 10, 'stock': 30}, 'toothpicks': {'price': 1, 'stock': 0}}

    def get_dependency(self, worker_ctx):
        if False:
            i = 10
            return i + 15
        return self.database

class StockService:
    name = 'stockservice'
    warehouse = Warehouse()

    @rpc
    def check_price(self, item_code):
        if False:
            i = 10
            return i + 15
        ' Check the price of an item.\n        '
        try:
            return self.warehouse[item_code]['price']
        except KeyError:
            raise ItemDoesNotExistError(item_code)

    @rpc
    def check_stock(self, item_code):
        if False:
            print('Hello World!')
        ' Check the stock level of an item.\n        '
        try:
            return self.warehouse[item_code]['stock']
        except KeyError:
            raise ItemDoesNotExistError(item_code)

    @rpc
    @timer(100)
    def monitor_stock(self):
        if False:
            print('Hello World!')
        " Periodic stock monitoring method. Can also be triggered manually\n        over RPC.\n\n        This is an expensive process that we don't want to exercise during\n        integration testing...\n        "
        raise NotImplementedError()

    @event_handler('acmeshopservice', 'checkout_complete')
    def dispatch_items(self, event_data):
        if False:
            return 10
        " Dispatch items from stock on successful checkouts.\n\n        This is an expensive process that we don't want to exercise during\n        integration testing...\n        "
        raise NotImplementedError()

class AddressBook(DependencyProvider):
    """ A database of user details, keyed on user_id.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self.address_book = {'wile_e_coyote': {'username': 'wile_e_coyote', 'fullname': 'Wile E Coyote', 'address': '12 Long Road, High Cliffs, Utah'}}

    def get_dependency(self, worker_ctx):
        if False:
            return 10

        def get_user_details():
            if False:
                print('Hello World!')
            try:
                user_id = worker_ctx.data['user_id']
            except KeyError:
                raise NotLoggedInError()
            return self.address_book.get(user_id)
        return get_user_details

class InvoiceService:
    name = 'invoiceservice'
    get_user_details = AddressBook()

    @rpc
    def prepare_invoice(self, amount):
        if False:
            return 10
        ' Prepare an invoice for ``amount`` for the current user.\n        '
        address = self.get_user_details().get('address')
        fullname = self.get_user_details().get('fullname')
        username = self.get_user_details().get('username')
        msg = 'Dear {}. Please pay ${} to ACME Corp.'.format(fullname, amount)
        invoice = {'message': msg, 'amount': amount, 'customer': username, 'address': address}
        return invoice

class PaymentService:
    name = 'paymentservice'

    @rpc
    def take_payment(self, invoice):
        if False:
            while True:
                i = 10
        " Take payment from a customer according to ``invoice``.\n\n        This is an expensive process that we don't want to exercise during\n        integration testing...\n        "
        raise NotImplementedError()

@pytest.fixture
def rpc_proxy_factory(rabbit_config):
    if False:
        print('Hello World!')
    ' Factory fixture for standalone RPC proxies.\n\n    Proxies are started automatically so they can be used without a ``with``\n    statement. All created proxies are stopped at the end of the test, when\n    this fixture closes.\n    '
    all_proxies = []

    def make_proxy(service_name, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        proxy = ServiceRpcProxy(service_name, rabbit_config, **kwargs)
        all_proxies.append(proxy)
        return proxy.start()
    yield make_proxy
    for proxy in all_proxies:
        proxy.stop()

def test_shop_checkout_integration(rabbit_config, runner_factory, rpc_proxy_factory):
    if False:
        print('Hello World!')
    " Simulate a checkout flow as an integration test.\n\n    Requires instances of AcmeShopService, StockService and InvoiceService\n    to be running. Explicitly replaces the rpc proxy to PaymentService so\n    that service doesn't need to be hosted.\n\n    Also replaces the event dispatcher dependency on AcmeShopService and\n    disables the timer entrypoint on StockService. Limiting the interactions\n    of services in this way reduces the scope of the integration test and\n    eliminates undesirable side-effects (e.g. processing events unnecessarily).\n    "
    context_data = {'user_id': 'wile_e_coyote'}
    shop = rpc_proxy_factory('acmeshopservice', context_data=context_data)
    runner = runner_factory(rabbit_config, AcmeShopService, StockService, InvoiceService)
    shop_container = get_container(runner, AcmeShopService)
    (fire_event, payment_rpc) = replace_dependencies(shop_container, 'fire_event', 'payment_rpc')
    stock_container = get_container(runner, StockService)
    restrict_entrypoints(stock_container, 'check_price', 'check_stock')
    runner.start()
    assert shop.add_to_basket('anvil') == 'anvil'
    assert shop.add_to_basket('invisible_paint') == 'invisible_paint'
    with pytest.raises(RemoteError) as exc_info:
        shop.add_to_basket('toothpicks')
    assert exc_info.value.exc_type == 'ItemOutOfStockError'
    payment_rpc.take_payment.return_value = 'Payment complete.'
    res = shop.checkout()
    total_amount = 100 + 10
    assert res == total_amount
    payment_rpc.take_payment.assert_called_once_with({'customer': 'wile_e_coyote', 'address': '12 Long Road, High Cliffs, Utah', 'amount': total_amount, 'message': 'Dear Wile E Coyote. Please pay $110 to ACME Corp.'})
    assert fire_event.call_count == 3
if __name__ == '__main__':
    import sys
    pytest.main(sys.argv)