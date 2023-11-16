from google.appengine.ext import ndb
import pytest
import shopping_cart

@pytest.fixture
def items(testbed):
    if False:
        print('Hello World!')
    account = shopping_cart.Account(id='123')
    account.put()
    items = [shopping_cart.InventoryItem(name='Item {}'.format(i)) for i in range(6)]
    special_items = [shopping_cart.InventoryItem(name='Special {}'.format(i)) for i in range(6)]
    for i in items + special_items:
        i.put()
    special_offers = [shopping_cart.SpecialOffer(inventory=item.key) for item in special_items]
    cart_items = [shopping_cart.CartItem(account=account.key, inventory=item.key, quantity=i) for (i, item) in enumerate(items[:6] + special_items[:6])]
    for i in special_offers + cart_items:
        i.put()
    return (account, items, special_items, cart_items, special_offers)

def test_get_cart_plus_offers(items):
    if False:
        for i in range(10):
            print('nop')
    (account, items, special_items, cart_items, special_offers) = items
    (cart, offers) = shopping_cart.get_cart_plus_offers(account)
    assert len(cart) == 12
    assert len(offers) == 6

def test_get_cart_plus_offers_async(items):
    if False:
        while True:
            i = 10
    (account, items, special_items, cart_items, special_offers) = items
    (cart, offers) = shopping_cart.get_cart_plus_offers_async(account)
    assert len(cart) == 12
    assert len(offers) == 6

def test_get_cart_tasklet(items):
    if False:
        for i in range(10):
            print('nop')
    (account, items, special_items, cart_items, special_offers) = items
    future = shopping_cart.get_cart_tasklet(account)
    cart = future.get_result()
    assert len(cart) == 12

def test_get_offers_tasklet(items):
    if False:
        print('Hello World!')
    (account, items, special_items, cart_items, special_offers) = items
    future = shopping_cart.get_offers_tasklet(account)
    offers = future.get_result()
    assert len(offers) == 6

def test_get_cart_plus_offers_tasklet(items):
    if False:
        while True:
            i = 10
    (account, items, special_items, cart_items, special_offers) = items
    future = shopping_cart.get_cart_plus_offers_tasklet(account)
    (cart, offers) = future.get_result()
    assert len(cart) == 12
    assert len(offers) == 6

def test_iterate_over_query_results_in_tasklet(items):
    if False:
        i = 10
        return i + 15
    (account, items, special_items, cart_items, special_offers) = items
    future = shopping_cart.iterate_over_query_results_in_tasklet(shopping_cart.InventoryItem, lambda item: '3' in item.name)
    assert '3' in future.get_result().name

def test_do_not_iterate_over_tasklet_like_this(items):
    if False:
        while True:
            i = 10
    (account, items, special_items, cart_items, special_offers) = items
    future = shopping_cart.blocking_iteration_over_query_results(shopping_cart.InventoryItem, lambda item: '3' in item.name)
    assert '3' in future.get_result().name

def test_get_google(testbed):
    if False:
        i = 10
        return i + 15
    testbed.init_urlfetch_stub()
    get_google = shopping_cart.define_get_google()
    future = get_google()
    assert 'Google' in future.get_result()

class Counter(ndb.Model):
    value = ndb.IntegerProperty()

def test_update_counter_async(testbed):
    if False:
        i = 10
        return i + 15
    counter_key = Counter(value=1).put()
    update_counter = shopping_cart.define_update_counter_async()
    future = update_counter(counter_key)
    assert counter_key.get().value == 1
    assert future.get_result() == 2
    assert counter_key.get().value == 2

def test_update_counter_tasklet(testbed):
    if False:
        while True:
            i = 10
    counter_key = Counter(value=1).put()
    update_counter = shopping_cart.define_update_counter_tasklet()
    future = update_counter(counter_key)
    assert counter_key.get().value == 1
    future.get_result()
    assert counter_key.get().value == 2

def test_get_first_ready(testbed):
    if False:
        while True:
            i = 10
    testbed.init_urlfetch_stub()
    content = shopping_cart.get_first_ready()
    assert 'html' in content.lower()