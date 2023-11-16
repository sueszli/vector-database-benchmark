from google.appengine.ext import ndb

class Account(ndb.Model):
    pass

class InventoryItem(ndb.Model):
    name = ndb.StringProperty()

class CartItem(ndb.Model):
    account = ndb.KeyProperty(kind=Account)
    inventory = ndb.KeyProperty(kind=InventoryItem)
    quantity = ndb.IntegerProperty()

class SpecialOffer(ndb.Model):
    inventory = ndb.KeyProperty(kind=InventoryItem)

def get_cart_plus_offers(acct):
    if False:
        for i in range(10):
            print('nop')
    cart = CartItem.query(CartItem.account == acct.key).fetch()
    offers = SpecialOffer.query().fetch(10)
    ndb.get_multi([item.inventory for item in cart] + [offer.inventory for offer in offers])
    return (cart, offers)

def get_cart_plus_offers_async(acct):
    if False:
        while True:
            i = 10
    cart_future = CartItem.query(CartItem.account == acct.key).fetch_async()
    offers_future = SpecialOffer.query().fetch_async(10)
    cart = cart_future.get_result()
    offers = offers_future.get_result()
    ndb.get_multi([item.inventory for item in cart] + [offer.inventory for offer in offers])
    return (cart, offers)

@ndb.tasklet
def get_cart_tasklet(acct):
    if False:
        return 10
    cart = (yield CartItem.query(CartItem.account == acct.key).fetch_async())
    yield ndb.get_multi_async([item.inventory for item in cart])
    raise ndb.Return(cart)

@ndb.tasklet
def get_offers_tasklet(acct):
    if False:
        return 10
    offers = (yield SpecialOffer.query().fetch_async(10))
    yield ndb.get_multi_async([offer.inventory for offer in offers])
    raise ndb.Return(offers)

@ndb.tasklet
def get_cart_plus_offers_tasklet(acct):
    if False:
        i = 10
        return i + 15
    (cart, offers) = (yield (get_cart_tasklet(acct), get_offers_tasklet(acct)))
    raise ndb.Return((cart, offers))

@ndb.tasklet
def iterate_over_query_results_in_tasklet(Model, is_the_entity_i_want):
    if False:
        i = 10
        return i + 15
    qry = Model.query()
    qit = qry.iter()
    while (yield qit.has_next_async()):
        entity = qit.next()
        if is_the_entity_i_want(entity):
            raise ndb.Return(entity)

@ndb.tasklet
def blocking_iteration_over_query_results(Model, is_the_entity_i_want):
    if False:
        for i in range(10):
            print('nop')
    qry = Model.query()
    for entity in qry:
        if is_the_entity_i_want(entity):
            raise ndb.Return(entity)

def define_get_google():
    if False:
        print('Hello World!')

    @ndb.tasklet
    def get_google():
        if False:
            print('Hello World!')
        context = ndb.get_context()
        result = (yield context.urlfetch('http://www.google.com/'))
        if result.status_code == 200:
            raise ndb.Return(result.content)
    return get_google

def define_update_counter_async():
    if False:
        print('Hello World!')

    @ndb.transactional_async
    def update_counter(counter_key):
        if False:
            for i in range(10):
                print('nop')
        counter = counter_key.get()
        counter.value += 1
        counter.put()
        return counter.value
    return update_counter

def define_update_counter_tasklet():
    if False:
        print('Hello World!')

    @ndb.transactional_tasklet
    def update_counter(counter_key):
        if False:
            while True:
                i = 10
        counter = (yield counter_key.get_async())
        counter.value += 1
        yield counter.put_async()
    return update_counter

def get_first_ready():
    if False:
        for i in range(10):
            print('nop')
    urls = ['http://www.google.com/', 'http://www.blogspot.com/']
    context = ndb.get_context()
    futures = [context.urlfetch(url) for url in urls]
    first_future = ndb.Future.wait_any(futures)
    return first_future.get_result().content