import gevent
from fast_wait import fast_wait
assert fast_wait

def nonterm_greenlet():
    if False:
        i = 10
        return i + 15
    while True:
        gevent.sleep(300)

def test_fast_wait():
    if False:
        print('Hello World!')
    'Annoy someone who causes fast-sleep test patching to regress.\n\n    Someone could break the test-only monkey-patching of gevent.sleep\n    without noticing and costing quite a bit of aggravation aggregated\n    over time waiting in tests, added bit by bit.\n\n    To avoid that, add this incredibly huge/annoying delay that can\n    only be avoided by monkey-patch to catch the regression.\n    '
    gevent.sleep(300)
    g = gevent.spawn(nonterm_greenlet)
    gevent.joinall([g], timeout=300)
    gevent.killall([g], timeout=300)