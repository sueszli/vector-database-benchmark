import time
from source_freshdesk.utils import CallCredit

def test_consume_one():
    if False:
        return 10
    'Multiple consumptions of 1 cred will reach limit'
    credit = CallCredit(balance=3, reload_period=1)
    ts_1 = time.time()
    for i in range(4):
        credit.consume(1)
    ts_2 = time.time()
    assert 1 <= ts_2 - ts_1 < 2

def test_consume_many():
    if False:
        return 10
    'Consumptions of N creds will reach limit and decrease balance'
    credit = CallCredit(balance=3, reload_period=1)
    ts_1 = time.time()
    credit.consume(1)
    credit.consume(3)
    ts_2 = time.time()
    credit.consume(1)
    ts_3 = time.time()
    assert 1 <= ts_2 - ts_1 < 2
    assert 1 <= ts_3 - ts_2 < 2