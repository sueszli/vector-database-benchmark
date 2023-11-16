import numpy as np
from jesse.config import config, reset_config
from jesse.factories import fake_candle, range_candles
from jesse.services.candle import generate_candle_from_one_minutes
from jesse.store import store
import jesse.helpers as jh

def set_up():
    if False:
        for i in range(10):
            print('nop')
    reset_config()
    from jesse.routes import router
    router.set_routes([{'exchange': 'Sandbox', 'symbol': 'BTC-USD', 'timeframe': '1m', 'strategy': 'Test01'}])
    router.set_extra_candles([{'exchange': 'Sandbox', 'symbol': 'BTC-USD', 'timeframe': '5m'}])
    config['app']['considering_timeframes'] = ['1m', '5m']
    config['app']['considering_symbols'] = ['BTC-USD']
    config['app']['considering_exchanges'] = ['Sandbox']
    store.reset(True)
    store.candles.init_storage()

def test_batch_add_candles():
    if False:
        i = 10
        return i + 15
    set_up()
    assert len(store.candles.get_candles('Sandbox', 'BTC-USD', '1m')) == 0
    candles_to_add = range_candles(100)
    assert len(candles_to_add) == 100
    store.candles.batch_add_candle(candles_to_add, 'Sandbox', 'BTC-USD', '1m')
    np.testing.assert_equal(store.candles.get_candles('Sandbox', 'BTC-USD', '1m'), candles_to_add)

def test_can_add_new_candle():
    if False:
        i = 10
        return i + 15
    set_up()
    np.testing.assert_equal(store.candles.get_candles('Sandbox', 'BTC-USD', '1m'), np.zeros((0, 6)))
    c1 = fake_candle()
    store.candles.add_candle(c1, 'Sandbox', 'BTC-USD', '1m')
    np.testing.assert_equal(store.candles.get_candles('Sandbox', 'BTC-USD', '1m')[0], c1)
    store.candles.add_candle(c1, 'Sandbox', 'BTC-USD', '1m')
    np.testing.assert_equal(store.candles.get_candles('Sandbox', 'BTC-USD', '1m')[0], c1)
    c2 = fake_candle()
    store.candles.add_candle(c2, 'Sandbox', 'BTC-USD', '1m')
    np.testing.assert_equal(store.candles.get_candles('Sandbox', 'BTC-USD', '1m'), np.array([c1, c2]))

def test_get_candles_including_forming():
    if False:
        while True:
            i = 10
    set_up()
    candles_to_add = range_candles(14)
    store.candles.batch_add_candle(candles_to_add, 'Sandbox', 'BTC-USD', '1m')
    store.candles.add_candle(generate_candle_from_one_minutes('5m', candles_to_add[0:5], False), 'Sandbox', 'BTC-USD', '5m')
    store.candles.add_candle(generate_candle_from_one_minutes('5m', candles_to_add[5:10], False), 'Sandbox', 'BTC-USD', '5m')
    assert len(store.candles.get_candles('Sandbox', 'BTC-USD', '5m')) == 3
    assert len(store.candles.get_candles('Sandbox', 'BTC-USD', '1m')) == 14
    candles = store.candles.get_candles('Sandbox', 'BTC-USD', '5m')
    assert candles[0][0] == candles_to_add[0][0]
    assert candles[-1][2] == candles_to_add[13][2]
    assert candles[-1][0] == candles_to_add[10][0]
    store.candles.add_candle(generate_candle_from_one_minutes('5m', candles_to_add[10:14], True), 'Sandbox', 'BTC-USD', '5m')
    assert len(store.candles.get_candles('Sandbox', 'BTC-USD', '5m')) == 3
    assert candles[-1][2] == candles_to_add[13][2]
    assert candles[-1][0] == candles_to_add[10][0]

def test_get_forming_candle():
    if False:
        print('Hello World!')
    set_up()
    candles_to_add = range_candles(13)
    store.candles.batch_add_candle(candles_to_add[0:4], 'Sandbox', 'BTC-USD', '1m')
    forming_candle = store.candles.get_current_candle('Sandbox', 'BTC-USD', '5m')
    assert forming_candle[0] == candles_to_add[0][0]
    assert forming_candle[1] == candles_to_add[0][1]
    assert forming_candle[2] == candles_to_add[3][2]
    store.candles.batch_add_candle(candles_to_add[4:], 'Sandbox', 'BTC-USD', '1m')
    store.candles.batch_add_candle(candles_to_add[0:5], 'Sandbox', 'BTC-USD', '5m')
    store.candles.batch_add_candle(candles_to_add[5:10], 'Sandbox', 'BTC-USD', '5m')
    forming_candle = store.candles.get_current_candle('Sandbox', 'BTC-USD', '5m')
    assert forming_candle[0] == candles_to_add[10][0]
    assert forming_candle[1] == candles_to_add[10][1]
    assert forming_candle[2] == candles_to_add[12][2]

def test_can_update_candle():
    if False:
        while True:
            i = 10
    set_up()
    np.testing.assert_equal(store.candles.get_candles('Sandbox', 'BTC-USD', '1m'), np.zeros((0, 6)))
    c1 = fake_candle()
    store.candles.add_candle(c1, 'Sandbox', 'BTC-USD', '1m')
    np.testing.assert_equal(store.candles.get_current_candle('Sandbox', 'BTC-USD', '1m'), c1)
    c2 = c1.copy()
    c2[1] = 1000
    store.candles.add_candle(c2, 'Sandbox', 'BTC-USD', '1m')
    np.testing.assert_equal(store.candles.get_current_candle('Sandbox', 'BTC-USD', '1m'), c2)
    assert len(store.candles.get_candles('Sandbox', 'BTC-USD', '1m')) == 1

def test_can_update_previous_candle():
    if False:
        for i in range(10):
            print('nop')
    set_up()
    c1 = fake_candle()
    store.candles.add_candle(c1, 'Sandbox', 'BTC-USD', '1m')
    c2 = fake_candle()
    store.candles.add_candle(c2, 'Sandbox', 'BTC-USD', '1m')
    c3 = fake_candle()
    store.candles.add_candle(c3, 'Sandbox', 'BTC-USD', '1m')
    new_c2 = c2.copy()
    new_c2[2] = 50
    assert store.candles.get_candles('Sandbox', 'BTC-USD', '1m')[-2][2] != c3[2]
    store.candles.add_candle(new_c2, 'Sandbox', 'BTC-USD', '1m')
    assert store.candles.get_candles('Sandbox', 'BTC-USD', '1m')[-2][2] == new_c2[2]