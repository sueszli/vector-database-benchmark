from datetime import datetime, timedelta, timezone
import pytest
from ccxt import DECIMAL_PLACES, ROUND, ROUND_DOWN, ROUND_UP, SIGNIFICANT_DIGITS, TICK_SIZE, TRUNCATE
from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException
from freqtrade.exchange import amount_to_contract_precision, amount_to_precision, date_minus_candles, price_to_precision, timeframe_to_minutes, timeframe_to_msecs, timeframe_to_next_date, timeframe_to_prev_date, timeframe_to_seconds
from freqtrade.exchange.check_exchange import check_exchange
from tests.conftest import log_has_re

def test_check_exchange(default_conf, caplog) -> None:
    if False:
        return 10
    default_conf['runmode'] = RunMode.DRY_RUN
    default_conf.get('exchange').update({'name': 'BITTREX'})
    assert check_exchange(default_conf)
    assert log_has_re('Exchange .* is officially supported by the Freqtrade development team\\.', caplog)
    caplog.clear()
    default_conf.get('exchange').update({'name': 'binance'})
    assert check_exchange(default_conf)
    assert log_has_re('Exchange \\"binance\\" is officially supported by the Freqtrade development team\\.', caplog)
    caplog.clear()
    default_conf.get('exchange').update({'name': 'binanceus'})
    assert check_exchange(default_conf)
    assert log_has_re('Exchange \\"binanceus\\" is officially supported by the Freqtrade development team\\.', caplog)
    caplog.clear()
    default_conf.get('exchange').update({'name': 'okex'})
    assert check_exchange(default_conf)
    assert log_has_re('Exchange \\"okex\\" is officially supported by the Freqtrade development team\\.', caplog)
    caplog.clear()
    default_conf.get('exchange').update({'name': 'huobipro'})
    assert check_exchange(default_conf)
    assert log_has_re('Exchange .* is known to the the ccxt library, available for the bot, but not officially supported by the Freqtrade development team\\. .*', caplog)
    caplog.clear()
    default_conf.get('exchange').update({'name': 'bitmex'})
    with pytest.raises(OperationalException, match='Exchange .* will not work with Freqtrade\\..*'):
        check_exchange(default_conf)
    caplog.clear()
    default_conf.get('exchange').update({'name': 'bitmex'})
    assert check_exchange(default_conf, False)
    assert log_has_re('Exchange .* is known to the the ccxt library, available for the bot, but not officially supported by the Freqtrade development team\\. .*', caplog)
    caplog.clear()
    default_conf.get('exchange').update({'name': 'unknown_exchange'})
    with pytest.raises(OperationalException, match='Exchange "unknown_exchange" is not known to the ccxt library and therefore not available for the bot.*'):
        check_exchange(default_conf)
    default_conf.get('exchange').update({'name': ''})
    default_conf['runmode'] = RunMode.PLOT
    assert check_exchange(default_conf)
    default_conf.get('exchange').update({'name': ''})
    default_conf['runmode'] = RunMode.UTIL_EXCHANGE
    with pytest.raises(OperationalException, match='This command requires a configured exchange.*'):
        check_exchange(default_conf)

def test_date_minus_candles():
    if False:
        while True:
            i = 10
    date = datetime(2019, 8, 12, 13, 25, 0, tzinfo=timezone.utc)
    assert date_minus_candles('5m', 3, date) == date - timedelta(minutes=15)
    assert date_minus_candles('5m', 5, date) == date - timedelta(minutes=25)
    assert date_minus_candles('1m', 6, date) == date - timedelta(minutes=6)
    assert date_minus_candles('1h', 3, date) == date - timedelta(hours=3, minutes=25)
    assert date_minus_candles('1h', 3) == timeframe_to_prev_date('1h') - timedelta(hours=3)

def test_timeframe_to_minutes():
    if False:
        i = 10
        return i + 15
    assert timeframe_to_minutes('5m') == 5
    assert timeframe_to_minutes('10m') == 10
    assert timeframe_to_minutes('1h') == 60
    assert timeframe_to_minutes('1d') == 1440

def test_timeframe_to_seconds():
    if False:
        i = 10
        return i + 15
    assert timeframe_to_seconds('5m') == 300
    assert timeframe_to_seconds('10m') == 600
    assert timeframe_to_seconds('1h') == 3600
    assert timeframe_to_seconds('1d') == 86400

def test_timeframe_to_msecs():
    if False:
        while True:
            i = 10
    assert timeframe_to_msecs('5m') == 300000
    assert timeframe_to_msecs('10m') == 600000
    assert timeframe_to_msecs('1h') == 3600000
    assert timeframe_to_msecs('1d') == 86400000

def test_timeframe_to_prev_date():
    if False:
        return 10
    date = datetime.fromtimestamp(1565616128, tz=timezone.utc)
    tf_list = [('5m', datetime(2019, 8, 12, 13, 20, 0, tzinfo=timezone.utc)), ('10m', datetime(2019, 8, 12, 13, 20, 0, tzinfo=timezone.utc)), ('1h', datetime(2019, 8, 12, 13, 0, 0, tzinfo=timezone.utc)), ('2h', datetime(2019, 8, 12, 12, 0, 0, tzinfo=timezone.utc)), ('4h', datetime(2019, 8, 12, 12, 0, 0, tzinfo=timezone.utc)), ('1d', datetime(2019, 8, 12, 0, 0, 0, tzinfo=timezone.utc))]
    for (interval, result) in tf_list:
        assert timeframe_to_prev_date(interval, date) == result
    date = datetime.now(tz=timezone.utc)
    assert timeframe_to_prev_date('5m') < date
    time = datetime(2019, 8, 12, 13, 20, 0, tzinfo=timezone.utc)
    assert timeframe_to_prev_date('5m', time) == time
    time = datetime(2019, 8, 12, 13, 0, 0, tzinfo=timezone.utc)
    assert timeframe_to_prev_date('1h', time) == time

def test_timeframe_to_next_date():
    if False:
        print('Hello World!')
    date = datetime.fromtimestamp(1565616128, tz=timezone.utc)
    tf_list = [('5m', datetime(2019, 8, 12, 13, 25, 0, tzinfo=timezone.utc)), ('10m', datetime(2019, 8, 12, 13, 30, 0, tzinfo=timezone.utc)), ('1h', datetime(2019, 8, 12, 14, 0, 0, tzinfo=timezone.utc)), ('2h', datetime(2019, 8, 12, 14, 0, 0, tzinfo=timezone.utc)), ('4h', datetime(2019, 8, 12, 16, 0, 0, tzinfo=timezone.utc)), ('1d', datetime(2019, 8, 13, 0, 0, 0, tzinfo=timezone.utc))]
    for (interval, result) in tf_list:
        assert timeframe_to_next_date(interval, date) == result
    date = datetime.now(tz=timezone.utc)
    assert timeframe_to_next_date('5m') > date
    date = datetime(2019, 8, 12, 13, 30, 0, tzinfo=timezone.utc)
    assert timeframe_to_next_date('5m', date) == date + timedelta(minutes=5)

@pytest.mark.parametrize('amount,precision_mode,precision,expected', [(2.34559, DECIMAL_PLACES, 4, 2.3455), (2.34559, DECIMAL_PLACES, 5, 2.34559), (2.34559, DECIMAL_PLACES, 3, 2.345), (2.9999, DECIMAL_PLACES, 3, 2.999), (2.9909, DECIMAL_PLACES, 3, 2.99), (2.9909, DECIMAL_PLACES, 0, 2), (29991.5555, DECIMAL_PLACES, 0, 29991), (29991.5555, DECIMAL_PLACES, -1, 29990), (29991.5555, DECIMAL_PLACES, -2, 29900), (2.34559, SIGNIFICANT_DIGITS, 4, 2.345), (2.34559, SIGNIFICANT_DIGITS, 5, 2.3455), (2.34559, SIGNIFICANT_DIGITS, 3, 2.34), (2.9999, SIGNIFICANT_DIGITS, 3, 2.99), (2.9909, SIGNIFICANT_DIGITS, 3, 2.99), (7.7723e-06, SIGNIFICANT_DIGITS, 5, 7.7723e-06), (7.7723e-06, SIGNIFICANT_DIGITS, 3, 7.77e-06), (7.7723e-06, SIGNIFICANT_DIGITS, 1, 7e-06), (2.34559, TICK_SIZE, 0.0001, 2.3455), (2.34559, TICK_SIZE, 1e-05, 2.34559), (2.34559, TICK_SIZE, 0.001, 2.345), (2.9999, TICK_SIZE, 0.001, 2.999), (2.9909, TICK_SIZE, 0.001, 2.99), (2.9909, TICK_SIZE, 0.005, 2.99), (2.9999, TICK_SIZE, 0.005, 2.995)])
def test_amount_to_precision(amount, precision_mode, precision, expected):
    if False:
        print('Hello World!')
    '\n    Test rounds down\n    '
    assert amount_to_precision(amount, precision, precision_mode) == expected

@pytest.mark.parametrize('price,precision_mode,precision,expected,rounding_mode', [(2.34559, DECIMAL_PLACES, 4, 2.3456, ROUND_UP), (2.34559, DECIMAL_PLACES, 5, 2.34559, ROUND_UP), (2.34559, DECIMAL_PLACES, 3, 2.346, ROUND_UP), (2.9999, DECIMAL_PLACES, 3, 3.0, ROUND_UP), (2.9909, DECIMAL_PLACES, 3, 2.991, ROUND_UP), (2.9901, DECIMAL_PLACES, 3, 2.991, ROUND_UP), (2.34559, DECIMAL_PLACES, 5, 2.34559, ROUND_DOWN), (2.34559, DECIMAL_PLACES, 4, 2.3455, ROUND_DOWN), (2.9901, DECIMAL_PLACES, 3, 2.99, ROUND_DOWN), (0.00299, DECIMAL_PLACES, 3, 0.002, ROUND_DOWN), (2.345600000000001, DECIMAL_PLACES, 4, 2.3456, ROUND), (2.345551, DECIMAL_PLACES, 4, 2.3456, ROUND), (2.49, DECIMAL_PLACES, 0, 2.0, ROUND), (2.51, DECIMAL_PLACES, 0, 3.0, ROUND), (5.1, DECIMAL_PLACES, -1, 10.0, ROUND), (4.9, DECIMAL_PLACES, -1, 0.0, ROUND), (7.222e-06, SIGNIFICANT_DIGITS, 1, 7e-06, ROUND), (7.222e-06, SIGNIFICANT_DIGITS, 2, 7.2e-06, ROUND), (7.777e-06, SIGNIFICANT_DIGITS, 2, 7.8e-06, ROUND), (2.34559, TICK_SIZE, 0.0001, 2.3456, ROUND_UP), (2.34559, TICK_SIZE, 1e-05, 2.34559, ROUND_UP), (2.34559, TICK_SIZE, 0.001, 2.346, ROUND_UP), (2.9999, TICK_SIZE, 0.001, 3.0, ROUND_UP), (2.9909, TICK_SIZE, 0.001, 2.991, ROUND_UP), (2.9909, TICK_SIZE, 0.001, 2.99, ROUND_DOWN), (2.9909, TICK_SIZE, 0.005, 2.995, ROUND_UP), (2.9973, TICK_SIZE, 0.005, 3.0, ROUND_UP), (2.9977, TICK_SIZE, 0.005, 3.0, ROUND_UP), (234.43, TICK_SIZE, 0.5, 234.5, ROUND_UP), (234.43, TICK_SIZE, 0.5, 234.0, ROUND_DOWN), (234.53, TICK_SIZE, 0.5, 235.0, ROUND_UP), (234.53, TICK_SIZE, 0.5, 234.5, ROUND_DOWN), (0.891534, TICK_SIZE, 0.0001, 0.8916, ROUND_UP), (64968.89, TICK_SIZE, 0.01, 64968.89, ROUND_UP), (3.483e-09, TICK_SIZE, 1e-12, 3.483e-09, ROUND_UP), (2.49, TICK_SIZE, 1.0, 2.0, ROUND), (2.51, TICK_SIZE, 1.0, 3.0, ROUND), (2.000000051, TICK_SIZE, 1e-07, 2.0000001, ROUND), (2.000000049, TICK_SIZE, 1e-07, 2.0, ROUND), (2.9909, TICK_SIZE, 0.005, 2.99, ROUND), (2.9973, TICK_SIZE, 0.005, 2.995, ROUND), (2.9977, TICK_SIZE, 0.005, 3.0, ROUND), (234.24, TICK_SIZE, 0.5, 234.0, ROUND), (234.26, TICK_SIZE, 0.5, 234.5, ROUND), (2.34559, DECIMAL_PLACES, 4, 2.3455, TRUNCATE), (2.34559, DECIMAL_PLACES, 5, 2.34559, TRUNCATE), (2.34559, DECIMAL_PLACES, 3, 2.345, TRUNCATE), (2.9999, DECIMAL_PLACES, 3, 2.999, TRUNCATE), (2.9909, DECIMAL_PLACES, 3, 2.99, TRUNCATE), (2.9909, TICK_SIZE, 0.001, 2.99, TRUNCATE), (2.9909, TICK_SIZE, 0.01, 2.99, TRUNCATE), (2.9909, TICK_SIZE, 0.1, 2.9, TRUNCATE), (2.34559, SIGNIFICANT_DIGITS, 4, 2.345, TRUNCATE), (2.34559, SIGNIFICANT_DIGITS, 5, 2.3455, TRUNCATE), (2.34559, SIGNIFICANT_DIGITS, 3, 2.34, TRUNCATE), (2.9999, SIGNIFICANT_DIGITS, 3, 2.99, TRUNCATE), (2.9909, SIGNIFICANT_DIGITS, 2, 2.9, TRUNCATE), (7.77e-06, SIGNIFICANT_DIGITS, 2, 7.7e-06, TRUNCATE), (7.29e-06, SIGNIFICANT_DIGITS, 2, 7.2e-06, TRUNCATE), (722.2, SIGNIFICANT_DIGITS, 1, 700.0, ROUND), (790.2, SIGNIFICANT_DIGITS, 1, 800.0, ROUND), (722.2, SIGNIFICANT_DIGITS, 2, 720.0, ROUND), (722.2, SIGNIFICANT_DIGITS, 1, 800.0, ROUND_UP), (722.2, SIGNIFICANT_DIGITS, 2, 730.0, ROUND_UP), (777.7, SIGNIFICANT_DIGITS, 2, 780.0, ROUND_UP), (777.7, SIGNIFICANT_DIGITS, 3, 778.0, ROUND_UP), (722.2, SIGNIFICANT_DIGITS, 1, 700.0, ROUND_DOWN), (722.2, SIGNIFICANT_DIGITS, 2, 720.0, ROUND_DOWN), (777.7, SIGNIFICANT_DIGITS, 2, 770.0, ROUND_DOWN), (777.7, SIGNIFICANT_DIGITS, 3, 777.0, ROUND_DOWN), (7.222e-06, SIGNIFICANT_DIGITS, 1, 8e-06, ROUND_UP), (7.222e-06, SIGNIFICANT_DIGITS, 2, 7.3e-06, ROUND_UP), (7.777e-06, SIGNIFICANT_DIGITS, 2, 7.8e-06, ROUND_UP), (7.222e-06, SIGNIFICANT_DIGITS, 1, 7e-06, ROUND_DOWN), (7.222e-06, SIGNIFICANT_DIGITS, 2, 7.2e-06, ROUND_DOWN), (7.777e-06, SIGNIFICANT_DIGITS, 2, 7.7e-06, ROUND_DOWN)])
def test_price_to_precision(price, precision_mode, precision, expected, rounding_mode):
    if False:
        while True:
            i = 10
    assert price_to_precision(price, precision, precision_mode, rounding_mode=rounding_mode) == expected

@pytest.mark.parametrize('amount,precision,precision_mode,contract_size,expected', [(1.17, 1.0, 4, 0.01, 1.17), (1.17, 1.0, 2, 0.01, 1.17), (1.16, 1.0, 4, 0.01, 1.16), (1.16, 1.0, 2, 0.01, 1.16), (1.13, 1.0, 2, 0.01, 1.13), (10.988, 1.0, 2, 10, 10), (10.988, 1.0, 4, 10, 10)])
def test_amount_to_contract_precision_standalone(amount, precision, precision_mode, contract_size, expected):
    if False:
        print('Hello World!')
    res = amount_to_contract_precision(amount, precision, precision_mode, contract_size)
    assert pytest.approx(res) == expected