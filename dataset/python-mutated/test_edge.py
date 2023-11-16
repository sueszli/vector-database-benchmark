import logging
import math
from datetime import timedelta
from unittest.mock import MagicMock
import numpy as np
import pytest
from pandas import DataFrame
from freqtrade.data.converter import ohlcv_to_dataframe
from freqtrade.edge import Edge, PairInfo
from freqtrade.enums import ExitType
from freqtrade.exceptions import OperationalException
from freqtrade.util.datetime_helpers import dt_ts, dt_utc
from tests.conftest import EXMS, get_patched_freqtradebot, log_has
from tests.optimize import BTContainer, BTrade, _build_backtest_dataframe, _get_frame_time_from_offset
tests_start_time = dt_utc(2018, 10, 3)
timeframe_in_minute = 60
tc0 = BTContainer(data=[[0, 5000, 5025, 4975, 4987, 6172, 1, 0], [1, 5000, 5025, 4975, 4987, 6172, 0, 1]], stop_loss=-0.99, roi={'0': float('inf')}, profit_perc=0.0, trades=[])
tc1 = BTContainer(data=[[0, 5000, 5025, 4975, 4987, 6172, 1, 0], [1, 5000, 5025, 4975, 4987, 6172, 0, 1], [2, 5000, 5025, 4975, 4987, 6172, 0, 0], [3, 5000, 5025, 4975, 4987, 6172, 1, 0], [4, 5000, 5025, 4975, 4987, 6172, 0, 0], [5, 5000, 5025, 4975, 4987, 6172, 0, 1], [6, 5000, 5025, 4975, 4987, 6172, 0, 0]], stop_loss=-0.99, roi={'0': float('inf')}, profit_perc=0.0, trades=[BTrade(exit_reason=ExitType.EXIT_SIGNAL, open_tick=1, close_tick=2), BTrade(exit_reason=ExitType.EXIT_SIGNAL, open_tick=4, close_tick=6)])
tc2 = BTContainer(data=[[0, 5000, 5025, 4975, 4987, 6172, 1, 0], [1, 5000, 5025, 4600, 4987, 6172, 0, 0], [2, 5000, 5025, 4975, 4987, 6172, 0, 0]], stop_loss=-0.01, roi={'0': float('inf')}, profit_perc=-0.01, trades=[BTrade(exit_reason=ExitType.STOP_LOSS, open_tick=1, close_tick=1)])
tc3 = BTContainer(data=[[0, 5000, 5025, 4975, 4987, 6172, 1, 0], [1, 5000, 5025, 4800, 4987, 6172, 0, 0], [2, 5000, 5025, 4975, 4987, 6172, 0, 0]], stop_loss=-0.03, roi={'0': float('inf')}, profit_perc=-0.03, trades=[BTrade(exit_reason=ExitType.STOP_LOSS, open_tick=1, close_tick=1)])
tc4 = BTContainer(data=[[0, 5000, 5025, 4975, 4987, 6172, 1, 0], [1, 5000, 5025, 4800, 4987, 6172, 0, 1], [2, 5000, 5025, 4975, 4987, 6172, 0, 0]], stop_loss=-0.03, roi={'0': float('inf')}, profit_perc=-0.03, trades=[BTrade(exit_reason=ExitType.STOP_LOSS, open_tick=1, close_tick=1)])
TESTS = [tc0, tc1, tc2, tc3, tc4]

@pytest.mark.parametrize('data', TESTS)
def test_edge_results(edge_conf, mocker, caplog, data) -> None:
    if False:
        print('Hello World!')
    '\n    run functional tests\n    '
    freqtrade = get_patched_freqtradebot(mocker, edge_conf)
    edge = Edge(edge_conf, freqtrade.exchange, freqtrade.strategy)
    frame = _build_backtest_dataframe(data.data)
    caplog.set_level(logging.DEBUG)
    edge.fee = 0
    trades = edge._find_trades_for_stoploss_range(frame, 'TEST/BTC', [data.stop_loss])
    results = edge._fill_calculable_fields(DataFrame(trades)) if trades else DataFrame()
    assert len(trades) == len(data.trades)
    if not results.empty:
        assert round(results['profit_ratio'].sum(), 3) == round(data.profit_perc, 3)
    for (c, trade) in enumerate(data.trades):
        res = results.iloc[c]
        assert res.exit_type == trade.exit_reason
        assert res.open_date == _get_frame_time_from_offset(trade.open_tick).replace(tzinfo=None)
        assert res.close_date == _get_frame_time_from_offset(trade.close_tick).replace(tzinfo=None)

def test_adjust(mocker, edge_conf):
    if False:
        return 10
    freqtrade = get_patched_freqtradebot(mocker, edge_conf)
    edge = Edge(edge_conf, freqtrade.exchange, freqtrade.strategy)
    mocker.patch('freqtrade.edge.Edge._cached_pairs', mocker.PropertyMock(return_value={'E/F': PairInfo(-0.01, 0.66, 3.71, 0.5, 1.71, 10, 60), 'C/D': PairInfo(-0.01, 0.66, 3.71, 0.5, 1.71, 10, 60), 'N/O': PairInfo(-0.01, 0.66, 3.71, 0.5, 1.71, 10, 60)}))
    pairs = ['A/B', 'C/D', 'E/F', 'G/H']
    assert edge.adjust(pairs) == ['E/F', 'C/D']

def test_edge_get_stoploss(mocker, edge_conf):
    if False:
        for i in range(10):
            print('nop')
    freqtrade = get_patched_freqtradebot(mocker, edge_conf)
    edge = Edge(edge_conf, freqtrade.exchange, freqtrade.strategy)
    mocker.patch('freqtrade.edge.Edge._cached_pairs', mocker.PropertyMock(return_value={'E/F': PairInfo(-0.01, 0.66, 3.71, 0.5, 1.71, 10, 60), 'C/D': PairInfo(-0.01, 0.66, 3.71, 0.5, 1.71, 10, 60), 'N/O': PairInfo(-0.01, 0.66, 3.71, 0.5, 1.71, 10, 60)}))
    assert edge.get_stoploss('E/F') == -0.01

def test_nonexisting_get_stoploss(mocker, edge_conf):
    if False:
        for i in range(10):
            print('nop')
    freqtrade = get_patched_freqtradebot(mocker, edge_conf)
    edge = Edge(edge_conf, freqtrade.exchange, freqtrade.strategy)
    mocker.patch('freqtrade.edge.Edge._cached_pairs', mocker.PropertyMock(return_value={'E/F': PairInfo(-0.01, 0.66, 3.71, 0.5, 1.71, 10, 60)}))
    assert edge.get_stoploss('N/O') == -0.1

def test_edge_stake_amount(mocker, edge_conf):
    if False:
        print('Hello World!')
    freqtrade = get_patched_freqtradebot(mocker, edge_conf)
    edge = Edge(edge_conf, freqtrade.exchange, freqtrade.strategy)
    mocker.patch('freqtrade.edge.Edge._cached_pairs', mocker.PropertyMock(return_value={'E/F': PairInfo(-0.02, 0.66, 3.71, 0.5, 1.71, 10, 60)}))
    assert edge._capital_ratio == 0.5
    assert edge.stake_amount('E/F', free_capital=100, total_capital=100, capital_in_trade=25) == 31.25
    assert edge.stake_amount('E/F', free_capital=20, total_capital=100, capital_in_trade=25) == 20
    assert edge.stake_amount('E/F', free_capital=0, total_capital=100, capital_in_trade=25) == 0
    edge._allowed_risk = 0.4
    edge._capital_ratio = 0.5
    assert edge.stake_amount('E/F', free_capital=100, total_capital=100, capital_in_trade=25) == 62.5
    assert edge.stake_amount('E/F', free_capital=100, total_capital=100, capital_in_trade=0) == 50
    edge._capital_ratio = 1
    assert edge.stake_amount('E/F', free_capital=100, total_capital=100, capital_in_trade=0) == 100
    assert edge.stake_amount('E/F', free_capital=0, total_capital=100, capital_in_trade=0) == 0

def test_nonexisting_stake_amount(mocker, edge_conf):
    if False:
        return 10
    freqtrade = get_patched_freqtradebot(mocker, edge_conf)
    edge = Edge(edge_conf, freqtrade.exchange, freqtrade.strategy)
    mocker.patch('freqtrade.edge.Edge._cached_pairs', mocker.PropertyMock(return_value={'E/F': PairInfo(-0.11, 0.66, 3.71, 0.5, 1.71, 10, 60)}))
    assert edge.stake_amount('N/O', 1, 2, 1) == 0.15

def test_edge_heartbeat_calculate(mocker, edge_conf):
    if False:
        for i in range(10):
            print('nop')
    freqtrade = get_patched_freqtradebot(mocker, edge_conf)
    edge = Edge(edge_conf, freqtrade.exchange, freqtrade.strategy)
    heartbeat = edge_conf['edge']['process_throttle_secs']
    edge._last_updated = dt_ts() - heartbeat + 1
    assert edge.calculate(edge_conf['exchange']['pair_whitelist']) is False

def mocked_load_data(datadir, pairs=[], timeframe='0m', timerange=None, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    hz = 0.1
    base = 0.001
    NEOBTC = [[dt_ts(tests_start_time + timedelta(minutes=x * timeframe_in_minute)), math.sin(x * hz) / 1000 + base, math.sin(x * hz) / 1000 + base + 0.0001, math.sin(x * hz) / 1000 + base - 0.0001, math.sin(x * hz) / 1000 + base, 123.45] for x in range(0, 500)]
    hz = 0.2
    base = 0.002
    LTCBTC = [[dt_ts(tests_start_time + timedelta(minutes=x * timeframe_in_minute)), math.sin(x * hz) / 1000 + base, math.sin(x * hz) / 1000 + base + 0.0001, math.sin(x * hz) / 1000 + base - 0.0001, math.sin(x * hz) / 1000 + base, 123.45] for x in range(0, 500)]
    pairdata = {'NEO/BTC': ohlcv_to_dataframe(NEOBTC, '1h', pair='NEO/BTC', fill_missing=True), 'LTC/BTC': ohlcv_to_dataframe(LTCBTC, '1h', pair='LTC/BTC', fill_missing=True)}
    return pairdata

def test_edge_process_downloaded_data(mocker, edge_conf):
    if False:
        return 10
    freqtrade = get_patched_freqtradebot(mocker, edge_conf)
    mocker.patch(f'{EXMS}.get_fee', MagicMock(return_value=0.001))
    mocker.patch('freqtrade.edge.edge_positioning.refresh_data', MagicMock())
    mocker.patch('freqtrade.edge.edge_positioning.load_data', mocked_load_data)
    edge = Edge(edge_conf, freqtrade.exchange, freqtrade.strategy)
    assert edge.calculate(edge_conf['exchange']['pair_whitelist'])
    assert len(edge._cached_pairs) == 2
    assert edge._last_updated <= dt_ts() + 2

def test_edge_process_no_data(mocker, edge_conf, caplog):
    if False:
        while True:
            i = 10
    freqtrade = get_patched_freqtradebot(mocker, edge_conf)
    mocker.patch(f'{EXMS}.get_fee', MagicMock(return_value=0.001))
    mocker.patch('freqtrade.edge.edge_positioning.refresh_data', MagicMock())
    mocker.patch('freqtrade.edge.edge_positioning.load_data', MagicMock(return_value={}))
    edge = Edge(edge_conf, freqtrade.exchange, freqtrade.strategy)
    assert not edge.calculate(edge_conf['exchange']['pair_whitelist'])
    assert len(edge._cached_pairs) == 0
    assert log_has('No data found. Edge is stopped ...', caplog)
    assert edge._last_updated == 0

def test_edge_process_no_trades(mocker, edge_conf, caplog):
    if False:
        return 10
    freqtrade = get_patched_freqtradebot(mocker, edge_conf)
    mocker.patch(f'{EXMS}.get_fee', return_value=0.001)
    mocker.patch('freqtrade.edge.edge_positioning.refresh_data')
    mocker.patch('freqtrade.edge.edge_positioning.load_data', mocked_load_data)
    mocker.patch('freqtrade.edge.Edge._find_trades_for_stoploss_range', return_value=[])
    edge = Edge(edge_conf, freqtrade.exchange, freqtrade.strategy)
    assert not edge.calculate(edge_conf['exchange']['pair_whitelist'])
    assert len(edge._cached_pairs) == 0
    assert log_has('No trades found.', caplog)

def test_edge_process_no_pairs(mocker, edge_conf, caplog):
    if False:
        while True:
            i = 10
    edge_conf['exchange']['pair_whitelist'] = []
    mocker.patch('freqtrade.freqtradebot.validate_config_consistency')
    freqtrade = get_patched_freqtradebot(mocker, edge_conf)
    fee_mock = mocker.patch(f'{EXMS}.get_fee', return_value=0.001)
    mocker.patch('freqtrade.edge.edge_positioning.refresh_data')
    mocker.patch('freqtrade.edge.edge_positioning.load_data', mocked_load_data)
    mocker.patch('freqtrade.edge.Edge._find_trades_for_stoploss_range', return_value=[])
    edge = Edge(edge_conf, freqtrade.exchange, freqtrade.strategy)
    assert fee_mock.call_count == 0
    assert edge.fee is None
    assert not edge.calculate(['XRP/USDT'])
    assert fee_mock.call_count == 1
    assert edge.fee == 0.001

def test_edge_init_error(mocker, edge_conf):
    if False:
        for i in range(10):
            print('nop')
    edge_conf['stake_amount'] = 0.5
    mocker.patch(f'{EXMS}.get_fee', MagicMock(return_value=0.001))
    with pytest.raises(OperationalException, match='Edge works only with unlimited stake amount'):
        get_patched_freqtradebot(mocker, edge_conf)

@pytest.mark.parametrize('fee,risk_reward_ratio,expectancy', [(0.0005, 306.5384615384, 101.5128205128), (0.001, 152.6923076923, 50.2307692308)])
def test_process_expectancy(mocker, edge_conf, fee, risk_reward_ratio, expectancy):
    if False:
        print('Hello World!')
    edge_conf['edge']['min_trade_number'] = 2
    freqtrade = get_patched_freqtradebot(mocker, edge_conf)

    def get_fee(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return fee
    freqtrade.exchange.get_fee = get_fee
    edge = Edge(edge_conf, freqtrade.exchange, freqtrade.strategy)
    trades = [{'pair': 'TEST/BTC', 'stoploss': -0.9, 'profit_percent': '', 'profit_abs': '', 'open_date': np.datetime64('2018-10-03T00:05:00.000000000'), 'close_date': np.datetime64('2018-10-03T00:10:00.000000000'), 'trade_duration': '', 'open_rate': 17, 'close_rate': 17, 'exit_type': 'exit_signal'}, {'pair': 'TEST/BTC', 'stoploss': -0.9, 'profit_percent': '', 'profit_abs': '', 'open_date': np.datetime64('2018-10-03T00:20:00.000000000'), 'close_date': np.datetime64('2018-10-03T00:25:00.000000000'), 'trade_duration': '', 'open_rate': 20, 'close_rate': 20, 'exit_type': 'exit_signal'}, {'pair': 'TEST/BTC', 'stoploss': -0.9, 'profit_percent': '', 'profit_abs': '', 'open_date': np.datetime64('2018-10-03T00:30:00.000000000'), 'close_date': np.datetime64('2018-10-03T00:40:00.000000000'), 'trade_duration': '', 'open_rate': 26, 'close_rate': 34, 'exit_type': 'exit_signal'}]
    trades_df = DataFrame(trades)
    trades_df = edge._fill_calculable_fields(trades_df)
    final = edge._process_expectancy(trades_df)
    assert len(final) == 1
    assert 'TEST/BTC' in final
    assert final['TEST/BTC'].stoploss == -0.9
    assert round(final['TEST/BTC'].winrate, 10) == 0.3333333333
    assert round(final['TEST/BTC'].risk_reward_ratio, 10) == risk_reward_ratio
    assert round(final['TEST/BTC'].required_risk_reward, 10) == 2.0
    assert round(final['TEST/BTC'].expectancy, 10) == expectancy
    trades.pop()
    trades_df = DataFrame(trades)
    trades_df = edge._fill_calculable_fields(trades_df)
    final = edge._process_expectancy(trades_df)
    assert len(final) == 0
    assert isinstance(final, dict)

def test_process_expectancy_remove_pumps(mocker, edge_conf, fee):
    if False:
        return 10
    edge_conf['edge']['min_trade_number'] = 2
    edge_conf['edge']['remove_pumps'] = True
    freqtrade = get_patched_freqtradebot(mocker, edge_conf)
    freqtrade.exchange.get_fee = fee
    edge = Edge(edge_conf, freqtrade.exchange, freqtrade.strategy)
    trades = [{'pair': 'TEST/BTC', 'stoploss': -0.9, 'profit_percent': '', 'profit_abs': '', 'open_date': np.datetime64('2018-10-03T00:05:00.000000000'), 'close_date': np.datetime64('2018-10-03T00:10:00.000000000'), 'open_index': 1, 'close_index': 1, 'trade_duration': '', 'open_rate': 17, 'close_rate': 15, 'exit_type': 'sell_signal'}, {'pair': 'TEST/BTC', 'stoploss': -0.9, 'profit_percent': '', 'profit_abs': '', 'open_date': np.datetime64('2018-10-03T00:20:00.000000000'), 'close_date': np.datetime64('2018-10-03T00:25:00.000000000'), 'open_index': 4, 'close_index': 4, 'trade_duration': '', 'open_rate': 20, 'close_rate': 10, 'exit_type': 'sell_signal'}, {'pair': 'TEST/BTC', 'stoploss': -0.9, 'profit_percent': '', 'profit_abs': '', 'open_date': np.datetime64('2018-10-03T00:20:00.000000000'), 'close_date': np.datetime64('2018-10-03T00:25:00.000000000'), 'open_index': 4, 'close_index': 4, 'trade_duration': '', 'open_rate': 20, 'close_rate': 10, 'exit_type': 'sell_signal'}, {'pair': 'TEST/BTC', 'stoploss': -0.9, 'profit_percent': '', 'profit_abs': '', 'open_date': np.datetime64('2018-10-03T00:20:00.000000000'), 'close_date': np.datetime64('2018-10-03T00:25:00.000000000'), 'open_index': 4, 'close_index': 4, 'trade_duration': '', 'open_rate': 20, 'close_rate': 10, 'exit_type': 'sell_signal'}, {'pair': 'TEST/BTC', 'stoploss': -0.9, 'profit_percent': '', 'profit_abs': '', 'open_date': np.datetime64('2018-10-03T00:20:00.000000000'), 'close_date': np.datetime64('2018-10-03T00:25:00.000000000'), 'open_index': 4, 'close_index': 4, 'trade_duration': '', 'open_rate': 20, 'close_rate': 10, 'exit_type': 'sell_signal'}, {'pair': 'TEST/BTC', 'stoploss': -0.9, 'profit_percent': '', 'profit_abs': '', 'open_date': np.datetime64('2018-10-03T00:30:00.000000000'), 'close_date': np.datetime64('2018-10-03T00:40:00.000000000'), 'open_index': 6, 'close_index': 7, 'trade_duration': '', 'open_rate': 26, 'close_rate': 134, 'exit_type': 'sell_signal'}]
    trades_df = DataFrame(trades)
    trades_df = edge._fill_calculable_fields(trades_df)
    final = edge._process_expectancy(trades_df)
    assert 'TEST/BTC' in final
    assert final['TEST/BTC'].stoploss == -0.9
    assert final['TEST/BTC'].nb_trades == len(trades_df) - 1
    assert round(final['TEST/BTC'].winrate, 10) == 0.0

def test_process_expectancy_only_wins(mocker, edge_conf, fee):
    if False:
        i = 10
        return i + 15
    edge_conf['edge']['min_trade_number'] = 2
    freqtrade = get_patched_freqtradebot(mocker, edge_conf)
    freqtrade.exchange.get_fee = fee
    edge = Edge(edge_conf, freqtrade.exchange, freqtrade.strategy)
    trades = [{'pair': 'TEST/BTC', 'stoploss': -0.9, 'profit_percent': '', 'profit_abs': '', 'open_date': np.datetime64('2018-10-03T00:05:00.000000000'), 'close_date': np.datetime64('2018-10-03T00:10:00.000000000'), 'open_index': 1, 'close_index': 1, 'trade_duration': '', 'open_rate': 15, 'close_rate': 17, 'exit_type': 'sell_signal'}, {'pair': 'TEST/BTC', 'stoploss': -0.9, 'profit_percent': '', 'profit_abs': '', 'open_date': np.datetime64('2018-10-03T00:20:00.000000000'), 'close_date': np.datetime64('2018-10-03T00:25:00.000000000'), 'open_index': 4, 'close_index': 4, 'trade_duration': '', 'open_rate': 10, 'close_rate': 20, 'exit_type': 'sell_signal'}, {'pair': 'TEST/BTC', 'stoploss': -0.9, 'profit_percent': '', 'profit_abs': '', 'open_date': np.datetime64('2018-10-03T00:30:00.000000000'), 'close_date': np.datetime64('2018-10-03T00:40:00.000000000'), 'open_index': 6, 'close_index': 7, 'trade_duration': '', 'open_rate': 26, 'close_rate': 134, 'exit_type': 'sell_signal'}]
    trades_df = DataFrame(trades)
    trades_df = edge._fill_calculable_fields(trades_df)
    final = edge._process_expectancy(trades_df)
    assert 'TEST/BTC' in final
    assert final['TEST/BTC'].stoploss == -0.9
    assert final['TEST/BTC'].nb_trades == len(trades_df)
    assert round(final['TEST/BTC'].winrate, 10) == 1.0
    assert round(final['TEST/BTC'].risk_reward_ratio, 10) == float('inf')
    assert round(final['TEST/BTC'].expectancy, 10) == float('inf')