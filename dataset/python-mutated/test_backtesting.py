import random
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock
import numpy as np
import pandas as pd
import pytest
from freqtrade import constants
from freqtrade.commands.optimize_commands import setup_optimize_configuration, start_backtesting
from freqtrade.configuration import TimeRange
from freqtrade.data import history
from freqtrade.data.btanalysis import BT_DATA_COLUMNS, evaluate_result_multi
from freqtrade.data.converter import clean_ohlcv_dataframe
from freqtrade.data.dataprovider import DataProvider
from freqtrade.data.history import get_timerange
from freqtrade.enums import CandleType, ExitType, RunMode
from freqtrade.exceptions import DependencyException, OperationalException
from freqtrade.exchange import timeframe_to_next_date, timeframe_to_prev_date
from freqtrade.optimize.backtest_caching import get_backtest_metadata_filename, get_strategy_run_id
from freqtrade.optimize.backtesting import Backtesting
from freqtrade.persistence import LocalTrade, Trade
from freqtrade.resolvers import StrategyResolver
from freqtrade.util.datetime_helpers import dt_utc
from tests.conftest import CURRENT_TEST_STRATEGY, EXMS, get_args, log_has, log_has_re, patch_exchange, patched_configuration_load_config_file
ORDER_TYPES = [{'entry': 'limit', 'exit': 'limit', 'stoploss': 'limit', 'stoploss_on_exchange': False}, {'entry': 'limit', 'exit': 'limit', 'stoploss': 'limit', 'stoploss_on_exchange': True}]

def trim_dictlist(dict_list, num):
    if False:
        return 10
    new = {}
    for (pair, pair_data) in dict_list.items():
        new[pair] = pair_data[num:].reset_index()
    return new

def load_data_test(what, testdatadir):
    if False:
        while True:
            i = 10
    timerange = TimeRange.parse_timerange('1510694220-1510700340')
    data = history.load_pair_history(pair='UNITTEST/BTC', datadir=testdatadir, timeframe='1m', timerange=timerange, drop_incomplete=False, fill_up_missing=False)
    base = 0.001
    if what == 'raise':
        data.loc[:, 'open'] = data.index * base
        data.loc[:, 'high'] = data.index * base + 0.0001
        data.loc[:, 'low'] = data.index * base - 0.0001
        data.loc[:, 'close'] = data.index * base
    if what == 'lower':
        data.loc[:, 'open'] = 1 - data.index * base
        data.loc[:, 'high'] = 1 - data.index * base + 0.0001
        data.loc[:, 'low'] = 1 - data.index * base - 0.0001
        data.loc[:, 'close'] = 1 - data.index * base
    if what == 'sine':
        hz = 0.1
        data.loc[:, 'open'] = np.sin(data.index * hz) / 1000 + base
        data.loc[:, 'high'] = np.sin(data.index * hz) / 1000 + base + 0.0001
        data.loc[:, 'low'] = np.sin(data.index * hz) / 1000 + base - 0.0001
        data.loc[:, 'close'] = np.sin(data.index * hz) / 1000 + base
    return {'UNITTEST/BTC': clean_ohlcv_dataframe(data, timeframe='1m', pair='UNITTEST/BTC', fill_missing=True, drop_incomplete=True)}

def _make_backtest_conf(mocker, datadir, conf=None, pair='UNITTEST/BTC'):
    if False:
        print('Hello World!')
    data = history.load_data(datadir=datadir, timeframe='1m', pairs=[pair])
    data = trim_dictlist(data, -201)
    patch_exchange(mocker)
    backtesting = Backtesting(conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    processed = backtesting.strategy.advise_all_indicators(data)
    (min_date, max_date) = get_timerange(processed)
    return {'processed': processed, 'start_date': min_date, 'end_date': max_date}

def _trend(signals, buy_value, sell_value):
    if False:
        return 10
    n = len(signals['low'])
    buy = np.zeros(n)
    sell = np.zeros(n)
    for i in range(0, len(signals['date'])):
        if random.random() > 0.5:
            buy[i] = buy_value
            sell[i] = sell_value
    signals['enter_long'] = buy
    signals['exit_long'] = sell
    signals['enter_short'] = 0
    signals['exit_short'] = 0
    return signals

def _trend_alternate(dataframe=None, metadata=None):
    if False:
        for i in range(10):
            print('nop')
    signals = dataframe
    low = signals['low']
    n = len(low)
    buy = np.zeros(n)
    sell = np.zeros(n)
    for i in range(0, len(buy)):
        if i % 2 == 0:
            buy[i] = 1
        else:
            sell[i] = 1
    signals['enter_long'] = buy
    signals['exit_long'] = sell
    signals['enter_short'] = 0
    signals['exit_short'] = 0
    return dataframe

def test_setup_optimize_configuration_without_arguments(mocker, default_conf, caplog) -> None:
    if False:
        return 10
    patched_configuration_load_config_file(mocker, default_conf)
    args = ['backtesting', '--config', 'config.json', '--strategy', CURRENT_TEST_STRATEGY, '--export', 'none']
    config = setup_optimize_configuration(get_args(args), RunMode.BACKTEST)
    assert 'max_open_trades' in config
    assert 'stake_currency' in config
    assert 'stake_amount' in config
    assert 'exchange' in config
    assert 'pair_whitelist' in config['exchange']
    assert 'datadir' in config
    assert log_has('Using data directory: {} ...'.format(config['datadir']), caplog)
    assert 'timeframe' in config
    assert not log_has_re('Parameter -i/--ticker-interval detected .*', caplog)
    assert 'position_stacking' not in config
    assert not log_has('Parameter --enable-position-stacking detected ...', caplog)
    assert 'timerange' not in config
    assert 'export' in config
    assert config['export'] == 'none'
    assert 'runmode' in config
    assert config['runmode'] == RunMode.BACKTEST

def test_setup_bt_configuration_with_arguments(mocker, default_conf, caplog) -> None:
    if False:
        return 10
    patched_configuration_load_config_file(mocker, default_conf)
    mocker.patch('freqtrade.configuration.configuration.create_datadir', lambda c, x: x)
    args = ['backtesting', '--config', 'config.json', '--strategy', CURRENT_TEST_STRATEGY, '--datadir', '/foo/bar', '--timeframe', '1m', '--enable-position-stacking', '--disable-max-market-positions', '--timerange', ':100', '--export-filename', 'foo_bar.json', '--fee', '0']
    config = setup_optimize_configuration(get_args(args), RunMode.BACKTEST)
    assert 'max_open_trades' in config
    assert 'stake_currency' in config
    assert 'stake_amount' in config
    assert 'exchange' in config
    assert 'pair_whitelist' in config['exchange']
    assert 'datadir' in config
    assert config['runmode'] == RunMode.BACKTEST
    assert log_has('Using data directory: {} ...'.format(config['datadir']), caplog)
    assert 'timeframe' in config
    assert log_has('Parameter -i/--timeframe detected ... Using timeframe: 1m ...', caplog)
    assert 'position_stacking' in config
    assert log_has('Parameter --enable-position-stacking detected ...', caplog)
    assert 'use_max_market_positions' in config
    assert log_has('Parameter --disable-max-market-positions detected ...', caplog)
    assert log_has('max_open_trades set to unlimited ...', caplog)
    assert 'timerange' in config
    assert log_has('Parameter --timerange detected: {} ...'.format(config['timerange']), caplog)
    assert 'export' in config
    assert 'exportfilename' in config
    assert isinstance(config['exportfilename'], Path)
    assert log_has('Storing backtest results to {} ...'.format(config['exportfilename']), caplog)
    assert 'fee' in config
    assert log_has('Parameter --fee detected, setting fee to: {} ...'.format(config['fee']), caplog)

def test_setup_optimize_configuration_stake_amount(mocker, default_conf, caplog) -> None:
    if False:
        for i in range(10):
            print('nop')
    patched_configuration_load_config_file(mocker, default_conf)
    args = ['backtesting', '--config', 'config.json', '--strategy', CURRENT_TEST_STRATEGY, '--stake-amount', '1', '--starting-balance', '2']
    conf = setup_optimize_configuration(get_args(args), RunMode.BACKTEST)
    assert isinstance(conf, dict)
    args = ['backtesting', '--config', 'config.json', '--strategy', CURRENT_TEST_STRATEGY, '--stake-amount', '1', '--starting-balance', '0.5']
    with pytest.raises(OperationalException, match='Starting balance .* smaller .*'):
        setup_optimize_configuration(get_args(args), RunMode.BACKTEST)

def test_start(mocker, fee, default_conf, caplog) -> None:
    if False:
        for i in range(10):
            print('nop')
    start_mock = MagicMock()
    mocker.patch(f'{EXMS}.get_fee', fee)
    patch_exchange(mocker)
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.start', start_mock)
    patched_configuration_load_config_file(mocker, default_conf)
    args = ['backtesting', '--config', 'config.json', '--strategy', CURRENT_TEST_STRATEGY]
    pargs = get_args(args)
    start_backtesting(pargs)
    assert log_has('Starting freqtrade in Backtesting mode', caplog)
    assert start_mock.call_count == 1

@pytest.mark.parametrize('order_types', ORDER_TYPES)
def test_backtesting_init(mocker, default_conf, order_types) -> None:
    if False:
        i = 10
        return i + 15
    '\n    Check that stoploss_on_exchange is set to False while backtesting\n    since backtesting assumes a perfect stoploss anyway.\n    '
    default_conf['order_types'] = order_types
    patch_exchange(mocker)
    get_fee = mocker.patch(f'{EXMS}.get_fee', MagicMock(return_value=0.5))
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    assert backtesting.config == default_conf
    assert backtesting.timeframe == '5m'
    assert callable(backtesting.strategy.advise_all_indicators)
    assert callable(backtesting.strategy.advise_entry)
    assert callable(backtesting.strategy.advise_exit)
    assert isinstance(backtesting.strategy.dp, DataProvider)
    get_fee.assert_called()
    assert backtesting.fee == 0.5
    assert not backtesting.strategy.order_types['stoploss_on_exchange']
    assert backtesting.strategy.bot_started is True

def test_backtesting_init_no_timeframe(mocker, default_conf, caplog) -> None:
    if False:
        for i in range(10):
            print('nop')
    patch_exchange(mocker)
    del default_conf['timeframe']
    default_conf['strategy_list'] = [CURRENT_TEST_STRATEGY, 'HyperoptableStrategy']
    mocker.patch(f'{EXMS}.get_fee', MagicMock(return_value=0.5))
    with pytest.raises(OperationalException, match='Timeframe needs to be set in either configuration'):
        Backtesting(default_conf)

def test_data_with_fee(default_conf, mocker) -> None:
    if False:
        return 10
    patch_exchange(mocker)
    default_conf['fee'] = 0.1234
    fee_mock = mocker.patch(f'{EXMS}.get_fee', MagicMock(return_value=0.5))
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    assert backtesting.fee == 0.1234
    assert fee_mock.call_count == 0
    default_conf['fee'] = 0.0
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    assert backtesting.fee == 0.0
    assert fee_mock.call_count == 0

def test_data_to_dataframe_bt(default_conf, mocker, testdatadir) -> None:
    if False:
        i = 10
        return i + 15
    patch_exchange(mocker)
    timerange = TimeRange.parse_timerange('1510694220-1510700340')
    data = history.load_data(testdatadir, '1m', ['UNITTEST/BTC'], timerange=timerange, fill_up_missing=True)
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    processed = backtesting.strategy.advise_all_indicators(data)
    assert len(processed['UNITTEST/BTC']) == 103
    strategy = StrategyResolver.load_strategy(default_conf)
    processed2 = strategy.advise_all_indicators(data)
    assert processed['UNITTEST/BTC'].equals(processed2['UNITTEST/BTC'])

def test_backtest_abort(default_conf, mocker, testdatadir) -> None:
    if False:
        return 10
    patch_exchange(mocker)
    backtesting = Backtesting(default_conf)
    backtesting.check_abort()
    backtesting.abort = True
    with pytest.raises(DependencyException, match='Stop requested'):
        backtesting.check_abort()
    assert backtesting.abort is False
    assert backtesting.progress.progress == 0

def test_backtesting_start(default_conf, mocker, caplog) -> None:
    if False:
        return 10

    def get_timerange(input1):
        if False:
            print('Hello World!')
        return (dt_utc(2017, 11, 14, 21, 17), dt_utc(2017, 11, 14, 22, 59))
    mocker.patch('freqtrade.data.history.get_timerange', get_timerange)
    patch_exchange(mocker)
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest')
    mocker.patch('freqtrade.optimize.backtesting.generate_backtest_stats')
    mocker.patch('freqtrade.optimize.backtesting.show_backtest_results')
    sbs = mocker.patch('freqtrade.optimize.backtesting.store_backtest_stats')
    sbc = mocker.patch('freqtrade.optimize.backtesting.store_backtest_analysis_results')
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist', PropertyMock(return_value=['UNITTEST/BTC']))
    default_conf['timeframe'] = '1m'
    default_conf['export'] = 'signals'
    default_conf['exportfilename'] = 'export.txt'
    default_conf['timerange'] = '-1510694220'
    default_conf['runmode'] = RunMode.BACKTEST
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.bot_loop_start = MagicMock()
    backtesting.strategy.bot_start = MagicMock()
    backtesting.start()
    exists = ['Backtesting with data from 2017-11-14 21:17:00 up to 2017-11-14 22:59:00 (0 days).']
    for line in exists:
        assert log_has(line, caplog)
    assert backtesting.strategy.dp._pairlists is not None
    assert backtesting.strategy.bot_start.call_count == 1
    assert backtesting.strategy.bot_loop_start.call_count == 0
    assert sbs.call_count == 1
    assert sbc.call_count == 1

def test_backtesting_start_no_data(default_conf, mocker, caplog, testdatadir) -> None:
    if False:
        return 10

    def get_timerange(input1):
        if False:
            while True:
                i = 10
        return (dt_utc(2017, 11, 14, 21, 17), dt_utc(2017, 11, 14, 22, 59))
    mocker.patch('freqtrade.data.history.history_utils.load_pair_history', MagicMock(return_value=pd.DataFrame()))
    mocker.patch('freqtrade.data.history.get_timerange', get_timerange)
    patch_exchange(mocker)
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest')
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist', PropertyMock(return_value=['UNITTEST/BTC']))
    default_conf['timeframe'] = '1m'
    default_conf['export'] = 'none'
    default_conf['timerange'] = '20180101-20180102'
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    with pytest.raises(OperationalException, match='No data found. Terminating.'):
        backtesting.start()

def test_backtesting_no_pair_left(default_conf, mocker, caplog, testdatadir) -> None:
    if False:
        return 10
    mocker.patch(f'{EXMS}.exchange_has', MagicMock(return_value=True))
    mocker.patch('freqtrade.data.history.history_utils.load_pair_history', MagicMock(return_value=pd.DataFrame()))
    mocker.patch('freqtrade.data.history.get_timerange', get_timerange)
    patch_exchange(mocker)
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest')
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist', PropertyMock(return_value=[]))
    default_conf['timeframe'] = '1m'
    default_conf['export'] = 'none'
    default_conf['timerange'] = '20180101-20180102'
    with pytest.raises(OperationalException, match='No pair in whitelist.'):
        Backtesting(default_conf)
    default_conf['pairlists'] = [{'method': 'VolumePairList', 'number_assets': 5}]
    with pytest.raises(OperationalException, match='VolumePairList not allowed for backtesting\\..*StaticPairList.*'):
        Backtesting(default_conf)
    default_conf.update({'pairlists': [{'method': 'StaticPairList'}], 'timeframe_detail': '1d'})
    with pytest.raises(OperationalException, match='Detail timeframe must be smaller than strategy timeframe.'):
        Backtesting(default_conf)

def test_backtesting_pairlist_list(default_conf, mocker, caplog, testdatadir, tickers) -> None:
    if False:
        i = 10
        return i + 15
    mocker.patch(f'{EXMS}.exchange_has', MagicMock(return_value=True))
    mocker.patch(f'{EXMS}.get_tickers', tickers)
    mocker.patch(f'{EXMS}.price_to_precision', lambda s, x, y: y)
    mocker.patch('freqtrade.data.history.get_timerange', get_timerange)
    patch_exchange(mocker)
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest')
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist', PropertyMock(return_value=['XRP/BTC']))
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.refresh_pairlist')
    default_conf['ticker_interval'] = '1m'
    default_conf['export'] = 'none'
    del default_conf['stoploss']
    default_conf['timerange'] = '20180101-20180102'
    default_conf['pairlists'] = [{'method': 'VolumePairList', 'number_assets': 5}]
    with pytest.raises(OperationalException, match='VolumePairList not allowed for backtesting\\..*StaticPairList.*'):
        Backtesting(default_conf)
    default_conf['pairlists'] = [{'method': 'StaticPairList'}, {'method': 'PerformanceFilter'}]
    with pytest.raises(OperationalException, match='PerformanceFilter not allowed for backtesting.'):
        Backtesting(default_conf)
    default_conf['pairlists'] = [{'method': 'StaticPairList'}, {'method': 'PrecisionFilter'}]
    Backtesting(default_conf)
    default_conf['strategy_list'] = [CURRENT_TEST_STRATEGY, 'StrategyTestV2']
    with pytest.raises(OperationalException, match='PrecisionFilter not allowed for backtesting multiple strategies.'):
        Backtesting(default_conf)

def test_backtest__enter_trade(default_conf, fee, mocker) -> None:
    if False:
        print('Hello World!')
    default_conf['use_exit_signal'] = False
    mocker.patch(f'{EXMS}.get_fee', fee)
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=1e-05)
    mocker.patch(f'{EXMS}.get_max_pair_stake_amount', return_value=float('inf'))
    patch_exchange(mocker)
    default_conf['stake_amount'] = 'unlimited'
    default_conf['max_open_trades'] = 2
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    pair = 'UNITTEST/BTC'
    row = [pd.Timestamp(year=2020, month=1, day=1, hour=5, minute=0), 1, 0.001, 0.0011, 0, 0.00099, 0.0012, '']
    trade = backtesting._enter_trade(pair, row=row, direction='long')
    assert isinstance(trade, LocalTrade)
    assert trade.stake_amount == 495
    LocalTrade.trades_open.append(trade)
    LocalTrade.trades_open.append(trade)
    backtesting.wallets.update()
    trade = backtesting._enter_trade(pair, row=row, direction='long')
    assert trade is None
    LocalTrade.trades_open.pop()
    trade = backtesting._enter_trade(pair, row=row, direction='long')
    assert trade is not None
    backtesting.strategy.custom_stake_amount = lambda **kwargs: 123.5
    backtesting.wallets.update()
    trade = backtesting._enter_trade(pair, row=row, direction='long')
    assert trade
    assert trade.stake_amount == 123.5
    backtesting.strategy.custom_stake_amount = lambda **kwargs: 20 / 0
    trade = backtesting._enter_trade(pair, row=row, direction='long')
    assert trade
    assert trade.stake_amount == 495
    assert trade.is_short is False
    trade = backtesting._enter_trade(pair, row=row, direction='short')
    assert trade
    assert trade.stake_amount == 495
    assert trade.is_short is True
    mocker.patch(f'{EXMS}.get_max_pair_stake_amount', return_value=300.0)
    trade = backtesting._enter_trade(pair, row=row, direction='long')
    assert trade
    assert trade.stake_amount == 300.0

def test_backtest__enter_trade_futures(default_conf_usdt, fee, mocker) -> None:
    if False:
        for i in range(10):
            print('nop')
    default_conf_usdt['use_exit_signal'] = False
    mocker.patch(f'{EXMS}.get_fee', fee)
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=1e-05)
    mocker.patch(f'{EXMS}.get_max_pair_stake_amount', return_value=float('inf'))
    mocker.patch(f'{EXMS}.get_max_leverage', return_value=100)
    mocker.patch('freqtrade.optimize.backtesting.price_to_precision', lambda p, *args: p)
    patch_exchange(mocker)
    default_conf_usdt['stake_amount'] = 300
    default_conf_usdt['max_open_trades'] = 2
    default_conf_usdt['trading_mode'] = 'futures'
    default_conf_usdt['margin_mode'] = 'isolated'
    default_conf_usdt['stake_currency'] = 'USDT'
    default_conf_usdt['exchange']['pair_whitelist'] = ['.*']
    backtesting = Backtesting(default_conf_usdt)
    backtesting._set_strategy(backtesting.strategylist[0])
    pair = 'ETH/USDT:USDT'
    row = [pd.Timestamp(year=2020, month=1, day=1, hour=5, minute=0), 0.1, 0.12, 0.099, 0.11, 1, 0, 1, 0, '', '', '']
    backtesting.strategy.leverage = MagicMock(return_value=5.0)
    mocker.patch(f'{EXMS}.get_maintenance_ratio_and_amt', return_value=(0.01, 0.01))
    trade = backtesting._enter_trade(pair, row=row, direction='long')
    assert pytest.approx(trade.liquidation_price) == 0.081767037
    trade = backtesting._enter_trade(pair, row=row, direction='short')
    assert pytest.approx(trade.liquidation_price) == 0.11787191
    assert pytest.approx(trade.orders[0].cost) == trade.stake_amount * trade.leverage + trade.fee_open
    assert pytest.approx(trade.orders[-1].stake_amount) == trade.stake_amount
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=600.0)
    trade = backtesting._enter_trade(pair, row=row, direction='long')
    assert trade is None
    mocker.patch('freqtrade.wallets.Wallets.get_trade_stake_amount', side_effect=DependencyException)
    trade = backtesting._enter_trade(pair, row=row, direction='long')
    assert trade is None

def test_backtest__check_trade_exit(default_conf, fee, mocker) -> None:
    if False:
        while True:
            i = 10
    default_conf['use_exit_signal'] = False
    mocker.patch(f'{EXMS}.get_fee', fee)
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=1e-05)
    mocker.patch(f'{EXMS}.get_max_pair_stake_amount', return_value=float('inf'))
    patch_exchange(mocker)
    default_conf['timeframe_detail'] = '1m'
    default_conf['max_open_trades'] = 2
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    pair = 'UNITTEST/BTC'
    row = [pd.Timestamp(year=2020, month=1, day=1, hour=4, minute=55, tzinfo=timezone.utc), 200, 201.5, 195, 201, 1, 0, 0, 0, '', '', '']
    trade = backtesting._enter_trade(pair, row=row, direction='long')
    assert isinstance(trade, LocalTrade)
    row_sell = [pd.Timestamp(year=2020, month=1, day=1, hour=5, minute=0, tzinfo=timezone.utc), 200, 210.5, 195, 201, 0, 0, 0, 0, '', '', '']
    res = backtesting._check_trade_exit(trade, row_sell, row_sell[0].to_pydatetime())
    assert res is not None
    assert res.exit_reason == ExitType.ROI.value
    assert res.close_date_utc == datetime(2020, 1, 1, 5, 0, tzinfo=timezone.utc)
    trade = backtesting._enter_trade(pair, row=row, direction='long')
    assert isinstance(trade, LocalTrade)
    backtesting.detail_data[pair] = pd.DataFrame([], columns=['date', 'open', 'high', 'low', 'close', 'enter_long', 'exit_long', 'enter_short', 'exit_short', 'long_tag', 'short_tag', 'exit_tag'])
    res = backtesting._check_trade_exit(trade, row, row[0].to_pydatetime())
    assert res is None

def test_backtest_one(default_conf, fee, mocker, testdatadir) -> None:
    if False:
        while True:
            i = 10
    default_conf['use_exit_signal'] = False
    default_conf['max_open_trades'] = 10
    mocker.patch(f'{EXMS}.get_fee', fee)
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=1e-05)
    mocker.patch(f'{EXMS}.get_max_pair_stake_amount', return_value=float('inf'))
    patch_exchange(mocker)
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    pair = 'UNITTEST/BTC'
    timerange = TimeRange('date', None, 1517227800, 0)
    data = history.load_data(datadir=testdatadir, timeframe='5m', pairs=['UNITTEST/BTC'], timerange=timerange)
    processed = backtesting.strategy.advise_all_indicators(data)
    (min_date, max_date) = get_timerange(processed)
    result = backtesting.backtest(processed=deepcopy(processed), start_date=min_date, end_date=max_date)
    results = result['results']
    assert not results.empty
    assert len(results) == 2
    expected = pd.DataFrame({'pair': [pair, pair], 'stake_amount': [0.001, 0.001], 'max_stake_amount': [0.001, 0.001], 'amount': [0.00957442, 0.0097064], 'open_date': pd.to_datetime([dt_utc(2018, 1, 29, 18, 40, 0), dt_utc(2018, 1, 30, 3, 30, 0)], utc=True), 'close_date': pd.to_datetime([dt_utc(2018, 1, 29, 22, 35, 0), dt_utc(2018, 1, 30, 4, 10, 0)], utc=True), 'open_rate': [0.104445, 0.10302485], 'close_rate': [0.104969, 0.103541], 'fee_open': [0.0025, 0.0025], 'fee_close': [0.0025, 0.0025], 'trade_duration': [235, 40], 'profit_ratio': [0.0, 0.0], 'profit_abs': [0.0, 0.0], 'exit_reason': [ExitType.ROI.value, ExitType.ROI.value], 'initial_stop_loss_abs': [0.0940005, 0.09272236], 'initial_stop_loss_ratio': [-0.1, -0.1], 'stop_loss_abs': [0.0940005, 0.09272236], 'stop_loss_ratio': [-0.1, -0.1], 'min_rate': [0.10370188, 0.10300000000000001], 'max_rate': [0.10501, 0.1038888], 'is_open': [False, False], 'enter_tag': [None, None], 'leverage': [1.0, 1.0], 'is_short': [False, False], 'open_timestamp': [1517251200000, 1517283000000], 'close_timestamp': [1517265300000, 1517285400000], 'orders': [[{'amount': 0.00957442, 'safe_price': 0.104445, 'ft_order_side': 'buy', 'order_filled_timestamp': 1517251200000, 'ft_is_entry': True}, {'amount': 0.00957442, 'safe_price': 0.10496853383458644, 'ft_order_side': 'sell', 'order_filled_timestamp': 1517265300000, 'ft_is_entry': False}], [{'amount': 0.0097064, 'safe_price': 0.10302485, 'ft_order_side': 'buy', 'order_filled_timestamp': 1517283000000, 'ft_is_entry': True}, {'amount': 0.0097064, 'safe_price': 0.10354126528822055, 'ft_order_side': 'sell', 'order_filled_timestamp': 1517285400000, 'ft_is_entry': False}]]})
    pd.testing.assert_frame_equal(results, expected)
    assert 'orders' in results.columns
    data_pair = processed[pair]
    for (_, t) in results.iterrows():
        assert len(t['orders']) == 2
        ln = data_pair.loc[data_pair['date'] == t['open_date']]
        assert not ln.empty
        assert round(ln.iloc[0]['open'], 6) == round(t['open_rate'], 6)
        ln1 = data_pair.loc[data_pair['date'] == t['close_date']]
        assert round(ln1.iloc[0]['open'], 6) == round(t['close_rate'], 6) or round(ln1.iloc[0]['low'], 6) < round(t['close_rate'], 6) < round(ln1.iloc[0]['high'], 6)

@pytest.mark.parametrize('use_detail', [True, False])
def test_backtest_one_detail(default_conf_usdt, fee, mocker, testdatadir, use_detail) -> None:
    if False:
        i = 10
        return i + 15
    default_conf_usdt['use_exit_signal'] = False
    mocker.patch(f'{EXMS}.get_fee', fee)
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=1e-05)
    mocker.patch(f'{EXMS}.get_max_pair_stake_amount', return_value=float('inf'))
    if use_detail:
        default_conf_usdt['timeframe_detail'] = '1m'
    patch_exchange(mocker)

    def advise_entry(df, *args, **kwargs):
        if False:
            print('Hello World!')
        df.loc[df['rsi'] < 40, 'enter_long'] = 1
        return df

    def custom_entry_price(proposed_rate, **kwargs):
        if False:
            print('Hello World!')
        return proposed_rate * 0.997
    default_conf_usdt['max_open_trades'] = 10
    backtesting = Backtesting(default_conf_usdt)
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.populate_entry_trend = advise_entry
    backtesting.strategy.custom_entry_price = custom_entry_price
    pair = 'XRP/ETH'
    timerange = TimeRange.parse_timerange('20191010-20191013')
    data = history.load_data(datadir=testdatadir, timeframe='5m', pairs=[pair], timerange=timerange)
    if use_detail:
        data_1m = history.load_data(datadir=testdatadir, timeframe='1m', pairs=[pair], timerange=timerange)
        backtesting.detail_data = data_1m
    processed = backtesting.strategy.advise_all_indicators(data)
    (min_date, max_date) = get_timerange(processed)
    result = backtesting.backtest(processed=deepcopy(processed), start_date=min_date, end_date=max_date)
    results = result['results']
    assert not results.empty
    assert len(results) == (2 if use_detail else 3)
    assert 'orders' in results.columns
    data_pair = processed[pair]
    data_1m_pair = data_1m[pair] if use_detail else pd.DataFrame()
    late_entry = 0
    for (_, t) in results.iterrows():
        assert len(t['orders']) == 2
        entryo = t['orders'][0]
        entry_ts = datetime.fromtimestamp(entryo['order_filled_timestamp'] // 1000, tz=timezone.utc)
        if entry_ts > t['open_date']:
            late_entry += 1
        ln = data_1m_pair.loc[data_1m_pair['date'] == entry_ts] if use_detail else data_pair.loc[data_pair['date'] == entry_ts]
        assert not ln.empty
        assert round(ln.iloc[0]['low'], 6) <= round(t['open_rate'], 6) <= round(ln.iloc[0]['high'], 6)
        ln1 = data_pair.loc[data_pair['date'] == t['close_date']]
        if use_detail:
            ln1_1m = data_1m_pair.loc[data_1m_pair['date'] == t['close_date']]
            assert not ln1.empty or not ln1_1m.empty
        else:
            assert not ln1.empty
        ln2 = ln1_1m if ln1.empty else ln1
        assert round(ln2.iloc[0]['low'], 6) <= round(t['close_rate'], 6) <= round(ln2.iloc[0]['high'], 6)
    assert late_entry > 0

@pytest.mark.parametrize('use_detail', [True, False])
def test_backtest_one_detail_futures(default_conf_usdt, fee, mocker, testdatadir, use_detail) -> None:
    if False:
        return 10
    default_conf_usdt['use_exit_signal'] = False
    default_conf_usdt['trading_mode'] = 'futures'
    default_conf_usdt['margin_mode'] = 'isolated'
    default_conf_usdt['candle_type_def'] = CandleType.FUTURES
    mocker.patch(f'{EXMS}.get_fee', fee)
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=1e-05)
    mocker.patch(f'{EXMS}.get_max_pair_stake_amount', return_value=float('inf'))
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist', PropertyMock(return_value=['XRP/USDT:USDT']))
    mocker.patch(f'{EXMS}.get_maintenance_ratio_and_amt', return_value=(0.01, 0.01))
    default_conf_usdt['timeframe'] = '1h'
    if use_detail:
        default_conf_usdt['timeframe_detail'] = '5m'
    patch_exchange(mocker)

    def advise_entry(df, *args, **kwargs):
        if False:
            print('Hello World!')
        df.loc[df['rsi'] < 40, 'enter_long'] = 1
        return df

    def custom_entry_price(proposed_rate, **kwargs):
        if False:
            return 10
        return proposed_rate * 0.997
    default_conf_usdt['max_open_trades'] = 10
    backtesting = Backtesting(default_conf_usdt)
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.populate_entry_trend = advise_entry
    backtesting.strategy.custom_entry_price = custom_entry_price
    pair = 'XRP/USDT:USDT'
    timerange = TimeRange.parse_timerange('20211117-20211119')
    data = history.load_data(datadir=Path(testdatadir), timeframe='1h', pairs=[pair], timerange=timerange, candle_type=CandleType.FUTURES)
    backtesting.load_bt_data_detail()
    processed = backtesting.strategy.advise_all_indicators(data)
    (min_date, max_date) = get_timerange(processed)
    result = backtesting.backtest(processed=deepcopy(processed), start_date=min_date, end_date=max_date)
    results = result['results']
    assert not results.empty
    assert len(results) == (5 if use_detail else 2)
    assert 'orders' in results.columns
    data_pair = processed[pair]
    data_1m_pair = backtesting.detail_data[pair] if use_detail else pd.DataFrame()
    late_entry = 0
    for (_, t) in results.iterrows():
        assert len(t['orders']) == 2
        entryo = t['orders'][0]
        entry_ts = datetime.fromtimestamp(entryo['order_filled_timestamp'] // 1000, tz=timezone.utc)
        if entry_ts > t['open_date']:
            late_entry += 1
        ln = data_1m_pair.loc[data_1m_pair['date'] == entry_ts] if use_detail else data_pair.loc[data_pair['date'] == entry_ts]
        assert not ln.empty
        assert round(ln.iloc[0]['low'], 6) <= round(t['open_rate'], 6) <= round(ln.iloc[0]['high'], 6)
        ln1 = data_pair.loc[data_pair['date'] == t['close_date']]
        if use_detail:
            ln1_1m = data_1m_pair.loc[data_1m_pair['date'] == t['close_date']]
            assert not ln1.empty or not ln1_1m.empty
        else:
            assert not ln1.empty
        ln2 = ln1_1m if ln1.empty else ln1
        assert round(ln2.iloc[0]['low'], 6) <= round(t['close_rate'], 6) <= round(ln2.iloc[0]['high'], 6)
    assert -0.0181 < Trade.trades[1].funding_fees < -0.01

@pytest.mark.parametrize('use_detail', [True, False])
def test_backtest_one_detail_futures_funding_fees(default_conf_usdt, fee, mocker, testdatadir, use_detail) -> None:
    if False:
        for i in range(10):
            print('nop')
    default_conf_usdt['use_exit_signal'] = False
    default_conf_usdt['trading_mode'] = 'futures'
    default_conf_usdt['margin_mode'] = 'isolated'
    default_conf_usdt['candle_type_def'] = CandleType.FUTURES
    default_conf_usdt['minimal_roi'] = {'0': 1}
    default_conf_usdt['dry_run_wallet'] = 100000
    mocker.patch(f'{EXMS}.get_fee', fee)
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=1e-05)
    mocker.patch(f'{EXMS}.get_max_pair_stake_amount', return_value=float('inf'))
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist', PropertyMock(return_value=['XRP/USDT:USDT']))
    mocker.patch(f'{EXMS}.get_maintenance_ratio_and_amt', return_value=(0.01, 0.01))
    default_conf_usdt['timeframe'] = '1h'
    if use_detail:
        default_conf_usdt['timeframe_detail'] = '5m'
    patch_exchange(mocker)

    def advise_entry(df, *args, **kwargs):
        if False:
            while True:
                i = 10
        df.loc[:, 'enter_long'] = 1
        return df

    def adjust_trade_position(trade, current_time, **kwargs):
        if False:
            return 10
        if current_time > datetime(2021, 11, 18, 2, 0, 0, tzinfo=timezone.utc):
            return None
        return default_conf_usdt['stake_amount']
    default_conf_usdt['max_open_trades'] = 1
    backtesting = Backtesting(default_conf_usdt)
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.populate_entry_trend = advise_entry
    backtesting.strategy.adjust_trade_position = adjust_trade_position
    backtesting.strategy.leverage = lambda **kwargs: 1
    backtesting.strategy.position_adjustment_enable = True
    pair = 'XRP/USDT:USDT'
    timerange = TimeRange.parse_timerange('20211117-20211119')
    data = history.load_data(datadir=Path(testdatadir), timeframe='1h', pairs=[pair], timerange=timerange, candle_type=CandleType.FUTURES)
    backtesting.load_bt_data_detail()
    processed = backtesting.strategy.advise_all_indicators(data)
    (min_date, max_date) = get_timerange(processed)
    result = backtesting.backtest(processed=deepcopy(processed), start_date=min_date, end_date=max_date)
    results = result['results']
    assert not results.empty
    assert len(results) == 1
    assert 'orders' in results.columns
    for t in Trade.trades:
        assert t.nr_of_successful_entries >= 6
        assert -1.81 < t.funding_fees < -0.1

def test_backtest_timedout_entry_orders(default_conf, fee, mocker, testdatadir) -> None:
    if False:
        for i in range(10):
            print('nop')
    default_conf['strategy'] = 'StrategyTestV3CustomEntryPrice'
    default_conf['startup_candle_count'] = 0
    default_conf['unfilledtimeout'] = {'entry': 4}
    mocker.patch(f'{EXMS}.get_fee', fee)
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=1e-05)
    mocker.patch(f'{EXMS}.get_max_pair_stake_amount', return_value=float('inf'))
    patch_exchange(mocker)
    default_conf['max_open_trades'] = 1
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    timerange = TimeRange('date', 'date', 1517227800, 1517231100)
    data = history.load_data(datadir=testdatadir, timeframe='5m', pairs=['UNITTEST/BTC'], timerange=timerange)
    (min_date, max_date) = get_timerange(data)
    result = backtesting.backtest(processed=deepcopy(data), start_date=min_date, end_date=max_date)
    assert result['timedout_entry_orders'] == 10

def test_backtest_1min_timeframe(default_conf, fee, mocker, testdatadir) -> None:
    if False:
        print('Hello World!')
    default_conf['use_exit_signal'] = False
    default_conf['max_open_trades'] = 1
    mocker.patch(f'{EXMS}.get_fee', fee)
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=1e-05)
    mocker.patch(f'{EXMS}.get_max_pair_stake_amount', return_value=float('inf'))
    patch_exchange(mocker)
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    timerange = TimeRange.parse_timerange('1510688220-1510700340')
    data = history.load_data(datadir=testdatadir, timeframe='1m', pairs=['UNITTEST/BTC'], timerange=timerange)
    processed = backtesting.strategy.advise_all_indicators(data)
    (min_date, max_date) = get_timerange(processed)
    results = backtesting.backtest(processed=processed, start_date=min_date, end_date=max_date)
    assert not results['results'].empty
    assert len(results['results']) == 1

def test_backtest_trim_no_data_left(default_conf, fee, mocker, testdatadir) -> None:
    if False:
        while True:
            i = 10
    default_conf['use_exit_signal'] = False
    default_conf['max_open_trades'] = 10
    mocker.patch(f'{EXMS}.get_fee', fee)
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=1e-05)
    mocker.patch(f'{EXMS}.get_max_pair_stake_amount', return_value=float('inf'))
    patch_exchange(mocker)
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    timerange = TimeRange('date', None, 1517227800, 0)
    backtesting.required_startup = 100
    backtesting.timerange = timerange
    data = history.load_data(datadir=testdatadir, timeframe='5m', pairs=['UNITTEST/BTC'], timerange=timerange)
    df = data['UNITTEST/BTC']
    df['date'] = df.loc[:, 'date'] - timedelta(days=1)
    df = df.iloc[:100]
    data['XRP/USDT'] = df
    processed = backtesting.strategy.advise_all_indicators(data)
    (min_date, max_date) = get_timerange(processed)
    backtesting.backtest(processed=deepcopy(processed), start_date=min_date, end_date=max_date)

def test_processed(default_conf, mocker, testdatadir) -> None:
    if False:
        print('Hello World!')
    patch_exchange(mocker)
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    dict_of_tickerrows = load_data_test('raise', testdatadir)
    dataframes = backtesting.strategy.advise_all_indicators(dict_of_tickerrows)
    dataframe = dataframes['UNITTEST/BTC']
    cols = dataframe.columns
    for col in ['close', 'high', 'low', 'open', 'date', 'ema10', 'rsi', 'fastd', 'plus_di']:
        assert col in cols

def test_backtest_dataprovider_analyzed_df(default_conf, fee, mocker, testdatadir) -> None:
    if False:
        for i in range(10):
            print('nop')
    default_conf['use_exit_signal'] = False
    default_conf['max_open_trades'] = 10
    mocker.patch(f'{EXMS}.get_fee', fee)
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=1e-05)
    mocker.patch(f'{EXMS}.get_max_pair_stake_amount', return_value=100000)
    patch_exchange(mocker)
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    timerange = TimeRange('date', None, 1517227800, 0)
    data = history.load_data(datadir=testdatadir, timeframe='5m', pairs=['UNITTEST/BTC'], timerange=timerange)
    processed = backtesting.strategy.advise_all_indicators(data)
    (min_date, max_date) = get_timerange(processed)
    count = 0

    def tmp_confirm_entry(pair, current_time, **kwargs):
        if False:
            print('Hello World!')
        nonlocal count
        dp = backtesting.strategy.dp
        (df, _) = dp.get_analyzed_dataframe(pair, backtesting.strategy.timeframe)
        current_candle = df.iloc[-1].squeeze()
        assert current_candle['enter_long'] == 1
        candle_date = timeframe_to_next_date(backtesting.strategy.timeframe, current_candle['date'])
        assert candle_date == current_time
        df = dp.get_pair_dataframe(pair, backtesting.strategy.timeframe)
        prior_time = timeframe_to_prev_date(backtesting.strategy.timeframe, candle_date - timedelta(seconds=1))
        assert prior_time == df.iloc[-1].squeeze()['date']
        assert df.iloc[-1].squeeze()['date'] < current_time
        count += 1
    backtesting.strategy.confirm_trade_entry = tmp_confirm_entry
    backtesting.backtest(processed=deepcopy(processed), start_date=min_date, end_date=max_date)
    assert count == 5

def test_backtest_pricecontours_protections(default_conf, fee, mocker, testdatadir) -> None:
    if False:
        for i in range(10):
            print('nop')
    patch_exchange(mocker)
    default_conf['protections'] = [{'method': 'CooldownPeriod', 'stop_duration': 3}]
    default_conf['enable_protections'] = True
    default_conf['timeframe'] = '1m'
    default_conf['max_open_trades'] = 1
    mocker.patch(f'{EXMS}.get_fee', fee)
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=1e-05)
    mocker.patch(f'{EXMS}.get_max_pair_stake_amount', return_value=float('inf'))
    tests = [['sine', 9], ['raise', 10], ['lower', 0], ['sine', 9], ['raise', 10]]
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    for [contour, numres] in tests:
        print(f'{contour}, {numres}')
        data = load_data_test(contour, testdatadir)
        processed = backtesting.strategy.advise_all_indicators(data)
        (min_date, max_date) = get_timerange(processed)
        assert isinstance(processed, dict)
        results = backtesting.backtest(processed=processed, start_date=min_date, end_date=max_date)
        assert len(results['results']) == numres

@pytest.mark.parametrize('protections,contour,expected', [(None, 'sine', 35), (None, 'raise', 19), (None, 'lower', 0), (None, 'sine', 35), (None, 'raise', 19), ([{'method': 'CooldownPeriod', 'stop_duration': 3}], 'sine', 9), ([{'method': 'CooldownPeriod', 'stop_duration': 3}], 'raise', 10), ([{'method': 'CooldownPeriod', 'stop_duration': 3}], 'lower', 0), ([{'method': 'CooldownPeriod', 'stop_duration': 3}], 'sine', 9), ([{'method': 'CooldownPeriod', 'stop_duration': 3}], 'raise', 10)])
def test_backtest_pricecontours(default_conf, fee, mocker, testdatadir, protections, contour, expected) -> None:
    if False:
        for i in range(10):
            print('nop')
    if protections:
        default_conf['protections'] = protections
        default_conf['enable_protections'] = True
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=1e-05)
    mocker.patch(f'{EXMS}.get_max_pair_stake_amount', return_value=float('inf'))
    mocker.patch(f'{EXMS}.get_fee', fee)
    patch_exchange(mocker)
    default_conf['timeframe'] = '1m'
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    data = load_data_test(contour, testdatadir)
    processed = backtesting.strategy.advise_all_indicators(data)
    (min_date, max_date) = get_timerange(processed)
    assert isinstance(processed, dict)
    backtesting.strategy.max_open_trades = 1
    backtesting.config.update({'max_open_trades': 1})
    results = backtesting.backtest(processed=processed, start_date=min_date, end_date=max_date)
    assert len(results['results']) == expected

def test_backtest_clash_buy_sell(mocker, default_conf, testdatadir):
    if False:
        return 10

    def fun(dataframe=None, pair=None):
        if False:
            i = 10
            return i + 15
        buy_value = 1
        sell_value = 1
        return _trend(dataframe, buy_value, sell_value)
    default_conf['max_open_trades'] = 10
    backtest_conf = _make_backtest_conf(mocker, conf=default_conf, datadir=testdatadir)
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.advise_entry = fun
    backtesting.strategy.advise_exit = fun
    result = backtesting.backtest(**backtest_conf)
    assert result['results'].empty

def test_backtest_only_sell(mocker, default_conf, testdatadir):
    if False:
        for i in range(10):
            print('nop')

    def fun(dataframe=None, pair=None):
        if False:
            i = 10
            return i + 15
        buy_value = 0
        sell_value = 1
        return _trend(dataframe, buy_value, sell_value)
    default_conf['max_open_trades'] = 10
    backtest_conf = _make_backtest_conf(mocker, conf=default_conf, datadir=testdatadir)
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.advise_entry = fun
    backtesting.strategy.advise_exit = fun
    result = backtesting.backtest(**backtest_conf)
    assert result['results'].empty

def test_backtest_alternate_buy_sell(default_conf, fee, mocker, testdatadir):
    if False:
        for i in range(10):
            print('nop')
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=1e-05)
    mocker.patch(f'{EXMS}.get_max_pair_stake_amount', return_value=float('inf'))
    mocker.patch(f'{EXMS}.get_fee', fee)
    default_conf['max_open_trades'] = 10
    backtest_conf = _make_backtest_conf(mocker, conf=default_conf, pair='UNITTEST/BTC', datadir=testdatadir)
    default_conf['timeframe'] = '1m'
    backtesting = Backtesting(default_conf)
    backtesting.required_startup = 0
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.advise_entry = _trend_alternate
    backtesting.strategy.advise_exit = _trend_alternate
    result = backtesting.backtest(**backtest_conf)
    results = result['results']
    assert len(results) == 100
    analyzed_df = backtesting.dataprovider.get_analyzed_dataframe('UNITTEST/BTC', '1m')[0]
    assert len(analyzed_df) == 200
    expected_last_candle_date = backtest_conf['end_date'] - timedelta(minutes=1)
    assert analyzed_df.iloc[-1]['date'].to_pydatetime() == expected_last_candle_date
    assert len(results.loc[results['is_open']]) == 0

@pytest.mark.parametrize('pair', ['ADA/BTC', 'LTC/BTC'])
@pytest.mark.parametrize('tres', [0, 20, 30])
def test_backtest_multi_pair(default_conf, fee, mocker, tres, pair, testdatadir):
    if False:
        return 10

    def _trend_alternate_hold(dataframe=None, metadata=None):
        if False:
            return 10
        '\n        Buy every xth candle - sell every other xth -2 (hold on to pairs a bit)\n        '
        if metadata['pair'] in ('ETH/BTC', 'LTC/BTC'):
            multi = 20
        else:
            multi = 18
        dataframe['enter_long'] = np.where(dataframe.index % multi == 0, 1, 0)
        dataframe['exit_long'] = np.where((dataframe.index + multi - 2) % multi == 0, 1, 0)
        dataframe['enter_short'] = 0
        dataframe['exit_short'] = 0
        return dataframe
    mocker.patch(f'{EXMS}.get_min_pair_stake_amount', return_value=1e-05)
    mocker.patch(f'{EXMS}.get_max_pair_stake_amount', return_value=float('inf'))
    mocker.patch(f'{EXMS}.get_fee', fee)
    patch_exchange(mocker)
    pairs = ['ADA/BTC', 'DASH/BTC', 'ETH/BTC', 'LTC/BTC', 'NXT/BTC']
    data = history.load_data(datadir=testdatadir, timeframe='5m', pairs=pairs)
    data = trim_dictlist(data, -500)
    if tres > 0:
        data[pair] = data[pair][tres:].reset_index()
    default_conf['timeframe'] = '5m'
    default_conf['max_open_trades'] = 3
    backtesting = Backtesting(default_conf)
    backtesting._set_strategy(backtesting.strategylist[0])
    backtesting.strategy.advise_entry = _trend_alternate_hold
    backtesting.strategy.advise_exit = _trend_alternate_hold
    processed = backtesting.strategy.advise_all_indicators(data)
    (min_date, max_date) = get_timerange(processed)
    backtest_conf = {'processed': deepcopy(processed), 'start_date': min_date, 'end_date': max_date}
    results = backtesting.backtest(**backtest_conf)
    assert len(evaluate_result_multi(results['results'], '5m', 2)) > 0
    assert len(evaluate_result_multi(results['results'], '5m', 3)) == 0
    offset = 1 if tres == 0 else 0
    removed_candles = len(data[pair]) - offset
    assert len(backtesting.dataprovider.get_analyzed_dataframe(pair, '5m')[0]) == removed_candles
    assert len(backtesting.dataprovider.get_analyzed_dataframe('NXT/BTC', '5m')[0]) == len(data['NXT/BTC']) - 1
    backtesting.strategy.max_open_trades = 1
    backtesting.config.update({'max_open_trades': 1})
    backtest_conf = {'processed': deepcopy(processed), 'start_date': min_date, 'end_date': max_date}
    results = backtesting.backtest(**backtest_conf)
    assert len(evaluate_result_multi(results['results'], '5m', 1)) == 0

def test_backtest_start_timerange(default_conf, mocker, caplog, testdatadir):
    if False:
        print('Hello World!')
    patch_exchange(mocker)
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest')
    mocker.patch('freqtrade.optimize.backtesting.generate_backtest_stats')
    mocker.patch('freqtrade.optimize.backtesting.show_backtest_results')
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist', PropertyMock(return_value=['UNITTEST/BTC']))
    patched_configuration_load_config_file(mocker, default_conf)
    args = ['backtesting', '--config', 'config.json', '--strategy', CURRENT_TEST_STRATEGY, '--datadir', str(testdatadir), '--timeframe', '1m', '--timerange', '1510694220-1510700340', '--enable-position-stacking', '--disable-max-market-positions']
    args = get_args(args)
    start_backtesting(args)
    exists = ['Parameter -i/--timeframe detected ... Using timeframe: 1m ...', 'Ignoring max_open_trades (--disable-max-market-positions was used) ...', 'Parameter --timerange detected: 1510694220-1510700340 ...', f'Using data directory: {testdatadir} ...', 'Loading data from 2017-11-14 20:57:00 up to 2017-11-14 22:59:00 (0 days).', 'Backtesting with data from 2017-11-14 21:17:00 up to 2017-11-14 22:59:00 (0 days).', 'Parameter --enable-position-stacking detected ...']
    for line in exists:
        assert log_has(line, caplog)

@pytest.mark.filterwarnings('ignore:deprecated')
def test_backtest_start_multi_strat(default_conf, mocker, caplog, testdatadir):
    if False:
        for i in range(10):
            print('nop')
    default_conf.update({'use_exit_signal': True, 'exit_profit_only': False, 'exit_profit_offset': 0.0, 'ignore_roi_if_entry_signal': False})
    patch_exchange(mocker)
    backtestmock = MagicMock(return_value={'results': pd.DataFrame(columns=BT_DATA_COLUMNS), 'config': default_conf, 'locks': [], 'rejected_signals': 20, 'timedout_entry_orders': 0, 'timedout_exit_orders': 0, 'canceled_trade_entries': 0, 'canceled_entry_orders': 0, 'replaced_entry_orders': 0, 'final_balance': 1000})
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist', PropertyMock(return_value=['UNITTEST/BTC']))
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest', backtestmock)
    text_table_mock = MagicMock()
    sell_reason_mock = MagicMock()
    strattable_mock = MagicMock()
    strat_summary = MagicMock()
    mocker.patch.multiple('freqtrade.optimize.optimize_reports.bt_output', text_table_bt_results=text_table_mock, text_table_strategy=strattable_mock)
    mocker.patch.multiple('freqtrade.optimize.optimize_reports.optimize_reports', generate_pair_metrics=MagicMock(), generate_exit_reason_stats=sell_reason_mock, generate_strategy_comparison=strat_summary, generate_daily_stats=MagicMock())
    patched_configuration_load_config_file(mocker, default_conf)
    args = ['backtesting', '--config', 'config.json', '--datadir', str(testdatadir), '--strategy-path', str(Path(__file__).parents[1] / 'strategy/strats'), '--timeframe', '1m', '--timerange', '1510694220-1510700340', '--enable-position-stacking', '--disable-max-market-positions', '--strategy-list', CURRENT_TEST_STRATEGY, 'StrategyTestV2']
    args = get_args(args)
    start_backtesting(args)
    assert backtestmock.call_count == 2
    assert text_table_mock.call_count == 4
    assert strattable_mock.call_count == 1
    assert sell_reason_mock.call_count == 2
    assert strat_summary.call_count == 1
    exists = ['Parameter -i/--timeframe detected ... Using timeframe: 1m ...', 'Ignoring max_open_trades (--disable-max-market-positions was used) ...', 'Parameter --timerange detected: 1510694220-1510700340 ...', f'Using data directory: {testdatadir} ...', 'Loading data from 2017-11-14 20:57:00 up to 2017-11-14 22:59:00 (0 days).', 'Backtesting with data from 2017-11-14 21:17:00 up to 2017-11-14 22:59:00 (0 days).', 'Parameter --enable-position-stacking detected ...', f'Running backtesting for Strategy {CURRENT_TEST_STRATEGY}', 'Running backtesting for Strategy StrategyTestV2']
    for line in exists:
        assert log_has(line, caplog)

def test_backtest_start_multi_strat_nomock(default_conf, mocker, caplog, testdatadir, capsys):
    if False:
        for i in range(10):
            print('nop')
    default_conf.update({'use_exit_signal': True, 'exit_profit_only': False, 'exit_profit_offset': 0.0, 'ignore_roi_if_entry_signal': False})
    patch_exchange(mocker)
    result1 = pd.DataFrame({'pair': ['XRP/BTC', 'LTC/BTC'], 'profit_ratio': [0.0, 0.0], 'profit_abs': [0.0, 0.0], 'open_date': pd.to_datetime(['2018-01-29 18:40:00', '2018-01-30 03:30:00'], utc=True), 'close_date': pd.to_datetime(['2018-01-29 20:45:00', '2018-01-30 05:35:00'], utc=True), 'trade_duration': [235, 40], 'is_open': [False, False], 'stake_amount': [0.01, 0.01], 'open_rate': [0.104445, 0.10302485], 'close_rate': [0.104969, 0.103541], 'is_short': [False, False], 'exit_reason': [ExitType.ROI, ExitType.ROI]})
    result2 = pd.DataFrame({'pair': ['XRP/BTC', 'LTC/BTC', 'ETH/BTC'], 'profit_ratio': [0.03, 0.01, 0.1], 'profit_abs': [0.01, 0.02, 0.2], 'open_date': pd.to_datetime(['2018-01-29 18:40:00', '2018-01-30 03:30:00', '2018-01-30 05:30:00'], utc=True), 'close_date': pd.to_datetime(['2018-01-29 20:45:00', '2018-01-30 05:35:00', '2018-01-30 08:30:00'], utc=True), 'trade_duration': [47, 40, 20], 'is_open': [False, False, False], 'stake_amount': [0.01, 0.01, 0.01], 'open_rate': [0.104445, 0.10302485, 0.122541], 'close_rate': [0.104969, 0.103541, 0.123541], 'is_short': [False, False, False], 'exit_reason': [ExitType.ROI, ExitType.ROI, ExitType.STOP_LOSS]})
    backtestmock = MagicMock(side_effect=[{'results': result1, 'config': default_conf, 'locks': [], 'rejected_signals': 20, 'timedout_entry_orders': 0, 'timedout_exit_orders': 0, 'canceled_trade_entries': 0, 'canceled_entry_orders': 0, 'replaced_entry_orders': 0, 'final_balance': 1000}, {'results': result2, 'config': default_conf, 'locks': [], 'rejected_signals': 20, 'timedout_entry_orders': 0, 'timedout_exit_orders': 0, 'canceled_trade_entries': 0, 'canceled_entry_orders': 0, 'replaced_entry_orders': 0, 'final_balance': 1000}])
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist', PropertyMock(return_value=['UNITTEST/BTC']))
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest', backtestmock)
    patched_configuration_load_config_file(mocker, default_conf)
    args = ['backtesting', '--config', 'config.json', '--datadir', str(testdatadir), '--strategy-path', str(Path(__file__).parents[1] / 'strategy/strats'), '--timeframe', '1m', '--timerange', '1510694220-1510700340', '--enable-position-stacking', '--disable-max-market-positions', '--breakdown', 'day', '--strategy-list', CURRENT_TEST_STRATEGY, 'StrategyTestV2']
    args = get_args(args)
    start_backtesting(args)
    exists = ['Parameter -i/--timeframe detected ... Using timeframe: 1m ...', 'Ignoring max_open_trades (--disable-max-market-positions was used) ...', 'Parameter --timerange detected: 1510694220-1510700340 ...', f'Using data directory: {testdatadir} ...', 'Loading data from 2017-11-14 20:57:00 up to 2017-11-14 22:59:00 (0 days).', 'Backtesting with data from 2017-11-14 21:17:00 up to 2017-11-14 22:59:00 (0 days).', 'Parameter --enable-position-stacking detected ...', f'Running backtesting for Strategy {CURRENT_TEST_STRATEGY}', 'Running backtesting for Strategy StrategyTestV2']
    for line in exists:
        assert log_has(line, caplog)
    captured = capsys.readouterr()
    assert 'BACKTESTING REPORT' in captured.out
    assert 'EXIT REASON STATS' in captured.out
    assert 'DAY BREAKDOWN' in captured.out
    assert 'LEFT OPEN TRADES REPORT' in captured.out
    assert '2017-11-14 21:17:00 -> 2017-11-14 22:59:00 | Max open trades : 1' in captured.out
    assert 'STRATEGY SUMMARY' in captured.out

@pytest.mark.filterwarnings('ignore:deprecated')
def test_backtest_start_futures_noliq(default_conf_usdt, mocker, caplog, testdatadir, capsys):
    if False:
        for i in range(10):
            print('nop')
    default_conf_usdt.update({'trading_mode': 'futures', 'margin_mode': 'isolated', 'use_exit_signal': True, 'exit_profit_only': False, 'exit_profit_offset': 0.0, 'ignore_roi_if_entry_signal': False, 'strategy': CURRENT_TEST_STRATEGY})
    patch_exchange(mocker)
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist', PropertyMock(return_value=['HULUMULU/USDT', 'XRP/USDT:USDT']))
    patched_configuration_load_config_file(mocker, default_conf_usdt)
    args = ['backtesting', '--config', 'config.json', '--datadir', str(testdatadir), '--strategy-path', str(Path(__file__).parents[1] / 'strategy/strats'), '--timeframe', '1h']
    args = get_args(args)
    with pytest.raises(OperationalException, match='Pairs .* got no leverage tiers available\\.'):
        start_backtesting(args)

@pytest.mark.filterwarnings('ignore:deprecated')
def test_backtest_start_nomock_futures(default_conf_usdt, mocker, caplog, testdatadir, capsys):
    if False:
        while True:
            i = 10
    default_conf_usdt.update({'trading_mode': 'futures', 'margin_mode': 'isolated', 'use_exit_signal': True, 'exit_profit_only': False, 'exit_profit_offset': 0.0, 'ignore_roi_if_entry_signal': False, 'strategy': CURRENT_TEST_STRATEGY})
    patch_exchange(mocker)
    result1 = pd.DataFrame({'pair': ['XRP/USDT:USDT', 'XRP/USDT:USDT'], 'profit_ratio': [0.0, 0.0], 'profit_abs': [0.0, 0.0], 'open_date': pd.to_datetime(['2021-11-18 18:00:00', '2021-11-18 03:00:00'], utc=True), 'close_date': pd.to_datetime(['2021-11-18 20:00:00', '2021-11-18 05:00:00'], utc=True), 'trade_duration': [235, 40], 'is_open': [False, False], 'is_short': [False, False], 'stake_amount': [0.01, 0.01], 'open_rate': [0.104445, 0.10302485], 'close_rate': [0.104969, 0.103541], 'exit_reason': [ExitType.ROI, ExitType.ROI]})
    result2 = pd.DataFrame({'pair': ['XRP/USDT:USDT', 'XRP/USDT:USDT', 'XRP/USDT:USDT'], 'profit_ratio': [0.03, 0.01, 0.1], 'profit_abs': [0.01, 0.02, 0.2], 'open_date': pd.to_datetime(['2021-11-19 18:00:00', '2021-11-19 03:00:00', '2021-11-19 05:00:00'], utc=True), 'close_date': pd.to_datetime(['2021-11-19 20:00:00', '2021-11-19 05:00:00', '2021-11-19 08:00:00'], utc=True), 'trade_duration': [47, 40, 20], 'is_open': [False, False, False], 'is_short': [False, False, False], 'stake_amount': [0.01, 0.01, 0.01], 'open_rate': [0.104445, 0.10302485, 0.122541], 'close_rate': [0.104969, 0.103541, 0.123541], 'exit_reason': [ExitType.ROI, ExitType.ROI, ExitType.STOP_LOSS]})
    backtestmock = MagicMock(side_effect=[{'results': result1, 'config': default_conf_usdt, 'locks': [], 'rejected_signals': 20, 'timedout_entry_orders': 0, 'timedout_exit_orders': 0, 'canceled_trade_entries': 0, 'canceled_entry_orders': 0, 'replaced_entry_orders': 0, 'final_balance': 1000}, {'results': result2, 'config': default_conf_usdt, 'locks': [], 'rejected_signals': 20, 'timedout_entry_orders': 0, 'timedout_exit_orders': 0, 'canceled_trade_entries': 0, 'canceled_entry_orders': 0, 'replaced_entry_orders': 0, 'final_balance': 1000}])
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist', PropertyMock(return_value=['XRP/USDT:USDT']))
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest', backtestmock)
    patched_configuration_load_config_file(mocker, default_conf_usdt)
    args = ['backtesting', '--config', 'config.json', '--datadir', str(testdatadir), '--strategy-path', str(Path(__file__).parents[1] / 'strategy/strats'), '--timeframe', '1h']
    args = get_args(args)
    start_backtesting(args)
    exists = ['Parameter -i/--timeframe detected ... Using timeframe: 1h ...', f'Using data directory: {testdatadir} ...', 'Loading data from 2021-11-17 01:00:00 up to 2021-11-21 04:00:00 (4 days).', 'Backtesting with data from 2021-11-17 21:00:00 up to 2021-11-21 04:00:00 (3 days).', 'XRP/USDT:USDT, funding_rate, 8h, data starts at 2021-11-18 00:00:00', 'XRP/USDT:USDT, mark, 8h, data starts at 2021-11-18 00:00:00', f'Running backtesting for Strategy {CURRENT_TEST_STRATEGY}']
    for line in exists:
        assert log_has(line, caplog)
    captured = capsys.readouterr()
    assert 'BACKTESTING REPORT' in captured.out
    assert 'EXIT REASON STATS' in captured.out
    assert 'LEFT OPEN TRADES REPORT' in captured.out

@pytest.mark.filterwarnings('ignore:deprecated')
def test_backtest_start_multi_strat_nomock_detail(default_conf, mocker, caplog, testdatadir, capsys):
    if False:
        print('Hello World!')
    default_conf.update({'use_exit_signal': True, 'exit_profit_only': False, 'exit_profit_offset': 0.0, 'ignore_roi_if_entry_signal': False})
    patch_exchange(mocker)
    result1 = pd.DataFrame({'pair': ['XRP/BTC', 'LTC/BTC'], 'profit_ratio': [0.0, 0.0], 'profit_abs': [0.0, 0.0], 'open_date': pd.to_datetime(['2018-01-29 18:40:00', '2018-01-30 03:30:00'], utc=True), 'close_date': pd.to_datetime(['2018-01-29 20:45:00', '2018-01-30 05:35:00'], utc=True), 'trade_duration': [235, 40], 'is_open': [False, False], 'is_short': [False, False], 'stake_amount': [0.01, 0.01], 'open_rate': [0.104445, 0.10302485], 'close_rate': [0.104969, 0.103541], 'exit_reason': [ExitType.ROI, ExitType.ROI]})
    result2 = pd.DataFrame({'pair': ['XRP/BTC', 'LTC/BTC', 'ETH/BTC'], 'profit_ratio': [0.03, 0.01, 0.1], 'profit_abs': [0.01, 0.02, 0.2], 'open_date': pd.to_datetime(['2018-01-29 18:40:00', '2018-01-30 03:30:00', '2018-01-30 05:30:00'], utc=True), 'close_date': pd.to_datetime(['2018-01-29 20:45:00', '2018-01-30 05:35:00', '2018-01-30 08:30:00'], utc=True), 'trade_duration': [47, 40, 20], 'is_open': [False, False, False], 'is_short': [False, False, False], 'stake_amount': [0.01, 0.01, 0.01], 'open_rate': [0.104445, 0.10302485, 0.122541], 'close_rate': [0.104969, 0.103541, 0.123541], 'exit_reason': [ExitType.ROI, ExitType.ROI, ExitType.STOP_LOSS]})
    backtestmock = MagicMock(side_effect=[{'results': result1, 'config': default_conf, 'locks': [], 'rejected_signals': 20, 'timedout_entry_orders': 0, 'timedout_exit_orders': 0, 'canceled_trade_entries': 0, 'canceled_entry_orders': 0, 'replaced_entry_orders': 0, 'final_balance': 1000}, {'results': result2, 'config': default_conf, 'locks': [], 'rejected_signals': 20, 'timedout_entry_orders': 0, 'timedout_exit_orders': 0, 'canceled_trade_entries': 0, 'canceled_entry_orders': 0, 'replaced_entry_orders': 0, 'final_balance': 1000}])
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist', PropertyMock(return_value=['XRP/ETH']))
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest', backtestmock)
    patched_configuration_load_config_file(mocker, default_conf)
    args = ['backtesting', '--config', 'config.json', '--datadir', str(testdatadir), '--strategy-path', str(Path(__file__).parents[1] / 'strategy/strats'), '--timeframe', '5m', '--timeframe-detail', '1m', '--strategy-list', CURRENT_TEST_STRATEGY]
    args = get_args(args)
    start_backtesting(args)
    exists = ['Parameter -i/--timeframe detected ... Using timeframe: 5m ...', 'Parameter --timeframe-detail detected, using 1m for intra-candle backtesting ...', f'Using data directory: {testdatadir} ...', 'Loading data from 2019-10-11 00:00:00 up to 2019-10-13 11:15:00 (2 days).', 'Backtesting with data from 2019-10-11 01:40:00 up to 2019-10-13 11:15:00 (2 days).', f'Running backtesting for Strategy {CURRENT_TEST_STRATEGY}']
    for line in exists:
        assert log_has(line, caplog)
    captured = capsys.readouterr()
    assert 'BACKTESTING REPORT' in captured.out
    assert 'EXIT REASON STATS' in captured.out
    assert 'LEFT OPEN TRADES REPORT' in captured.out

@pytest.mark.filterwarnings('ignore:deprecated')
@pytest.mark.parametrize('run_id', ['2', 'changed'])
@pytest.mark.parametrize('start_delta', [{'days': 0}, {'days': 1}, {'weeks': 1}, {'weeks': 4}])
@pytest.mark.parametrize('cache', constants.BACKTEST_CACHE_AGE)
def test_backtest_start_multi_strat_caching(default_conf, mocker, caplog, testdatadir, run_id, start_delta, cache):
    if False:
        while True:
            i = 10
    default_conf.update({'use_exit_signal': True, 'exit_profit_only': False, 'exit_profit_offset': 0.0, 'ignore_roi_if_entry_signal': False})
    patch_exchange(mocker)
    backtestmock = MagicMock(return_value={'results': pd.DataFrame(columns=BT_DATA_COLUMNS), 'config': default_conf, 'locks': [], 'rejected_signals': 20, 'timedout_entry_orders': 0, 'timedout_exit_orders': 0, 'canceled_trade_entries': 0, 'canceled_entry_orders': 0, 'replaced_entry_orders': 0, 'final_balance': 1000})
    mocker.patch('freqtrade.plugins.pairlistmanager.PairListManager.whitelist', PropertyMock(return_value=['UNITTEST/BTC']))
    mocker.patch('freqtrade.optimize.backtesting.Backtesting.backtest', backtestmock)
    mocker.patch('freqtrade.optimize.backtesting.show_backtest_results', MagicMock())
    now = min_backtest_date = datetime.now(tz=timezone.utc)
    start_time = now - timedelta(**start_delta) + timedelta(hours=1)
    if cache == 'none':
        min_backtest_date = now + timedelta(days=1)
    elif cache == 'day':
        min_backtest_date = now - timedelta(days=1)
    elif cache == 'week':
        min_backtest_date = now - timedelta(weeks=1)
    elif cache == 'month':
        min_backtest_date = now - timedelta(weeks=4)
    load_backtest_metadata = MagicMock(return_value={'StrategyTestV2': {'run_id': '1', 'backtest_start_time': now.timestamp()}, 'StrategyTestV3': {'run_id': run_id, 'backtest_start_time': start_time.timestamp()}})
    load_backtest_stats = MagicMock(side_effect=[{'metadata': {'StrategyTestV2': {'run_id': '1'}}, 'strategy': {'StrategyTestV2': {}}, 'strategy_comparison': [{'key': 'StrategyTestV2'}]}, {'metadata': {'StrategyTestV3': {'run_id': '2'}}, 'strategy': {'StrategyTestV3': {}}, 'strategy_comparison': [{'key': 'StrategyTestV3'}]}])
    mocker.patch('pathlib.Path.glob', return_value=[Path(datetime.strftime(datetime.now(), 'backtest-result-%Y-%m-%d_%H-%M-%S.json'))])
    mocker.patch.multiple('freqtrade.data.btanalysis', load_backtest_metadata=load_backtest_metadata, load_backtest_stats=load_backtest_stats)
    mocker.patch('freqtrade.optimize.backtesting.get_strategy_run_id', side_effect=['1', '2', '2'])
    patched_configuration_load_config_file(mocker, default_conf)
    args = ['backtesting', '--config', 'config.json', '--datadir', str(testdatadir), '--strategy-path', str(Path(__file__).parents[1] / 'strategy/strats'), '--timeframe', '1m', '--timerange', '1510694220-1510700340', '--enable-position-stacking', '--disable-max-market-positions', '--cache', cache, '--strategy-list', 'StrategyTestV2', 'StrategyTestV3']
    args = get_args(args)
    start_backtesting(args)
    exists = ['Parameter -i/--timeframe detected ... Using timeframe: 1m ...', 'Parameter --timerange detected: 1510694220-1510700340 ...', f'Using data directory: {testdatadir} ...', 'Loading data from 2017-11-14 20:57:00 up to 2017-11-14 22:59:00 (0 days).', 'Parameter --enable-position-stacking detected ...']
    for line in exists:
        assert log_has(line, caplog)
    if cache == 'none':
        assert backtestmock.call_count == 2
        exists = ['Running backtesting for Strategy StrategyTestV2', 'Running backtesting for Strategy StrategyTestV3', 'Ignoring max_open_trades (--disable-max-market-positions was used) ...', 'Backtesting with data from 2017-11-14 21:17:00 up to 2017-11-14 22:59:00 (0 days).']
    elif run_id == '2' and min_backtest_date < start_time:
        assert backtestmock.call_count == 0
        exists = ['Reusing result of previous backtest for StrategyTestV2', 'Reusing result of previous backtest for StrategyTestV3']
    else:
        exists = ['Reusing result of previous backtest for StrategyTestV2', 'Running backtesting for Strategy StrategyTestV3', 'Ignoring max_open_trades (--disable-max-market-positions was used) ...', 'Backtesting with data from 2017-11-14 21:17:00 up to 2017-11-14 22:59:00 (0 days).']
        assert backtestmock.call_count == 1
    for line in exists:
        assert log_has(line, caplog)

def test_get_strategy_run_id(default_conf_usdt):
    if False:
        while True:
            i = 10
    default_conf_usdt.update({'strategy': 'StrategyTestV2', 'max_open_trades': float('inf')})
    strategy = StrategyResolver.load_strategy(default_conf_usdt)
    x = get_strategy_run_id(strategy)
    assert isinstance(x, str)

def test_get_backtest_metadata_filename():
    if False:
        while True:
            i = 10
    filename = Path('backtest_results.json')
    expected = Path('backtest_results.meta.json')
    assert get_backtest_metadata_filename(filename) == expected
    filename = Path('/path/to/backtest.results.json')
    expected = Path('/path/to/backtest.results.meta.json')
    assert get_backtest_metadata_filename(filename) == expected
    filename = Path('backtest_results.json')
    expected = Path('backtest_results.meta.json')
    assert get_backtest_metadata_filename(filename) == expected
    filename = '/path/to/backtest_results.json'
    expected = Path('/path/to/backtest_results.meta.json')
    assert get_backtest_metadata_filename(filename) == expected
    filename = '/path/to/backtest_results'
    expected = Path('/path/to/backtest_results.meta')
    assert get_backtest_metadata_filename(filename) == expected
    filename = '/path/to/backtest.results.json'
    expected = Path('/path/to/backtest.results.meta.json')
    assert get_backtest_metadata_filename(filename) == expected
    filename = 'backtest_results.json'
    expected = Path('backtest_results.meta.json')
    assert get_backtest_metadata_filename(filename) == expected