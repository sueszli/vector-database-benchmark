from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock
import pytest
from pandas import DataFrame, DateOffset, Timestamp, to_datetime
from freqtrade.configuration import TimeRange
from freqtrade.constants import LAST_BT_RESULT_FN
from freqtrade.data.btanalysis import BT_DATA_COLUMNS, analyze_trade_parallelism, extract_trades_of_period, get_latest_backtest_filename, get_latest_hyperopt_file, load_backtest_data, load_backtest_metadata, load_trades, load_trades_from_db
from freqtrade.data.history import load_data, load_pair_history
from freqtrade.data.metrics import calculate_cagr, calculate_calmar, calculate_csum, calculate_expectancy, calculate_market_change, calculate_max_drawdown, calculate_sharpe, calculate_sortino, calculate_underwater, combine_dataframes_with_mean, create_cum_profit
from freqtrade.exceptions import OperationalException
from freqtrade.util import dt_utc
from tests.conftest import CURRENT_TEST_STRATEGY, create_mock_trades
from tests.conftest_trades import MOCK_TRADE_COUNT

def test_get_latest_backtest_filename(testdatadir, mocker):
    if False:
        return 10
    with pytest.raises(ValueError, match='Directory .* does not exist\\.'):
        get_latest_backtest_filename(testdatadir / 'does_not_exist')
    with pytest.raises(ValueError, match='Directory .* does not seem to contain .*'):
        get_latest_backtest_filename(testdatadir)
    testdir_bt = testdatadir / 'backtest_results'
    res = get_latest_backtest_filename(testdir_bt)
    assert res == 'backtest-result.json'
    res = get_latest_backtest_filename(str(testdir_bt))
    assert res == 'backtest-result.json'
    mocker.patch('freqtrade.data.btanalysis.json_load', return_value={})
    with pytest.raises(ValueError, match="Invalid '.last_result.json' format."):
        get_latest_backtest_filename(testdir_bt)

def test_get_latest_hyperopt_file(testdatadir):
    if False:
        print('Hello World!')
    res = get_latest_hyperopt_file(testdatadir / 'does_not_exist', 'testfile.pickle')
    assert res == testdatadir / 'does_not_exist/testfile.pickle'
    res = get_latest_hyperopt_file(testdatadir.parent)
    assert res == testdatadir.parent / 'hyperopt_results.pickle'
    res = get_latest_hyperopt_file(str(testdatadir.parent))
    assert res == testdatadir.parent / 'hyperopt_results.pickle'
    with pytest.raises(OperationalException, match='--hyperopt-filename expects only the filename, not an absolute path.'):
        get_latest_hyperopt_file(str(testdatadir.parent), str(testdatadir.parent))

def test_load_backtest_metadata(mocker, testdatadir):
    if False:
        print('Hello World!')
    res = load_backtest_metadata(testdatadir / 'nonexistant.file.json')
    assert res == {}
    mocker.patch('freqtrade.data.btanalysis.get_backtest_metadata_filename')
    mocker.patch('freqtrade.data.btanalysis.json_load', side_effect=Exception())
    with pytest.raises(OperationalException, match='Unexpected error.*loading backtest metadata\\.'):
        load_backtest_metadata(testdatadir / 'nonexistant.file.json')

def test_load_backtest_data_old_format(testdatadir, mocker):
    if False:
        print('Hello World!')
    filename = testdatadir / 'backtest-result_test222.json'
    mocker.patch('freqtrade.data.btanalysis.load_backtest_stats', return_value=[])
    with pytest.raises(OperationalException, match='Backtest-results with only trades data are no longer supported.'):
        load_backtest_data(filename)

def test_load_backtest_data_new_format(testdatadir):
    if False:
        for i in range(10):
            print('nop')
    filename = testdatadir / 'backtest_results/backtest-result.json'
    bt_data = load_backtest_data(filename)
    assert isinstance(bt_data, DataFrame)
    assert set(bt_data.columns) == set(BT_DATA_COLUMNS)
    assert len(bt_data) == 179
    bt_data2 = load_backtest_data(str(filename))
    assert bt_data.equals(bt_data2)
    bt_data3 = load_backtest_data(testdatadir / 'backtest_results')
    assert bt_data.equals(bt_data3)
    with pytest.raises(ValueError, match='File .* does not exist\\.'):
        load_backtest_data('filename' + 'nofile')
    with pytest.raises(ValueError, match='Unknown dataformat.'):
        load_backtest_data(testdatadir / 'backtest_results' / LAST_BT_RESULT_FN)

def test_load_backtest_data_multi(testdatadir):
    if False:
        print('Hello World!')
    filename = testdatadir / 'backtest_results/backtest-result_multistrat.json'
    for strategy in ('StrategyTestV2', 'TestStrategy'):
        bt_data = load_backtest_data(filename, strategy=strategy)
        assert isinstance(bt_data, DataFrame)
        assert set(bt_data.columns) == set(BT_DATA_COLUMNS)
        assert len(bt_data) == 179
        bt_data2 = load_backtest_data(str(filename), strategy=strategy)
        assert bt_data.equals(bt_data2)
    with pytest.raises(ValueError, match='Strategy XYZ not available in the backtest result\\.'):
        load_backtest_data(filename, strategy='XYZ')
    with pytest.raises(ValueError, match='Detected backtest result with more than one strategy.*'):
        load_backtest_data(filename)

@pytest.mark.usefixtures('init_persistence')
@pytest.mark.parametrize('is_short', [False, True])
def test_load_trades_from_db(default_conf, fee, is_short, mocker):
    if False:
        while True:
            i = 10
    create_mock_trades(fee, is_short)
    init_mock = mocker.patch('freqtrade.data.btanalysis.init_db', MagicMock())
    trades = load_trades_from_db(db_url=default_conf['db_url'])
    assert init_mock.call_count == 1
    assert len(trades) == MOCK_TRADE_COUNT
    assert isinstance(trades, DataFrame)
    assert 'pair' in trades.columns
    assert 'open_date' in trades.columns
    assert 'profit_ratio' in trades.columns
    for col in BT_DATA_COLUMNS:
        if col not in ['index', 'open_at_end']:
            assert col in trades.columns
    trades = load_trades_from_db(db_url=default_conf['db_url'], strategy=CURRENT_TEST_STRATEGY)
    assert len(trades) == 4
    trades = load_trades_from_db(db_url=default_conf['db_url'], strategy='NoneStrategy')
    assert len(trades) == 0

def test_extract_trades_of_period(testdatadir):
    if False:
        print('Hello World!')
    pair = 'UNITTEST/BTC'
    timerange = TimeRange('date', None, 1510639620, 0)
    data = load_pair_history(pair=pair, timeframe='1m', datadir=testdatadir, timerange=timerange)
    trades = DataFrame({'pair': [pair, pair, pair, pair], 'profit_ratio': [0.0, 0.1, -0.2, -0.5], 'profit_abs': [0.0, 1, -2, -5], 'open_date': to_datetime([datetime(2017, 11, 13, 15, 40, 0, tzinfo=timezone.utc), datetime(2017, 11, 14, 9, 41, 0, tzinfo=timezone.utc), datetime(2017, 11, 14, 14, 20, 0, tzinfo=timezone.utc), datetime(2017, 11, 15, 3, 40, 0, tzinfo=timezone.utc)], utc=True), 'close_date': to_datetime([datetime(2017, 11, 13, 16, 40, 0, tzinfo=timezone.utc), datetime(2017, 11, 14, 10, 41, 0, tzinfo=timezone.utc), datetime(2017, 11, 14, 15, 25, 0, tzinfo=timezone.utc), datetime(2017, 11, 15, 3, 55, 0, tzinfo=timezone.utc)], utc=True)})
    trades1 = extract_trades_of_period(data, trades)
    assert len(trades1) == 2
    assert trades1.iloc[0].open_date == datetime(2017, 11, 14, 9, 41, 0, tzinfo=timezone.utc)
    assert trades1.iloc[0].close_date == datetime(2017, 11, 14, 10, 41, 0, tzinfo=timezone.utc)
    assert trades1.iloc[-1].open_date == datetime(2017, 11, 14, 14, 20, 0, tzinfo=timezone.utc)
    assert trades1.iloc[-1].close_date == datetime(2017, 11, 14, 15, 25, 0, tzinfo=timezone.utc)

def test_analyze_trade_parallelism(testdatadir):
    if False:
        return 10
    filename = testdatadir / 'backtest_results/backtest-result.json'
    bt_data = load_backtest_data(filename)
    res = analyze_trade_parallelism(bt_data, '5m')
    assert isinstance(res, DataFrame)
    assert 'open_trades' in res.columns
    assert res['open_trades'].max() == 3
    assert res['open_trades'].min() == 0

def test_load_trades(default_conf, mocker):
    if False:
        print('Hello World!')
    db_mock = mocker.patch('freqtrade.data.btanalysis.load_trades_from_db', MagicMock())
    bt_mock = mocker.patch('freqtrade.data.btanalysis.load_backtest_data', MagicMock())
    load_trades('DB', db_url=default_conf.get('db_url'), exportfilename=default_conf.get('exportfilename'), no_trades=False, strategy=CURRENT_TEST_STRATEGY)
    assert db_mock.call_count == 1
    assert bt_mock.call_count == 0
    db_mock.reset_mock()
    bt_mock.reset_mock()
    default_conf['exportfilename'] = Path('testfile.json')
    load_trades('file', db_url=default_conf.get('db_url'), exportfilename=default_conf.get('exportfilename'))
    assert db_mock.call_count == 0
    assert bt_mock.call_count == 1
    db_mock.reset_mock()
    bt_mock.reset_mock()
    default_conf['exportfilename'] = 'testfile.json'
    load_trades('file', db_url=default_conf.get('db_url'), exportfilename=default_conf.get('exportfilename'), no_trades=True)
    assert db_mock.call_count == 0
    assert bt_mock.call_count == 0

def test_calculate_market_change(testdatadir):
    if False:
        i = 10
        return i + 15
    pairs = ['ETH/BTC', 'ADA/BTC']
    data = load_data(datadir=testdatadir, pairs=pairs, timeframe='5m')
    result = calculate_market_change(data)
    assert isinstance(result, float)
    assert pytest.approx(result) == 0.01100002

def test_combine_dataframes_with_mean(testdatadir):
    if False:
        i = 10
        return i + 15
    pairs = ['ETH/BTC', 'ADA/BTC']
    data = load_data(datadir=testdatadir, pairs=pairs, timeframe='5m')
    df = combine_dataframes_with_mean(data)
    assert isinstance(df, DataFrame)
    assert 'ETH/BTC' in df.columns
    assert 'ADA/BTC' in df.columns
    assert 'mean' in df.columns

def test_combine_dataframes_with_mean_no_data(testdatadir):
    if False:
        print('Hello World!')
    pairs = ['ETH/BTC', 'ADA/BTC']
    data = load_data(datadir=testdatadir, pairs=pairs, timeframe='6m')
    with pytest.raises(ValueError, match='No objects to concatenate'):
        combine_dataframes_with_mean(data)

def test_create_cum_profit(testdatadir):
    if False:
        while True:
            i = 10
    filename = testdatadir / 'backtest_results/backtest-result.json'
    bt_data = load_backtest_data(filename)
    timerange = TimeRange.parse_timerange('20180110-20180112')
    df = load_pair_history(pair='TRX/BTC', timeframe='5m', datadir=testdatadir, timerange=timerange)
    cum_profits = create_cum_profit(df.set_index('date'), bt_data[bt_data['pair'] == 'TRX/BTC'], 'cum_profits', timeframe='5m')
    assert 'cum_profits' in cum_profits.columns
    assert cum_profits.iloc[0]['cum_profits'] == 0
    assert pytest.approx(cum_profits.iloc[-1]['cum_profits']) == 9.0225563e-05

def test_create_cum_profit1(testdatadir):
    if False:
        return 10
    filename = testdatadir / 'backtest_results/backtest-result.json'
    bt_data = load_backtest_data(filename)
    bt_data['close_date'] = bt_data.loc[:, 'close_date'] + DateOffset(seconds=20)
    timerange = TimeRange.parse_timerange('20180110-20180112')
    df = load_pair_history(pair='TRX/BTC', timeframe='5m', datadir=testdatadir, timerange=timerange)
    cum_profits = create_cum_profit(df.set_index('date'), bt_data[bt_data['pair'] == 'TRX/BTC'], 'cum_profits', timeframe='5m')
    assert 'cum_profits' in cum_profits.columns
    assert cum_profits.iloc[0]['cum_profits'] == 0
    assert pytest.approx(cum_profits.iloc[-1]['cum_profits']) == 9.0225563e-05
    with pytest.raises(ValueError, match='Trade dataframe empty.'):
        create_cum_profit(df.set_index('date'), bt_data[bt_data['pair'] == 'NOTAPAIR'], 'cum_profits', timeframe='5m')

def test_calculate_max_drawdown(testdatadir):
    if False:
        print('Hello World!')
    filename = testdatadir / 'backtest_results/backtest-result.json'
    bt_data = load_backtest_data(filename)
    (_, hdate, lowdate, hval, lval, drawdown) = calculate_max_drawdown(bt_data, value_col='profit_abs')
    assert isinstance(drawdown, float)
    assert pytest.approx(drawdown) == 0.29753914
    assert isinstance(hdate, Timestamp)
    assert isinstance(lowdate, Timestamp)
    assert isinstance(hval, float)
    assert isinstance(lval, float)
    assert hdate == Timestamp('2018-01-16 19:30:00', tz='UTC')
    assert lowdate == Timestamp('2018-01-16 22:25:00', tz='UTC')
    underwater = calculate_underwater(bt_data)
    assert isinstance(underwater, DataFrame)
    with pytest.raises(ValueError, match='Trade dataframe empty.'):
        calculate_max_drawdown(DataFrame())
    with pytest.raises(ValueError, match='Trade dataframe empty.'):
        calculate_underwater(DataFrame())

def test_calculate_csum(testdatadir):
    if False:
        return 10
    filename = testdatadir / 'backtest_results/backtest-result.json'
    bt_data = load_backtest_data(filename)
    (csum_min, csum_max) = calculate_csum(bt_data)
    assert isinstance(csum_min, float)
    assert isinstance(csum_max, float)
    assert csum_min < csum_max
    assert csum_min < 0.0001
    assert csum_max > 0.0002
    (csum_min1, csum_max1) = calculate_csum(bt_data, 5)
    assert csum_min1 == csum_min + 5
    assert csum_max1 == csum_max + 5
    with pytest.raises(ValueError, match='Trade dataframe empty.'):
        (csum_min, csum_max) = calculate_csum(DataFrame())

def test_calculate_expectancy(testdatadir):
    if False:
        for i in range(10):
            print('nop')
    filename = testdatadir / 'backtest_results/backtest-result.json'
    bt_data = load_backtest_data(filename)
    (expectancy, expectancy_ratio) = calculate_expectancy(DataFrame())
    assert expectancy == 0.0
    assert expectancy_ratio == 100
    (expectancy, expectancy_ratio) = calculate_expectancy(bt_data)
    assert isinstance(expectancy, float)
    assert isinstance(expectancy_ratio, float)
    assert pytest.approx(expectancy) == 5.820687070932315e-06
    assert pytest.approx(expectancy_ratio) == 0.07151374226574791
    data = {'profit_abs': [100, 200, 50, -150, 300, -100, 80, -30]}
    df = DataFrame(data)
    (expectancy, expectancy_ratio) = calculate_expectancy(df)
    assert pytest.approx(expectancy) == 56.25
    assert pytest.approx(expectancy_ratio) == 0.60267857

def test_calculate_sortino(testdatadir):
    if False:
        print('Hello World!')
    filename = testdatadir / 'backtest_results/backtest-result.json'
    bt_data = load_backtest_data(filename)
    sortino = calculate_sortino(DataFrame(), None, None, 0)
    assert sortino == 0.0
    sortino = calculate_sortino(bt_data, bt_data['open_date'].min(), bt_data['close_date'].max(), 0.01)
    assert isinstance(sortino, float)
    assert pytest.approx(sortino) == 35.17722

def test_calculate_sharpe(testdatadir):
    if False:
        i = 10
        return i + 15
    filename = testdatadir / 'backtest_results/backtest-result.json'
    bt_data = load_backtest_data(filename)
    sharpe = calculate_sharpe(DataFrame(), None, None, 0)
    assert sharpe == 0.0
    sharpe = calculate_sharpe(bt_data, bt_data['open_date'].min(), bt_data['close_date'].max(), 0.01)
    assert isinstance(sharpe, float)
    assert pytest.approx(sharpe) == 44.5078669

def test_calculate_calmar(testdatadir):
    if False:
        return 10
    filename = testdatadir / 'backtest_results/backtest-result.json'
    bt_data = load_backtest_data(filename)
    calmar = calculate_calmar(DataFrame(), None, None, 0)
    assert calmar == 0.0
    calmar = calculate_calmar(bt_data, bt_data['open_date'].min(), bt_data['close_date'].max(), 0.01)
    assert isinstance(calmar, float)
    assert pytest.approx(calmar) == 559.040508

@pytest.mark.parametrize('start,end,days, expected', [(64900, 176000, 3 * 365, 0.3945), (64900, 176000, 365, 1.7119), (1000, 1000, 365, 0.0), (1000, 1500, 365, 0.5), (1000, 1500, 100, 3.3927), (0.01, 0.01762792, 120, 4.6087)])
def test_calculate_cagr(start, end, days, expected):
    if False:
        i = 10
        return i + 15
    assert round(calculate_cagr(days, start, end), 4) == expected

def test_calculate_max_drawdown2():
    if False:
        i = 10
        return i + 15
    values = [0.01158, 0.010048, 0.01134, 0.012161, 0.010416, 0.010009, 0.020024, -0.024662, -0.02235, 0.020496, -0.029859, -0.030511, 0.010041, 0.010872, -0.025782, 0.0104, 0.012374, 0.012467, 0.114741, 0.010303, 0.010088, -0.033961, 0.01068, 0.010886, -0.029274, 0.011178, 0.010693, 0.010711]
    dates = [dt_utc(2020, 1, 1) + timedelta(days=i) for i in range(len(values))]
    df = DataFrame(zip(values, dates), columns=['profit', 'open_date'])
    df = df.sort_values('profit').reset_index(drop=True)
    df1 = df.copy()
    (drawdown, hdate, ldate, hval, lval, drawdown_rel) = calculate_max_drawdown(df, date_col='open_date', value_col='profit')
    assert df.equals(df1)
    assert isinstance(drawdown, float)
    assert isinstance(drawdown_rel, float)
    assert hdate < ldate
    assert hval > lval
    assert drawdown == 0.091755
    df = DataFrame(zip(values[:5], dates[:5]), columns=['profit', 'open_date'])
    with pytest.raises(ValueError, match='No losing trade, therefore no drawdown.'):
        calculate_max_drawdown(df, date_col='open_date', value_col='profit')

@pytest.mark.parametrize('profits,relative,highd,lowd,result,result_rel', [([0.0, -500.0, 500.0, 10000.0, -1000.0], False, 3, 4, 1000.0, 0.090909), ([0.0, -500.0, 500.0, 10000.0, -1000.0], True, 0, 1, 500.0, 0.5)])
def test_calculate_max_drawdown_abs(profits, relative, highd, lowd, result, result_rel):
    if False:
        i = 10
        return i + 15
    '\n    Test case from issue https://github.com/freqtrade/freqtrade/issues/6655\n    [1000, 500,  1000, 11000, 10000] # absolute results\n    [1000, 50%,  0%,   0%,       ~9%]   # Relative drawdowns\n    '
    init_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
    dates = [init_date + timedelta(days=i) for i in range(len(profits))]
    df = DataFrame(zip(profits, dates), columns=['profit_abs', 'open_date'])
    df = df.sort_values('profit_abs').reset_index(drop=True)
    df1 = df.copy()
    (drawdown, hdate, ldate, hval, lval, drawdown_rel) = calculate_max_drawdown(df, date_col='open_date', starting_balance=1000, relative=relative)
    assert df.equals(df1)
    assert isinstance(drawdown, float)
    assert isinstance(drawdown_rel, float)
    assert hdate == init_date + timedelta(days=highd)
    assert ldate == init_date + timedelta(days=lowd)
    assert hdate < ldate
    assert hval > lval
    assert drawdown == result
    assert pytest.approx(drawdown_rel) == result_rel