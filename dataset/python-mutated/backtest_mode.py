import time
from typing import Dict, Union, List
import arrow
import numpy as np
import pandas as pd
import jesse.helpers as jh
import jesse.services.metrics as stats
import jesse.services.required_candles as required_candles
import jesse.services.selectors as selectors
from jesse import exceptions
from jesse.config import config
from jesse.enums import timeframes, order_types
from jesse.models import Candle, Order, Position
from jesse.modes.utils import save_daily_portfolio_balance
from jesse.routes import router
from jesse.services import charts
from jesse.services import quantstats
from jesse.services import report
from jesse.services.cache import cache
from jesse.services.candle import generate_candle_from_one_minutes, print_candle, candle_includes_price, split_candle
from jesse.services.file import store_logs
from jesse.services.validators import validate_routes
from jesse.store import store
from jesse.services import logger
from jesse.services.failure import register_custom_exception_handler
from jesse.services.redis import sync_publish, process_status
from timeloop import Timeloop
from datetime import timedelta
from jesse.services.progressbar import Progressbar

def run(debug_mode, user_config: dict, routes: List[Dict[str, str]], extra_routes: List[Dict[str, str]], start_date: str, finish_date: str, candles: dict=None, chart: bool=False, tradingview: bool=False, full_reports: bool=False, csv: bool=False, json: bool=False) -> None:
    if False:
        return 10
    if not jh.is_unit_testing():
        status_checker = Timeloop()

        @status_checker.job(interval=timedelta(seconds=1))
        def handle_time():
            if False:
                for i in range(10):
                    print('nop')
            if process_status() != 'started':
                raise exceptions.Termination
        status_checker.start()
    from jesse.config import config, set_config
    config['app']['trading_mode'] = 'backtest'
    config['app']['debug_mode'] = debug_mode
    if not jh.is_unit_testing():
        set_config(user_config)
    router.initiate(routes, extra_routes)
    store.app.set_session_id()
    register_custom_exception_handler()
    validate_routes(router)
    store.candles.init_storage(5000)
    if candles is None:
        candles = load_candles(start_date, finish_date)
    if not jh.should_execute_silently():
        sync_publish('general_info', {'session_id': jh.get_session_id(), 'debug_mode': str(config['app']['debug_mode'])})
        key = f"{config['app']['considering_candles'][0][0]}-{config['app']['considering_candles'][0][1]}"
        sync_publish('candles_info', stats.candles_info(candles[key]['candles']))
        sync_publish('routes_info', stats.routes(router.routes))
    result = simulator(candles, run_silently=jh.should_execute_silently(), generate_charts=chart, generate_tradingview=tradingview, generate_quantstats=full_reports, generate_csv=csv, generate_json=json, generate_equity_curve=True, generate_hyperparameters=True)
    if not jh.should_execute_silently():
        sync_publish('alert', {'message': f"Successfully executed backtest simulation in: {result['execution_duration']} seconds", 'type': 'success'})
        sync_publish('hyperparameters', result['hyperparameters'])
        sync_publish('metrics', result['metrics'])
        sync_publish('equity_curve', result['equity_curve'])
    from jesse.services.db import database
    database.close_connection()

def _generate_quantstats_report(candles_dict: dict) -> str:
    if False:
        while True:
            i = 10
    if store.completed_trades.count == 0:
        return None
    price_data = []
    timestamps = []
    for (index, c) in enumerate(config['app']['considering_candles']):
        (exchange, symbol) = (c[0], c[1])
        if exchange in config['app']['trading_exchanges'] and symbol in config['app']['trading_symbols']:
            candles = candles_dict[jh.key(exchange, symbol)]['candles']
            if timestamps == []:
                timestamps = candles[:, 0]
            price_data.append(candles[:, 1])
    price_data = np.transpose(price_data)
    price_df = pd.DataFrame(price_data, index=pd.to_datetime(timestamps, unit='ms'), dtype=float).resample('D').mean()
    price_pct_change = price_df.pct_change(1).fillna(0)
    buy_and_hold_daily_returns_all_routes = price_pct_change.mean(1)
    study_name = _get_study_name()
    res = quantstats.quantstats_tearsheet(buy_and_hold_daily_returns_all_routes, study_name)
    return res

def _get_study_name() -> str:
    if False:
        return 10
    routes_count = len(router.routes)
    more = f'-and-{routes_count - 1}-more' if routes_count > 1 else ''
    if type(router.routes[0].strategy_name) is str:
        strategy_name = router.routes[0].strategy_name
    else:
        strategy_name = router.routes[0].strategy_name.__name__
    study_name = f'{strategy_name}-{router.routes[0].exchange}-{router.routes[0].symbol}-{router.routes[0].timeframe}{more}'
    return study_name

def load_candles(start_date_str: str, finish_date_str: str) -> Dict[str, Dict[str, Union[str, np.ndarray]]]:
    if False:
        i = 10
        return i + 15
    start_date = jh.date_to_timestamp(start_date_str)
    finish_date = jh.date_to_timestamp(finish_date_str) - 60000
    if start_date == finish_date:
        raise ValueError('start_date and finish_date cannot be the same.')
    if start_date > finish_date:
        raise ValueError('start_date cannot be bigger than finish_date.')
    if finish_date > arrow.utcnow().int_timestamp * 1000:
        raise ValueError("Can't load candle data from the future! The finish-date can be up to yesterday's date at most.")
    if jh.is_backtesting():
        for c in config['app']['considering_candles']:
            (exchange, symbol) = (c[0], c[1])
            required_candles.inject_required_candles_to_store(required_candles.load_required_candles(exchange, symbol, start_date_str, finish_date_str), exchange, symbol)
    candles = {}
    for c in config['app']['considering_candles']:
        (exchange, symbol) = (c[0], c[1])
        key = jh.key(exchange, symbol)
        cache_key = f'{start_date_str}-{finish_date_str}-{key}'
        cached_value = cache.get_value(cache_key)
        candles_tuple = cached_value or Candle.select(Candle.timestamp, Candle.open, Candle.close, Candle.high, Candle.low, Candle.volume).where(Candle.exchange == exchange, Candle.symbol == symbol, Candle.timeframe == '1m' or Candle.timeframe.is_null(), Candle.timestamp.between(start_date, finish_date)).order_by(Candle.timestamp.asc()).tuples()
        required_candles_count = (finish_date - start_date) / 60000
        if len(candles_tuple) == 0 or candles_tuple[-1][0] != finish_date or candles_tuple[0][0] != start_date:
            raise exceptions.CandleNotFoundInDatabase(f'Not enough candles for {symbol}. You need to import candles.')
        elif len(candles_tuple) != required_candles_count + 1:
            raise exceptions.CandleNotFoundInDatabase(f'There are missing candles between {start_date_str} => {finish_date_str}')
        cache.set_value(cache_key, tuple(candles_tuple), expire_seconds=60 * 60 * 24 * 7)
        candles[key] = {'exchange': exchange, 'symbol': symbol, 'candles': np.array(candles_tuple)}
    return candles

def simulator(candles: dict, run_silently: bool, hyperparameters: dict=None, generate_charts: bool=False, generate_tradingview: bool=False, generate_quantstats: bool=False, generate_csv: bool=False, generate_json: bool=False, generate_equity_curve: bool=False, generate_hyperparameters: bool=False) -> dict:
    if False:
        while True:
            i = 10
    result = {}
    begin_time_track = time.time()
    key = f"{config['app']['considering_candles'][0][0]}-{config['app']['considering_candles'][0][1]}"
    first_candles_set = candles[key]['candles']
    length = len(first_candles_set)
    try:
        store.app.starting_time = first_candles_set[0][0]
    except IndexError:
        raise IndexError('Check your "warm_up_candles" config value')
    store.app.time = first_candles_set[0][0]
    for r in router.routes:
        if isinstance(r.strategy_name, str):
            StrategyClass = jh.get_strategy_class(r.strategy_name)
        else:
            StrategyClass = r.strategy_name
        try:
            r.strategy = StrategyClass()
        except TypeError:
            raise exceptions.InvalidStrategy('Looks like the structure of your strategy directory is incorrect. Make sure to include the strategy INSIDE the __init__.py file. Another reason for this error might be that your strategy is missing the mandatory methods such as should_long(), go_long(), and should_cancel_entry(). \nIf you need working examples, check out: https://github.com/jesse-ai/example-strategies')
        except:
            raise
        r.strategy.name = r.strategy_name
        r.strategy.exchange = r.exchange
        r.strategy.symbol = r.symbol
        r.strategy.timeframe = r.timeframe
        if len(r.strategy.dna()) > 0 and hyperparameters is None:
            hyperparameters = jh.dna_to_hp(r.strategy.hyperparameters(), r.strategy.dna())
        if hyperparameters is not None:
            r.strategy.hp = hyperparameters
        r.strategy._init_objects()
        selectors.get_position(r.exchange, r.symbol).strategy = r.strategy
    save_daily_portfolio_balance()
    progressbar = Progressbar(length, step=60)
    for i in range(length):
        store.app.time = first_candles_set[i][0] + 60000
        for j in candles:
            short_candle = candles[j]['candles'][i]
            if i != 0:
                previous_short_candle = candles[j]['candles'][i - 1]
                short_candle = _get_fixed_jumped_candle(previous_short_candle, short_candle)
            exchange = candles[j]['exchange']
            symbol = candles[j]['symbol']
            store.candles.add_candle(short_candle, exchange, symbol, '1m', with_execution=False, with_generation=False)
            if jh.is_debuggable('shorter_period_candles'):
                print_candle(short_candle, True, symbol)
            _simulate_price_change_effect(short_candle, exchange, symbol)
            for timeframe in config['app']['considering_timeframes']:
                if timeframe == '1m':
                    continue
                count = jh.timeframe_to_one_minutes(timeframe)
                if (i + 1) % count == 0:
                    generated_candle = generate_candle_from_one_minutes(timeframe, candles[j]['candles'][i - (count - 1):i + 1])
                    store.candles.add_candle(generated_candle, exchange, symbol, timeframe, with_execution=False, with_generation=False)
        if not run_silently and i % 60 == 0:
            progressbar.update()
            sync_publish('progressbar', {'current': progressbar.current, 'estimated_remaining_seconds': progressbar.estimated_remaining_seconds})
        for r in router.routes:
            count = jh.timeframe_to_one_minutes(r.timeframe)
            if r.timeframe == timeframes.MINUTE_1:
                r.strategy._execute()
            elif (i + 1) % count == 0:
                if jh.is_debuggable('trading_candles'):
                    print_candle(store.candles.get_current_candle(r.exchange, r.symbol, r.timeframe), False, r.symbol)
                r.strategy._execute()
        store.orders.execute_pending_market_orders()
        if i != 0 and i % 1440 == 0:
            save_daily_portfolio_balance()
    if not run_silently:
        finish_time_track = time.time()
        result['execution_duration'] = round(finish_time_track - begin_time_track, 2)
    for r in router.routes:
        r.strategy._terminate()
        store.orders.execute_pending_market_orders()
    save_daily_portfolio_balance()
    if generate_hyperparameters:
        result['hyperparameters'] = stats.hyperparameters(router.routes)
    result['metrics'] = report.portfolio_metrics()
    logs_path = store_logs(generate_json, generate_tradingview, generate_csv)
    if generate_json:
        result['json'] = logs_path['json']
    if generate_tradingview:
        result['tradingview'] = logs_path['tradingview']
    if generate_csv:
        result['csv'] = logs_path['csv']
    if generate_charts:
        result['charts'] = charts.portfolio_vs_asset_returns(_get_study_name())
    if generate_equity_curve:
        result['equity_curve'] = charts.equity_curve()
    if generate_quantstats:
        result['quantstats'] = _generate_quantstats_report(candles)
    return result

def _get_fixed_jumped_candle(previous_candle: np.ndarray, candle: np.ndarray) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    "\n    A little workaround for the times that the price has jumped and the opening\n    price of the current candle is not equal to the previous candle's close!\n\n    :param previous_candle: np.ndarray\n    :param candle: np.ndarray\n    "
    if previous_candle[2] < candle[1]:
        candle[1] = previous_candle[2]
        candle[4] = min(previous_candle[2], candle[4])
    elif previous_candle[2] > candle[1]:
        candle[1] = previous_candle[2]
        candle[3] = max(previous_candle[2], candle[3])
    return candle

def _simulate_price_change_effect(real_candle: np.ndarray, exchange: str, symbol: str) -> None:
    if False:
        while True:
            i = 10
    orders = store.orders.get_orders(exchange, symbol)
    current_temp_candle = real_candle.copy()
    executed_order = False
    while True:
        if len(orders) == 0:
            executed_order = False
        else:
            for (index, order) in enumerate(orders):
                if index == len(orders) - 1 and (not order.is_active):
                    executed_order = False
                if not order.is_active:
                    continue
                if candle_includes_price(current_temp_candle, order.price):
                    (storable_temp_candle, current_temp_candle) = split_candle(current_temp_candle, order.price)
                    store.candles.add_candle(storable_temp_candle, exchange, symbol, '1m', with_execution=False, with_generation=False)
                    p = selectors.get_position(exchange, symbol)
                    p.current_price = storable_temp_candle[2]
                    executed_order = True
                    order.execute()
                    break
                else:
                    executed_order = False
        if not executed_order:
            store.candles.add_candle(real_candle, exchange, symbol, '1m', with_execution=False, with_generation=False)
            p = selectors.get_position(exchange, symbol)
            if p:
                p.current_price = real_candle[2]
            break
    _check_for_liquidations(real_candle, exchange, symbol)

def _check_for_liquidations(candle: np.ndarray, exchange: str, symbol: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    p: Position = selectors.get_position(exchange, symbol)
    if not p:
        return
    if p.mode != 'isolated':
        return
    if candle_includes_price(candle, p.liquidation_price):
        closing_order_side = jh.closing_side(p.type)
        order = Order({'id': jh.generate_unique_id(), 'symbol': symbol, 'exchange': exchange, 'side': closing_order_side, 'type': order_types.MARKET, 'reduce_only': True, 'qty': jh.prepare_qty(p.qty, closing_order_side), 'price': p.bankruptcy_price})
        store.orders.add_order(order)
        store.app.total_liquidations += 1
        logger.info(f'{p.symbol} liquidated at {p.liquidation_price}')
        order.execute()