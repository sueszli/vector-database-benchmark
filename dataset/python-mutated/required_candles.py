import arrow
import numpy as np
import jesse.helpers as jh
from jesse.config import config
from jesse.exceptions import CandleNotFoundInDatabase
from jesse.models import Candle
from jesse.services.cache import cache
from jesse.services.candle import generate_candle_from_one_minutes
from jesse.store import store

def load_required_candles(exchange: str, symbol: str, start_date_str: str, finish_date_str: str) -> np.ndarray:
    if False:
        print('Hello World!')
    '\n    loads initial candles that required before executing strategies.\n    210 for the biggest timeframe and more for the rest\n    '
    start_date = jh.arrow_to_timestamp(arrow.get(start_date_str, 'YYYY-MM-DD'))
    finish_date = jh.arrow_to_timestamp(arrow.get(finish_date_str, 'YYYY-MM-DD')) - 60000
    if start_date == finish_date:
        raise ValueError('start_date and finish_date cannot be the same.')
    if start_date > finish_date:
        raise ValueError('start_date cannot be bigger than finish_date.')
    if finish_date > arrow.utcnow().int_timestamp * 1000:
        raise ValueError("Can't backtest the future!")
    max_timeframe = jh.max_timeframe(config['app']['considering_timeframes'])
    short_candles_count = jh.get_config('env.data.warmup_candles_num', 210) * jh.timeframe_to_one_minutes(max_timeframe)
    pre_finish_date = start_date - 60000
    pre_start_date = pre_finish_date - short_candles_count * 60000
    pre_start_date = jh.timestamp_to_arrow(pre_start_date).floor('day').int_timestamp * 1000
    short_candles_count = int((pre_finish_date - pre_start_date) / 60000)
    key = jh.key(exchange, symbol)
    cache_key = f'{jh.timestamp_to_date(pre_start_date)}-{jh.timestamp_to_date(pre_finish_date)}-{key}'
    cached_value = cache.get_value(cache_key)
    if cached_value:
        candles_tuple = cached_value
    else:
        candles_tuple = tuple(Candle.select(Candle.timestamp, Candle.open, Candle.close, Candle.high, Candle.low, Candle.volume).where(Candle.exchange == exchange, Candle.symbol == symbol, Candle.timeframe == '1m' or Candle.timeframe.is_null(), Candle.timestamp.between(pre_start_date, pre_finish_date)).order_by(Candle.timestamp.asc()).tuples())
        cache.set_value(cache_key, candles_tuple, expire_seconds=60 * 60 * 24 * 7)
    candles = np.array(candles_tuple)
    if len(candles) < short_candles_count + 1:
        first_existing_candle = tuple(Candle.select(Candle.timestamp).where(Candle.exchange == exchange, Candle.symbol == symbol, Candle.timeframe == '1m' or Candle.timeframe.is_null()).order_by(Candle.timestamp.asc()).limit(1).tuples())
        if not len(first_existing_candle):
            raise CandleNotFoundInDatabase(f'No candle for {exchange} {symbol} is present in the database. Try importing candles.')
        first_existing_candle = first_existing_candle[0][0]
        last_existing_candle = tuple(Candle.select(Candle.timestamp).where(Candle.exchange == exchange, Candle.symbol == symbol, Candle.timeframe == '1m' or Candle.timeframe.is_null()).order_by(Candle.timestamp.desc()).limit(1).tuples())[0][0]
        first_backtestable_timestamp = first_existing_candle + (pre_finish_date - pre_start_date) + 60000 * 1440
        if first_backtestable_timestamp > jh.today_to_timestamp():
            raise CandleNotFoundInDatabase(f'Not enough candle for {exchange} {symbol} is present in the database. Jesse requires "210 * biggest_timeframe" warm-up candles. Try importing more candles from an earlier date.')
        raise CandleNotFoundInDatabase(f'Not enough candles for {exchange} {symbol} exists to run backtest from {start_date_str} => {finish_date_str}. \nAre you considering the warmup candles? For more info please read:\n https://jesse.trade/help/faq/i-imported-candles-but-keep-getting-not-enough-candles')
    return candles

def inject_required_candles_to_store(candles: np.ndarray, exchange: str, symbol: str) -> None:
    if False:
        while True:
            i = 10
    '\n    generate and add required candles to the candle store\n    '
    store.candles.batch_add_candle(candles, exchange, symbol, '1m', with_generation=False)
    for i in range(len(candles)):
        for timeframe in config['app']['considering_timeframes']:
            if timeframe == '1m':
                continue
            num = jh.timeframe_to_one_minutes(timeframe)
            if (i + 1) % num == 0:
                generated_candle = generate_candle_from_one_minutes(timeframe, candles[i - (num - 1):i + 1], True)
                store.candles.add_candle(generated_candle, exchange, symbol, timeframe, with_execution=False, with_generation=False)