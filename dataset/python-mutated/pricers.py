"""Computes option prices using TensorFlow Finance."""
import datetime
import time
import numpy as np
import tensorflow as tf
import tf_quant_finance as tff

class Timer:
    """A simple timer."""

    def __init__(self):
        if False:
            print('Hello World!')
        self.start_time = 0
        self.end_time = 0

    def __enter__(self) -> 'Timer':
        if False:
            return 10
        self.start_time = time.time()
        return self

    def __exit__(self, unused_type, unused_value, unused_traceback):
        if False:
            print('Hello World!')
        del unused_type, unused_value, unused_traceback
        self.end_time = time.time()

    @property
    def elapsed_ms(self) -> float:
        if False:
            i = 10
            return i + 15
        'Returns the elapsed time in milliseconds.'
        return (self.end_time - self.start_time) * 1000

def _price(spot_mkt, vol_mkt, rate_mkt, underliers, strikes, call_put_flag, expiry_ordinals):
    if False:
        i = 10
        return i + 15
    'Prices the options.'
    spots = tf.gather(spot_mkt, underliers)
    vols = tf.gather(vol_mkt, underliers)
    rates = tf.gather(rate_mkt, underliers)
    expiry_ordinals = tf.cast(expiry_ordinals, dtype=tf.int32)
    expiry_dates = tff.datetime.dates_from_ordinals(expiry_ordinals)
    pricing_date = tff.datetime.dates_from_datetimes([datetime.date.today()])
    expiry_times = tff.datetime.daycount_actual_360(start_date=pricing_date, end_date=expiry_dates, dtype=np.float64)
    prices = tff.black_scholes.option_price(volatilities=vols, strikes=strikes, expiries=expiry_times, spots=spots, discount_rates=rates, is_call_options=call_put_flag)
    return prices

class TffOptionPricer:
    """Prices options using TFF."""

    def __init__(self, batch_size=1000000, num_assets=1000):
        if False:
            while True:
                i = 10
        dtype = np.float64
        self._pricer = tf.function(_price)
        if batch_size is not None and num_assets is not None:
            self._pricer(np.zeros([num_assets], dtype=dtype), np.zeros([num_assets], dtype=dtype), np.zeros([num_assets], dtype=dtype), np.zeros([batch_size], dtype=np.int32), np.zeros([batch_size], dtype=dtype), np.zeros([batch_size], dtype=bool), np.ones([batch_size], dtype=np.int32))

    def price(self, spot_mkt, vol_mkt, rate_mkt, underliers, strikes, call_put_flag, expiry_ordinals):
        if False:
            while True:
                i = 10
        'Prices options.'
        prices = self._pricer(spot_mkt, vol_mkt, rate_mkt, underliers, strikes, call_put_flag, expiry_ordinals)
        return prices