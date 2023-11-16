__author__ = 'saeedamen'
import numpy as np
import pandas as pd
from numba import guvectorize
from findatapy.timeseries import Calendar
from findatapy.util import LoggerManager
from finmarketpy.util.marketconstants import MarketConstants
from finmarketpy.curve.abstractpricer import AbstractPricer
from finmarketpy.curve.rates.fxforwardspricer import FXForwardsPricer
from financepy.finutils.FinDate import FinDate
from financepy.models.FinModelBlackScholes import FinModelBlackScholes
from financepy.products.fx.FinFXVanillaOption import FinFXVanillaOption
from financepy.finutils.FinGlobalTypes import FinOptionTypes
from financepy.products.fx.FinFXMktConventions import *
market_constants = MarketConstants()

class FXOptionsPricer(AbstractPricer):
    """Prices various vanilla FX options, using FinancePy underneath.
    """

    def __init__(self, fx_vol_surface=None, premium_output=market_constants.fx_options_premium_output, delta_output=market_constants.fx_options_delta_output):
        if False:
            i = 10
            return i + 15
        self._calendar = Calendar()
        self._fx_vol_surface = fx_vol_surface
        self._fx_forwards_pricer = FXForwardsPricer()
        self._premium_output = premium_output
        self._delta_output = delta_output

    def price_instrument(self, cross, horizon_date, strike, expiry_date=None, vol=None, notional=1000000, contract_type='european-call', tenor=None, fx_vol_surface=None, premium_output=None, delta_output=None, depo_tenor=None, use_atm_quoted=False, return_as_df=True):
        if False:
            return 10
        "Prices FX options for horizon dates/expiry dates given by the user from FX spot rates, FX volatility surface\n        and deposit rates.\n\n        Parameters\n        ----------\n        cross : str\n            Currency pair\n\n        horizon_date : DateTimeIndex\n            Horizon dates for options\n\n        strike : np.ndarray, float or str\n            Strike of option\n\n            eg. 'atm' - at-the-money\n            eg. 'atmf' - at-the-money forward\n            eg. 'atms' - at-the-money spot\n            eg. '25d-otm' - out-of-the-money 25d\n            eg. '10d-otm\n\n        expiry_date : DateTimeIndex (optional)\n            Expiry dates for options\n\n        vol : np.ndarray (optional)\n            Umplied vol for options\n\n        notional : float\n            Notional in base currency of the option\n\n        contract_type : str\n            What type of option are we pricing?\n\n            eg. 'european-call'\n\n        tenor : str (optional)\n            Tenor of option\n\n        fx_vol_surface : FXVolSurface\n            Interpolates FX vol surface\n\n        premium_output : str\n            'pct-for' (in base currency pct) or 'pct-dom' (in terms currency pct)\n\n        delta_output : bool\n            Also output delta of options\n\n        depo_tenor : str\n            Tenor of the deposit to use in the option pricing\n\n        use_atm_quoted : bool\n            True - takes the direct market quote\n            False - uses interpolated version\n\n        return_as_df : bool\n            True - returns output as DataFrame\n            False - returns output as np.ndarray\n\n        Returns\n        -------\n        DataFrame\n        "
        if fx_vol_surface is None:
            fx_vol_surface = self._fx_vol_surface
        if premium_output is None:
            premium_output = self._premium_output
        if delta_output is None:
            delta_output = self._delta_output
        logger = LoggerManager().getLogger(__name__)
        field = fx_vol_surface._field
        if isinstance(horizon_date, pd.Timestamp):
            horizon_date = pd.DatetimeIndex([horizon_date])
        else:
            horizon_date = pd.DatetimeIndex(horizon_date)
        if expiry_date is not None:
            if isinstance(expiry_date, pd.Timestamp):
                expiry_date = pd.DatetimeIndex([expiry_date])
            else:
                expiry_date = pd.DatetimeIndex(expiry_date)
        else:
            expiry_date = self._calendar.get_expiry_date_from_horizon_date(horizon_date, tenor, cal=cross)
        if not isinstance(strike, np.ndarray):
            old_strike = strike
            if isinstance(strike, str):
                strike = np.empty(len(horizon_date), dtype=object)
            else:
                strike = np.empty(len(horizon_date))
            strike.fill(old_strike)
        if not isinstance(vol, np.ndarray):
            if vol is None:
                vol = np.nan
            old_vol = vol
            vol = np.empty(len(horizon_date))
            vol.fill(old_vol)
        option_values = np.zeros(len(horizon_date))
        spot = np.zeros(len(horizon_date))
        delta = np.zeros(len(horizon_date))
        intrinsic_values = np.zeros(len(horizon_date))

        def _price_option(contract_type_, contract_type_fin_):
            if False:
                print('Hello World!')
            for i in range(len(expiry_date)):
                built_vol_surface = False
                if isinstance(strike[i], str):
                    if not built_vol_surface:
                        fx_vol_surface.build_vol_surface(horizon_date[i])
                        fx_vol_surface.extract_vol_surface(num_strike_intervals=None)
                        built_vol_surface = True
                    if strike[i] == 'atm':
                        strike[i] = fx_vol_surface.get_atm_strike(tenor)
                        if use_atm_quoted:
                            vol[i] = fx_vol_surface.get_atm_quoted_vol(tenor) / 100.0
                        else:
                            vol[i] = fx_vol_surface.get_atm_vol(tenor) / 100.0
                    elif strike[i] == 'atms':
                        strike[i] = fx_vol_surface.get_spot()
                    elif strike[i] == 'atmf':
                        strike[i] = float(fx_vol_surface.get_all_market_data()[cross + '.close'][horizon_date[i]]) + float(fx_vol_surface.get_all_market_data()[cross + tenor + '.close'][horizon_date[i]]) / self._fx_forwards_pricer.get_forwards_divisor(cross[3:6])
                    elif strike[i] == '25d-otm':
                        if 'call' in contract_type_:
                            strike[i] = fx_vol_surface.get_25d_call_strike(tenor)
                            vol[i] = fx_vol_surface.get_25d_call_vol(tenor) / 100.0
                        elif 'put' in contract_type_:
                            strike[i] = fx_vol_surface.get_25d_put_strike(tenor)
                            vol[i] = fx_vol_surface.get_25d_put_vol(tenor) / 100.0
                    elif strike[i] == '10d-otm':
                        if 'call' in contract_type_:
                            strike[i] = fx_vol_surface.get_10d_call_strike(tenor)
                            vol[i] = fx_vol_surface.get_10d_call_vol(tenor) / 100.0
                        elif 'put' in contract_type_:
                            strike[i] = fx_vol_surface.get_10d_put_strike(tenor)
                            vol[i] = fx_vol_surface.get_10d_put_vol(tenor) / 100.0
                if not built_vol_surface:
                    try:
                        fx_vol_surface.build_vol_surface(horizon_date[i])
                    except:
                        logger.warn('Failed to build vol surface for ' + str(horizon_date) + ", won't be able to interpolate vol")
                if np.isnan(vol[i]):
                    if tenor is None:
                        vol[i] = fx_vol_surface.calculate_vol_for_strike_expiry(strike[i], expiry_date=expiry_date[i], tenor=None)
                    else:
                        vol[i] = fx_vol_surface.calculate_vol_for_strike_expiry(strike[i], expiry_date=None, tenor=tenor)
                model = FinModelBlackScholes(float(vol[i]))
                logger.info('Pricing ' + contract_type_ + ' option, horizon date = ' + str(horizon_date[i]) + ', expiry date = ' + str(expiry_date[i]))
                option = FinFXVanillaOption(self._findate(expiry_date[i]), strike[i], cross, contract_type_fin_, notional, cross[0:3])
                spot[i] = fx_vol_surface.get_spot()
                ' FinancePy will return the value in the following dictionary for values\n                    {\'v\': vdf,\n                    "cash_dom": cash_dom,\n                    "cash_for": cash_for,\n                    "pips_dom": pips_dom,\n                    "pips_for": pips_for,\n                    "pct_dom": pct_dom,\n                    "pct_for": pct_for,\n                    "not_dom": notional_dom,\n                    "not_for": notional_for,\n                    "ccy_dom": self._domName,\n                    "ccy_for": self._forName}\n                '
                option_values[i] = option_values[i] + option.value(self._findate(horizon_date[i]), spot[i], fx_vol_surface.get_dom_discount_curve(), fx_vol_surface.get_for_discount_curve(), model)[premium_output.replace('-', '_')]
                intrinsic_values[i] = intrinsic_values[i] + option.value(self._findate(expiry_date[i]), spot[i], fx_vol_surface.get_dom_discount_curve(), fx_vol_surface.get_for_discount_curve(), model)[premium_output.replace('-', '_')]
                'FinancePy returns this dictionary for deltas\n                    {"pips_spot_delta": pips_spot_delta,\n                    "pips_fwd_delta": pips_fwd_delta,\n                    "pct_spot_delta_prem_adj": pct_spot_delta_prem_adj,\n                    "pct_fwd_delta_prem_adj": pct_fwd_delta_prem_adj}\n                '
                delta[i] = delta[i] + option.delta(self._findate(horizon_date[i]), spot[i], fx_vol_surface.get_dom_discount_curve(), fx_vol_surface.get_for_discount_curve(), model)[delta_output.replace('-', '_')]
        if contract_type == 'european-call':
            contract_type_fin = FinOptionTypes.EUROPEAN_CALL
            _price_option(contract_type, contract_type_fin)
        elif contract_type == 'european-put':
            contract_type_fin = FinOptionTypes.EUROPEAN_PUT
            _price_option(contract_type, contract_type_fin)
        elif contract_type == 'european-straddle' or contract_type == 'european-strangle':
            contract_type = 'european-call'
            contract_type_fin = FinOptionTypes.EUROPEAN_CALL
            _price_option(contract_type, contract_type_fin)
            contract_type = 'european-put'
            contract_type_fin = FinOptionTypes.EUROPEAN_PUT
            _price_option(contract_type, contract_type_fin)
        if return_as_df:
            option_prices_df = pd.DataFrame(index=horizon_date)
            option_prices_df[cross + '-option-price.' + field] = option_values
            option_prices_df[cross + '.' + field] = spot
            option_prices_df[cross + '-strike.' + field] = strike
            option_prices_df[cross + '-vol.' + field] = vol
            option_prices_df[cross + '-delta.' + field] = delta
            option_prices_df[cross + '.expiry-date'] = expiry_date
            option_prices_df[cross + '-intrinsic-value.' + field] = intrinsic_values
            return option_prices_df
        return (option_values, spot, strike, vol, delta, expiry_date, intrinsic_values)

    def get_day_count_conv(self, currency):
        if False:
            for i in range(10):
                print('nop')
        if currency in market_constants.currencies_with_365_basis:
            return 365.0
        return 360.0

    def _findate(self, timestamp):
        if False:
            while True:
                i = 10
        return FinDate(timestamp.day, timestamp.month, timestamp.year, hh=timestamp.hour, mm=timestamp.minute, ss=timestamp.second)