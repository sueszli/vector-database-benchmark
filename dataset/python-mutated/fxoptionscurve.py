__author__ = 'saeedamen'
import numpy as np
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay, CustomBusinessMonthEnd
from findatapy.market import Market, MarketDataRequest
from findatapy.timeseries import Calculations, Calendar, Filter
from findatapy.util.dataconstants import DataConstants
from findatapy.util.fxconv import FXConv
from finmarketpy.curve.volatility.fxoptionspricer import FXOptionsPricer
from finmarketpy.curve.volatility.fxvolsurface import FXVolSurface
from finmarketpy.util.marketconstants import MarketConstants
data_constants = DataConstants()
market_constants = MarketConstants()

class FXOptionsCurve(object):
    """Constructs continuous forwards time series total return indices from underlying forwards contracts.

    """

    def __init__(self, market_data_generator=None, fx_vol_surface=None, enter_trading_dates=None, fx_options_trading_tenor=market_constants.fx_options_trading_tenor, roll_days_before=market_constants.fx_options_roll_days_before, roll_event=market_constants.fx_options_roll_event, construct_via_currency='no', fx_options_tenor_for_interpolation=market_constants.fx_options_tenor_for_interpolation, base_depos_tenor=data_constants.base_depos_tenor, roll_months=market_constants.fx_options_roll_months, cum_index=market_constants.fx_options_cum_index, strike=market_constants.fx_options_index_strike, contract_type=market_constants.fx_options_index_contract_type, premium_output=market_constants.fx_options_index_premium_output, position_multiplier=1, depo_tenor_for_option=market_constants.fx_options_depo_tenor, freeze_implied_vol=market_constants.fx_options_freeze_implied_vol, tot_label='', cal=None, output_calculation_fields=market_constants.output_calculation_fields):
        if False:
            for i in range(10):
                print('nop')
        "Initializes FXForwardsCurve\n\n        Parameters\n        ----------\n        market_data_generator : MarketDataGenerator\n            Used for downloading market data\n\n        fx_vol_surface : FXVolSurface\n            We can specify the FX vol surface beforehand if we want\n\n        fx_options_trading_tenor : str\n            What is primary forward contract being used to trade (default - '1M')\n\n        roll_days_before : int\n            Number of days before roll event to enter into a new forwards contract\n\n        roll_event : str\n            What constitutes a roll event? ('month-end', 'quarter-end', 'year-end', 'expiry')\n\n        cum_index : str\n            In total return index, do we compute in additive or multiplicative way ('add' or 'mult')\n\n        construct_via_currency : str\n            What currency should we construct the forward via? Eg. if we asked for AUDJPY we can construct it via\n            AUDUSD & JPYUSD forwards, as opposed to AUDJPY forwards (default - 'no')\n\n        fx_options_tenor_for_interpolation : str(list)\n            Which forwards should we use for interpolation\n\n        base_depos_tenor : str(list)\n            Which base deposits tenors do we need (this is only necessary if we want to start inferring depos)\n\n        roll_months : int\n            After how many months should we initiate a roll. Typically for trading 1M this should 1, 3M this should be 3\n            etc.\n\n        tot_label : str\n            Postfix for the total returns field\n\n        cal : str\n            Calendar to use for expiry (if None, uses that of FX pair)\n\n        output_calculation_fields : bool\n            Also output additional data should forward expiries etc. alongside total returns indices\n        "
        self._market_data_generator = market_data_generator
        self._calculations = Calculations()
        self._calendar = Calendar()
        self._filter = Filter()
        self._fx_vol_surface = fx_vol_surface
        self._enter_trading_dates = enter_trading_dates
        self._fx_options_trading_tenor = fx_options_trading_tenor
        self._roll_days_before = roll_days_before
        self._roll_event = roll_event
        self._construct_via_currency = construct_via_currency
        self._fx_options_tenor_for_interpolation = fx_options_tenor_for_interpolation
        self._base_depos_tenor = base_depos_tenor
        self._roll_months = roll_months
        self._cum_index = cum_index
        self._contact_type = contract_type
        self._strike = strike
        self._premium_output = premium_output
        self._position_multiplier = position_multiplier
        self._depo_tenor_for_option = depo_tenor_for_option
        self._freeze_implied_vol = freeze_implied_vol
        self._tot_label = tot_label
        self._cal = cal
        self._output_calculation_fields = output_calculation_fields

    def generate_key(self):
        if False:
            while True:
                i = 10
        from findatapy.market.ioengine import SpeedCache
        return SpeedCache().generate_key(self, ['_market_data_generator', '_calculations', '_calendar', '_filter'])

    def fetch_continuous_time_series(self, md_request, market_data_generator, fx_vol_surface=None, enter_trading_dates=None, fx_options_trading_tenor=None, roll_days_before=None, roll_event=None, construct_via_currency=None, fx_options_tenor_for_interpolation=None, base_depos_tenor=None, roll_months=None, cum_index=None, strike=None, contract_type=None, premium_output=None, position_multiplier=None, depo_tenor_for_option=None, freeze_implied_vol=None, tot_label=None, cal=None, output_calculation_fields=None):
        if False:
            for i in range(10):
                print('nop')
        if fx_vol_surface is None:
            fx_vol_surface = self._fx_vol_surface
        if enter_trading_dates is None:
            enter_trading_dates = self._enter_trading_dates
        if market_data_generator is None:
            market_data_generator = self._market_data_generator
        if fx_options_trading_tenor is None:
            fx_options_trading_tenor = self._fx_options_trading_tenor
        if roll_days_before is None:
            roll_days_before = self._roll_days_before
        if roll_event is None:
            roll_event = self._roll_event
        if construct_via_currency is None:
            construct_via_currency = self._construct_via_currency
        if fx_options_tenor_for_interpolation is None:
            fx_options_tenor_for_interpolation = self._fx_options_tenor_for_interpolation
        if base_depos_tenor is None:
            base_depos_tenor = self._base_depos_tenor
        if roll_months is None:
            roll_months = self._roll_months
        if strike is None:
            strike = self._strike
        if contract_type is None:
            contract_type = self._contact_type
        if premium_output is None:
            premium_output = self._premium_output
        if position_multiplier is None:
            position_multiplier = self._position_multiplier
        if depo_tenor_for_option is None:
            depo_tenor_for_option = self._depo_tenor_for_option
        if freeze_implied_vol is None:
            freeze_implied_vol = self._freeze_implied_vol
        if tot_label is None:
            tot_label = self._tot_label
        if cal is None:
            cal = self._cal
        if output_calculation_fields is None:
            output_calculation_fields = self._output_calculation_fields
        if construct_via_currency == 'no':
            if fx_vol_surface is None:
                market = Market(market_data_generator=market_data_generator)
                md_request_download = MarketDataRequest(md_request=md_request)
                fx_conv = FXConv()
                md_request_download.tickers = [fx_conv.correct_notation(x) for x in md_request.tickers]
                md_request_download.category = 'fx-vol-market'
                md_request_download.fields = 'close'
                md_request_download.abstract_curve = None
                md_request_download.fx_options_tenor = fx_options_tenor_for_interpolation
                md_request_download.base_depos_tenor = base_depos_tenor
                forwards_market_df = market.fetch_market(md_request_download)
            else:
                forwards_market_df = None
            return self.construct_total_return_index(md_request.tickers, forwards_market_df, fx_vol_surface=fx_vol_surface, enter_trading_dates=enter_trading_dates, fx_options_trading_tenor=fx_options_trading_tenor, roll_days_before=roll_days_before, roll_event=roll_event, fx_options_tenor_for_interpolation=fx_options_tenor_for_interpolation, roll_months=roll_months, cum_index=cum_index, strike=strike, contract_type=contract_type, premium_output=premium_output, position_multiplier=position_multiplier, freeze_implied_vol=freeze_implied_vol, depo_tenor_for_option=depo_tenor_for_option, tot_label=tot_label, cal=cal, output_calculation_fields=output_calculation_fields)
        else:
            total_return_indices = []
            for tick in md_request.tickers:
                base = tick[0:3]
                terms = tick[3:6]
                md_request_base = MarketDataRequest(md_request=md_request)
                md_request_base.tickers = base + construct_via_currency
                md_request_terms = MarketDataRequest(md_request=md_request)
                md_request_terms.tickers = terms + construct_via_currency
                base_vals = self.fetch_continuous_time_series(md_request_base, market_data_generator, fx_vol_surface=fx_vol_surface, enter_trading_dates=enter_trading_dates, fx_options_trading_tenor=fx_options_trading_tenor, roll_days_before=roll_days_before, roll_event=roll_event, fx_options_tenor_for_interpolation=fx_options_tenor_for_interpolation, base_depos_tenor=base_depos_tenor, roll_months=roll_months, cum_index=cum_index, strike=strike, contract_type=contract_type, premium_output=premium_output, position_multiplier=position_multiplier, depo_tenor_for_option=depo_tenor_for_option, freeze_implied_vol=freeze_implied_vol, tot_label=tot_label, cal=cal, output_calculation_fields=output_calculation_fields, construct_via_currency='no')
                terms_vals = self.fetch_continuous_time_series(md_request_terms, market_data_generator, fx_vol_surface=fx_vol_surface, enter_trading_dates=enter_trading_dates, fx_options_trading_tenor=fx_options_trading_tenor, roll_days_before=roll_days_before, roll_event=roll_event, fx_options_tenor_for_interpolation=fx_options_tenor_for_interpolation, base_depos_tenor=base_depos_tenor, roll_months=roll_months, cum_index=cum_index, strike=strike, contract_type=contract_type, position_multiplier=position_multiplier, depo_tenor_for_option=depo_tenor_for_option, freeze_implied_vol=freeze_implied_vol, tot_label=tot_label, cal=cal, output_calculation_fields=output_calculation_fields, construct_via_currency='no')
                if base + terms == construct_via_currency + construct_via_currency:
                    base_rets = self._calculations.calculate_returns(base_vals)
                    cross_rets = pd.DataFrame(0, index=base_rets.index, columns=base_rets.columns)
                elif base + construct_via_currency == construct_via_currency + construct_via_currency:
                    cross_rets = -self._calculations.calculate_returns(terms_vals)
                elif terms + construct_via_currency == construct_via_currency + construct_via_currency:
                    cross_rets = self._calculations.calculate_returns(base_vals)
                else:
                    base_rets = self._calculations.calculate_returns(base_vals)
                    terms_rets = self._calculations.calculate_returns(terms_vals)
                    cross_rets = base_rets.sub(terms_rets.iloc[:, 0], axis=0)
                cross_rets.iloc[0] = 0
                cross_vals = self._calculations.create_mult_index(cross_rets)
                cross_vals.columns = [tick + '-option-tot.close']
                total_return_indices.append(cross_vals)
            return self._calculations.join(total_return_indices, how='outer')

    def unhedged_asset_fx(self, assets_df, asset_currency, home_curr, start_date, finish_date, spot_df=None):
        if False:
            return 10
        pass

    def hedged_asset_fx(self, assets_df, asset_currency, home_curr, start_date, finish_date, spot_df=None, total_return_indices_df=None):
        if False:
            return 10
        pass

    def get_day_count_conv(self, currency):
        if False:
            for i in range(10):
                print('nop')
        if currency in market_constants.currencies_with_365_basis:
            return 365.0
        return 360.0

    def construct_total_return_index(self, cross_fx, market_df, fx_vol_surface=None, enter_trading_dates=None, fx_options_trading_tenor=None, roll_days_before=None, roll_event=None, roll_months=None, cum_index=None, strike=None, contract_type=None, premium_output=None, position_multiplier=None, fx_options_tenor_for_interpolation=None, freeze_implied_vol=None, depo_tenor_for_option=None, tot_label=None, cal=None, output_calculation_fields=None):
        if False:
            while True:
                i = 10
        if fx_vol_surface is None:
            fx_vol_surface = self._fx_vol_surface
        if enter_trading_dates is None:
            enter_trading_dates = self._enter_trading_dates
        if fx_options_trading_tenor is None:
            fx_options_trading_tenor = self._fx_options_trading_tenor
        if roll_days_before is None:
            roll_days_before = self._roll_days_before
        if roll_event is None:
            roll_event = self._roll_event
        if roll_months is None:
            roll_months = self._roll_months
        if cum_index is None:
            cum_index = self._cum_index
        if strike is None:
            strike = self._strike
        if contract_type is None:
            contract_type = self._contact_type
        if premium_output is None:
            premium_output = self._premium_output
        if position_multiplier is None:
            position_multiplier = self._position_multiplier
        if fx_options_tenor_for_interpolation is None:
            fx_options_tenor_for_interpolation = self._fx_options_tenor_for_interpolation
        if freeze_implied_vol is None:
            freeze_implied_vol = self._freeze_implied_vol
        if depo_tenor_for_option is None:
            depo_tenor_for_option = self._depo_tenor_for_option
        if tot_label is None:
            tot_label = self._tot_label
        if cal is None:
            cal = self._cal
        if output_calculation_fields is None:
            output_calculation_fields = self._output_calculation_fields
        if not isinstance(cross_fx, list):
            cross_fx = [cross_fx]
        total_return_index_df_agg = []
        if market_df is not None:
            market_df = market_df.dropna(how='all', axis=1)
        fx_options_pricer = FXOptionsPricer(premium_output=premium_output)

        def get_roll_date(horizon_d, expiry_d, asset_hols, month_adj=0):
            if False:
                while True:
                    i = 10
            if roll_event == 'month-end':
                roll_d = horizon_d + CustomBusinessMonthEnd(roll_months + month_adj, holidays=asset_hols)
                if roll_days_before > 0:
                    return roll_d - CustomBusinessDay(n=roll_days_before, holidays=asset_hols)
            elif roll_event == 'expiry-date':
                roll_d = expiry_d
                if roll_days_before > 0:
                    return roll_d - CustomBusinessDay(n=roll_days_before, holidays=asset_hols)
            return roll_d
        for cross in cross_fx:
            if cal is None:
                cal = cross
            if cross[0:3] == cross[3:6]:
                total_return_index_df_agg.append(pd.DataFrame(100, index=market_df.index, columns=[cross + '-option-tot.close']))
            else:
                old_cross = cross
                cross = FXConv().correct_notation(cross)
                if old_cross != cross:
                    pass
                if fx_vol_surface is None:
                    fx_vol_surface = FXVolSurface(market_df=market_df, asset=cross, tenors=fx_options_tenor_for_interpolation, depo_tenor=depo_tenor_for_option)
                    market_df = fx_vol_surface.get_all_market_data()
                horizon_date = market_df.index
                expiry_date = np.zeros(len(horizon_date), dtype=object)
                roll_date = np.zeros(len(horizon_date), dtype=object)
                new_trade = np.full(len(horizon_date), False, dtype=bool)
                exit_trade = np.full(len(horizon_date), False, dtype=bool)
                has_position = np.full(len(horizon_date), False, dtype=bool)
                asset_holidays = self._calendar.get_holidays(cal=cross)
                if enter_trading_dates is None:
                    expiry_date[0] = self._calendar.get_expiry_date_from_horizon_date(pd.DatetimeIndex([horizon_date[0]]), fx_options_trading_tenor, cal=cal, asset_class='fx-vol')[0]
                    roll_date[0] = get_roll_date(horizon_date[0], expiry_date[0], asset_holidays, month_adj=0)
                    new_trade[0] = True
                    exit_trade[0] = False
                    has_position[0] = True
                    for i in range(1, len(horizon_date)):
                        has_position[i] = True
                        if (horizon_date[i] - roll_date[i - 1]).days >= 0:
                            new_trade[i] = True
                        else:
                            new_trade[i] = False
                        if new_trade[i]:
                            exp = self._calendar.get_expiry_date_from_horizon_date(pd.DatetimeIndex([horizon_date[i]]), fx_options_trading_tenor, cal=cal, asset_class='fx-vol')[0]
                            if exp not in market_df.index:
                                exp_index = market_df.index.searchsorted(exp)
                                if exp_index < len(market_df.index):
                                    exp_index = min(exp_index, len(market_df.index))
                                    exp = market_df.index[exp_index]
                            expiry_date[i] = exp
                            roll_date[i] = get_roll_date(horizon_date[i], expiry_date[i], asset_holidays)
                            exit_trade[i] = True
                        elif horizon_date[i] <= expiry_date[i - 1]:
                            expiry_date[i] = expiry_date[i - 1]
                            roll_date[i] = roll_date[i - 1]
                            exit_trade[i] = False
                        else:
                            exit_trade[i] = True
                else:
                    new_trade[horizon_date.searchsorted(enter_trading_dates)] = True
                    has_position[horizon_date.searchsorted(enter_trading_dates)] = True
                    for i in range(0, len(horizon_date)):
                        if new_trade[i]:
                            exp = self._calendar.get_expiry_date_from_horizon_date(pd.DatetimeIndex([horizon_date[i]]), fx_options_trading_tenor, cal=cal, asset_class='fx-vol')[0]
                            if exp not in market_df.index:
                                exp_index = market_df.index.searchsorted(exp)
                                if exp_index < len(market_df.index):
                                    exp_index = min(exp_index, len(market_df.index))
                                    exp = market_df.index[exp_index]
                            expiry_date[i] = exp
                            exit_trade[i] = False
                        elif i > 0:
                            if expiry_date[i - 1] == 0:
                                has_position[i] = False
                            else:
                                if horizon_date[i] <= expiry_date[i - 1]:
                                    expiry_date[i] = expiry_date[i - 1]
                                    has_position[i] = True
                                if horizon_date[i] == expiry_date[i]:
                                    exit_trade[i] = True
                                else:
                                    exit_trade[i] = False
                mtm = np.zeros(len(horizon_date))
                calculated_strike = np.zeros(len(horizon_date))
                interpolated_option = np.zeros(len(horizon_date))
                implied_vol = np.zeros(len(horizon_date))
                delta = np.zeros(len(horizon_date))
                df_temp = pd.DataFrame()
                df_temp['expiry-date'] = expiry_date
                df_temp['horizon-date'] = horizon_date
                df_temp['roll-date'] = roll_date
                df_temp['new-trade'] = new_trade
                df_temp['exit-trade'] = exit_trade
                df_temp['has-position'] = has_position
                if has_position[0]:
                    (option_values_, spot_, strike_, vol_, delta_, expiry_date_, intrinsic_values_) = fx_options_pricer.price_instrument(cross, horizon_date[0], strike, expiry_date[0], contract_type=contract_type, tenor=fx_options_trading_tenor, fx_vol_surface=fx_vol_surface, return_as_df=False)
                    interpolated_option[0] = option_values_
                    calculated_strike[0] = strike_
                    implied_vol[0] = vol_
                mtm[0] = 0
                for i in range(1, len(horizon_date)):
                    if exit_trade[i]:
                        (option_values_, spot_, strike_, vol_, delta_, expiry_date_, intrinsic_values_) = fx_options_pricer.price_instrument(cross, horizon_date[i], calculated_strike[i - 1], expiry_date[i - 1], contract_type=contract_type, tenor=fx_options_trading_tenor, fx_vol_surface=fx_vol_surface, return_as_df=False)
                        mtm[i] = option_values_
                        delta[i] = 0
                        calculated_strike[i] = calculated_strike[i - 1]
                    if new_trade[i]:
                        (option_values_, spot_, strike_, vol_, delta_, expiry_date_, intrinsic_values_) = fx_options_pricer.price_instrument(cross, horizon_date[i], strike, expiry_date[i], contract_type=contract_type, tenor=fx_options_trading_tenor, fx_vol_surface=fx_vol_surface, return_as_df=False)
                        calculated_strike[i] = strike_
                        implied_vol[i] = vol_
                        interpolated_option[i] = option_values_
                        delta[i] = delta_
                    elif has_position[i] and (not exit_trade[i]):
                        calculated_strike[i] = calculated_strike[i - 1]
                        if freeze_implied_vol:
                            frozen_vol = implied_vol[i - 1]
                        else:
                            frozen_vol = None
                        (option_values_, spot_, strike_, vol_, delta_, expiry_date_, intrinsic_values_) = fx_options_pricer.price_instrument(cross, horizon_date[i], calculated_strike[i], expiry_date[i], vol=frozen_vol, contract_type=contract_type, tenor=fx_options_trading_tenor, fx_vol_surface=fx_vol_surface, return_as_df=False)
                        interpolated_option[i] = option_values_
                        implied_vol[i] = vol_
                        mtm[i] = interpolated_option[i]
                        delta[i] = delta_
                spot_rets = (market_df[cross + '.close'] / market_df[cross + '.close'].shift(1) - 1).values
                if tot_label == '':
                    tot_rets = spot_rets
                else:
                    tot_rets = (market_df[cross + '-' + tot_label + '.close'] / market_df[cross + '-' + tot_label + '.close'].shift(1) - 1).values
                delta_hedging_pnl = -np.roll(delta, 1) * tot_rets * position_multiplier
                delta_hedging_pnl[0] = 0
                option_rets = (mtm - np.roll(interpolated_option, 1)) * position_multiplier
                option_rets[0] = 0
                option_delta_rets = delta_hedging_pnl + option_rets
                if cum_index == 'mult':
                    cum_rets = 100 * np.cumprod(1.0 + option_rets)
                    cum_delta_rets = 100 * np.cumprod(1.0 + delta_hedging_pnl)
                    cum_option_delta_rets = 100 * np.cumprod(1.0 + option_delta_rets)
                elif cum_index == 'add':
                    cum_rets = 100 + 100 * np.cumsum(option_rets)
                    cum_delta_rets = 100 + 100 * np.cumsum(delta_hedging_pnl)
                    cum_option_delta_rets = 100 + 100 * np.cumsum(option_delta_rets)
                total_return_index_df = pd.DataFrame(index=horizon_date, columns=[cross + '-option-tot.close'])
                total_return_index_df[cross + '-option-tot.close'] = cum_rets
                if output_calculation_fields:
                    total_return_index_df[cross + '-interpolated-option.close'] = interpolated_option
                    total_return_index_df[cross + '-mtm.close'] = mtm
                    total_return_index_df[cross + '.close'] = market_df[cross + '.close'].values
                    total_return_index_df[cross + '-implied-vol.close'] = implied_vol
                    total_return_index_df[cross + '-new-trade.close'] = new_trade
                    total_return_index_df[cross + '.roll-date'] = roll_date
                    total_return_index_df[cross + '-exit-trade.close'] = exit_trade
                    total_return_index_df[cross + '.expiry-date'] = expiry_date
                    total_return_index_df[cross + '-calculated-strike.close'] = calculated_strike
                    total_return_index_df[cross + '-option-return.close'] = option_rets
                    total_return_index_df[cross + '-spot-return.close'] = spot_rets
                    total_return_index_df[cross + '-tot-return.close'] = tot_rets
                    total_return_index_df[cross + '-delta.close'] = delta
                    total_return_index_df[cross + '-delta-pnl-return.close'] = delta_hedging_pnl
                    total_return_index_df[cross + '-delta-pnl-index.close'] = cum_delta_rets
                    total_return_index_df[cross + '-option-delta-return.close'] = option_delta_rets
                    total_return_index_df[cross + '-option-delta-tot.close'] = cum_option_delta_rets
                total_return_index_df_agg.append(total_return_index_df)
        return self._calculations.join(total_return_index_df_agg, how='outer')

    def apply_tc_signals_to_total_return_index(self, cross_fx, total_return_index_orig_df, option_tc_bp, spot_tc_bp, signal_df=None, cum_index=None):
        if False:
            return 10
        if cum_index is None:
            cum_index = self._cum_index
        total_return_index_df_agg = []
        if not isinstance(cross_fx, list):
            cross_fx = [cross_fx]
        option_tc = option_tc_bp / (2 * 100 * 100)
        spot_tc = spot_tc_bp / (2 * 100 * 100)
        total_return_index_df = total_return_index_orig_df.copy()
        for cross in cross_fx:
            total_return_index_df[cross + '-option-return-with-tc.close'] = total_return_index_df[cross + '-option-return.close'] - abs(total_return_index_df[cross + '-new-trade.close'].shift(1)) * option_tc
            total_return_index_df[cross + '-delta-pnl-return-with-tc.close'] = total_return_index_df[cross + '-delta-pnl-return.close'] - abs(total_return_index_df[cross + '-delta.close'] - total_return_index_df[cross + '-delta.close'].shift(1)) * spot_tc
            total_return_index_df[cross + '-option-return-with-tc.close'][0] = 0
            total_return_index_df[cross + '-delta-pnl-return-with-tc.close'][0] = 0
            total_return_index_df[cross + '-option-delta-return-with-tc.close'] = total_return_index_df[cross + '-option-return-with-tc.close'] + total_return_index_df[cross + '-delta-pnl-return-with-tc.close']
            if cum_index == 'mult':
                cum_rets = 100 * np.cumprod(1.0 + total_return_index_df[cross + '-option-return-with-tc.close'].values)
                cum_delta_rets = 100 * np.cumprod(1.0 + total_return_index_df[cross + '-delta-pnl-return-with-tc.close'].values)
                cum_option_delta_rets = 100 * np.cumprod(1.0 + total_return_index_df[cross + '-option-delta-return-with-tc.close'].values)
            elif cum_index == 'add':
                cum_rets = 100 + 100 * np.cumsum(total_return_index_df[cross + '-option-return-with-tc.close'].values)
                cum_delta_rets = 100 + 100 * np.cumsum(total_return_index_df[cross + '-delta-pnl-return-with-tc.close'].values)
                cum_option_delta_rets = 100 + 100 * np.cumsum(total_return_index_df[cross + '-option-delta-return-with-tc.close'].values)
            total_return_index_df[cross + '-option-tot-with-tc.close'] = cum_rets
            total_return_index_df[cross + '-delta-pnl-index-with-tc.close'] = cum_delta_rets
            total_return_index_df[cross + '-option-delta-tot-with-tc.close'] = cum_option_delta_rets
            total_return_index_df_agg.append(total_return_index_df)
        return self._calculations.join(total_return_index_df_agg, how='outer')