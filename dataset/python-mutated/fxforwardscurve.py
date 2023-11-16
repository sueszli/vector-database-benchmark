__author__ = 'saeedamen'
import numpy as np
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay, CustomBusinessMonthEnd
from findatapy.market import Market, MarketDataRequest
from findatapy.timeseries import Calculations, Calendar, Filter
from findatapy.util.dataconstants import DataConstants
from findatapy.util.fxconv import FXConv
from finmarketpy.curve.rates.fxforwardspricer import FXForwardsPricer
from finmarketpy.util.marketconstants import MarketConstants
data_constants = DataConstants()
market_constants = MarketConstants()

class FXForwardsCurve(object):
    """Constructs continuous forwards time series total return indices from underlying forwards contracts.

    """

    def __init__(self, market_data_generator=None, fx_forwards_trading_tenor=market_constants.fx_forwards_trading_tenor, roll_days_before=market_constants.fx_forwards_roll_days_before, roll_event=market_constants.fx_forwards_roll_event, construct_via_currency='no', fx_forwards_tenor_for_interpolation=market_constants.fx_forwards_tenor_for_interpolation, base_depos_tenor=data_constants.base_depos_tenor, roll_months=market_constants.fx_forwards_roll_months, cum_index=market_constants.fx_forwards_cum_index, output_calculation_fields=market_constants.output_calculation_fields, field='close'):
        if False:
            i = 10
            return i + 15
        "Initializes FXForwardsCurve\n\n        Parameters\n        ----------\n        market_data_generator : MarketDataGenerator\n            Used for downloading market data\n\n        fx_forwards_trading_tenor : str\n            What is primary forward contract being used to trade (default - '1M')\n\n        roll_days_before : int\n            Number of days before roll event to enter into a new forwards contract\n\n        roll_event : str\n            What constitutes a roll event? ('month-end', 'quarter-end', 'year-end', 'expiry')\n\n        construct_via_currency : str\n            What currency should we construct the forward via? Eg. if we asked for AUDJPY we can construct it via\n            AUDUSD & JPYUSD forwards, as opposed to AUDJPY forwards (default - 'no')\n\n        fx_forwards_tenor_for_interpolation : str(list)\n            Which forwards should we use for interpolation\n\n        base_depos_tenor : str(list)\n            Which base deposits tenors do we need (this is only necessary if we want to start inferring depos)\n\n        roll_months : int\n            After how many months should we initiate a roll. Typically for trading 1M this should 1, 3M this should be 3\n            etc.\n\n        cum_index : str\n            In total return index, do we compute in additive or multiplicative way ('add' or 'mult')\n\n        output_calculation_fields : bool\n            Also output additional data should forward expiries etc. alongside total returns indices\n        "
        self._market_data_generator = market_data_generator
        self._calculations = Calculations()
        self._calendar = Calendar()
        self._filter = Filter()
        self._fx_forwards_trading_tenor = fx_forwards_trading_tenor
        self._roll_days_before = roll_days_before
        self._roll_event = roll_event
        self._construct_via_currency = construct_via_currency
        self._fx_forwards_tenor_for_interpolation = fx_forwards_tenor_for_interpolation
        self._base_depos_tenor = base_depos_tenor
        self._roll_months = roll_months
        self._cum_index = cum_index
        self._output_calcultion_fields = output_calculation_fields
        self._field = field

    def generate_key(self):
        if False:
            for i in range(10):
                print('nop')
        from findatapy.market.ioengine import SpeedCache
        return SpeedCache().generate_key(self, ['_market_data_generator', '_calculations', '_calendar', '_filter'])

    def fetch_continuous_time_series(self, md_request, market_data_generator, fx_forwards_trading_tenor=None, roll_days_before=None, roll_event=None, construct_via_currency=None, fx_forwards_tenor_for_interpolation=None, base_depos_tenor=None, roll_months=None, cum_index=None, output_calculation_fields=False, field=None):
        if False:
            while True:
                i = 10
        if market_data_generator is None:
            market_data_generator = self._market_data_generator
        if fx_forwards_trading_tenor is None:
            fx_forwards_trading_tenor = self._fx_forwards_trading_tenor
        if roll_days_before is None:
            roll_days_before = self._roll_days_before
        if roll_event is None:
            roll_event = self._roll_event
        if construct_via_currency is None:
            construct_via_currency = self._construct_via_currency
        if fx_forwards_tenor_for_interpolation is None:
            fx_forwards_tenor_for_interpolation = self._fx_forwards_tenor_for_interpolation
        if base_depos_tenor is None:
            base_depos_tenor = self._base_depos_tenor
        if roll_months is None:
            roll_months = self._roll_months
        if cum_index is None:
            cum_index = self._cum_index
        if output_calculation_fields is None:
            output_calculation_fields = self._output_calcultion_fields
        if field is None:
            field = self._field
        if construct_via_currency == 'no':
            market = Market(market_data_generator=market_data_generator)
            md_request_download = MarketDataRequest(md_request=md_request)
            fx_conv = FXConv()
            md_request_download.tickers = [fx_conv.correct_notation(x) for x in md_request.tickers]
            md_request_download.category = 'fx-forwards-market'
            md_request_download.fields = field
            md_request_download.abstract_curve = None
            md_request_download.fx_forwards_tenor = fx_forwards_tenor_for_interpolation
            md_request_download.base_depos_tenor = base_depos_tenor
            forwards_market_df = market.fetch_market(md_request_download)
            return self.construct_total_return_index(md_request.tickers, forwards_market_df, fx_forwards_trading_tenor=fx_forwards_trading_tenor, roll_days_before=roll_days_before, roll_event=roll_event, fx_forwards_tenor_for_interpolation=fx_forwards_tenor_for_interpolation, roll_months=roll_months, cum_index=cum_index, output_calculation_fields=output_calculation_fields, field=field)
        else:
            total_return_indices = []
            for tick in md_request.tickers:
                base = tick[0:3]
                terms = tick[3:6]
                md_request_base = MarketDataRequest(md_request=md_request)
                md_request_base.tickers = base + construct_via_currency
                md_request_terms = MarketDataRequest(md_request=md_request)
                md_request_terms.tickers = terms + construct_via_currency
                base_vals = self.fetch_continuous_time_series(md_request_base, market_data_generator, fx_forwards_trading_tenor=fx_forwards_trading_tenor, roll_days_before=roll_days_before, roll_event=roll_event, fx_forwards_tenor_for_interpolation=fx_forwards_tenor_for_interpolation, base_depos_tenor=base_depos_tenor, roll_months=roll_months, output_calculation_fields=False, cum_index=cum_index, construct_via_currency='no', field=field)
                terms_vals = self.fetch_continuous_time_series(md_request_terms, market_data_generator, fx_forwards_trading_tenor=fx_forwards_trading_tenor, roll_days_before=roll_days_before, roll_event=roll_event, fx_forwards_tenor_for_interpolation=fx_forwards_tenor_for_interpolation, base_depos_tenor=base_depos_tenor, roll_months=roll_months, cum_index=cum_index, output_calculation_fields=False, construct_via_currency='no', field=field)
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
                cross_vals.columns = [tick + '-forward-tot.' + field]
                total_return_indices.append(cross_vals)
            return self._calculations.join(total_return_indices, how='outer')

    def unhedged_asset_fx(self, assets_df, asset_currency, home_curr, start_date, finish_date, spot_df=None):
        if False:
            for i in range(10):
                print('nop')
        pass

    def hedged_asset_fx(self, assets_df, asset_currency, home_curr, start_date, finish_date, spot_df=None, total_return_indices_df=None):
        if False:
            for i in range(10):
                print('nop')
        pass

    def get_day_count_conv(self, currency):
        if False:
            for i in range(10):
                print('nop')
        if currency in market_constants.currencies_with_365_basis:
            return 365.0
        return 360.0

    def construct_total_return_index(self, cross_fx, forwards_market_df, fx_forwards_trading_tenor=None, roll_days_before=None, roll_event=None, roll_months=None, fx_forwards_tenor_for_interpolation=None, cum_index=None, output_calculation_fields=None, field=None):
        if False:
            while True:
                i = 10
        if not isinstance(cross_fx, list):
            cross_fx = [cross_fx]
        if fx_forwards_trading_tenor is None:
            fx_forwards_trading_tenor = self._fx_forwards_trading_tenor
        if roll_days_before is None:
            roll_days_before = self._roll_days_before
        if roll_event is None:
            roll_event = self._roll_event
        if roll_months is None:
            roll_months = self._roll_months
        if fx_forwards_tenor_for_interpolation is None:
            fx_forwards_tenor_for_interpolation = self._fx_forwards_tenor_for_interpolation
        if cum_index is None:
            cum_index = self._cum_index
        if field is None:
            field = self._field
        total_return_index_df_agg = []
        forwards_market_df = forwards_market_df.dropna(how='all', axis=1)
        fx_forwards_pricer = FXForwardsPricer()

        def get_roll_date(horizon_d, delivery_d, asset_hols, month_adj=1):
            if False:
                print('Hello World!')
            if roll_event == 'month-end':
                roll_d = horizon_d + CustomBusinessMonthEnd(roll_months + month_adj, holidays=asset_hols)
            elif roll_event == 'delivery-date':
                roll_d = delivery_d
            return roll_d - CustomBusinessDay(n=roll_days_before, holidays=asset_hols)
        for cross in cross_fx:
            if cross[0:3] == cross[3:6]:
                total_return_index_df_agg.append(pd.DataFrame(100, index=forwards_market_df.index, columns=[cross + '-forward-tot.close']))
            else:
                old_cross = cross
                cross = FXConv().correct_notation(cross)
                horizon_date = forwards_market_df.index
                delivery_date = []
                roll_date = []
                new_trade = np.full(len(horizon_date), False, dtype=bool)
                asset_holidays = self._calendar.get_holidays(cal=cross)
                delivery_date.append(self._calendar.get_delivery_date_from_horizon_date(horizon_date[0], fx_forwards_trading_tenor, cal=cross, asset_class='fx')[0])
                roll_date.append(get_roll_date(horizon_date[0], delivery_date[0], asset_holidays, month_adj=0))
                new_trade[0] = True
                for i in range(1, len(horizon_date)):
                    if (horizon_date[i] - roll_date[i - 1]).days == 0:
                        new_trade[i] = True
                    if new_trade[i]:
                        delivery_date.append(self._calendar.get_delivery_date_from_horizon_date(horizon_date[i], fx_forwards_trading_tenor, cal=cross, asset_class='fx')[0])
                        roll_date.append(get_roll_date(horizon_date[i], delivery_date[i], asset_holidays))
                    else:
                        delivery_date.append(delivery_date[i - 1])
                        roll_date.append(roll_date[i - 1])
                interpolated_forward = fx_forwards_pricer.price_instrument(cross, horizon_date, delivery_date, market_df=forwards_market_df, fx_forwards_tenor_for_interpolation=fx_forwards_tenor_for_interpolation)[cross + '-interpolated-outright-forward.' + field].values
                mtm = np.copy(interpolated_forward)
                for i in range(1, len(horizon_date)):
                    if new_trade[i]:
                        mtm[i] = fx_forwards_pricer.price_instrument(cross, horizon_date[i], delivery_date[i - 1], market_df=forwards_market_df, fx_forwards_tenor_for_interpolation=fx_forwards_tenor_for_interpolation)[cross + '-interpolated-outright-forward.' + field].values
                if old_cross != cross:
                    mtm = 1.0 / mtm
                    interpolated_forward = 1.0 / interpolated_forward
                forward_rets = mtm / np.roll(interpolated_forward, 1) - 1.0
                forward_rets[0] = 0
                if cum_index == 'mult':
                    cum_rets = 100 * np.cumprod(1.0 + forward_rets)
                elif cum_index == 'add':
                    cum_rets = 100 + 100 * np.cumsum(forward_rets)
                total_return_index_df = pd.DataFrame(index=horizon_date, columns=[cross + '-forward-tot.' + field])
                total_return_index_df[cross + '-forward-tot.' + field] = cum_rets
                if output_calculation_fields:
                    total_return_index_df[cross + '-interpolated-outright-forward.' + field] = interpolated_forward
                    total_return_index_df[cross + '-mtm.close'] = mtm
                    total_return_index_df[cross + '-roll.close'] = new_trade
                    total_return_index_df[cross + '.roll-date'] = roll_date
                    total_return_index_df[cross + '.delivery-date'] = delivery_date
                    total_return_index_df[cross + '-forward-return.' + field] = forward_rets
                total_return_index_df_agg.append(total_return_index_df)
        return self._calculations.join(total_return_index_df_agg, how='outer')