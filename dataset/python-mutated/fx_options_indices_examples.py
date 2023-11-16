__author__ = 'saeedamen'
'\nShows how to use finmarketpy to total return indices for FX vanilla options (uses FinancePy underneath), so we can \nsee the historical P&L from for example, rolling a 1M call option etc.\n\nNote, you will need to have a Bloomberg terminal (with blpapi Python library) to download the FX market data in order\nto generate the FX option prices, which are used underneath (FX spot, FX forwards, FX implied volatility quotes and deposit rates)\n'
import pandas as pd
from chartpy import Chart, Style
from findatapy.market import Market, MarketDataGenerator, MarketDataRequest
from findatapy.timeseries import Filter, Calculations, RetStats
from findatapy.util.loggermanager import LoggerManager
from finmarketpy.curve.fxoptionscurve import FXOptionsCurve
from finmarketpy.curve.volatility.fxvolsurface import FXVolSurface
from finmarketpy.curve.volatility.fxoptionspricer import FXOptionsPricer
logger = LoggerManager().getLogger(__name__)
chart = Chart(engine='plotly')
market = Market(market_data_generator=MarketDataGenerator())
run_example = 0

def prepare_indices(cross, df_option_tot=None, df_option_tc=None, df_spot_tot=None):
    if False:
        return 10
    df_list = []
    if df_option_tot is not None:
        df_list.append(pd.DataFrame(df_option_tot[cross + '-option-tot.close']))
        df_list.append(pd.DataFrame(df_option_tot[cross + '-option-delta-tot.close']))
        df_list.append(pd.DataFrame(df_option_tot[cross + '-delta-pnl-index.close']))
    if df_option_tc is not None:
        df_list.append(pd.DataFrame(df_option_tc[cross + '-option-tot-with-tc.close']))
        df_list.append(pd.DataFrame(df_option_tc[cross + '-option-delta-tot-with-tc.close']))
        df_list.append(pd.DataFrame(df_option_tc[cross + '-delta-pnl-index-with-tc.close']))
    if df_spot_tot is not None:
        df_list.append(df_spot_tot)
    df = calculations.join(df_list, how='outer').fillna(method='ffill')
    return calculations.create_mult_index_from_prices(df)
if __name__ == '__main__':
    if run_example == 1 or run_example == 0:
        start_date = '01 Jan 2007'
        finish_date = '31 Dec 2020'
        cross = 'AUDUSD'
        fx_options_trading_tenor = '1M'
        md_request = MarketDataRequest(start_date=start_date, finish_date=finish_date, data_source='bloomberg', cut='BGN', category='fx-vol-market', tickers=cross, fx_vol_tenor=['1W', '1M', '3M'], cache_algo='cache_algo_return', base_depos_currencies=[cross[0:3], cross[3:6]])
        df_vol_market = market.fetch_market(md_request)
        df_vol_market = df_vol_market.fillna(method='ffill')
        df_vol_market = Filter().filter_time_series_by_holidays(df_vol_market, cal='FX')
        md_request = MarketDataRequest(start_date=start_date, finish_date=finish_date, data_source='bloomberg', cut='NYC', category='fx-tot', tickers=cross, cache_algo='cache_algo_return')
        df_tot = market.fetch_market(md_request)
        df_vol_market = df_vol_market.join(df_tot, how='left')
        df_vol_market = df_vol_market.fillna(method='ffill')
        fx_options_curve = FXOptionsCurve(fx_options_trading_tenor=fx_options_trading_tenor, roll_days_before=0, roll_event='expiry-date', roll_months=1, fx_options_tenor_for_interpolation=['1W', '1M'], strike='atm', contract_type='european-call', depo_tenor_for_option='1M', position_multiplier=1.0, tot_label='tot', output_calculation_fields=True)
        df_cuemacro_option_call_tot = fx_options_curve.construct_total_return_index(cross, df_vol_market)
        df_cuemacro_option_call_tc = fx_options_curve.apply_tc_signals_to_total_return_index(cross, df_cuemacro_option_call_tot, option_tc_bp=5, spot_tc_bp=2)
        df_cuemacro_option_put_tot = fx_options_curve.construct_total_return_index(cross, df_vol_market, contract_type='european-put', strike='10d-otm', position_multiplier=1.0)
        df_cuemacro_option_put_tc = fx_options_curve.apply_tc_signals_to_total_return_index(cross, df_cuemacro_option_put_tot, option_tc_bp=5, spot_tc_bp=2)
        df_bbg_tot = df_tot
        df_bbg_tot.columns = [x + '-bbg' for x in df_bbg_tot.columns]
        calculations = Calculations()
        ret_stats = RetStats()
        df_hedged = calculations.join([df_bbg_tot[cross + '-tot.close-bbg'].to_frame(), df_cuemacro_option_put_tc[cross + '-option-tot-with-tc.close'].to_frame()], how='outer')
        df_hedged = df_hedged.fillna(method='ffill')
        df_hedged = df_hedged.pct_change()
        df_hedged['Spot + 2*option put hedge'] = df_hedged[cross + '-tot.close-bbg'] + df_hedged[cross + '-option-tot-with-tc.close']
        df_hedged.columns = RetStats(returns_df=df_hedged, ann_factor=252).summary()
        chart.plot(calculations.create_mult_index_from_prices(prepare_indices(cross=cross, df_option_tot=df_cuemacro_option_call_tot, df_option_tc=df_cuemacro_option_call_tc, df_spot_tot=df_bbg_tot)))
        chart.plot(calculations.create_mult_index_from_prices(prepare_indices(cross=cross, df_option_tot=df_cuemacro_option_put_tot, df_option_tc=df_cuemacro_option_put_tc, df_spot_tot=df_bbg_tot)))
        chart.plot(calculations.create_mult_index_from_prices(prepare_indices(cross=cross, df_option_tc=df_cuemacro_option_put_tc, df_spot_tot=df_bbg_tot)))
        chart.plot(calculations.create_mult_index(df_hedged))
        chart.plot(df_cuemacro_option_put_tot[cross + '-delta.close'])
    if run_example == 2 or run_example == 0:
        start_date = '08 Mar 2007'
        finish_date = '31 Dec 2020'
        cross = 'USDJPY'
        fx_options_trading_tenor = '1W'
        md_request = MarketDataRequest(start_date=start_date, finish_date=finish_date, data_source='bloomberg', cut='10AM', category='fx-vol-market', tickers=cross, fx_vol_tenor=['1W', '1M'], base_depos_tenor=['1W', '1M'], cache_algo='cache_algo_return', base_depos_currencies=[cross[0:3], cross[3:6]])
        df = market.fetch_market(md_request)
        df = df.resample('B').last().fillna(method='ffill')
        df = df[df.index >= '09 Mar 2007']
        cal = 'WKD'
        fx_options_curve = FXOptionsCurve(fx_options_trading_tenor=fx_options_trading_tenor, roll_days_before=0, roll_event='expiry-date', roll_months=1, fx_options_tenor_for_interpolation=['1W', '1M'], strike='atm', contract_type='european-straddle', position_multiplier=-1.0, output_calculation_fields=True, freeze_implied_vol=True, cal=cal, cum_index='mult')
        df_cuemacro_option_straddle_tot = fx_options_curve.construct_total_return_index(cross, df, depo_tenor_for_option=fx_options_trading_tenor)
        df_cuemacro_option_straddle_tc = fx_options_curve.apply_tc_signals_to_total_return_index(cross, df_cuemacro_option_straddle_tot, option_tc_bp=10, spot_tc_bp=2)
        md_request.abstract_curve = None
        md_request.category = 'fx-tot'
        md_request.cut = 'NYC'
        df_bbg_tot = market.fetch_market(md_request)
        df_bbg_tot.columns = [x + '-bbg' for x in df_bbg_tot.columns]
        calculations = Calculations()
        df_index = calculations.create_mult_index_from_prices(prepare_indices(cross=cross, df_option_tc=df_cuemacro_option_straddle_tc, df_spot_tot=df_bbg_tot))
        from finmarketpy.economics.quickchart import QuickChart
        QuickChart(engine='plotly').plot_chart_with_ret_stats(df=df_index, plotly_plot_mode='offline_html', scale_factor=-1.5)