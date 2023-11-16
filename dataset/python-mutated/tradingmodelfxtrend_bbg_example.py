__author__ = 'saeedamen'
import datetime
from findatapy.market import Market, MarketDataGenerator, MarketDataRequest
from finmarketpy.backtest import TradingModel, BacktestRequest
from finmarketpy.economics import TechIndicator

class TradingModelFXTrend_BBG_Example(TradingModel):
    """Shows how to create a simple FX CTA style strategy, using the TradingModel abstract class (backtest_examples.py
    is a lower level way of doing this). Uses BBG total returns data.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(TradingModel, self).__init__()
        self.market = Market(market_data_generator=MarketDataGenerator())
        self.DUMP_PATH = ''
        self.FINAL_STRATEGY = 'FX trend'
        self.SCALE_FACTOR = 1
        self.DEFAULT_PLOT_ENGINE = 'matplotlib'
        self.br = self.load_parameters()
        return

    def load_parameters(self, br=None):
        if False:
            i = 10
            return i + 15
        if br is not None:
            return br
        br = BacktestRequest()
        br.start_date = '04 Jan 1989'
        br.finish_date = datetime.datetime.utcnow().date()
        br.spot_tc_bp = 0.5
        br.ann_factor = 252
        br.plot_start = '01 Apr 2015'
        br.calc_stats = True
        br.write_csv = False
        br.plot_interim = True
        br.include_benchmark = True
        br.signal_vol_adjust = True
        br.signal_vol_target = 0.1
        br.signal_vol_max_leverage = 5
        br.signal_vol_periods = 20
        br.signal_vol_obs_in_year = 252
        br.signal_vol_rebalance_freq = 'BM'
        br.signal_vol_resample_freq = None
        br.portfolio_vol_adjust = True
        br.portfolio_vol_target = 0.1
        br.portfolio_vol_max_leverage = 5
        br.portfolio_vol_periods = 20
        br.portfolio_vol_obs_in_year = 252
        br.portfolio_vol_rebalance_freq = 'BM'
        br.portfolio_vol_resample_freq = None
        br.tech_params.sma_period = 200
        return br

    def load_assets(self, br=None):
        if False:
            while True:
                i = 10
        from findatapy.util.loggermanager import LoggerManager
        logger = LoggerManager().getLogger(__name__)
        full_bkt = ['EURUSD', 'USDJPY', 'GBPUSD', 'AUDUSD', 'USDCAD', 'NZDUSD', 'USDCHF', 'USDNOK', 'USDSEK']
        basket_dict = {}
        for i in range(0, len(full_bkt)):
            basket_dict[full_bkt[i]] = [full_bkt[i]]
        basket_dict['FX trend'] = full_bkt
        br = self.load_parameters(br=br)
        logger.info('Loading asset data...')
        vendor_tickers = [x + 'CR CMPN Curncy' for x in full_bkt]
        market_data_request = MarketDataRequest(start_date=br.start_date, finish_date=br.finish_date, freq='daily', data_source='bloomberg', tickers=full_bkt, fields=['close'], vendor_tickers=vendor_tickers, vendor_fields=['PX_LAST'], cache_algo='internet_load_return')
        asset_df = self.market.fetch_market(market_data_request)
        if asset_df is None:
            import pandas
            asset_df = pandas.read_csv('d:/fxcta.csv', index_col=0, parse_dates=['Date'], date_parser=lambda x: pandas.datetime.strptime(x, '%Y-%m-%d'))
        spot_df = asset_df
        spot_df2 = None
        return (asset_df, spot_df, spot_df2, basket_dict)

    def construct_signal(self, spot_df, spot_df2, tech_params, br, run_in_parallel=False):
        if False:
            while True:
                i = 10
        tech_ind = TechIndicator()
        tech_ind.create_tech_ind(spot_df, 'SMA', tech_params)
        signal_df = tech_ind.get_signal()
        return signal_df

    def construct_strategy_benchmark(self):
        if False:
            return 10
        tsr_indices = MarketDataRequest(start_date=self.br.start_date, finish_date=self.br.finish_date, freq='daily', data_source='bloomberg', tickers=['EURUSD'], vendor_tickers=['EURUSDCR CMPN Curncy'], fields=['close'], vendor_fields=['PX_LAST'], cache_algo='internet_load_return')
        df = self.market.fetch_market(tsr_indices)
        df.columns = [x.split('.')[0] for x in df.columns]
        return df
if __name__ == '__main__':
    if True:
        model = TradingModelFXTrend_BBG_Example()
        model.construct_strategy()
        model.plot_strategy_pnl()
        model.plot_strategy_leverage()
        model.plot_strategy_group_pnl_trades()
        model.plot_strategy_group_benchmark_pnl()
        model.plot_strategy_group_benchmark_pnl_ir()
        model.plot_strategy_group_leverage()
        from finmarketpy.backtest import TradeAnalysis
        ta = TradeAnalysis()
        ta.run_strategy_returns_stats(model, engine='finmarketpy')
    if True:
        strategy = TradingModelFXTrend_BBG_Example()
        from finmarketpy.backtest import TradeAnalysis
        ta = TradeAnalysis()
        ta.run_strategy_returns_stats(model, engine='finmarketpy')
        parameter_list = [{'portfolio_vol_adjust': True, 'signal_vol_adjust': True}, {'portfolio_vol_adjust': False, 'signal_vol_adjust': False}]
        pretty_portfolio_names = ['Vol target', 'No vol target']
        parameter_type = 'vol target'
        ta.run_arbitrary_sensitivity(strategy, parameter_list=parameter_list, pretty_portfolio_names=pretty_portfolio_names, parameter_type=parameter_type)
        tc = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2.0]
        ta.run_tc_shock(strategy, tc=tc)
        ta.run_day_of_month_analysis(strategy)
    if False:
        from finmarketpy.backtest import TradeAnalysis
        model = TradingModelFXTrend_BBG_Example()
        model.construct_strategy()
        tradeanalysis = TradeAnalysis()
        tradeanalysis.run_strategy_returns_stats(strategy)