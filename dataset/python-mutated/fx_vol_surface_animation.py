__author__ = 'saeedamen'
'\nShows how to load up FX vol surfaces from Bloomberg and then plot an animation of them. Note, this does not do\nany interpolation.\n'
from findatapy.market import Market, MarketDataRequest, MarketDataGenerator, FXVolFactory
from chartpy import Chart, Style

def plot_animated_vol_market():
    if False:
        i = 10
        return i + 15
    market = Market(market_data_generator=MarketDataGenerator())
    cross = ['EURUSD']
    start_date = '01 Mar 2017'
    finish_date = '21 Apr 2017'
    sampling = 'no'
    md_request = MarketDataRequest(start_date=start_date, finish_date=finish_date, data_source='bloomberg', cut='NYC', category='fx-implied-vol', tickers=cross, cache_algo='cache_algo_return')
    df = market.fetch_market(md_request)
    if sampling != 'no':
        df = df.resample(sampling).mean()
    fxvf = FXVolFactory()
    df_vs = []
    for i in range(0, len(df.index)):
        df_vs.append(fxvf.extract_vol_surface_for_date(df, cross[0], i))
    style = Style(title='FX vol surface of ' + cross[0], source='chartpy', color='Blues')
    Chart(df=df_vs[0], chart_type='surface', style=style).plot(engine='plotly')
    style = Style(title='FX vol surface of ' + cross[0], source='chartpy', color='Blues', animate_figure=True, animate_titles=df.index, animate_frame_ms=500, normalize_colormap=False)
    Chart(df=df_vs, chart_type='surface', style=style).plot(engine='plotly')
    Chart(df=df_vs, chart_type='surface', style=style).plot(engine='plotly')
if __name__ == '__main__':
    plot_animated_vol_market()