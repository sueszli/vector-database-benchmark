import logging
import numpy as np
from neuralprophet.forecaster import NeuralProphet
log = logging.getLogger('NP.forecaster')

class TorchProphet(NeuralProphet):
    """
    Prophet wrapper for the NeuralProphet forecaster.

    Parameters
    ----------
    growth: String 'linear' or 'flat' to specify a linear or
        flat trend. Note: 'flat' is equivalent to 'off' in NeuralProphet.
    changepoints: List of dates at which to include potential changepoints. If
        not specified, potential changepoints are selected automatically.
    n_changepoints: Number of potential changepoints to include. Not used
        if input `changepoints` is supplied. If `changepoints` is not supplied,
        then n_changepoints potential changepoints are selected uniformly from
        the first `changepoint_range` proportion of the history.
    changepoint_range: Proportion of history in which trend changepoints will
        be estimated. Defaults to 0.8 for the first 80%. Not used if
        `changepoints` is specified.
    yearly_seasonality: Fit yearly seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    weekly_seasonality: Fit weekly seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    daily_seasonality: Fit daily seasonality.
        Can be 'auto', True, False, or a number of Fourier terms to generate.
    holidays: pd.DataFrame with columns holiday (string) and ds (date type)
        and optionally columns lower_window and upper_window which specify a
        range of days around the date to be included as holidays.
        lower_window=-2 will include 2 days prior to the date as holidays. Also
        optionally can have a column prior_scale specifying the prior scale for
        that holiday.
    seasonality_mode: 'additive' (default) or 'multiplicative'.
    seasonality_prior_scale: Not supported for regularisation in NeuralProphet,
        please use the `seasonality_reg` arg instead.
    holidays_prior_scale: Not supported for regularisation in NeuralProphet.
    changepoint_prior_scale: Not supported for regularisation in NeuralProphet,
        please use the `trend_reg` arg instead.
    mcmc_samples: Not required for NeuralProphet
    interval_width: Float, width of the uncertainty intervals provided
        for the forecast. Converted to list of quantiles for NeuralProphet. Use
        the quantiles arg to pass quantiles directly to NeuralProphet.
    uncertainty_samples: Not required for NeuralProphet.
    stan_backend: Not supported by NeuralProphet.
    """

    def __init__(self, growth='linear', changepoints=None, n_changepoints=25, changepoint_range=0.8, yearly_seasonality='auto', weekly_seasonality='auto', daily_seasonality='auto', holidays=None, seasonality_mode='additive', seasonality_prior_scale=None, holidays_prior_scale=None, changepoint_prior_scale=None, mcmc_samples=None, interval_width=0.8, uncertainty_samples=None, stan_backend=None, **kwargs):
        if False:
            while True:
                i = 10
        if seasonality_prior_scale or holidays_prior_scale or changepoint_prior_scale:
            log.error('Using `_prior_scale` is unsupported for regularisation in NeuralProphet, please use the corresponding `_reg` arg instead.')
        if mcmc_samples or uncertainty_samples:
            log.warning('Providing the number of samples for Bayesian inference or Uncertainty estimation is not required in NeuralProphet.')
        if stan_backend:
            log.warning('A stan_backend is not used in NeuralProphet. Please remove the parameter')
        if growth == 'flat':
            log.warning("Using 'flat' growth is equivalent to 'off' in NeuralProphet.")
            growth = 'off'
        if 'quantiles' not in kwargs:
            alpha = 1 - interval_width
            quantiles = [np.round(alpha / 2, 4), np.round(1 - alpha / 2, 4)]
        super(TorchProphet, self).__init__(growth=growth, changepoints=changepoints, n_changepoints=n_changepoints, changepoints_range=changepoint_range, yearly_seasonality=yearly_seasonality, weekly_seasonality=weekly_seasonality, daily_seasonality=daily_seasonality, seasonality_mode=seasonality_mode, quantiles=quantiles, **kwargs)
        if holidays is not None:
            self.add_events(events=list(holidays['holiday'].unique()), lower_window=holidays['lower_window'].max(), upper_window=holidays['upper_window'].max())
            self.events_df = holidays.copy()
            self.events_df.rename(columns={'holiday': 'event'}, inplace=True)
            self.events_df.drop(['lower_window', 'upper_window'], axis=1, errors='ignore', inplace=True)
        self.name = 'TorchProphet'
        self.history = None
        self.train_holiday_names = None

    def validate_inputs(self):
        if False:
            while True:
                i = 10
        '\n        Validates the inputs to NeuralProphet.\n        '
        log.error('Not required in NeuralProphet as all inputs are automatically checked.')

    def validate_column_name(self, name, check_holidays=True, check_seasonalities=True, check_regressors=True):
        if False:
            for i in range(10):
                print('nop')
        'Validates the name of a seasonality, holiday, or regressor.\n\n        Parameters\n        ----------\n        name: string\n        check_holidays: bool check if name already used for holiday\n        check_seasonalities: bool check if name already used for seasonality\n        check_regressors: bool check if name already used for regressor\n        '
        super(TorchProphet, self)._validate_column_name(name=name, events=check_holidays, seasons=check_seasonalities, regressors=check_regressors, covariates=check_regressors)

    def setup_dataframe(self, df, initialize_scales=False):
        if False:
            print('Hello World!')
        '\n        Dummy function that raises an error.\n\n        This function is not supported in NeuralProphet.\n        '
        log.error('Not required in NeuralProphet as the dataframe is automatically prepared using the private `_normalize` function.')

    def fit(self, df, **kwargs):
        if False:
            i = 10
            return i + 15
        "Fit the NeuralProphet model.\n\n        This sets self.params to contain the fitted model parameters. It is a\n        dictionary parameter names as keys and the following items:\n            k (Mx1 array): M posterior samples of the initial slope.\n            m (Mx1 array): The initial intercept.\n            delta (MxN array): The slope change at each of N changepoints.\n            beta (MxK matrix): Coefficients for K seasonality features.\n            sigma_obs (Mx1 array): Noise level.\n        Note that M=1 if MAP estimation.\n\n        Parameters\n        ----------\n        df: pd.DataFrame containing the history. Must have columns ds (date\n            type) and y, the time series. If self.growth is 'logistic', then\n            df must also have a column cap that specifies the capacity at\n            each ds.\n        kwargs: Additional arguments passed to the optimizing or sampling\n            functions in Stan.\n\n        Returns\n        -------\n        The fitted NeuralProphet object.\n        "
        if 'cap' in df.columns:
            raise NotImplementedError('Saturating forecasts using cap is not supported in NeuralProphet.')
        if 'show_progress' in kwargs:
            del kwargs['show_progress']
        if hasattr(self, 'events_df'):
            df = self.create_df_with_events(df, self.events_df)
        metrics_df = super(TorchProphet, self).fit(df=df, **kwargs)
        self.history = df
        return metrics_df

    def predict(self, df=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Predict using the NeuralProphet model.\n\n        Parameters\n        ----------\n        df: pd.DataFrame with dates for predictions (column ds), and capacity\n            (column cap) if logistic growth. If not provided, predictions are\n            made on the history.\n\n        Returns\n        -------\n        A pd.DataFrame with the forecast components.\n        '
        if df is None:
            df = self.history.copy()
        df = super(TorchProphet, self).predict(df=df, **kwargs)
        for column in df.columns:
            if 'event_' in column:
                df[column.replace('event_', '')] = df[column]
        return df

    def predict_trend(self, df):
        if False:
            i = 10
            return i + 15
        'Predict trend using the NeuralProphet model.\n\n        Parameters\n        ----------\n        df: Prediction dataframe.\n\n        Returns\n        -------\n        Vector with trend on prediction dates.\n        '
        df = super(TorchProphet, self).predict_trend(self, df, quantile=0.5)
        return df['trend'].to_numpy()

    def make_future_dataframe(self, periods, freq='D', include_history=True, **kwargs):
        if False:
            print('Hello World!')
        "Simulate the trend using the extrapolated generative model.\n\n        Parameters\n        ----------\n        periods: Int number of periods to forecast forward.\n        freq: Any valid frequency for pd.date_range, such as 'D' or 'M'.\n        include_history: Boolean to include the historical dates in the data\n            frame for predictions.\n\n        Returns\n        -------\n        pd.Dataframe that extends forward from the end of self.history for the\n        requested number of periods.\n        "
        if freq == 'M':
            periods = periods * 30
        if hasattr(self, 'events_df'):
            df_future = super(TorchProphet, self).make_future_dataframe(df=self.history, events_df=self.events_df, periods=periods, n_historic_predictions=include_history, **kwargs)
        else:
            df_future = super(TorchProphet, self).make_future_dataframe(df=self.history, periods=periods, n_historic_predictions=include_history, **kwargs)
        return df_future

    def add_seasonality(self, name, period, fourier_order, prior_scale=None, mode=None, condition_name=None, **kwargs):
        if False:
            return 10
        "Add a seasonal component with specified period, number of Fourier\n        components, and prior scale.\n\n        Increasing the number of Fourier components allows the seasonality to\n        change more quickly (at risk of overfitting). Default values for yearly\n        and weekly seasonalities are 10 and 3 respectively.\n\n        Increasing prior scale will allow this seasonality component more\n        flexibility, decreasing will dampen it. If not provided, will use the\n        seasonality_prior_scale provided on initialization (defaults\n        to 10).\n\n        Mode can be specified as either 'additive' or 'multiplicative'. If not\n        specified, self.seasonality_mode will be used (defaults to additive).\n        Additive means the seasonality will be added to the trend,\n        multiplicative means it will multiply the trend.\n\n        If condition_name is provided, the dataframe passed to `fit` and\n        `predict` should have a column with the specified condition_name\n        containing booleans which decides when to apply seasonality.\n\n        Parameters\n        ----------\n        name: string name of the seasonality component.\n        period: float number of days in one period.\n        fourier_order: int number of Fourier components to use.\n        prior_scale: Not supported in NeuralProphet.\n        mode: optional 'additive' or 'multiplicative'\n        condition_name: Not supported in NeuralProphet.\n\n        Returns\n        -------\n        The NeuralProphet object.\n        "
        if condition_name:
            raise NotImplementedError('Conditioning on seasonality is not supported in NeuralProphet.')
        if prior_scale:
            log.warning('Prior scale is not supported in NeuralProphet. Use the `regularisation` parameter for regularisation.')
        try:
            self.season_config.mode = mode
        except AttributeError:
            log.warning('Cannot set the seasonality mode attribute in NeuralProphet. Pleas inspect manually.')
        return super(TorchProphet, self).add_seasonality(name, period, fourier_order, **kwargs)

    def add_regressor(self, name, prior_scale=None, standardize='auto', mode='additive', **kwargs):
        if False:
            return 10
        "Add an additional (future) regressor to be used for fitting and predicting.\n\n        Parameters\n        ----------\n        name: string name of the regressor.\n        prior_scale: Not supported in NeuralProphet.\n        standardize: optional, specify whether this regressor will be\n            standardized prior to fitting. Can be 'auto' (standardize if not\n            binary), True, or False.\n        mode: optional, 'additive' or 'multiplicative'. Defaults to\n            self.seasonality_mode. Not supported in NeuralProphet.\n\n        Returns\n        -------\n        The NeuralProphet object.\n        "
        if prior_scale:
            log.warning('Prior scale is not supported in NeuralProphet. Use the `regularisation` parameter for regularisation.')
        super(TorchProphet, self).add_future_regressor(name, normalize=standardize, **kwargs)
        return self

    def add_country_holidays(self, country_name, **kwargs):
        if False:
            while True:
                i = 10
        "Add in built-in holidays for the specified country.\n\n        These holidays will be included in addition to any specified on model\n        initialization.\n\n        Holidays will be calculated for arbitrary date ranges in the history\n        and future. See the online documentation for the list of countries with\n        built-in holidays.\n\n        Built-in country holidays can only be set for a single country.\n\n        Parameters\n        ----------\n        country_name: Name of the country, like 'UnitedStates' or 'US'\n\n        Returns\n        -------\n        The NeuralProphet object.\n        "
        super(TorchProphet, self).add_country_holidays(country_name=country_name, **kwargs)

    def plot(self, fcst, ax=None, uncertainty=True, plot_cap=True, xlabel='ds', ylabel='y', figsize=(10, 6), include_legend=False, **kwargs):
        if False:
            i = 10
            return i + 15
        'Plot the NeuralProphet forecast.\n\n        Parameters\n        ----------\n        fcst: pd.DataFrame output of self.predict.\n        ax: Optional matplotlib axes on which to plot.\n        uncertainty: Not supported in NeuralProphet.\n        plot_cap: Not supported in NeuralProphet.\n        xlabel: Optional label name on X-axis\n        ylabel: Optional label name on Y-axis\n        figsize: Optional tuple width, height in inches.\n        include_legend: Not supported in NeuralProphet.\n\n        Returns\n        -------\n        A matplotlib figure.\n        '
        log.warning('The attributes `uncertainty`, `plot_cap` and `include_legend` are not supported by NeuralProphet')
        fig = super(TorchProphet, self).plot(fcst=fcst, ax=ax, xlabel=xlabel, ylabel=ylabel, figsize=figsize, **kwargs)
        return fig

    def plot_components(self, fcst, uncertainty=True, plot_cap=True, weekly_start=0, yearly_start=0, figsize=None, **kwargs):
        if False:
            i = 10
            return i + 15
        'Plot the NeuralProphet forecast components.\n\n        Will plot whichever are available of: trend, holidays, weekly\n        seasonality, and yearly seasonality.\n\n        Parameters\n        ----------\n        fcst: pd.DataFrame output of self.predict.\n        uncertainty: Not supported in NeuralProphet.\n        plot_cap: Not supported in NeuralProphet.\n        weekly_start: Not supported in NeuralProphet.\n        yearly_start: Not supported in NeuralProphet.\n        figsize: Optional tuple width, height in inches.\n\n        Returns\n        -------\n        A matplotlib figure.\n        '
        log.warning('The attributes `uncertainty`, `plot_cap`, `weekly_start` and `yearly_start` are not supported by NeuralProphet')
        fig = super(TorchProphet, self).plot_components(fcst=fcst, figsize=figsize, **kwargs)
        return fig

def plot(self, fcst, ax=None, uncertainty=True, plot_cap=True, xlabel='ds', ylabel='y', figsize=(10, 6), include_legend=False, **kwargs):
    if False:
        i = 10
        return i + 15
    'Plot the NeuralProphet forecast.\n\n    Parameters\n    ----------\n    fcst: pd.DataFrame output of self.predict.\n    ax: Optional matplotlib axes on which to plot.\n    uncertainty: Not supported in NeuralProphet.\n    plot_cap: Not supported in NeuralProphet.\n    xlabel: Optional label name on X-axis\n    ylabel: Optional label name on Y-axis\n    figsize: Optional tuple width, height in inches.\n    include_legend: Not supported in NeuralProphet.\n\n    Returns\n    -------\n    A matplotlib figure.\n    '
    log.warning('The attributes `uncertainty`, `plot_cap` and `include_legend` are not supported by NeuralProphet')
    fig = super(TorchProphet, self).plot(fcst=fcst, ax=ax, xlabel=xlabel, ylabel=ylabel, figsize=figsize, **kwargs)
    return fig

def plot_plotly(self, fcst, ax=None, uncertainty=True, plot_cap=True, xlabel='ds', ylabel='y', figsize=(10, 6), include_legend=False, **kwargs):
    if False:
        i = 10
        return i + 15
    'Plot the NeuralProphet forecast.\n\n    Parameters\n    ----------\n    fcst: pd.DataFrame output of self.predict.\n    ax: Optional matplotlib axes on which to plot.\n    uncertainty: Not supported in NeuralProphet.\n    plot_cap: Not supported in NeuralProphet.\n    xlabel: Optional label name on X-axis\n    ylabel: Optional label name on Y-axis\n    figsize: Optional tuple width, height in inches.\n    include_legend: Not supported in NeuralProphet.\n\n    Returns\n    -------\n    A matplotlib figure.\n    '
    log.warning('The attributes `uncertainty`, `plot_cap` and `include_legend` are not supported by NeuralProphet')
    fig = super(TorchProphet, self).plot(fcst=fcst, ax=ax, xlabel=xlabel, ylabel=ylabel, figsize=figsize, plotting_backend='plotly', **kwargs)
    return fig

def plot_components(m, fcst, uncertainty=True, plot_cap=True, weekly_start=0, yearly_start=0, figsize=None, **kwargs):
    if False:
        print('Hello World!')
    '\n    Plot the NeuralProphet forecast components.\n\n    Will plot whichever are available of: trend, holidays, weekly\n    seasonality, yearly seasonality, and additive and multiplicative extra\n    regressors.\n\n    Parameters\n    ----------\n    m: NeuralProphet model.\n    fcst: pd.DataFrame output of m.predict.\n    uncertainty: Not supported in NeuralProphet.\n    plot_cap: Not supported in NeuralProphet.\n    weekly_start: Not supported in NeuralProphet.\n    yearly_start: Not supported in NeuralProphet.\n    figsize: Optional tuple width, height in inches.\n\n    Returns\n    -------\n    A matplotlib figure.\n    '
    log.warning('The attributes `uncertainty`, `plot_cap`, `weekly_start` and `yearly_start` are not supported by NeuralProphet')
    fig = m.plot_components(fcst, **kwargs)
    return fig

def plot_components_plotly(m, fcst, uncertainty=True, plot_cap=True, figsize=(900, 200), **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Plot the NeuralProphet forecast components using Plotly.\n    See plot_plotly() for Plotly setup instructions\n\n    Will plot whichever are available of: trend, holidays, weekly\n    seasonality, yearly seasonality, and additive and multiplicative extra\n    regressors.\n\n    Parameters\n    ----------\n    m: NeuralProphet model.\n    fcst: pd.DataFrame output of m.predict.\n    uncertainty: Not supported in NeuralProphet.\n    plot_cap: Not supported in NeuralProphet.\n    figsize: Not supported in NeuralProphet.\n    Returns\n    -------\n    A Plotly Figure.\n    '
    log.warning('The attributes `uncertainty`, `plot_cap`, `weekly_start` and `yearly_start` are not supported by NeuralProphet')
    fig = m.plot_components(fcst, figsize=None, plotting_backend='plotly', **kwargs)
    return fig