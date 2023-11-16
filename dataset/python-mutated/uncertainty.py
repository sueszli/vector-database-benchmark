import re
from dataclasses import dataclass
from typing import Any, List, Tuple, Union
import matplotlib
import numpy as np
import pandas as pd
from neuralprophet.plot_forecast_matplotlib import plot_interval_width_per_timestep, plot_nonconformity_scores
from neuralprophet.plot_forecast_plotly import plot_interval_width_per_timestep as plot_interval_width_per_timestep_plotly
from neuralprophet.plot_forecast_plotly import plot_nonconformity_scores as plot_nonconformity_scores_plotly
from neuralprophet.plot_utils import log_warning_deprecation_plotly, select_plotting_backend

@dataclass
class Conformal:
    """Conformal prediction dataclass

    Parameters
    ----------
    alpha : float or tuple
        user-specified significance level of the prediction interval, float if coverage error spread arbitrarily over
        left and right tails, tuple of two floats for different coverage error over left and right tails respectively
    method : str
        name of conformal prediction technique used

        Options
            * ``naive``: Naive or Absolute Residual
            * ``cqr``: Conformalized Quantile Regression
    n_forecasts : int
        optional, number of steps ahead of prediction time step to forecast
    quantiles : list
        optional, list of quantiles for quantile regression uncertainty estimate

    """
    alpha: Union[float, Tuple[float, float]]
    method: str
    n_forecasts: int
    quantiles: List[float]

    def __post_init__(self):
        if False:
            print('Hello World!')
        if isinstance(self.alpha, float):
            self.symmetrical = True
            self.q_hats = pd.DataFrame(columns=['q_hat_sym'])
        elif self.method == 'naive':
            raise ValueError("Asymmetrical coverage errors are not available for the naive method. Please use one alpha or method='cqr'.")
        else:
            self.symmetrical = False
            (self.alpha_lo, self.alpha_hi) = self.alpha
            self.q_hats = pd.DataFrame(columns=['q_hat_lo', 'q_hat_hi'])
        self.noncon_scores = dict()

    def predict(self, df: pd.DataFrame, df_cal: pd.DataFrame, show_all_PI: bool=False) -> pd.DataFrame:
        if False:
            return 10
        'Apply a given conformal prediction technique to get the uncertainty prediction intervals (or q-hat) for test\n        dataframe.\n\n        Parameters\n        ----------\n            df : pd.DataFrame\n                test dataframe\n            df_cal : pd.DataFrame\n                calibration dataframe\n            show_all_PI : bool\n                whether to return all prediction intervals (including quantile regression and conformal prediction)\n\n            Returns\n            -------\n                pd.DataFrame\n                    test dataframe with uncertainty prediction intervals\n\n        '
        df_qr = df.copy()
        for step_number in range(1, self.n_forecasts + 1):
            noncon_scores = self._get_nonconformity_scores(df_cal, step_number)
            q_hat = self._get_q_hat(df_cal, noncon_scores)
            y_hat_col = f'yhat{step_number}'
            y_hat_lo_col = f'{y_hat_col} {min(self.quantiles) * 100}%'
            y_hat_hi_col = f'{y_hat_col} {max(self.quantiles) * 100}%'
            if self.method == 'naive' and self.symmetrical:
                q_hat_sym = q_hat['q_hat_sym']
                df[y_hat_lo_col] = df[y_hat_col] - q_hat_sym
                df[y_hat_hi_col] = df[y_hat_col] + q_hat_sym
            elif self.method == 'cqr' and self.symmetrical:
                q_hat_sym = q_hat['q_hat_sym']
                df[y_hat_lo_col] = df[y_hat_lo_col] - q_hat_sym
                df[y_hat_hi_col] = df[y_hat_hi_col] + q_hat_sym
            elif self.method == 'cqr' and (not self.symmetrical):
                q_hat_lo = q_hat['q_hat_lo']
                q_hat_hi = q_hat['q_hat_hi']
                df[y_hat_lo_col] = df[y_hat_lo_col] - q_hat_lo
                df[y_hat_hi_col] = df[y_hat_hi_col] + q_hat_hi
            else:
                raise ValueError(f"Unknown conformal prediction method '{self.method}'. Please input either 'naive' or 'cqr'.")
            if step_number == 1:
                self.noncon_scores = noncon_scores
            q_hat_df = pd.DataFrame([q_hat])
            self.q_hats = pd.concat([self.q_hats, q_hat_df], ignore_index=True)
            if show_all_PI:
                df_quantiles = [col for col in df_qr.columns if '%' in col and f'yhat{step_number}' in col]
                df_add = df_qr[df_quantiles]
                if self.method == 'naive':
                    cp_lo_col = f'yhat{step_number} - qhat{step_number}'
                    cp_hi_col = f'yhat{step_number} + qhat{step_number}'
                    df.rename(columns={y_hat_lo_col: cp_lo_col, y_hat_hi_col: cp_hi_col}, inplace=True)
                elif self.method == 'cqr':
                    qr_lo_col = f'yhat{step_number} {max(self.quantiles) * 100}% - qhat{step_number}'
                    qr_hi_col = f'yhat{step_number} {min(self.quantiles) * 100}% + qhat{step_number}'
                    df.rename(columns={y_hat_lo_col: qr_lo_col, y_hat_hi_col: qr_hi_col}, inplace=True)
                df = pd.concat([df, df_add], axis=1, ignore_index=False)
        return df

    def _get_nonconformity_scores(self, df_cal: pd.DataFrame, step_number: int) -> dict:
        if False:
            return 10
        'Get the nonconformity scores using the given conformal prediction technique.\n\n        Parameters\n        ----------\n            df_cal : pd.DataFrame\n                calibration dataframe\n            step_number : int\n                i-th step ahead forecast\n\n            Returns\n            -------\n                Dict[str, np.ndarray]\n                    dictionary with one entry (symmetrical) or two entries (asymmetrical) of nonconformity scores\n\n        '
        y_hat_col = f'yhat{step_number}'
        if self.method == 'cqr':
            quantile_lo = str(min(self.quantiles) * 100)
            quantile_hi = str(max(self.quantiles) * 100)
            quantile_lo_col = f'{y_hat_col} {quantile_lo}%'
            quantile_hi_col = f'{y_hat_col} {quantile_hi}%'
            if self.symmetrical:

                def cqr_scoring_func(row):
                    if False:
                        while True:
                            i = 10
                    return [None, None] if row[quantile_lo_col] is None or row[quantile_hi_col] is None else [max(row[quantile_lo_col] - row['y'], row['y'] - row[quantile_hi_col]), 0 if row[quantile_lo_col] - row['y'] > row['y'] - row[quantile_hi_col] else 1]
                scores_df = df_cal.apply(cqr_scoring_func, axis=1, result_type='expand')
                scores_df.columns = ['scores', 'arg']
                noncon_scores = scores_df['scores'].values
            else:

                def cqr_scoring_func(row):
                    if False:
                        return 10
                    return [None, None] if row[quantile_lo_col] is None or row[quantile_hi_col] is None else [row[quantile_lo_col] - row['y'], row['y'] - row[quantile_hi_col], 0 if row[quantile_lo_col] - row['y'] > row['y'] - row[quantile_hi_col] else 1]
                scores_df = df_cal.apply(cqr_scoring_func, axis=1, result_type='expand')
                scores_df.columns = ['scores_lo', 'scores_hi', 'arg']
                noncon_scores_lo = scores_df['scores_lo'].values
                noncon_scores_hi = scores_df['scores_hi'].values
                noncon_scores_lo: Any = noncon_scores_lo[~pd.isnull(noncon_scores_lo)]
                noncon_scores_hi: Any = noncon_scores_hi[~pd.isnull(noncon_scores_hi)]
                noncon_scores_lo.sort()
                noncon_scores_hi.sort()
                return {'noncon_scores_hi': noncon_scores_lo, 'noncon_scores_lo': noncon_scores_hi}
        else:
            noncon_scores = abs(df_cal['y'] - df_cal[y_hat_col]).values
        noncon_scores: Any = noncon_scores[~pd.isnull(noncon_scores)]
        noncon_scores.sort()
        return {'noncon_scores': noncon_scores}

    def _get_q_hat(self, df_cal: pd.DataFrame, noncon_scores: dict) -> dict:
        if False:
            while True:
                i = 10
        'Get the q_hat that is derived from the nonconformity scores.\n\n        Parameters\n        ----------\n            df_cal : pd.DataFrame\n                calibration dataframe\n            noncon_scores : dict\n                dictionary with one entry (symmetrical) or two entries (asymmetrical) of nonconformity scores\n\n            Returns\n            -------\n                Dict[str, float]\n                    upper and lower q_hat value, or the one-sided prediction interval width\n\n        '
        if self.method == 'cqr' and self.symmetrical is False:
            noncon_scores_lo = noncon_scores['noncon_scores_lo']
            noncon_scores_hi = noncon_scores['noncon_scores_hi']
            q_hat_idx_lo = int(len(noncon_scores_lo) * self.alpha_lo)
            q_hat_idx_hi = int(len(noncon_scores_hi) * self.alpha_hi)
            q_hat_lo = noncon_scores_lo[-q_hat_idx_lo]
            q_hat_hi = noncon_scores_hi[-q_hat_idx_hi]
            return {'q_hat_lo': q_hat_lo, 'q_hat_hi': q_hat_hi}
        else:
            noncon_scores = noncon_scores['noncon_scores']
            q_hat_idx = int(len(noncon_scores) * self.alpha)
            q_hat = noncon_scores[-q_hat_idx]
            return {'q_hat_sym': q_hat}

    def plot(self, plotting_backend=None):
        if False:
            return 10
        "Apply a given conformal prediction technique to get the uncertainty prediction intervals (or q-hats).\n\n        Parameters\n        ----------\n            plotting_backend : str\n                specifies the plotting backend for the nonconformity scores plot, if any\n\n                Options\n                * ``plotly-resampler``: Use the plotly backend for plotting in resample mode. This mode uses the\n                    plotly-resampler package to accelerate visualizing large data by resampling it. For some\n                    environments (colab, pycharm interpreter) plotly-resampler might not properly vizualise the figures.\n                    In this case, consider switching to 'plotly-auto'.\n                * ``plotly``: Use the plotly backend for plotting\n                * ``matplotlib``: use matplotlib for plotting\n                * (default) None: Plotting backend ist set automatically. Use plotly with resampling for jupyterlab\n                    notebooks and vscode notebooks. Automatically switch to plotly without resampling for all other\n                    environments.\n\n        "
        method = self.method.upper() if 'cqr' in self.method.lower() else self.method.title()
        plotting_backend = select_plotting_backend(model=self, plotting_backend=plotting_backend)
        log_warning_deprecation_plotly(plotting_backend)
        initial_q_hat = self.q_hats['q_hat_sym'][0] if self.symmetrical else [self.q_hats['q_hat_lo'][0], self.q_hats['q_hat_hi'][0]]
        if plotting_backend.startswith('plotly'):
            if self.n_forecasts == 1:
                fig = plot_nonconformity_scores_plotly(self.noncon_scores, self.alpha, initial_q_hat, method, resampler_active=plotting_backend == 'plotly-resampler')
            else:
                fig = plot_interval_width_per_timestep_plotly(self.q_hats, method, resampler_active=False)
            fig.show()
        elif self.n_forecasts == 1:
            fig = plot_nonconformity_scores(self.noncon_scores, self.alpha, initial_q_hat, method)
        else:
            fig = plot_interval_width_per_timestep(self.q_hats, method)
        if plotting_backend in ['matplotlib', 'plotly', 'plotly-resampler'] and matplotlib.is_interactive():
            fig

def uncertainty_evaluate(df_forecast: pd.DataFrame) -> pd.DataFrame:
    if False:
        print('Hello World!')
    'Evaluate conformal prediction on test dataframe.\n\n    Parameters\n    ----------\n        df_forecast : pd.DataFrame\n            forecast dataframe with the conformal prediction intervals\n\n    Returns\n    -------\n        pd.DataFrame\n            table containing evaluation metrics such as interval_width and miscoverage_rate\n    '
    df_forecast_eval = df_forecast.dropna(subset=['y', 'yhat1']).reset_index(drop=True)
    df_eval = pd.DataFrame()
    cols = df_forecast_eval.columns
    yhat_cols = [col for col in cols if '%' in col]
    n_forecasts = int(re.search('yhat(\\d+)', yhat_cols[-1]).group(1))
    quantiles = []
    for col in yhat_cols:
        match = re.search('\\d+\\.\\d+', col)
        if match:
            quantiles.append(float(match.group()))
    quantiles = sorted(set(quantiles))
    for step_number in range(1, n_forecasts + 1):
        y = df_forecast_eval['y'].values
        if len([col for col in cols if 'qhat' in col]) > 0:
            qhat_cols = [col for col in cols if f'qhat{step_number}' in col]
            yhat_lo = df_forecast_eval[qhat_cols[0]].values
            yhat_hi = df_forecast_eval[qhat_cols[-1]].values
        else:
            yhat_lo = df_forecast_eval[f'yhat{step_number} {quantiles[0]}%'].values
            yhat_hi = df_forecast_eval[f'yhat{step_number} {quantiles[-1]}%'].values
        (interval_width, miscoverage_rate) = _get_evaluate_metrics_from_dataset(y, yhat_lo, yhat_hi)
        col_names = ['interval_width', 'miscoverage_rate']
        row = [interval_width, miscoverage_rate]
        df_row = pd.DataFrame([row], columns=pd.MultiIndex.from_product([[f'yhat{step_number}'], col_names]))
        df_eval = pd.concat([df_eval, df_row], axis=1)
    return df_eval

def _get_evaluate_metrics_from_dataset(y: np.ndarray, yhat_lo: np.ndarray, yhat_hi: np.ndarray) -> Tuple[float, float]:
    if False:
        print('Hello World!')
    'Infers evaluation parameters based on the evaluation dataframe columns.\n\n    Parameters\n    ----------\n        df_forecast_eval : pd.DataFrame\n            forecast dataframe with the conformal prediction intervals\n\n    Returns\n    -------\n        float, float\n            conformal prediction evaluation metrics\n    '
    quantile_lo_mean = np.mean(yhat_lo)
    quantile_hi_mean = np.mean(yhat_hi)
    interval_width = quantile_hi_mean - quantile_lo_mean
    n_covered = np.sum((y >= yhat_lo) & (y <= yhat_hi))
    coverage_rate = n_covered / len(y)
    miscoverage_rate = 1 - coverage_rate
    return (interval_width, miscoverage_rate)