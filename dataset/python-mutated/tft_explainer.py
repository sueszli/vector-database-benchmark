"""
TFT Explainer for Temporal Fusion Transformer (TFTModel)
------------------------------------

The `TFTExplainer` uses a trained :class:`TFTModel <darts.models.forecasting.tft_model.TFTModel>` and extracts the
explainability information from the model.

- :func:`plot_variable_selection() <TFTExplainer.plot_variable_selection>` plots the variable selection weights for
  each of the input features.
  - encoder importance: historic part of target, past covariates and historic part of future covariates
  - decoder importance: future part of future covariates
  - static covariates importance: the numeric and catageorical static covariates importance

- :func:`plot_attention() <TFTExplainer.plot_attention>` plots the transformer attention that the `TFTModel` applies
  on the given past and future input. The attention is aggregated over all attention heads.

The attention and feature importance values can be extracted using the :class:`TFTExplainabilityResult
<darts.explainability.explainability_result.TFTExplainabilityResult>` returned by
:func:`explain() <TFTExplainer.explain>`. An example of this is shown in the method description.

We also show how to use the `TFTExplainer` in the example notebook of the `TFTModel` `here
<https://unit8co.github.io/darts/examples/13-TFT-examples.html#Explainability>`_.
"""
from typing import Dict, List, Optional, Sequence, Union
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch import Tensor
from darts import TimeSeries
from darts.explainability import TFTExplainabilityResult
from darts.explainability.explainability import _ForecastingModelExplainer
from darts.logging import get_logger, raise_log
from darts.models import TFTModel
from darts.utils.timeseries_generation import generate_index
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
logger = get_logger(__name__)

class TFTExplainer(_ForecastingModelExplainer):
    model: TFTModel

    def __init__(self, model: TFTModel, background_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, background_past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, background_future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Explainer class for the `TFTModel`.\n\n        **Definitions**\n\n        - A background series is a `TimeSeries` that is used as a default for generating the explainability result\n          (if no `foreground` is passed to :func:`explain() <TFTExplainer.explain>`).\n        - A foreground series is a `TimeSeries` that can be passed to :func:`explain() <TFTExplainer.explain>` to use\n          instead of the background for generating the explainability result.\n\n        Parameters\n        ----------\n        model\n            The fitted `TFTModel` to be explained.\n        background_series\n            Optionally, a series or list of series to use as a default target series for the explanations.\n            Optional if `model` was trained on a single target series. By default, it is the `series` used at fitting\n            time.\n            Mandatory if `model` was trained on multiple (sequence of) target series.\n        background_past_covariates\n            Optionally, a past covariates series or list of series to use as a default past covariates series\n            for the explanations. The same requirements apply as for `background_series` .\n        background_future_covariates\n            Optionally, a future covariates series or list of series to use as a default future covariates series\n            for the explanations. The same requirements apply as for `background_series`.\n\n        Examples\n        --------\n        >>> from darts.datasets import AirPassengersDataset\n        >>> from darts.explainability.tft_explainer import TFTExplainer\n        >>> from darts.models import TFTModel\n        >>> series = AirPassengersDataset().load()\n        >>> model = TFTModel(\n        >>>     input_chunk_length=12,\n        >>>     output_chunk_length=6,\n        >>>     add_encoders={"cyclic": {"future": ["hour"]}}\n        >>> )\n        >>> model.fit(series)\n        >>> # create the explainer and generate explanations\n        >>> explainer = TFTExplainer(model)\n        >>> results = explainer.explain()\n        >>> # plot the results\n        >>> explainer.plot_attention(results, plot_type="all")\n        >>> explainer.plot_variable_selection(results)\n        '
        super().__init__(model, background_series=background_series, background_past_covariates=background_past_covariates, background_future_covariates=background_future_covariates, requires_background=True, requires_covariates_encoding=False, check_component_names=False, test_stationarity=False)
        if model.add_relative_index:
            if self.future_covariates_components is not None:
                self.future_covariates_components.append('add_relative_index')
            else:
                self.future_covariates_components = ['add_relative_index']

    def explain(self, foreground_series: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, foreground_past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, foreground_future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]]=None, horizons: Optional[Sequence[int]]=None, target_components: Optional[Sequence[str]]=None) -> TFTExplainabilityResult:
        if False:
            i = 10
            return i + 15
        'Returns the :class:`TFTExplainabilityResult\n        <darts.explainability.explainability_result.TFTExplainabilityResult>` result for all series in\n        `foreground_series`. If `foreground_series` is `None`, will use the `background` input\n        from `TFTExplainer` creation (either the `background` passed to creation, or the series stored in the\n        `TFTModel` in case it was only trained on a single series).\n        For each series, the results contain the attention heads, encoder variable importances, decoder variable\n        importances, and static covariates importances.\n\n        Parameters\n        ----------\n        foreground_series\n            Optionally, one or a sequence of target `TimeSeries` to be explained. Can be multivariate.\n            If not provided, the background `TimeSeries` will be explained instead.\n        foreground_past_covariates\n            Optionally, one or a sequence of past covariates `TimeSeries` if required by the forecasting model.\n        foreground_future_covariates\n            Optionally, one or a sequence of future covariates `TimeSeries` if required by the forecasting model.\n        horizons\n            This parameter is not used by the `TFTExplainer`.\n        target_components\n            This parameter is not used by the `TFTExplainer`.\n\n        Returns\n        -------\n        TFTExplainabilityResult\n            The explainability result containing the attention heads, encoder variable importances, decoder variable\n            importances, and static covariates importances.\n\n        Examples\n        --------\n        >>> explainer = TFTExplainer(model)  # requires `background` if model was trained on multiple series\n\n        Optionally, give a foreground input to generate the explanation on a new input.\n        Otherwise, leave it empty to compute the explanation on the background from `TFTExplainer` creation\n\n        >>> explain_results = explainer.explain(\n        >>>     foreground_series=foreground_series,\n        >>>     foreground_past_covariates=foreground_past_covariates,\n        >>>     foreground_future_covariates=foreground_future_covariates,\n        >>> )\n        >>> attn = explain_results.get_attention()\n        >>> importances = explain_results.get_feature_importances()\n        '
        if target_components is not None or horizons is not None:
            logger.warning('`horizons`, and `target_components` are not supported by `TFTExplainer` and will be ignored.')
        super().explain(foreground_series, foreground_past_covariates, foreground_future_covariates)
        (foreground_series, foreground_past_covariates, foreground_future_covariates, _, _, _, _) = self._process_foreground(foreground_series, foreground_past_covariates, foreground_future_covariates)
        (horizons, _) = self._process_horizons_and_targets(None, None)
        preds = self.model.predict(n=self.n, series=foreground_series, past_covariates=foreground_past_covariates, future_covariates=foreground_future_covariates)
        attention_heads = self.model.model._attn_out_weights.detach().cpu().numpy().sum(axis=-2)
        encoder_importance = self._encoder_importance
        decoder_importance = self._decoder_importance
        static_covariates_importance = self._static_covariates_importance
        horizon_idx = [h - 1 for h in horizons]
        results = []
        icl = self.model.input_chunk_length
        for (idx, (series, pred_series)) in enumerate(zip(foreground_series, preds)):
            times = series.time_index[-icl:].union(pred_series.time_index)
            attention = TimeSeries.from_times_and_values(values=np.take(attention_heads[idx], horizon_idx, axis=0).T, times=times, columns=[f'horizon {str(i)}' for i in horizons])
            results.append({'attention': attention, 'encoder_importance': encoder_importance.iloc[idx:idx + 1], 'decoder_importance': decoder_importance.iloc[idx:idx + 1], 'static_covariates_importance': static_covariates_importance.iloc[idx:idx + 1]})
        return TFTExplainabilityResult(explanations=results[0] if len(results) == 1 else results)

    def plot_variable_selection(self, expl_result: TFTExplainabilityResult, fig_size=None, max_nr_series: int=5):
        if False:
            for i in range(10):
                print('nop')
        'Plots the variable selection / feature importances of the `TFTModel` based on the input.\n        The figure includes three subplots:\n\n        - encoder importances: contains the past target, past covariates, and historic future covariates importance\n          on the encoder (input chunk)\n        - decoder importances: contains the future covariates importance on the decoder (output chunk)\n        - static covariates importances: contains the numeric and / or categorical static covariates importance\n\n        Parameters\n        ----------\n        expl_result\n            A `TFTExplainabilityResult` object. Corresponds to the output of :func:`explain() <TFTExplainer.explain>`.\n        fig_size\n            The size of the figure to be plotted.\n        max_nr_series\n            The maximum number of plots to show in case `expl_result` was computed on multiple series.\n        '
        encoder_importance = expl_result.get_encoder_importance()
        decoder_importance = expl_result.get_decoder_importance()
        static_covariates_importance = expl_result.get_static_covariates_importance()
        if not isinstance(encoder_importance, list):
            encoder_importance = [encoder_importance]
            decoder_importance = [decoder_importance]
            static_covariates_importance = [static_covariates_importance]
        uses_static_covariates = not static_covariates_importance[0].empty
        for (idx, (enc_imp, dec_imp, stc_imp)) in enumerate(zip(encoder_importance, decoder_importance, static_covariates_importance)):
            (fig, axes) = plt.subplots(nrows=3 if uses_static_covariates else 2, sharex=True, figsize=fig_size)
            self._plot_cov_selection(enc_imp, title='Encoder variable importance', ax=axes[0])
            axes[0].set_xlabel('')
            self._plot_cov_selection(dec_imp, title='Decoder variable importance', ax=axes[1])
            if uses_static_covariates:
                axes[1].set_xlabel('')
                self._plot_cov_selection(stc_imp, title='Static variable importance', ax=axes[2])
            fig.tight_layout()
            plt.show()
            if idx + 1 == max_nr_series:
                break

    def plot_attention(self, expl_result: TFTExplainabilityResult, plot_type: Optional[Literal['all', 'time', 'heatmap']]='all', show_index_as: Literal['relative', 'time']='relative', ax: Optional[matplotlib.axes.Axes]=None, max_nr_series: int=5, show_plot: bool=True) -> matplotlib.axes.Axes:
        if False:
            i = 10
            return i + 15
        'Plots the attention heads of the `TFTModel`.\n\n        Parameters\n        ----------\n        expl_result\n            A `TFTExplainabilityResult` object. Corresponds to the output of :func:`explain() <TFTExplainer.explain>`.\n        plot_type\n            The type of attention head plot. One of ("all", "time", "heatmap").\n            If "all", will plot the attention per horizon (given the horizons in the `TFTExplainabilityResult`).\n            The maximum horizon corresponds to the `output_chunk_length` of the trained `TFTModel`.\n            If "time", will plot the mean attention over all horizons.\n            If "heatmap", will plot the attention per horizon on a heat map. The horizons are shown on the y-axis,\n            and times / relative indices on the x-axis.\n        show_index_as\n            The type of index to be shown. One of ("relative", "time").\n            If "relative", will plot the x-axis from `(-input_chunk_length, output_chunk_length - 1)`. `0` corresponds\n            to the first prediction point.\n            If "time", will plot the x-axis with the actual time index (or range index) of the corresponding\n            `TFTExplainabilityResult`.\n        ax\n            Optionally, an axis to plot on. Only effective on a single `expl_result`.\n        max_nr_series\n            The maximum number of plots to show in case `expl_result` was computed on multiple series.\n        show_plot\n            Whether to show the plot.\n        '
        single_series = False
        attentions = expl_result.get_explanation(component='attention')
        if isinstance(attentions, TimeSeries):
            attentions = [attentions]
            single_series = True
        for (idx, attention) in enumerate(attentions):
            if ax is None or not single_series:
                (fig, ax) = plt.subplots()
            if show_index_as == 'relative':
                x_ticks = generate_index(start=-self.model.input_chunk_length, end=self.n - 1)
                attention = TimeSeries.from_times_and_values(times=generate_index(start=-self.model.input_chunk_length, end=self.n - 1), values=attention.values(copy=False), columns=attention.components)
                x_label = 'Index relative to first prediction point'
            elif show_index_as == 'time':
                x_ticks = attention.time_index
                x_label = 'Time index'
            else:
                (x_label, x_ticks) = (None, None)
                raise_log(ValueError("`show_index_as` must either be 'relative', or 'time'."))
            prediction_start_color = 'red'
            if plot_type == 'all':
                ax_title = 'Attention per Horizon'
                y_label = 'Attention'
                attention.plot(max_nr_components=-1, ax=ax)
            elif plot_type == 'time':
                ax_title = 'Mean Attention'
                y_label = 'Attention'
                attention.mean(1).plot(label='Mean Attention Head', ax=ax)
            elif plot_type == 'heatmap':
                ax_title = 'Attention Heat Map'
                y_label = 'Horizon'
                (x, y) = np.meshgrid(x_ticks, np.arange(1, self.n + 1))
                c = ax.pcolormesh(x, y, attention.values().transpose(), cmap='hot')
                ax.axis([x.min(), x.max(), y.max(), y.min()])
                prediction_start_color = 'lightblue'
                fig.colorbar(c, ax=ax, orientation='horizontal')
            else:
                raise raise_log(ValueError("`plot_type` must be either 'all', 'time' or 'heatmap'"), logger=logger)
            (y_min, y_max) = ax.get_ylim()
            ax.vlines(x=x_ticks[-self.n], ymin=y_min, ymax=y_max, label='prediction start', ls='dashed', lw=2, colors=prediction_start_color)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            title_suffix = '' if single_series else f': series index {idx}'
            ax.set_title(ax_title + title_suffix)
            ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
            if show_plot:
                plt.show()
            if idx + 1 == max_nr_series:
                break
        return ax

    @property
    def _encoder_importance(self) -> pd.DataFrame:
        if False:
            print('Hello World!')
        'Returns the encoder variable importance of the TFT model.\n\n        The encoder_weights are calculated for the past inputs of the model.\n        The encoder_importance contains the weights of the encoder variable selection network.\n        The encoder variable selection network is used to select the most important static and time dependent\n        covariates. It provides insights which variable are most significant for the prediction problem.\n        See section 4.2 of the paper for more details.\n\n        Returns\n        -------\n        pd.DataFrame\n            The encoder variable importance.\n        '
        return self._get_importance(weight=self.model.model._encoder_sparse_weights, names=self.model.model.encoder_variables)

    @property
    def _decoder_importance(self) -> pd.DataFrame:
        if False:
            i = 10
            return i + 15
        'Returns the decoder variable importance of the TFT model.\n\n        The decoder_weights are calculated for the known future inputs of the model.\n        The decoder_importance contains the weights of the decoder variable selection network.\n        The decoder variable selection network is used to select the most important static and time dependent\n        covariates. It provides insights which variable are most significant for the prediction problem.\n        See section 4.2 of the paper for more details.\n\n        Returns\n        -------\n        pd.DataFrame\n            The importance of the decoder variables.\n        '
        return self._get_importance(weight=self.model.model._decoder_sparse_weights, names=self.model.model.decoder_variables)

    @property
    def _static_covariates_importance(self) -> pd.DataFrame:
        if False:
            print('Hello World!')
        'Returns the static covariates importance of the TFT model.\n\n        The static covariate importances are calculated for the static inputs of the model (numeric and / or\n        categorical). The static variable selection network is used to select the most important static covariates.\n        It provides insights which variable are most significant for the prediction problem.\n        See section 4.2, and 4.3 of the paper for more details.\n\n        Returns\n        -------\n        pd.DataFrame\n            The static covariates importance.\n        '
        return self._get_importance(weight=self.model.model._static_covariate_var, names=self.model.model.static_variables)

    def _get_importance(self, weight: Tensor, names: List[str], n_decimals=3) -> pd.DataFrame:
        if False:
            print('Hello World!')
        'Returns the encoder or decoder variable of the TFT model.\n\n        Parameters\n        ----------\n        weights\n            The weights of the encoder or decoder of the trained TFT model.\n        names\n            The encoder or decoder names saved in the TFT model class.\n        n_decimals\n            The number of decimals to round the importance to.\n\n        Returns\n        -------\n        pd.DataFrame\n            The importance of the variables.\n        '
        if weight is None:
            return pd.DataFrame()
        if weight.ndim == 3:
            weight = weight.unsqueeze(1)
        weights_percentage = weight.detach().cpu().numpy().mean(axis=1).squeeze(axis=1).round(n_decimals) * 100
        name_mapping = self._name_mapping
        importance = pd.DataFrame(weights_percentage, columns=[name_mapping[name] for name in names])
        return importance.transpose().sort_values(0, ascending=True).transpose()

    @property
    def _name_mapping(self) -> Dict[str, str]:
        if False:
            print('Hello World!')
        "Returns the feature name mapping of the TFT model.\n\n        Returns\n        -------\n        Dict[str, str]\n            The feature name mapping. For example\n            {\n                'target_0': 'ice cream',\n                'past_covariate_0': 'heater',\n                'past_covariate_1': 'year',\n                'past_covariate_2': 'month',\n                'future_covariate_0': 'darts_enc_fc_cyc_month_sin',\n                'future_covariate_1': 'darts_enc_fc_cyc_month_cos',\n             }\n        "

        def map_cols(comps, name, suffix):
            if False:
                return 10
            comps = comps if comps is not None else []
            return {f'{name}_{i}': colname + f'_{suffix}' for (i, colname) in enumerate(comps)}
        return {**map_cols(self.target_components, 'target', 'target'), **map_cols(self.static_covariates_components, 'static_covariate', 'statcov'), **map_cols(self.past_covariates_components, 'past_covariate', 'pastcov'), **map_cols(self.future_covariates_components, 'future_covariate', 'futcov')}

    @staticmethod
    def _plot_cov_selection(importance: pd.DataFrame, title: str='Variable importance', ax: Optional[matplotlib.axes.Axes]=None):
        if False:
            while True:
                i = 10
        'Plots the variable importance of the TFT model.\n\n        Parameters\n        ----------\n        importance\n            The encoder / decoder importance.\n        title\n            The title of the plot.\n        ax\n            Optionally, an axis to plot on. Otherwise, will create and plot on a new axis.\n        '
        if ax is None:
            (_, ax) = plt.subplots()
        ax.barh(importance.columns.tolist(), importance.values[0].tolist())
        ax.set_title(title)
        ax.set_ylabel('Variable', fontsize=12)
        ax.set_xlabel('Variable importance in %')
        return ax