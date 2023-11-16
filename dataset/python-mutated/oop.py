from typing import Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import pycaret.internal.patches.sklearn
import pycaret.internal.patches.yellowbrick
import pycaret.internal.persistence
import pycaret.internal.preprocess
from pycaret.containers.metrics.clustering import get_all_metric_containers
from pycaret.containers.models.clustering import get_all_model_containers
from pycaret.internal.logging import get_logger
from pycaret.internal.pycaret_experiment.unsupervised_experiment import _UnsupervisedExperiment
from pycaret.utils.generic import MLUsecase
LOGGER = get_logger()

class ClusteringExperiment(_UnsupervisedExperiment):

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self._ml_usecase = MLUsecase.CLUSTERING
        self.exp_name_log = 'cluster-default-name'
        self._available_plots = {'pipeline': 'Pipeline Plot', 'cluster': 't-SNE (3d) Dimension Plot', 'tsne': 'Cluster t-SNE (3d)', 'elbow': 'Elbow Plot', 'silhouette': 'Silhouette Plot', 'distance': 'Distance Plot', 'distribution': 'Distribution Plot'}

    def _get_models(self, raise_errors: bool=True) -> Tuple[dict, dict]:
        if False:
            print('Hello World!')
        all_models = {k: v for (k, v) in get_all_model_containers(self, raise_errors=raise_errors).items() if not v.is_special}
        all_models_internal = get_all_model_containers(self, raise_errors=raise_errors)
        return (all_models, all_models_internal)

    def _get_metrics(self, raise_errors: bool=True) -> dict:
        if False:
            i = 10
            return i + 15
        return get_all_metric_containers(self.variables, raise_errors=raise_errors)

    def _get_default_plots_to_log(self) -> List[str]:
        if False:
            while True:
                i = 10
        return ['cluster', 'distribution', 'elbow']

    def predict_model(self, estimator, data: pd.DataFrame, ml_usecase: Optional[MLUsecase]=None) -> pd.DataFrame:
        if False:
            print('Hello World!')
        "\n        This function generates cluster labels using a trained model.\n\n        Example\n        -------\n        >>> from pycaret.datasets import get_data\n        >>> jewellery = get_data('jewellery')\n        >>> from pycaret.clustering import *\n        >>> exp_name = setup(data = jewellery)\n        >>> kmeans = create_model('kmeans')\n        >>> kmeans_predictions = predict_model(model = kmeans, data = unseen_data)\n\n\n        model: scikit-learn compatible object\n            Trained Model Object.\n\n\n        data : pandas.DataFrame\n            Shape (n_samples, n_features) where n_samples is the number of samples and\n            n_features is the number of features.\n\n\n        Returns:\n            pandas.DataFrame\n\n\n        Warnings\n        --------\n        - Models that do not support 'predict' method cannot be used in the ``predict_model``.\n\n        - The behavior of the predict_model is changed in version 2.1 without backward compatibility.\n        As such, the pipelines trained using the version (<= 2.0), may not work for inference\n        with version >= 2.1. You can either retrain your models with a newer version or downgrade\n        the version for inference.\n\n\n        "
        return super().predict_model(estimator, data, ml_usecase)

    def plot_model(self, estimator, plot: str='auc', scale: float=1, save: Union[str, bool]=False, fold: Optional[Union[int, Any]]=None, fit_kwargs: Optional[dict]=None, plot_kwargs: Optional[dict]=None, groups: Optional[Union[str, Any]]=None, feature_name: Optional[str]=None, label: bool=False, verbose: bool=True, display_format: Optional[str]=None) -> Optional[str]:
        if False:
            return 10
        "\n        This function analyzes the performance of a trained model.\n\n\n        Example\n        -------\n        >>> from pycaret.datasets import get_data\n        >>> jewellery = get_data('jewellery')\n        >>> from pycaret.clustering import *\n        >>> exp_name = setup(data = jewellery)\n        >>> kmeans = create_model('kmeans')\n        >>> plot_model(kmeans, plot = 'cluster')\n\n\n        model: scikit-learn compatible object\n            Trained Model Object\n\n\n        plot: str, default = 'cluster'\n            List of available plots (ID - Name):\n\n            * 'cluster' - Cluster PCA Plot (2d)\n            * 'tsne' - Cluster t-SNE (3d)\n            * 'elbow' - Elbow Plot\n            * 'silhouette' - Silhouette Plot\n            * 'distance' - Distance Plot\n            * 'distribution' - Distribution Plot\n\n\n        feature: str, default = None\n            Feature to be evaluated when plot = 'distribution'. When ``plot`` type is\n            'cluster' or 'tsne' feature column is used as a hoverover tooltip and/or\n            label when the ``label`` param is set to True. When the ``plot`` type is\n            'cluster' or 'tsne' and feature is None, first column of the dataset is\n            used.\n\n\n        label: bool, default = False\n            Name of column to be used as data labels. Ignored when ``plot`` is not\n            'cluster' or 'tsne'.\n\n\n        scale: float, default = 1\n            The resolution scale of the figure.\n\n\n        save: bool, default = False\n            When set to True, plot is saved in the current working directory.\n\n\n        display_format: str, default = None\n            To display plots in Streamlit (https://www.streamlit.io/), set this to 'streamlit'.\n            Currently, not all plots are supported.\n\n\n        Returns:\n            Path to saved file, if any.\n\n        "
        return super().plot_model(estimator, plot, scale, save, fold, fit_kwargs, plot_kwargs, groups, feature_name, label, verbose, display_format)

    def get_metrics(self, reset: bool=False, include_custom: bool=True, raise_errors: bool=True) -> pd.DataFrame:
        if False:
            i = 10
            return i + 15
        "\n        Returns table of metrics available.\n\n\n        Example\n        -------\n        >>> from pycaret.datasets import get_data\n        >>> jewellery = get_data('jewellery')\n        >>> from pycaret.clustering import *\n        >>> exp_name = setup(data = jewellery)\n        >>> all_metrics = get_metrics()\n\n\n        reset: bool, default = False\n            If True, will reset all changes made using add_metric() and get_metric().\n\n\n        include_custom: bool, default = True\n            Whether to include user added (custom) metrics or not.\n\n\n        raise_errors: bool, default = True\n            If False, will suppress all exceptions, ignoring models\n            that couldn't be created.\n\n\n        Returns:\n            pandas.DataFrame\n\n        "
        if reset and (not self._setup_ran):
            raise ValueError('setup() needs to be ran first.')
        np.random.seed(self.seed)
        if reset:
            self._all_metrics = self._get_metrics(raise_errors=raise_errors)
        metric_containers = self._all_metrics
        rows = [v.get_dict() for (k, v) in metric_containers.items()]
        df = pd.DataFrame(rows)
        df.set_index('ID', inplace=True, drop=True)
        if not include_custom:
            df = df[df['Custom'] is False]
        return df

    def add_metric(self, id: str, name: str, score_func: type, greater_is_better: bool=True, needs_ground_truth: bool=False, **kwargs) -> pd.Series:
        if False:
            for i in range(10):
                print('nop')
        '\n        Adds a custom metric to be used in all functions.\n\n\n        id: str\n            Unique id for the metric.\n\n\n        name: str\n            Display name of the metric.\n\n\n        score_func: type\n            Score function (or loss function) with signature ``score_func(y, y_pred, **kwargs)``.\n\n\n        greater_is_better: bool, default = True\n            Whether score_func is a score function (default), meaning high is good,\n            or a loss function, meaning low is good. In the latter case, the\n            scorer object will sign-flip the outcome of the score_func.\n\n\n        multiclass: bool, default = True\n            Whether the metric supports multiclass problems.\n\n\n        **kwargs:\n            Arguments to be passed to score function.\n\n        Returns:\n            pandas.Series\n\n        '
        if not self._setup_ran:
            raise ValueError('setup() needs to be ran first.')
        if id in self._all_metrics:
            raise ValueError('id already present in metrics dataframe.')
        new_metric = pycaret.containers.metrics.clustering.ClusterMetricContainer(id=id, name=name, score_func=score_func, args=kwargs, display_name=name, greater_is_better=greater_is_better, needs_ground_truth=needs_ground_truth, is_custom=True)
        self._all_metrics[id] = new_metric
        new_metric = new_metric.get_dict()
        new_metric = pd.Series(new_metric, name=id.replace(' ', '_')).drop('ID')
        return new_metric

    def remove_metric(self, name_or_id: str):
        if False:
            print('Hello World!')
        "\n        Removes a metric used for evaluation.\n\n\n        Example\n        -------\n        >>> from pycaret.datasets import get_data\n        >>> jewellery = get_data('jewellery')\n        >>> from pycaret.clustering import *\n        >>> exp_name = setup(data = jewellery)\n        >>> remove_metric('cs')\n\n\n        name_or_id: str\n            Display name or ID of the metric.\n\n\n        Returns:\n            None\n\n        "
        if not self._setup_ran:
            raise ValueError('setup() needs to be ran first.')
        try:
            self._all_metrics.pop(name_or_id)
            return
        except Exception:
            pass
        try:
            k_to_remove = next((k for (k, v) in self._all_metrics.items() if v.name == name_or_id))
            self._all_metrics.pop(k_to_remove)
            return
        except Exception:
            pass
        raise ValueError(f"No metric 'Display Name' or 'ID' (index) {name_or_id} present in the metrics repository.")