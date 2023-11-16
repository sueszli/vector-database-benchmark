"""
Static Covariates Transformer
------
"""
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from darts.logging import get_logger, raise_log
from darts.timeseries import TimeSeries
from .fittable_data_transformer import FittableDataTransformer
from .invertible_data_transformer import InvertibleDataTransformer
logger = get_logger(__name__)

class StaticCovariatesTransformer(FittableDataTransformer, InvertibleDataTransformer):

    def __init__(self, transformer_num=None, transformer_cat=None, cols_num: Optional[List[str]]=None, cols_cat: Optional[List[str]]=None, name='StaticCovariatesTransformer', n_jobs: int=1, verbose: bool=False):
        if False:
            i = 10
            return i + 15
        'Generic wrapper class for scalers/encoders/transformers of static covariates. This transformer acts\n        only on static covariates of the series passed to ``fit()``, ``transform()``, ``fit_transform()``, and\n        ``inverse_transform()``. It can both scale numerical features, as well as encode categorical features.\n\n        The underlying ``transformer_num`` and ``transformer_cat`` have to implement the ``fit()``, ``transform()``,\n        and ``inverse_transform()`` methods (typically from scikit-learn).\n\n        By default, numerical and categorical columns/features are inferred and allocated to ``transformer_num`` and\n        ``transformer_cat``, respectively. Alternatively, specify which columns to scale/transform with ``cols_num``\n        and ``cols_cat``.\n\n        Both ``transformer_num`` and ``transformer_cat`` are fit globally on static covariate data from all series\n        passed to :class:`StaticCovariatesTransformer.fit()`\n\n        Parameters\n        ----------\n        transformer_num\n            The transformer to transform numeric static covariate columns with. It must provide ``fit()``,\n            ``transform()`` and ``inverse_transform()`` methods.\n            Default: :class:`sklearn.preprocessing.MinMaxScaler(feature_range=(0, 1))`; this will scale all\n            values between 0 and 1.\n        transformer_cat\n            The encoder to transform categorical static covariate columns with. It must provide ``fit()``,\n            ``transform()`` and ``inverse_transform()`` methods.\n            Default: :class:`sklearn.preprocessing.OrdinalEncoder()`; this will convert categories\n            into integer valued arrays where each integer stands for a specific category.\n        cols_num\n            Optionally, a list of column names for which to apply the numeric transformer ``transformer_num``.\n            By default, the transformer will infer all numerical features based on types, and scale them with\n            `transformer_num`. If an empty list, no column will be scaled.\n        cols_cat\n            Optionally, a list of column names for which to apply the categorical transformer `transformer_cat`.\n            By default, the transformer will infer all categorical features based on types, and transform them with\n            `transformer_cat`. If an empty list, no column will be transformed.\n        name\n            A specific name for the :class:`StaticCovariatesTransformer`.\n        n_jobs\n            The number of jobs to run in parallel. Parallel jobs are created only when a ``Sequence[TimeSeries]`` is\n            passed as input to a method, parallelising operations regarding different ``TimeSeries``. Defaults to `1`\n            (sequential). Setting the parameter to `-1` means using all the available processors.\n            Note: for a small amount of data, the parallelisation overhead could end up increasing the total\n            required amount of time.\n        verbose\n            Optionally, whether to print operations progress\n\n        Examples\n        --------\n        >>> import numpy as np\n        >>> import pandas as pd\n        >>> from darts import TimeSeries\n        >>> from darts.dataprocessing.transformers import StaticCovariatesTransformer\n        >>> static_covs = pd.DataFrame(data={"num": [0, 2, 1], "cat": ["a", "c", "b"]})\n        >>> series = TimeSeries.from_values(\n        >>>     values=np.random.random((10, 3)),\n        >>>     columns=["comp1", "comp2", "comp3"],\n        >>>     static_covariates=static_covs,\n        >>> )\n        >>> transformer = StaticCovariatesTransformer()\n        >>> series_transformed = transformer.fit_transform(series)\n        >>> print(series.static_covariates)\n        static_covariates  num cat\n        component\n        comp1               0.0   a\n        comp2               2.0   c\n        comp3               1.0   b\n        >>> print(series_transformed.static_covariates)\n        static_covariates  num  cat\n        component\n        comp1               0.0  0.0\n        comp2               1.0  2.0\n        comp3               0.5  1.0\n        '
        self.transformer_num = MinMaxScaler() if transformer_num is None else transformer_num
        self.transformer_cat = OrdinalEncoder() if transformer_cat is None else transformer_cat
        for (transformer, transformer_name) in zip([self.transformer_num, self.transformer_cat], ['transformer_num', 'transformer_cat']):
            if not callable(getattr(transformer, 'fit', None)) or not callable(getattr(transformer, 'transform', None)) or (not callable(getattr(transformer, 'inverse_transform', None))):
                raise_log(ValueError(f'The provided `{transformer_name}` object must have fit(), transform() and inverse_transform() methods'), logger)
        (self.cols_num, self.cols_cat) = (cols_num, cols_cat)
        super().__init__(name=name, n_jobs=n_jobs, verbose=verbose, mask_components=False, global_fit=True)

    @staticmethod
    def ts_fit(series: Sequence[TimeSeries], params: Dict[str, Dict[str, Any]], *args, **kwargs):
        if False:
            return 10
        '\n        Collates static covariates of all provided `TimeSeries` and fits the following parameters:\n            1. `transformer_num`, the fitted numerical static covariate transformer.\n            2. `transformer_cat`, the fitted categorical static covariate transformer.\n            3. `mask_num`, a dictionary containing two boolean arrays: one that indicates which\n            components of the *untransformed* static covariates are numerical, and another that\n            indicates which components of the *transformed* static covariates are numerical.\n            4. `mask_cat`, a dictionary containing two boolean arrays: one that indicates which\n            components of the *untransformed* static covariates are categorical, and another that\n            indicates which components of the *transformed* static covariates are categorical.\n            5. `n_cat_cols`, a dictionary that stores the number of categorical columns\n            we should expect in the untransformed and in the transformed static covariates.\n        '
        fixed_params = params['fixed']
        transformer_num = fixed_params['transformer_num']
        transformer_cat = fixed_params['transformer_cat']
        cols_num = fixed_params['cols_num']
        cols_cat = fixed_params['cols_cat']
        stat_covs = pd.concat([s.static_covariates for s in series], axis=0)
        (cols_num, cols_cat) = StaticCovariatesTransformer._infer_static_cov_dtypes(stat_covs, cols_num, cols_cat)
        (mask_num, mask_cat) = StaticCovariatesTransformer._create_component_masks(stat_covs, cols_num, cols_cat)
        stat_covs = stat_covs.to_numpy(copy=False)
        if mask_num.any():
            transformer_num = transformer_num.fit(stat_covs[:, mask_num])
        if mask_cat.any():
            transformer_cat = transformer_cat.fit(stat_covs[:, mask_cat])
        (cat_mapping, inv_cat_mapping) = StaticCovariatesTransformer._create_category_mappings(stat_covs, transformer_cat, mask_cat, cols_cat)
        (inv_mask_num, inv_mask_cat) = StaticCovariatesTransformer._create_inv_component_masks(mask_num, mask_cat, cat_mapping, cols_cat)
        mask_num_dict = {'transform': mask_num, 'inverse_transform': inv_mask_num}
        mask_cat_dict = {'transform': mask_cat, 'inverse_transform': inv_mask_cat}
        col_map_cat_dict = {'transform': cat_mapping, 'inverse_transform': inv_cat_mapping}
        n_cat_cols = {method: len(col_map_cat_dict[method]) for method in ('transform', 'inverse_transform')}
        return {'transformer_num': transformer_num, 'transformer_cat': transformer_cat, 'mask_num': mask_num_dict, 'mask_cat': mask_cat_dict, 'col_map_cat': col_map_cat_dict, 'n_cat_cols': n_cat_cols}

    @staticmethod
    def _infer_static_cov_dtypes(stat_covs: pd.DataFrame, cols_num: Optional[Sequence[str]], cols_cat: Optional[Sequence[str]]):
        if False:
            return 10
        '\n        Returns a list of names of numerical static covariates and a list\n        of names of categorical/ordinal static covariates.\n        '
        if cols_num is None:
            mask_num = stat_covs.columns.isin(stat_covs.select_dtypes(include=np.number).columns)
            cols_num = stat_covs.columns[mask_num]
        if cols_cat is None:
            mask_cat = stat_covs.columns.isin(stat_covs.select_dtypes(exclude=np.number).columns)
            cols_cat = stat_covs.columns[mask_cat]
        return (cols_num, cols_cat)

    @staticmethod
    def _create_component_masks(untransformed_stat_covs: pd.DataFrame, cols_num: Sequence[str], cols_cat: Sequence[str]):
        if False:
            while True:
                i = 10
        "\n        Returns a boolean array indicating which components of the UNTRANSFORMED\n        `stat_covs` are numerical and a boolean array indicating which components\n        of the UNTRANSFORMED `stat_covs` are categoical.\n\n        It's important to recognise that these masks only apply to the UNTRANSFORMED\n        static covariates since some transformations can generate multiple new components\n        from a single component (e.g. one-hot encoding).\n        "
        mask_num = untransformed_stat_covs.columns.isin(cols_num)
        mask_cat = untransformed_stat_covs.columns.isin(cols_cat)
        return (mask_num, mask_cat)

    @staticmethod
    def _create_category_mappings(untransformed_stat_covs: np.ndarray, transformer_cat, mask_cat: np.ndarray, cols_cat: Sequence[str]):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns mapping from names of untransformed categorical static covariates names\n        and names of transformed categorical static covariate names (i.e. `col_map_cat`), as well\n        as a mapping from the transformed categorical static covariate names to the untransformed\n        ones (i.e. `inv_col_map_cat`).\n\n        These mappings will be many-to-one/one-to-many if a transformation that generates\n        multiple components from a single categorical variable is being used (e.g. one-hot\n        encoding).\n        '
        if mask_cat.any():
            n_cat_out = transformer_cat.transform(np.expand_dims(untransformed_stat_covs[0, mask_cat], 0)).shape[-1]
            if n_cat_out == sum(mask_cat):
                col_map_cat = inv_col_map_cat = OrderedDict({col: [col] for col in cols_cat})
            else:
                col_map_cat = OrderedDict()
                inv_col_map_cat = OrderedDict()
                for (col, categories) in zip(cols_cat, transformer_cat.categories_):
                    col_map_cat_i = []
                    for cat in categories:
                        col_map_cat_i.append(cat)
                        if len(categories) > 1:
                            cat_col_name = str(col) + '_' + str(cat)
                            inv_col_map_cat[cat_col_name] = [col]
                        else:
                            inv_col_map_cat[cat] = [col]
                    col_map_cat[col] = col_map_cat_i
        else:
            col_map_cat = {}
            inv_col_map_cat = {}
        return (col_map_cat, inv_col_map_cat)

    @staticmethod
    def _create_inv_component_masks(mask_num: np.ndarray, mask_cat: np.ndarray, cat_mapping: Dict[str, str], cols_cat: Sequence[str]):
        if False:
            return 10
        "\n        Returns a boolean array indicating which components of the TRANSFORMED\n        `stat_covs` are numerical and a boolean array indicating which components\n        of the TRANSFORMED `stat_covs` are categoical.\n\n        It's important to recognise that these masks only apply to the UNTRANSFORMED\n        static covariates since some transformations can generate multiple new components\n        from a single component (e.g. one-hot encoding).\n        "
        cat_idx = 0
        (inv_mask_num, inv_mask_cat) = ([], [])
        for (is_num, is_cat) in zip(mask_num, mask_cat):
            if is_num:
                inv_mask_num.append(True)
                inv_mask_cat.append(False)
            elif is_cat:
                cat_name = cols_cat[cat_idx]
                num_cat_outputs = len(cat_mapping[cat_name])
                inv_mask_num += num_cat_outputs * [False]
                inv_mask_cat += num_cat_outputs * [True]
                cat_idx += 1
            else:
                inv_mask_num.append(False)
                inv_mask_cat.append(False)
        inv_mask_num = np.array(inv_mask_num, dtype=bool)
        inv_mask_cat = np.array(inv_mask_cat, dtype=bool)
        return (inv_mask_num, inv_mask_cat)

    @staticmethod
    def ts_transform(series: TimeSeries, params: Dict[str, Any], *args, **kwargs) -> TimeSeries:
        if False:
            for i in range(10):
                print('nop')
        return StaticCovariatesTransformer._transform_static_covs(series, params['fitted'], method='transform')

    @staticmethod
    def ts_inverse_transform(series: TimeSeries, params: Dict[str, Any], *args, **kwargs) -> TimeSeries:
        if False:
            while True:
                i = 10
        return StaticCovariatesTransformer._transform_static_covs(series, params['fitted'], method='inverse_transform')

    @staticmethod
    def _transform_static_covs(series: TimeSeries, fitted_params: Dict[str, Any], method: Literal['transform', 'inverse_transform']):
        if False:
            return 10
        "\n        Transforms the static covariates of a `series` if `method = 'transform'`, and inverse\n        transforms the static covariates of a `series` if `method = 'inverse_transform'`.\n        "
        transformer_num = fitted_params['transformer_num']
        transformer_cat = fitted_params['transformer_cat']
        mask_num = fitted_params['mask_num'][method]
        mask_cat = fitted_params['mask_cat'][method]
        col_map_cat = fitted_params['col_map_cat'][method]
        n_cat_cols = fitted_params['n_cat_cols'][method]
        (vals_num, vals_cat) = StaticCovariatesTransformer._extract_static_covs(series, mask_num, mask_cat)
        (tr_out_num, tr_out_cat) = (None, None)
        if mask_num.any():
            tr_out_num = getattr(transformer_num, method)(vals_num)
        if mask_cat.any():
            tr_out_cat = getattr(transformer_cat, method)(vals_cat)
            if isinstance(tr_out_cat, csr_matrix):
                tr_out_cat = tr_out_cat.toarray()
        n_vals_cat_cols = 0 if vals_cat is None else vals_cat.shape[1]
        if method == 'inverse_transform' and n_vals_cat_cols != n_cat_cols:
            raise_log(ValueError(f'Expected `{n_cat_cols}` categorical value columns but only encountered `{n_vals_cat_cols}`'), logger)
        series = StaticCovariatesTransformer._add_back_static_covs(series, tr_out_num, tr_out_cat, mask_num, mask_cat, col_map_cat)
        return series

    @staticmethod
    def _extract_static_covs(series: TimeSeries, mask_num: np.ndarray, mask_cat: np.ndarray) -> Tuple[np.array, np.array]:
        if False:
            i = 10
            return i + 15
        '\n        Extracts all static covariates from a `TimeSeries`, and then extracts the numerical\n        and categorical components to transform from these static covariates.\n        '
        vals = series.static_covariates_values(copy=False)
        return (vals[:, mask_num], vals[:, mask_cat])

    @staticmethod
    def _add_back_static_covs(series: TimeSeries, vals_num: np.ndarray, vals_cat: np.ndarray, mask_num: np.ndarray, mask_cat: np.ndarray, col_map_cat: Dict[str, str]) -> pd.DataFrame:
        if False:
            i = 10
            return i + 15
        '\n        Adds transformed static covariates back to original `TimeSeries`. The categorical component\n        mapping is used to correctly name categorical components with a one-to-many mapping\n        between their untransformed and transformed versions (e.g. components generated using\n        one-hot encoding).\n        '
        data = {}
        (idx_num, idx_cat) = (0, 0)
        static_cov_columns = []
        for (col, is_num, is_cat) in zip(series.static_covariates.columns, mask_num, mask_cat):
            if is_num:
                data[col] = vals_num[:, idx_num]
                static_cov_columns.append(col)
                idx_num += 1
            elif is_cat:
                for col_name in col_map_cat[col]:
                    if len(col_map_cat[col]) > 1:
                        col_name = str(col) + '_' + str(col_name)
                    if col_name not in static_cov_columns:
                        data[col_name] = vals_cat[:, idx_cat]
                        static_cov_columns.append(col_name)
                        idx_cat += 1
            else:
                data[col] = series.static_covariates[col]
                static_cov_columns.append(col)
        transformed_static_covs = pd.DataFrame(data, columns=static_cov_columns, index=series.static_covariates.index)
        return series.with_static_covariates(transformed_static_covs)