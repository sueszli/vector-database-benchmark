from lux.core.frame import LuxDataFrame
from lux.vis.Vis import Vis
from lux.executor.PandasExecutor import PandasExecutor
from lux.utils import utils
import pandas as pd
import numpy as np
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from scipy.spatial.distance import euclidean
import lux
from lux.utils.utils import get_filter_specs
from lux.interestingness.similarity import preprocess, euclidean_dist
from lux.vis.VisList import VisList
import warnings

def interestingness(vis: Vis, ldf: LuxDataFrame) -> int:
    if False:
        return 10
    '\n    Compute the interestingness score of the vis.\n    The interestingness metric is dependent on the vis type.\n\n    Parameters\n    ----------\n    vis : Vis\n    ldf : LuxDataFrame\n\n    Returns\n    -------\n    int\n            Interestingness Score\n    '
    if vis.data is None or len(vis.data) == 0:
        return -1
    try:
        filter_specs = utils.get_filter_specs(vis._inferred_intent)
        vis_attrs_specs = utils.get_attrs_specs(vis._inferred_intent)
        n_dim = vis._ndim
        n_msr = vis._nmsr
        n_filter = len(filter_specs)
        attr_specs = [clause for clause in vis_attrs_specs if clause.attribute != 'Record']
        dimension_lst = vis.get_attr_by_data_model('dimension')
        measure_lst = vis.get_attr_by_data_model('measure')
        v_size = len(vis.data)
        if n_dim == 1 and (n_msr == 0 or n_msr == 1) and (ldf.current_vis is not None) and (vis.get_attr_by_channel('y')[0].data_type == 'quantitative') and (len(ldf.current_vis) == 1) and (ldf.current_vis[0].mark == 'line') and (len(get_filter_specs(ldf.intent)) > 0):
            query_vc = VisList(ldf.current_vis, ldf)
            query_vis = query_vc[0]
            preprocess(query_vis)
            preprocess(vis)
            return 1 - euclidean_dist(query_vis, vis)
        if n_dim == 1 and (n_msr == 0 or n_msr == 1):
            if v_size < 2:
                return -1
            if vis.mark == 'geographical':
                return n_distinct(vis, dimension_lst, measure_lst)
            if n_filter == 0:
                return unevenness(vis, ldf, measure_lst, dimension_lst)
            elif n_filter == 1:
                return deviation_from_overall(vis, ldf, filter_specs, measure_lst[0].attribute)
        elif n_dim == 0 and n_msr == 1:
            if v_size < 2:
                return -1
            if n_filter == 0 and 'Number of Records' in vis.data:
                if 'Number of Records' in vis.data:
                    v = vis.data['Number of Records']
                    return skewness(v)
            elif n_filter == 1 and 'Number of Records' in vis.data:
                return deviation_from_overall(vis, ldf, filter_specs, 'Number of Records')
            return -1
        elif n_dim == 0 and n_msr == 2:
            if v_size < 10:
                return -1
            if vis.mark == 'heatmap':
                return weighted_correlation(vis.data['xBinStart'], vis.data['yBinStart'], vis.data['count'])
            if n_filter == 1:
                v_filter_size = get_filtered_size(filter_specs, vis.data)
                sig = v_filter_size / v_size
            else:
                sig = 1
            return sig * monotonicity(vis, attr_specs)
        elif n_dim == 1 and n_msr == 2:
            if v_size < 10:
                return -1
            color_attr = vis.get_attr_by_channel('color')[0].attribute
            C = ldf.cardinality[color_attr]
            if C < 40:
                return 1 / C
            else:
                return -1
        elif n_dim == 1 and n_msr == 2:
            return 0.2
        elif n_msr == 3:
            return 0.1
        elif vis.mark == 'line' and n_dim == 2:
            return 0.15
        elif vis.mark == 'bar' and n_dim == 2:
            from scipy.stats import chi2_contingency
            measure_column = vis.get_attr_by_data_model('measure')[0].attribute
            dimension_columns = vis.get_attr_by_data_model('dimension')
            groupby_column = dimension_columns[0].attribute
            color_column = dimension_columns[1].attribute
            contingency_tbl = pd.crosstab(vis.data[groupby_column], vis.data[color_column], values=vis.data[measure_column], aggfunc=sum)
            try:
                color_cardinality = ldf.cardinality[color_column]
                groupby_cardinality = ldf.cardinality[groupby_column]
                score = chi2_contingency(contingency_tbl)[0] * 0.9 ** (color_cardinality + groupby_cardinality)
            except (ValueError, KeyError):
                score = -1
            return score
        else:
            return -1
    except:
        if lux.config.interestingness_fallback:
            warnings.warn(f'An error occurred when computing interestingness for: {vis}')
            return -1
        else:
            raise

def get_filtered_size(filter_specs, ldf):
    if False:
        return 10
    filter_intents = filter_specs[0]
    result = PandasExecutor.apply_filter(ldf, filter_intents.attribute, filter_intents.filter_op, filter_intents.value)
    return len(result)

def skewness(v):
    if False:
        i = 10
        return i + 15
    from scipy.stats import skew
    return skew(v)

def weighted_avg(x, w):
    if False:
        return 10
    return np.average(x, weights=w)

def weighted_cov(x, y, w):
    if False:
        while True:
            i = 10
    return np.sum(w * (x - weighted_avg(x, w)) * (y - weighted_avg(y, w))) / np.sum(w)

def weighted_correlation(x, y, w):
    if False:
        print('Hello World!')
    return weighted_cov(x, y, w) / np.sqrt(weighted_cov(x, x, w) * weighted_cov(y, y, w))

def deviation_from_overall(vis: Vis, ldf: LuxDataFrame, filter_specs: list, msr_attribute: str, exclude_nan: bool=True) -> int:
    if False:
        for i in range(10):
            print('nop')
    '\n    Difference in bar chart/histogram shape from overall chart\n    Note: this function assumes that the filtered vis.data is operating on the same range as the unfiltered vis.data.\n\n    Parameters\n    ----------\n    vis : Vis\n    ldf : LuxDataFrame\n    filter_specs : list\n            List of filters from the Vis\n    msr_attribute : str\n            The attribute name of the measure value of the chart\n    exclude_nan: bool\n            Whether to include/exclude NaN values as part of the deviation calculation\n\n    Returns\n    -------\n    int\n            Score describing how different the vis is from the overall vis\n    '
    if lux.config.executor.name == 'PandasExecutor':
        if exclude_nan:
            vdata = vis.data.dropna()
        else:
            vdata = vis.data
        v_filter_size = get_filtered_size(filter_specs, ldf)
        v_size = len(vis.data)
    else:
        from lux.executor.SQLExecutor import SQLExecutor
        v_filter_size = SQLExecutor.get_filtered_size(filter_specs, ldf)
        v_size = len(ldf)
        vdata = vis.data
    v_filter = vdata[msr_attribute]
    total = v_filter.sum()
    v_filter = v_filter / total
    if total == 0:
        return 0
    import copy
    unfiltered_vis = copy.copy(vis)
    unfiltered_vis._inferred_intent = utils.get_attrs_specs(vis._inferred_intent)
    lux.config.executor.execute([unfiltered_vis], ldf)
    if exclude_nan:
        uv = unfiltered_vis.data.dropna()
    else:
        uv = unfiltered_vis.data
    v = uv[msr_attribute]
    v = v / v.sum()
    assert len(v) == len(v_filter), 'Data for filtered and unfiltered vis have unequal length.'
    sig = v_filter_size / v_size
    rankSig = 1
    if vis.mark == 'bar':
        dimList = vis.get_attr_by_data_model('dimension')
        v_rank = uv.rank()
        v_filter_rank = vdata.rank()
        numCategories = ldf.cardinality[dimList[0].attribute]
        for r in range(0, numCategories - 1):
            if v_rank[msr_attribute][r] != v_filter_rank[msr_attribute][r]:
                rankSig += 1
        rankSig = rankSig / numCategories
    from scipy.spatial.distance import euclidean
    return sig * rankSig * euclidean(v, v_filter)

def unevenness(vis: Vis, ldf: LuxDataFrame, measure_lst: list, dimension_lst: list) -> int:
    if False:
        print('Hello World!')
    '\n    Measure the unevenness of a bar chart vis.\n    If a bar chart is highly uneven across the possible values, then it may be interesting. (e.g., USA produces lots of cars compared to Japan and Europe)\n    Likewise, if a bar chart shows that the measure is the same for any possible values the dimension attribute could take on, then it may not very informative.\n    (e.g., The cars produced across all Origins (Europe, Japan, and USA) has approximately the same average Acceleration.)\n\n    Parameters\n    ----------\n    vis : Vis\n    ldf : LuxDataFrame\n    measure_lst : list\n            List of measures\n    dimension_lst : list\n            List of dimensions\n    Returns\n    -------\n    int\n            Score describing how uneven the bar chart is.\n    '
    v = vis.data[measure_lst[0].attribute]
    v = v / v.sum()
    v = v.fillna(0)
    attr = dimension_lst[0].attribute
    if isinstance(attr, pd._libs.tslibs.timestamps.Timestamp):
        attr = str(attr._date_repr)
    C = ldf.cardinality[attr]
    D = 0.9 ** C
    v_flat = pd.Series([1 / C] * len(v))
    if is_datetime(v):
        v = v.astype('int')
    return D * euclidean(v, v_flat)

def mutual_information(v_x: list, v_y: list) -> int:
    if False:
        for i in range(10):
            print('nop')
    from sklearn.metrics import mutual_info_score
    return mutual_info_score(v_x, v_y)

def monotonicity(vis: Vis, attr_specs: list, ignore_identity: bool=True) -> int:
    if False:
        i = 10
        return i + 15
    '\n    Monotonicity measures there is a monotonic trend in the scatterplot, whether linear or not.\n    This score is computed as the Pearson\'s correlation on the ranks of x and y.\n    See "Graph-Theoretic Scagnostics", Wilkinson et al 2005: https://research.tableau.com/sites/default/files/Wilkinson_Infovis-05.pdf\n    Parameters\n    ----------\n    vis : Vis\n    attr_spec: list\n            List of attribute Clause objects\n\n    ignore_identity: bool\n            Boolean flag to ignore items with the same x and y attribute (score as -1)\n\n    Returns\n    -------\n    int\n            Score describing the strength of monotonic relationship in vis\n    '
    from scipy.stats import pearsonr
    msr1 = attr_specs[0].attribute
    msr2 = attr_specs[1].attribute
    if ignore_identity and msr1 == msr2:
        return -1
    vxy = vis.data.dropna()
    v_x = vxy[msr1]
    v_y = vxy[msr2]
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            score = np.abs(pearsonr(v_x, v_y)[0])
        except:
            score = -1
    if pd.isnull(score):
        return -1
    else:
        return score

def n_distinct(vis: Vis, dimension_lst: list, measure_lst: list) -> int:
    if False:
        while True:
            i = 10
    '\n    Computes how many unique values there are for a dimensional data type.\n    Ignores attributes that are latitude or longitude coordinates.\n\n    For example, if a dataset displayed earthquake magnitudes across 48 states and\n    3 countries, return 48 and 3 respectively.\n\n    Parameters\n    ----------\n    vis : Vis\n    dimension_lst: list\n            List of dimension Clause objects.\n    measure_lst: list\n            List of measure Clause objects.\n\n    Returns\n    -------\n    int\n            Score describing the number of unique values in the dimension.\n    '
    if measure_lst[0].get_attr() in {'longitude', 'latitude'}:
        return -1
    return vis.data[dimension_lst[0].get_attr()].nunique()