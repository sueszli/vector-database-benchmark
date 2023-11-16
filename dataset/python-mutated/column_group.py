import lux
from lux.interestingness.interestingness import interestingness
from lux.processor.Compiler import Compiler
from lux.utils import utils
from lux.vis.Vis import Vis
from lux.vis.VisList import VisList
import pandas as pd

def column_group(ldf):
    if False:
        print('Hello World!')
    recommendation = {'action': 'Column Groups', 'description': 'Shows charts of possible visualizations with respect to the column-wise index.', 'long_description': 'A column index can be thought of as an extra column that indicates the values that the user is interested in.             Lux focuses on visualizing named dataframe indices, i.e., indices with a non-null name property, as a proxy of the attribute                 that the user is interested in or have operated on (e.g., group-by attribute). In particular, dataframes with named indices                     are often pre-aggregated, so Lux visualizes exactly the values that the dataframe portrays.                          <a href="https://lux-api.readthedocs.io/en/latest/source/advanced/indexgroup.html" target="_blank">More details</a>'}
    collection = []
    ldf_flat = ldf
    if isinstance(ldf.columns, pd.DatetimeIndex):
        ldf_flat.columns = ldf_flat.columns.format()
    ldf_flat = ldf_flat.reset_index()
    if ldf.index.nlevels == 1:
        if ldf.index.name:
            index_column_name = ldf.index.name
        else:
            index_column_name = 'index'
        if isinstance(ldf.columns, pd.DatetimeIndex):
            ldf.columns = ldf.columns.to_native_types()
        for attribute in ldf.columns:
            if ldf[attribute].dtype != 'object' and attribute != 'index':
                vis = Vis([lux.Clause(attribute=index_column_name, data_type='nominal', data_model='dimension', aggregation=''), lux.Clause(attribute=attribute, data_type='quantitative', data_model='measure', aggregation=None)])
                collection.append(vis)
    vlst = VisList(collection, ldf_flat)
    recommendation['collection'] = vlst
    return recommendation