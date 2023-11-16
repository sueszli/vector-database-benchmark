import lux
from lux.interestingness.interestingness import interestingness
from lux.processor.Compiler import Compiler
from lux.utils import utils
from lux.vis.Vis import Vis
from lux.vis.VisList import VisList
import pandas as pd

def row_group(ldf):
    if False:
        for i in range(10):
            print('nop')
    recommendation = {'action': 'Row Groups', 'description': 'Shows charts of possible visualizations with respect to the row-wise index.', 'long_description': 'A row index can be thought of as an extra row that indicates the values that the user is interested in.             Lux focuses on visualizing named dataframe indices, i.e., indices with a non-null name property, as a proxy of the attribute                 that the user is interested in or have operated on (e.g., group-by attribute). In particular, dataframes with named indices                     are often pre-aggregated, so Lux visualizes exactly the values that the dataframe portrays.                         <a href="https://lux-api.readthedocs.io/en/latest/source/advanced/indexgroup.html" target="_blank">More details</a>'}
    collection = []
    if ldf.index.nlevels == 1:
        if ldf.columns.name is not None:
            dim_name = ldf.columns.name
        else:
            dim_name = 'index'
        for row_id in range(len(ldf)):
            row = ldf.iloc[row_id,]
            rowdf = row.reset_index()
            vis = Vis([dim_name, lux.Clause(row.name, data_model='measure', aggregation=None)], rowdf)
            collection.append(vis)
    vlst = VisList(collection)
    recommendation['collection'] = vlst
    return recommendation