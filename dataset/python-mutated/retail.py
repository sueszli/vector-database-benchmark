import pandas as pd
from woodwork.logical_types import NaturalLanguage
import featuretools as ft

def load_retail(id='demo_retail_data', nrows=None, return_single_table=False):
    if False:
        i = 10
        return i + 15
    'Returns the retail entityset example.\n    The original dataset can be found `here <https://archive.ics.uci.edu/ml/datasets/online+retail>`_.\n\n    We have also made some modifications to the data. We\n    changed the column names, converted the ``customer_id``\n    to a unique fake ``customer_name``, dropped duplicates,\n    added columns for ``total`` and ``cancelled`` and\n    converted amounts from GBP to USD. You can download the modified CSV in gz `compressed (7 MB)\n    <https://oss.alteryx.com/datasets/online-retail-logs-2018-08-28.csv.gz>`_\n    or `uncompressed (43 MB)\n    <https://oss.alteryx.com/datasets/online-retail-logs-2018-08-28.csv>`_ formats.\n\n    Args:\n        id (str):  Id to assign to EntitySet.\n        nrows (int):  Number of rows to load of the underlying CSV.\n            If None, load all.\n        return_single_table (bool): If True, return a CSV rather than an EntitySet. Default is False.\n\n    Examples:\n\n        .. ipython::\n            :verbatim:\n\n            In [1]: import featuretools as ft\n\n            In [2]: es = ft.demo.load_retail()\n\n            In [3]: es\n            Out[3]:\n            Entityset: demo_retail_data\n              DataFrames:\n                orders (shape = [22190, 3])\n                products (shape = [3684, 3])\n                customers (shape = [4372, 2])\n                order_products (shape = [401704, 7])\n\n        Load in subset of data\n\n        .. ipython::\n            :verbatim:\n\n            In [4]: es = ft.demo.load_retail(nrows=1000)\n\n            In [5]: es\n            Out[5]:\n            Entityset: demo_retail_data\n              DataFrames:\n                orders (shape = [67, 5])\n                products (shape = [606, 3])\n                customers (shape = [50, 2])\n                order_products (shape = [1000, 7])\n    '
    es = ft.EntitySet(id)
    csv_s3_gz = 'https://oss.alteryx.com/datasets/online-retail-logs-2018-08-28.csv.gz?library=featuretools&version=' + ft.__version__
    csv_s3 = 'https://oss.alteryx.com/datasets/online-retail-logs-2018-08-28.csv?library=featuretools&version=' + ft.__version__
    try:
        df = pd.read_csv(csv_s3_gz, nrows=nrows, parse_dates=['order_date'])
    except Exception:
        df = pd.read_csv(csv_s3, nrows=nrows, parse_dates=['order_date'])
    if return_single_table:
        return df
    es.add_dataframe(dataframe_name='order_products', dataframe=df, index='order_product_id', make_index=True, time_index='order_date', logical_types={'description': NaturalLanguage})
    es.normalize_dataframe(new_dataframe_name='products', base_dataframe_name='order_products', index='product_id', additional_columns=['description'])
    es.normalize_dataframe(new_dataframe_name='orders', base_dataframe_name='order_products', index='order_id', additional_columns=['customer_name', 'country', 'cancelled'])
    es.normalize_dataframe(new_dataframe_name='customers', base_dataframe_name='orders', index='customer_name')
    es.add_last_time_indexes()
    return es