import logging
import pandas as pd
from featuretools.computational_backends.utils import get_ww_types_from_features
from featuretools.utils.gen_utils import make_tqdm_iterator
logger = logging.getLogger('featuretools')
DEFAULT_TOP_N = 10

def encode_features(feature_matrix, features, top_n=DEFAULT_TOP_N, include_unknown=True, to_encode=None, inplace=False, drop_first=False, verbose=False):
    if False:
        print('Hello World!')
    'Encode categorical features\n\n    Args:\n        feature_matrix (pd.DataFrame): Dataframe of features.\n        features (list[PrimitiveBase]): Feature definitions in feature_matrix.\n        top_n (int or dict[string -> int]): Number of top values to include.\n            If dict[string -> int] is used, key is feature name and value is\n            the number of top values to include for that feature.\n            If a feature\'s name is not in dictionary, a default value of 10 is used.\n        include_unknown (pd.DataFrame): Add feature encoding an unknown class.\n            defaults to True\n        to_encode (list[str]): List of feature names to encode.\n            features not in this list are unencoded in the output matrix\n            defaults to encode all necessary features.\n        inplace (bool): Encode feature_matrix in place. Defaults to False.\n        drop_first (bool): Whether to get k-1 dummies out of k categorical\n                levels by removing the first level.\n                defaults to False\n        verbose (str): Print progress info.\n\n    Returns:\n        (pd.Dataframe, list) : encoded feature_matrix, encoded features\n\n    Example:\n        .. ipython:: python\n            :suppress:\n\n            from featuretools.tests.testing_utils import make_ecommerce_entityset\n            import featuretools as ft\n            es = make_ecommerce_entityset()\n\n        .. ipython:: python\n\n            f1 = ft.Feature(es["log"].ww["product_id"])\n            f2 = ft.Feature(es["log"].ww["purchased"])\n            f3 = ft.Feature(es["log"].ww["value"])\n\n            features = [f1, f2, f3]\n            ids = [0, 1, 2, 3, 4, 5]\n            feature_matrix = ft.calculate_feature_matrix(features, es,\n                                                         instance_ids=ids)\n\n            fm_encoded, f_encoded = ft.encode_features(feature_matrix,\n                                                       features)\n            f_encoded\n\n            fm_encoded, f_encoded = ft.encode_features(feature_matrix,\n                                                       features, top_n=2)\n            f_encoded\n\n            fm_encoded, f_encoded = ft.encode_features(feature_matrix, features,\n                                                       include_unknown=False)\n            f_encoded\n\n            fm_encoded, f_encoded = ft.encode_features(feature_matrix, features,\n                                                       to_encode=[\'purchased\'])\n            f_encoded\n\n            fm_encoded, f_encoded = ft.encode_features(feature_matrix, features,\n                                                       drop_first=True)\n            f_encoded\n    '
    if not isinstance(feature_matrix, pd.DataFrame):
        msg = 'feature_matrix must be a Pandas DataFrame'
        raise TypeError(msg)
    if inplace:
        X = feature_matrix
    else:
        X = feature_matrix.copy()
    old_feature_names = set()
    for feature in features:
        for fname in feature.get_feature_names():
            assert fname in X.columns, 'Feature %s not found in feature matrix' % fname
            old_feature_names.add(fname)
    pass_through = [col for col in X.columns if col not in old_feature_names]
    if verbose:
        iterator = make_tqdm_iterator(iterable=features, total=len(features), desc='Encoding pass 1', unit='feature')
    else:
        iterator = features
    new_feature_list = []
    kept_columns = []
    encoded_columns = []
    columns_info = feature_matrix.ww.columns
    for f in iterator:
        is_discrete = {'category', 'foreign_key'}.intersection(f.column_schema.semantic_tags)
        if f.number_output_features > 1 or not is_discrete:
            if f.number_output_features > 1:
                logger.warning('Feature %s has multiple columns and will not be encoded.  This may result in a matrix with non-numeric values.' % f)
            new_feature_list.append(f)
            kept_columns.extend(f.get_feature_names())
            continue
        if to_encode is not None and f.get_name() not in to_encode:
            new_feature_list.append(f)
            kept_columns.extend(f.get_feature_names())
            continue
        val_counts = X[f.get_name()].value_counts()
        val_counts = val_counts[val_counts > 0].to_frame()
        index_name = val_counts.index.name
        val_counts = val_counts.rename(columns={val_counts.columns[0]: 'count'})
        if index_name is None:
            if 'index' in val_counts.columns:
                index_name = 'level_0'
            else:
                index_name = 'index'
        val_counts.reset_index(inplace=True)
        val_counts = val_counts.sort_values(['count', index_name], ascending=False)
        val_counts.set_index(index_name, inplace=True)
        select_n = top_n
        if isinstance(top_n, dict):
            select_n = top_n.get(f.get_name(), DEFAULT_TOP_N)
        if drop_first:
            select_n = min(len(val_counts), top_n)
            select_n = max(select_n - 1, 1)
        unique = val_counts.head(select_n).index.tolist()
        for label in unique:
            add = f == label
            add_name = add.get_name()
            new_feature_list.append(add)
            new_col = X[f.get_name()] == label
            new_col.rename(add_name, inplace=True)
            encoded_columns.append(new_col)
        if include_unknown:
            unknown = f.isin(unique).NOT().rename(f.get_name() + ' is unknown')
            unknown_name = unknown.get_name()
            new_feature_list.append(unknown)
            new_col = ~X[f.get_name()].isin(unique)
            new_col.rename(unknown_name, inplace=True)
            encoded_columns.append(new_col)
        if inplace:
            X.drop(f.get_name(), axis=1, inplace=True)
    kept_columns.extend(pass_through)
    if inplace:
        for encoded_column in encoded_columns:
            X[encoded_column.name] = encoded_column
    else:
        X = pd.concat([X[kept_columns]] + encoded_columns, axis=1)
    entityset = new_feature_list[0].entityset
    ww_init_kwargs = get_ww_types_from_features(new_feature_list, entityset)
    for column in kept_columns:
        ww_init_kwargs['logical_types'][column] = columns_info[column].logical_type
        ww_init_kwargs['semantic_tags'][column] = columns_info[column].semantic_tags
        ww_init_kwargs['column_origins'][column] = columns_info[column].origin
    X.ww.init(**ww_init_kwargs)
    return (X, new_feature_list)