from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from turicreate.toolkits._main import ToolkitError
import logging as _logging

def _validate_row_label(label, column_type_map):
    if False:
        print('Hello World!')
    '\n    Validate a row label column.\n\n    Parameters\n    ----------\n    label : str\n        Name of the row label column.\n\n    column_type_map : dict[str, type]\n        Dictionary mapping the name of each column in an SFrame to the type of\n        the values in the column.\n    '
    if not isinstance(label, str):
        raise TypeError('The row label column name must be a string.')
    if not label in column_type_map.keys():
        raise ToolkitError('Row label column not found in the dataset.')
    if not column_type_map[label] in (str, int):
        raise TypeError('Row labels must be integers or strings.')

def _robust_column_name(base_name, column_names):
    if False:
        i = 10
        return i + 15
    "\n    Generate a new column name that is guaranteed not to conflict with an\n    existing set of column names.\n\n    Parameters\n    ----------\n    base_name : str\n        The base of the new column name. Usually this does not conflict with\n        the existing column names, in which case this function simply returns\n        `base_name`.\n\n    column_names : list[str]\n        List of existing column names.\n\n    Returns\n    -------\n    robust_name : str\n        The new column name. If `base_name` isn't in `column_names`, then\n        `robust_name` is the same as `base_name`. If there are conflicts, a\n        numeric suffix is added to `base_name` until it no longer conflicts\n        with the column names.\n    "
    robust_name = base_name
    i = 1
    while robust_name in column_names:
        robust_name = base_name + '.{}'.format(i)
        i += 1
    return robust_name

def _select_valid_features(dataset, features, valid_feature_types, target_column=None):
    if False:
        return 10
    "\n    Utility function for selecting columns of only valid feature types.\n\n    Parameters\n    ----------\n    dataset: SFrame\n        The input SFrame containing columns of potential features.\n\n    features: list[str]\n        List of feature column names.  If None, the candidate feature set is\n        taken to be all the columns in the dataset.\n\n    valid_feature_types: list[type]\n        List of Python types that represent valid features.  If type is array.array,\n        then an extra check is done to ensure that the individual elements of the array\n        are of numeric type.  If type is dict, then an extra check is done to ensure\n        that dictionary values are numeric.\n\n    target_column: str\n        Name of the target column.  If not None, the target column is excluded\n        from the list of valid feature columns.\n\n    Returns\n    -------\n    out: list[str]\n        List of valid feature column names.  Warnings are given for each candidate\n        feature column that is excluded.\n\n    Examples\n    --------\n    # Select all the columns of type `str` in sf, excluding the target column named\n    # 'rating'\n    >>> valid_columns = _select_valid_features(sf, None, [str], target_column='rating')\n\n    # Select the subset of columns 'X1', 'X2', 'X3' that has dictionary type or defines\n    # numeric array type\n    >>> valid_columns = _select_valid_features(sf, ['X1', 'X2', 'X3'], [dict, array.array])\n    "
    if features is not None:
        if not hasattr(features, '__iter__'):
            raise TypeError("Input 'features' must be an iterable type.")
        if not all([isinstance(x, str) for x in features]):
            raise TypeError("Input 'features' must contain only strings.")
    if features is None:
        features = dataset.column_names()
    col_type_map = {col_name: col_type for (col_name, col_type) in zip(dataset.column_names(), dataset.column_types())}
    valid_features = []
    for col_name in features:
        if col_name not in dataset.column_names():
            _logging.warning("Column '{}' is not in the input dataset.".format(col_name))
        elif col_name == target_column:
            _logging.warning('Excluding target column ' + target_column + ' as a feature.')
        elif col_type_map[col_name] not in valid_feature_types:
            _logging.warning("Column '{}' is excluded as a ".format(col_name) + 'feature due to invalid column type.')
        else:
            valid_features.append(col_name)
    if len(valid_features) == 0:
        raise ValueError('The dataset does not contain any valid feature columns. ' + 'Accepted feature types are ' + str(valid_feature_types) + '.')
    return valid_features

def _check_elements_equal(lst):
    if False:
        return 10
    '\n    Returns true if all of the elements in the list are equal.\n    '
    assert isinstance(lst, list), 'Input value must be a list.'
    return not lst or lst.count(lst[0]) == len(lst)

def _validate_lists(sa, allowed_types=[str], require_same_type=True, require_equal_length=False, num_to_check=10):
    if False:
        for i in range(10):
            print('nop')
    '\n    For a list-typed SArray, check whether the first elements are lists that\n    - contain only the provided types\n    - all have the same lengths (optionally)\n\n    Parameters\n    ----------\n    sa : SArray\n        An SArray containing lists.\n\n    allowed_types : list\n        A list of types that are allowed in each list.\n\n    require_same_type : bool\n        If true, the function returns false if more than one type of object\n        exists in the examined lists.\n\n    require_equal_length : bool\n        If true, the function requires false when the list lengths differ.\n\n    Returns\n    -------\n    out : bool\n        Returns true if all elements are lists of equal length and containing\n        only ints or floats. Otherwise returns false.\n    '
    if len(sa) == 0:
        return True
    first_elements = sa.head(num_to_check)
    if first_elements.dtype != list:
        raise ValueError('Expected an SArray of lists when type-checking lists.')
    list_lengths = list(first_elements.item_length())
    same_length = _check_elements_equal(list_lengths)
    if require_equal_length and (not same_length):
        return False
    if len(first_elements[0]) == 0:
        return True
    types = first_elements.apply(lambda xs: [str(type(x)) for x in xs])
    same_type = [_check_elements_equal(x) for x in types]
    all_same_type = _check_elements_equal(same_type)
    if require_same_type and (not all_same_type):
        return False
    first_types = [t[0] for t in types if t]
    all_same_type = _check_elements_equal(first_types)
    if require_same_type and (not all_same_type):
        return False
    allowed_type_strs = [str(x) for x in allowed_types]
    for list_element_types in types:
        for t in list_element_types:
            if t not in allowed_type_strs:
                return False
    return True

def _summarize_accessible_fields(field_descriptions, width=40, section_title='Accessible fields'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create a summary string for the accessible fields in a model. Unlike\n    `_toolkit_repr_print`, this function does not look up the values of the\n    fields, it just formats the names and descriptions.\n\n    Parameters\n    ----------\n    field_descriptions : dict{str: str}\n        Name of each field and its description, in a dictionary. Keys and\n        values should be strings.\n\n    width : int, optional\n        Width of the names. This is usually determined and passed by the\n        calling `__repr__` method.\n\n    section_title : str, optional\n        Name of the accessible fields section in the summary string.\n\n    Returns\n    -------\n    out : str\n    '
    key_str = '{:<{}}: {}'
    items = []
    items.append(section_title)
    items.append('-' * len(section_title))
    for (field_name, field_desc) in field_descriptions.items():
        items.append(key_str.format(field_name, width, field_desc))
    return '\n'.join(items)