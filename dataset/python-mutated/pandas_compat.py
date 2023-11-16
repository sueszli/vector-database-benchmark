"""Pandas DataFrame↔Table conversion helpers"""
from unittest.mock import patch
import numpy as np
from scipy import sparse as sp
from scipy.sparse import csr_matrix
import pandas as pd
from pandas.core.arrays import SparseArray
from pandas.api.types import is_object_dtype, is_datetime64_any_dtype, is_numeric_dtype, is_integer_dtype
from Orange.data import Table, Domain, DiscreteVariable, StringVariable, TimeVariable, ContinuousVariable
from Orange.data.table import Role
__all__ = ['table_from_frame', 'table_to_frame']

class OrangeDataFrame(pd.DataFrame):
    _metadata = ['orange_variables', 'orange_weights', 'orange_attributes', 'orange_role']

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        "\n        A pandas DataFrame wrapper for one of Table's numpy arrays:\n            - sets index values corresponding to Orange's global row indices\n              e.g. ['_o1', '_o2'] (allows Orange to handle selection)\n            - remembers the array's role in the Table (attribute, class var, meta)\n            - keeps the Variable objects, and uses them in back-to-table conversion,\n              should a column name match a variable's name\n            - stores weight values (legacy)\n\n        Parameters\n        ----------\n        table : Table\n        orange_role : Role, (default=Role.Attribute)\n            When converting back to an orange table, the DataFrame will\n            convert to the right role (attrs, class vars, or metas)\n        "
        if len(args) <= 0 or not isinstance(args[0], Table):
            super().__init__(*args, **kwargs)
            return
        table = args[0]
        if 'orange_role' in kwargs:
            role = kwargs.pop('orange_role')
        elif len(args) >= 2:
            role = args[1]
        else:
            role = Role.Attribute
        if role == Role.Attribute:
            data = table.X
            vars_ = table.domain.attributes
        elif role == Role.ClassAttribute:
            data = table.Y
            vars_ = table.domain.class_vars
        else:
            data = table.metas
            vars_ = table.domain.metas
        index = ['_o' + str(id_) for id_ in table.ids]
        varsdict = {var._name: var for var in vars_}
        columns = varsdict.keys()
        if sp.issparse(data):
            data = data.asformat('csc')
            sparrays = [SparseArray.from_spmatrix(data[:, i]) for i in range(data.shape[1])]
            data = dict(enumerate(sparrays))
            super().__init__(data, index=index, **kwargs)
            self.columns = columns
            self.sparse.to_dense = self.__patch_constructor(self.sparse.to_dense)
        else:
            super().__init__(data=data, index=index, columns=columns, **kwargs)
        self.orange_role = role
        self.orange_variables = varsdict
        self.orange_weights = dict(zip(index, table.W)) if table.W.size > 0 else {}
        self.orange_attributes = table.attributes

    def __patch_constructor(self, method):
        if False:
            i = 10
            return i + 15

        def new_method(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            with patch('pandas.DataFrame', OrangeDataFrame):
                df = method(*args, **kwargs)
            df.__finalize__(self)
            return df
        return new_method

    @property
    def _constructor(self):
        if False:
            i = 10
            return i + 15
        return OrangeDataFrame

    def to_orange_table(self):
        if False:
            print('Hello World!')
        return table_from_frame(self)

    def __finalize__(self, other, method=None, **_):
        if False:
            return 10
        '\n        propagate metadata from other to self\n\n        Parameters\n        ----------\n        other : the object from which to get the attributes that we are going\n            to propagate\n        method : optional, a passed method name ; possibly to take different\n            types of propagation actions based on this\n\n        '
        if method == 'concat':
            objs = other.objs
        elif method == 'merge':
            objs = (other.left, other.right)
        else:
            objs = [other]
        orange_role = getattr(self, 'orange_role', None)
        dicts = {dname: getattr(self, dname, {}) for dname in ('orange_variables', 'orange_weights', 'orange_attributes')}
        for obj in objs:
            other_role = getattr(obj, 'orange_role', None)
            if other_role is not None:
                orange_role = other_role
            for (dname, dict_) in dicts.items():
                other_dict = getattr(obj, dname, {})
                dict_.update(other_dict)
        object.__setattr__(self, 'orange_role', orange_role)
        for (dname, dict_) in dicts.items():
            object.__setattr__(self, dname, dict_)
        return self
    pd.DataFrame.__finalize__ = __finalize__

def _reset_index(df: pd.DataFrame) -> pd.DataFrame:
    if False:
        return 10
    'If df index is not a simple RangeIndex (or similar), include it into a table'
    if not (is_integer_dtype(df.index) and (df.index.is_monotonic_increasing or df.index.is_monotonic_decreasing)) and (isinstance(df.index, pd.MultiIndex) or not is_object_dtype(df.index) or (not any((str(i).startswith('_o') for i in df.index)))):
        df = df.reset_index()
    return df

def _is_discrete(s, force_nominal):
    if False:
        while True:
            i = 10
    return isinstance(s.dtype, pd.CategoricalDtype) or (is_object_dtype(s) and (force_nominal or s.nunique() < s.size ** 0.666))

def _is_datetime(s):
    if False:
        print('Hello World!')
    if is_datetime64_any_dtype(s):
        return True
    try:
        if is_object_dtype(s):
            try:
                pd.to_numeric(s)
                return False
            except (ValueError, TypeError):
                pass
            pd.to_datetime(s, utc=True)
            return True
    except Exception:
        pass
    return False

def _convert_datetime(series, var):
    if False:
        for i in range(10):
            print('nop')

    def col_type(dt):
        if False:
            print('Hello World!')
        'Test if is date, time or datetime'
        dt_nonnat = dt[~pd.isnull(dt)]
        if (dt_nonnat.dt.floor('d') == dt_nonnat).all():
            return (1, 0)
        elif (dt_nonnat.dt.date == pd.Timestamp('now').date()).all():
            return (0, 1)
        else:
            return (1, 1)
    try:
        dt = pd.to_datetime(series)
    except ValueError:
        dt = pd.to_datetime(series, utc=True)
    (var.have_date, var.have_time) = col_type(dt)
    if dt.dt.tz is not None:
        var.timezone = dt.dt.tz
        dt = dt.dt.tz_convert('UTC')
    if var.have_time and (not var.have_date):
        return ((dt.dt.tz_localize(None) - pd.Timestamp('now').normalize()) / pd.Timedelta('1s')).values
    return ((dt.dt.tz_localize(None) - pd.Timestamp('1970-01-01')) / pd.Timedelta('1s')).values

def to_categorical(s, _):
    if False:
        i = 10
        return i + 15
    x = s.astype('category').cat.codes
    x = x.where(x != -1, np.nan)
    return np.asarray(x)

def vars_from_df(df, role=None, force_nominal=False):
    if False:
        for i in range(10):
            print('nop')
    if role is None and hasattr(df, 'orange_role'):
        role = df.orange_role
    df = _reset_index(df)
    cols = ([], [], [])
    exprs = ([], [], [])
    vars_ = ([], [], [])
    for column in df.columns:
        s = df[column]
        _role = Role.Attribute if role is None else role
        if hasattr(df, 'orange_variables') and column in df.orange_variables:
            original_var = df.orange_variables[column]
            var = original_var.copy(compute_value=None)
            expr = None
        elif _is_datetime(s):
            var = TimeVariable(str(column))
            expr = _convert_datetime
        elif _is_discrete(s, force_nominal):
            discrete = s.astype('category').cat
            var = DiscreteVariable(str(column), discrete.categories.astype(str).tolist())
            expr = to_categorical
        elif is_numeric_dtype(s):
            var = ContinuousVariable(str(column), number_of_decimals=0 if is_integer_dtype(s) else None)
            expr = None
        else:
            if role is not None and role != Role.Meta:
                raise ValueError('String variable must be in metas.')
            _role = Role.Meta
            var = StringVariable(str(column))
            expr = lambda s, _: np.asarray(s.astype(object).fillna(StringVariable.Unknown).astype(str), dtype=object)
        cols[_role].append(column)
        exprs[_role].append(expr)
        vars_[_role].append(var)
    xym = []
    for (a_vars, a_cols, a_expr) in zip(vars_, cols, exprs):
        if not a_cols:
            arr = None if a_cols != cols[0] else np.empty((df.shape[0], 0))
        elif not any(a_expr):
            a_df = df if all((c in a_cols for c in df.columns)) else df[a_cols]
            if all((isinstance(a, pd.SparseDtype) for a in a_df.dtypes)):
                arr = csr_matrix(a_df.sparse.to_coo())
            else:
                arr = np.asarray(a_df)
        else:
            arr = np.array([expr(df[col], var) if expr else np.asarray(df[col]) for (var, col, expr) in zip(a_vars, a_cols, a_expr)]).T
        xym.append(arr)
    if xym[1] is not None and xym[1].ndim == 2 and (xym[1].shape[1] == 1):
        xym[1] = xym[1][:, 0]
    return (xym, Domain(*vars_))

def table_from_frame(df, *, force_nominal=False):
    if False:
        for i in range(10):
            print('nop')
    (XYM, domain) = vars_from_df(df, force_nominal=force_nominal)
    if hasattr(df, 'orange_weights') and hasattr(df, 'orange_attributes'):
        W = [df.orange_weights[i] for i in df.index if i in df.orange_weights]
        if len(W) != len(df.index):
            W = None
        attributes = df.orange_attributes
        if isinstance(df.index, pd.MultiIndex) or not is_object_dtype(df.index):
            ids = None
        else:
            ids = [int(i[2:]) if str(i).startswith('_o') and i[2:].isdigit() else Table.new_id() for i in df.index]
    else:
        W = None
        attributes = None
        ids = None
    return Table.from_numpy(domain, *XYM, W=W, attributes=attributes, ids=ids)

def table_from_frames(xdf, ydf, mdf):
    if False:
        i = 10
        return i + 15
    if not (xdf.index.equals(ydf.index) and xdf.index.equals(mdf.index)):
        raise ValueError('Indexes not equal. Make sure that all three dataframes have equal index')
    xdf = xdf.reset_index(drop=True)
    ydf = ydf.reset_index(drop=True)
    dfs = (xdf, ydf, mdf)
    if not all((df.shape[0] == xdf.shape[0] for df in dfs)):
        raise ValueError(f'Leading dimension mismatch (not {xdf.shape[0]} == {ydf.shape[0]} == {mdf.shape[0]})')
    (xXYM, xDomain) = vars_from_df(xdf, role=Role.Attribute)
    (yXYM, yDomain) = vars_from_df(ydf, role=Role.ClassAttribute)
    (mXYM, mDomain) = vars_from_df(mdf, role=Role.Meta)
    XYM = (xXYM[0], yXYM[1], mXYM[2])
    domain = Domain(xDomain.attributes, yDomain.class_vars, mDomain.metas)
    ids = [int(idx[2:]) if str(idx).startswith('_o') and idx[2:].isdigit() else Table.new_id() for idx in mdf.index]
    attributes = {}
    W = None
    for df in dfs:
        if isinstance(df, OrangeDataFrame):
            W = [df.orange_weights[i] for i in df.index if i in df.orange_weights]
            if len(W) != len(df.index):
                W = None
            attributes.update(df.orange_attributes)
        else:
            W = None
    return Table.from_numpy(domain, *XYM, W=W, attributes=attributes, ids=ids)

def table_to_frame(tab, include_metas=False):
    if False:
        i = 10
        return i + 15
    '\n    Convert Orange.data.Table to pandas.DataFrame\n\n    Parameters\n    ----------\n    tab : Table\n\n    include_metas : bool, (default=False)\n        Include table metas into dataframe.\n\n    Returns\n    -------\n    pandas.DataFrame\n    '

    def _column_to_series(col, vals):
        if False:
            for i in range(10):
                print('nop')
        result = ()
        if col.is_discrete:
            codes = pd.Series(vals).fillna(-1).astype(int)
            result = (col.name, pd.Categorical.from_codes(codes=codes, categories=col.values, ordered=True))
        elif col.is_time:
            result = (col.name, pd.to_datetime(vals, unit='s').to_series().reset_index()[0])
        elif col.is_continuous:
            dt = float
            if col.number_of_decimals == 0 and (not np.any(pd.isnull(vals))):
                dt = int
            result = (col.name, pd.Series(vals).astype(dt))
        elif col.is_string:
            result = (col.name, pd.Series(vals))
        return result

    def _columns_to_series(cols, vals):
        if False:
            for i in range(10):
                print('nop')
        return [_column_to_series(col, vals[:, i]) for (i, col) in enumerate(cols)]
    (x, y, metas) = ([], [], [])
    domain = tab.domain
    if domain.attributes:
        x = _columns_to_series(domain.attributes, tab.X)
    if domain.class_vars:
        y_values = tab.Y.reshape(tab.Y.shape[0], len(domain.class_vars))
        y = _columns_to_series(domain.class_vars, y_values)
    if domain.metas:
        metas = _columns_to_series(domain.metas, tab.metas)
    all_series = dict(x + y + metas)
    all_vars = tab.domain.variables
    if include_metas:
        all_vars += tab.domain.metas
    original_column_order = [var.name for var in all_vars]
    unsorted_columns_df = pd.DataFrame(all_series)
    return unsorted_columns_df[original_column_order]

def table_to_frames(table):
    if False:
        print('Hello World!')
    xdf = OrangeDataFrame(table, Role.Attribute)
    ydf = OrangeDataFrame(table, Role.ClassAttribute)
    mdf = OrangeDataFrame(table, Role.Meta)
    return (xdf, ydf, mdf)

def amend_table_with_frame(table, df, role):
    if False:
        for i in range(10):
            print('nop')
    arr = Role.get_arr(role, table)
    if arr.shape[0] != df.shape[0]:
        raise ValueError(f'Leading dimension mismatch (not {arr.shape[0]} == {df.shape[0]})')
    (XYM, domain) = vars_from_df(df, role=role)
    if role == Role.Attribute:
        table.domain = Domain(domain.attributes, table.domain.class_vars, table.domain.metas)
        table.X = XYM[0]
    elif role == Role.ClassAttribute:
        table.domain = Domain(table.domain.attributes, domain.class_vars, table.domain.metas)
        table.Y = XYM[1]
    else:
        table.domain = Domain(table.domain.attributes, table.domain.class_vars, domain.metas)
        table.metas = XYM[2]
    if isinstance(df, OrangeDataFrame):
        table.attributes.update(df.orange_attributes)