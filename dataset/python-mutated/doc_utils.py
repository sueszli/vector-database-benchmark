"""Module contains decorators for documentation of the query compiler methods."""
from functools import partial
from modin.utils import align_indents, append_to_docstring, format_string
_one_column_warning = '\n.. warning::\n    This method is supported only by one-column query compilers.\n'
_deprecation_warning = '\n.. warning::\n    This method duplicates logic of ``{0}`` and will be removed soon.\n'
_refer_to_note = '\nNotes\n-----\nPlease refer to ``modin.pandas.{0}`` for more information\nabout parameters and output format.\n'
add_one_column_warning = append_to_docstring(_one_column_warning)

def add_deprecation_warning(replacement_method):
    if False:
        for i in range(10):
            print('nop')
    "\n    Build decorator which appends deprecation warning to the function's docstring.\n\n    Appended warning indicates that the current method duplicates functionality of\n    some other method and so is slated to be removed in the future.\n\n    Parameters\n    ----------\n    replacement_method : str\n        Name of the method to use instead of deprecated.\n\n    Returns\n    -------\n    callable\n    "
    message = _deprecation_warning.format(replacement_method)
    return append_to_docstring(message)

def add_refer_to(method):
    if False:
        for i in range(10):
            print('nop')
    "\n    Build decorator which appends link to the high-level equivalent method to the function's docstring.\n\n    Parameters\n    ----------\n    method : str\n        Method name in ``modin.pandas`` module to refer to.\n\n    Returns\n    -------\n    callable\n    "
    note = _refer_to_note.format(method)
    return append_to_docstring(note)

def doc_qc_method(template, params=None, refer_to=None, refer_to_module_name=None, one_column_method=False, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Build decorator which adds docstring for query compiler method.\n\n    Parameters\n    ----------\n    template : str\n        Method docstring in the NumPy docstyle format. Must contain {params}\n        placeholder.\n    params : str, optional\n        Method parameters in the NumPy docstyle format to substitute\n        in the `template`. `params` string should not include the "Parameters"\n        header.\n    refer_to : str, optional\n        Method name in `refer_to_module_name` module to refer to for more information\n        about parameters and output format.\n    refer_to_module_name : str, optional\n    one_column_method : bool, default: False\n        Whether to append note that this method is for one-column\n        query compilers only.\n    **kwargs : dict\n        Values to substitute in the `template`.\n\n    Returns\n    -------\n    callable\n    '
    params_template = '\n\n        Parameters\n        ----------\n        {params}\n        '
    params = format_string(params_template, params=params) if params else ''
    substituted = format_string(template, params=params, refer_to=refer_to, **kwargs)
    if refer_to_module_name:
        refer_to = f'{refer_to_module_name}.{refer_to}'

    def decorator(func):
        if False:
            while True:
                i = 10
        func.__doc__ = substituted
        appendix = ''
        if refer_to:
            appendix += _refer_to_note.format(refer_to)
        if one_column_method:
            appendix += _one_column_warning
        if appendix:
            func = append_to_docstring(appendix)(func)
        return func
    return decorator

def doc_binary_method(operation, sign, self_on_right=False, op_type='arithmetic'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Build decorator which adds docstring for binary method.\n\n    Parameters\n    ----------\n    operation : str\n        Name of the binary operation.\n    sign : str\n        Sign which represents specified binary operation.\n    self_on_right : bool, default: False\n        Whether `self` is the right operand.\n    op_type : {"arithmetic", "logical", "comparison"}, default: "arithmetic"\n        Type of the binary operation.\n\n    Returns\n    -------\n    callable\n    '
    template = "\n    Perform element-wise {operation} (``{verbose}``).\n\n    If axes are not equal, perform frames alignment first.\n\n    Parameters\n    ----------\n    other : BaseQueryCompiler, scalar or array-like\n        Other operand of the binary operation.\n    broadcast : bool, default: False\n        If `other` is a one-column query compiler, indicates whether it is a Series or not.\n        Frames and Series have to be processed differently, however we can't distinguish them\n        at the query compiler level, so this parameter is a hint that is passed from a high-level API.\n    {extra_params}**kwargs : dict\n        Serves the compatibility purpose. Does not affect the result.\n\n    Returns\n    -------\n    BaseQueryCompiler\n        Result of binary operation.\n    "
    extra_params = {'logical': '\n        level : int or label\n            In case of MultiIndex match index values on the passed level.\n        axis : {{0, 1}}\n            Axis to match indices along for 1D `other` (list or QueryCompiler that represents Series).\n            0 is for index, when 1 is for columns.\n        ', 'arithmetic': '\n        level : int or label\n            In case of MultiIndex match index values on the passed level.\n        axis : {{0, 1}}\n            Axis to match indices along for 1D `other` (list or QueryCompiler that represents Series).\n            0 is for index, when 1 is for columns.\n        fill_value : float or None\n            Value to fill missing elements during frame alignment.\n        '}
    verbose_substitution = f'other {sign} self' if self_on_right else f'self {sign} other'
    params_substitution = extra_params.get(op_type, '')
    return doc_qc_method(template, extra_params=params_substitution, operation=operation, verbose=verbose_substitution)

def doc_reduce_agg(method, refer_to, params=None, extra_params=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Build decorator which adds docstring for the reduce method.\n\n    Parameters\n    ----------\n    method : str\n        The result of the method.\n    refer_to : str\n        Method name in ``modin.pandas.DataFrame`` module to refer to for\n        more information about parameters and output format.\n    params : str, optional\n        Method parameters in the NumPy docstyle format to substitute\n        to the docstring template.\n    extra_params : sequence of str, optional\n        Method parameter names to append to the docstring template. Parameter\n        type and description will be grabbed from ``extra_params_map`` (Please\n        refer to the source code of this function to explore the map).\n\n    Returns\n    -------\n    callable\n    '
    template = '\n        Get the {method} for each column or row.\n        {params}\n        Returns\n        -------\n        BaseQueryCompiler\n            One-column QueryCompiler with index labels of the specified axis,\n            where each row contains the {method} for the corresponding\n            row or column.\n        '
    if params is None:
        params = '\n        axis : {{0, 1}}\n        numeric_only : bool, optional'
    extra_params_map = {'skipna': '\n        skipna : bool, default: True', 'min_count': '\n        min_count : int', 'ddof': '\n        ddof : int', '*args': '\n        *args : iterable\n            Serves the compatibility purpose. Does not affect the result.', '**kwargs': '\n        **kwargs : dict\n            Serves the compatibility purpose. Does not affect the result.'}
    params += ''.join([align_indents(source=params, target=extra_params_map.get(param, f'\n{param} : object')) for param in extra_params or []])
    return doc_qc_method(template, params=params, method=method, refer_to=f'DataFrame.{refer_to}')
doc_cum_agg = partial(doc_qc_method, template='\n    Get cumulative {method} for every row or column.\n\n    Parameters\n    ----------\n    fold_axis : {{0, 1}}\n    skipna : bool\n    **kwargs : dict\n        Serves the compatibility purpose. Does not affect the result.\n\n    Returns\n    -------\n    BaseQueryCompiler\n        QueryCompiler of the same shape as `self`, where each element is the {method}\n        of all the previous values in this row or column.\n    ', refer_to_module_name='DataFrame')
doc_resample = partial(doc_qc_method, template='\n    Resample time-series data and apply aggregation on it.\n\n    Group data into intervals by time-series row/column with\n    a specified frequency and {action}.\n\n    Parameters\n    ----------\n    resample_kwargs : dict\n        Resample parameters as expected by ``modin.pandas.DataFrame.resample`` signature.\n    {extra_params}\n    Returns\n    -------\n    BaseQueryCompiler\n        New QueryCompiler containing the result of resample aggregation built by the\n        following rules:\n\n        {build_rules}\n    ', refer_to_module_name='resample.Resampler')

def doc_resample_reduce(result, refer_to, params=None, compatibility_params=True):
    if False:
        i = 10
        return i + 15
    '\n    Build decorator which adds docstring for the resample reduce method.\n\n    Parameters\n    ----------\n    result : str\n        The result of the method.\n    refer_to : str\n        Method name in ``modin.pandas.resample.Resampler`` module to refer to for\n        more information about parameters and output format.\n    params : str, optional\n        Method parameters in the NumPy docstyle format to substitute\n        to the docstring template.\n    compatibility_params : bool, default: True\n        Whether method takes `*args` and `**kwargs` that do not affect\n        the result.\n\n    Returns\n    -------\n    callable\n    '
    action = f'compute {result} for each group'
    params_substitution = '\n        *args : iterable\n            Serves the compatibility purpose. Does not affect the result.\n        **kwargs : dict\n            Serves the compatibility purpose. Does not affect the result.\n        ' if compatibility_params else ''
    if params:
        params_substitution = format_string('{params}\n{params_substitution}', params=params, params_substitution=params_substitution)
    build_rules = f'\n            - Labels on the specified axis are the group names (time-stamps)\n            - Labels on the opposite of specified axis are preserved.\n            - Each element of QueryCompiler is the {result} for the\n              corresponding group and column/row.'
    return doc_resample(action=action, extra_params=params_substitution, build_rules=build_rules, refer_to=refer_to)

def doc_resample_agg(action, output, refer_to, params=None):
    if False:
        print('Hello World!')
    '\n    Build decorator which adds docstring for the resample aggregation method.\n\n    Parameters\n    ----------\n    action : str\n        What method does with the resampled data.\n    output : str\n        What is the content of column names in the result.\n    refer_to : str\n        Method name in ``modin.pandas.resample.Resampler`` module to refer to for\n        more information about parameters and output format.\n    params : str, optional\n        Method parameters in the NumPy docstyle format to substitute\n        to the docstring template.\n\n    Returns\n    -------\n    callable\n    '
    action = f'{action} for each group over the specified axis'
    params_substitution = '\n        *args : iterable\n            Positional arguments to pass to the aggregation function.\n        **kwargs : dict\n            Keyword arguments to pass to the aggregation function.\n        '
    if params:
        params_substitution = format_string('{params}\n{params_substitution}', params=params, params_substitution=params_substitution)
    build_rules = f'\n            - Labels on the specified axis are the group names (time-stamps)\n            - Labels on the opposite of specified axis are a MultiIndex, where first level\n              contains preserved labels of this axis and the second level is the {output}.\n            - Each element of QueryCompiler is the result of corresponding function for the\n              corresponding group and column/row.'
    return doc_resample(action=action, extra_params=params_substitution, build_rules=build_rules, refer_to=refer_to)

def doc_resample_fillna(method, refer_to, params=None, overwrite_template_params=False):
    if False:
        return 10
    '\n    Build decorator which adds docstring for the resample fillna query compiler method.\n\n    Parameters\n    ----------\n    method : str\n        Fillna method name.\n    refer_to : str\n        Method name in ``modin.pandas.resample.Resampler`` module to refer to for\n        more information about parameters and output format.\n    params : str, optional\n        Method parameters in the NumPy docstyle format to substitute\n        to the docstring template.\n    overwrite_template_params : bool, default: False\n        If `params` is specified indicates whether to overwrite method parameters in\n        the docstring template or append then at the end.\n\n    Returns\n    -------\n    callable\n    '
    action = f'fill missing values in each group independently using {method} method'
    params_substitution = 'limit : int\n'
    if params:
        params_substitution = params if overwrite_template_params else format_string('{params}\n{params_substitution}', params=params, params_substitution=params_substitution)
    build_rules = '- QueryCompiler contains unsampled data with missing values filled.'
    return doc_resample(action=action, extra_params=params_substitution, build_rules=build_rules, refer_to=refer_to)
doc_dt = partial(doc_qc_method, template='\n    Get {prop} for each {dt_type} value.\n    {params}\n    Returns\n    -------\n    BaseQueryCompiler\n        New QueryCompiler with the same shape as `self`, where each element is\n        {prop} for the corresponding {dt_type} value.\n    ', one_column_method=True, refer_to_module_name='Series.dt')
doc_dt_timestamp = partial(doc_dt, dt_type='datetime')
doc_dt_interval = partial(doc_dt, dt_type='interval')
doc_dt_period = partial(doc_dt, dt_type='period')
doc_dt_round = partial(doc_qc_method, template='\n    Perform {refer_to} operation on the underlying time-series data to the specified `freq`.\n\n    Parameters\n    ----------\n    freq : str\n    ambiguous : {{"raise", "infer", "NaT"}} or bool mask, default: "raise"\n    nonexistent : {{"raise", "shift_forward", "shift_backward", "NaT"}} or timedelta, default: "raise"\n\n    Returns\n    -------\n    BaseQueryCompiler\n        New QueryCompiler with performed {refer_to} operation on every element.\n    ', one_column_method=True, refer_to_module_name='Series.dt')
doc_str_method = partial(doc_qc_method, template='\n    Apply "{refer_to}" function to each string value in QueryCompiler.\n    {params}\n    Returns\n    -------\n    BaseQueryCompiler\n        New QueryCompiler containing the result of execution of the "{refer_to}" function\n        against each string element.\n    ', one_column_method=True, refer_to_module_name='Series.str')

def doc_window_method(window_cls_name, result, refer_to, action=None, win_type='rolling window', params=None, build_rules='aggregation'):
    if False:
        while True:
            i = 10
    '\n    Build decorator which adds docstring for a window method.\n\n    Parameters\n    ----------\n    window_cls_name : str\n        The Window class the method is on.\n    result : str\n        The result of the method.\n    refer_to : str\n        Method name in ``modin.pandas.window.Window`` module to refer to\n        for more information about parameters and output format.\n    action : str, optional\n        What method does with the created window.\n    win_type : str, default: "rolling_window"\n        Type of window that the method creates.\n    params : str, optional\n        Method parameters in the NumPy docstyle format to substitute\n        to the docstring template.\n    build_rules : str, default: "aggregation"\n        Description of the data output format.\n\n    Returns\n    -------\n    callable\n    '
    template = '\n        Create {win_type} and {action} for each window over the given axis.\n\n        Parameters\n        ----------\n        fold_axis : {{0, 1}}\n        {window_args_name} : list\n            Rolling windows arguments with the same signature as ``modin.pandas.DataFrame.rolling``.\n        {extra_params}\n        Returns\n        -------\n        BaseQueryCompiler\n            New QueryCompiler containing {result} for each window, built by the following\n            rules:\n\n            {build_rules}\n        '
    doc_build_rules = {'aggregation': f'\n            - Output QueryCompiler has the same shape and axes labels as the source.\n            - Each element is the {result} for the corresponding window.', 'udf_aggregation': '\n            - Labels on the specified axis are preserved.\n            - Labels on the opposite of specified axis are MultiIndex, where first level\n              contains preserved labels of this axis and the second level has the function names.\n            - Each element of QueryCompiler is the result of corresponding function for the\n              corresponding window and column/row.'}
    if action is None:
        action = f'compute {result}'
    if win_type == 'rolling window':
        window_args_name = 'rolling_kwargs'
    elif win_type == 'expanding window':
        window_args_name = 'expanding_args'
    else:
        window_args_name = 'window_kwargs'
    if params and params[-1] != '\n':
        params += '\n'
    if params is None:
        params = ''
    return doc_qc_method(template, result=result, action=action, win_type=win_type, extra_params=params, build_rules=doc_build_rules.get(build_rules, build_rules), refer_to=f'{window_cls_name}.{refer_to}', window_args_name=window_args_name)

def doc_groupby_method(result, refer_to, action=None):
    if False:
        i = 10
        return i + 15
    '\n    Build decorator which adds docstring for the groupby reduce method.\n\n    Parameters\n    ----------\n    result : str\n        The result of reduce.\n    refer_to : str\n        Method name in ``modin.pandas.groupby`` module to refer to\n        for more information about parameters and output format.\n    action : str, optional\n        What method does with groups.\n\n    Returns\n    -------\n    callable\n    '
    template = '\n    Group QueryCompiler data and {action} for every group.\n\n    Parameters\n    ----------\n    by : BaseQueryCompiler, column or index label, Grouper or list of such\n        Object that determine groups.\n    axis : {{0, 1}}\n        Axis to group and apply aggregation function along.\n        0 is for index, when 1 is for columns.\n    groupby_kwargs : dict\n        GroupBy parameters as expected by ``modin.pandas.DataFrame.groupby`` signature.\n    agg_args : list-like\n        Positional arguments to pass to the `agg_func`.\n    agg_kwargs : dict\n        Key arguments to pass to the `agg_func`.\n    drop : bool, default: False\n        If `by` is a QueryCompiler indicates whether or not by-data came\n        from the `self`.\n\n    Returns\n    -------\n    BaseQueryCompiler\n        QueryCompiler containing the result of groupby reduce built by the\n        following rules:\n\n        - Labels on the opposite of specified axis are preserved.\n        - If groupby_args["as_index"] is True then labels on the specified axis\n          are the group names, otherwise labels would be default: 0, 1 ... n.\n        - If groupby_args["as_index"] is False, then first N columns/rows of the frame\n          contain group names, where N is the columns/rows to group on.\n        - Each element of QueryCompiler is the {result} for the\n          corresponding group and column/row.\n\n    .. warning\n        `map_args` and `reduce_args` parameters are deprecated. They\'re leaked here from\n        ``PandasQueryCompiler.groupby_*``, pandas storage format implements groupby via TreeReduce\n        approach, but for other storage formats these parameters make no sense, and so they\'ll be removed in the future.\n    '
    if action is None:
        action = f'compute {result}'
    return doc_qc_method(template, result=result, action=action, refer_to=f'GroupBy.{refer_to}')