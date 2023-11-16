"""
Builtin aggregators for SFrame groupby operator.
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from .util import _is_non_string_iterable

def SUM(src_column):
    if False:
        for i in range(10):
            print('nop')
    '\n    Builtin sum aggregator for groupby.\n\n    Example: Get the sum of the rating column for each user. If\n    src_column is of array type, if array\'s do not match in length a NoneType is\n    returned in the destination column.\n\n    >>> sf.groupby("user",\n    ...            {\'rating_sum\':tc.aggregate.SUM(\'rating\')})\n\n    '
    return ('__builtin__sum__', [src_column])

def ARGMAX(agg_column, out_column):
    if False:
        print('Hello World!')
    '\n    Builtin arg maximum aggregator for groupby\n\n    Example: Get the movie with maximum rating per user.\n\n    >>> sf.groupby("user",\n    ...            {\'best_movie\':tc.aggregate.ARGMAX(\'rating\',\'movie\')})\n    '
    return ('__builtin__argmax__', [agg_column, out_column])

def ARGMIN(agg_column, out_column):
    if False:
        print('Hello World!')
    '\n    Builtin arg minimum aggregator for groupby\n\n    Example: Get the movie with minimum rating per user.\n\n    >>> sf.groupby("user",\n    ...            {\'best_movie\':tc.aggregate.ARGMIN(\'rating\',\'movie\')})\n    '
    return ('__builtin__argmin__', [agg_column, out_column])

def MAX(src_column):
    if False:
        for i in range(10):
            print('nop')
    '\n    Builtin maximum aggregator for groupby\n\n    Example: Get the maximum rating of each user.\n\n    >>> sf.groupby("user",\n    ...            {\'rating_max\':tc.aggregate.MAX(\'rating\')})\n\n    '
    return ('__builtin__max__', [src_column])

def MIN(src_column):
    if False:
        return 10
    '\n    Builtin minimum aggregator for groupby\n\n    Example: Get the minimum rating of each user.\n\n    >>> sf.groupby("user",\n    ...            {\'rating_min\':tc.aggregate.MIN(\'rating\')})\n\n    '
    return ('__builtin__min__', [src_column])

def COUNT():
    if False:
        for i in range(10):
            print('nop')
    '\n    Builtin count aggregator for groupby\n\n    Example: Get the number of occurrences of each user.\n\n    >>> sf.groupby("user",\n    ...            {\'count\':tc.aggregate.COUNT()})\n    '
    return ('__builtin__count__', [''])

def AVG(src_column):
    if False:
        return 10
    '\n    Builtin average aggregator for groupby. Synonym for tc.aggregate.MEAN. If\n    src_column is of array type, and if array\'s do not match in length a NoneType is\n    returned in the destination column.\n\n    Example: Get the average rating of each user.\n\n    >>> sf.groupby("user",\n    ...            {\'rating_avg\':tc.aggregate.AVG(\'rating\')})\n    '
    return ('__builtin__avg__', [src_column])

def MEAN(src_column):
    if False:
        return 10
    '\n    Builtin average aggregator for groupby. Synonym for tc.aggregate.AVG. If\n    src_column is of array type, and if array\'s do not match in length a NoneType is\n    returned in the destination column.\n\n    Example: Get the average rating of each user.\n\n    >>> sf.groupby("user",\n    ...            {\'rating_mean\':tc.aggregate.MEAN(\'rating\')})\n\n    '
    return ('__builtin__avg__', [src_column])

def VAR(src_column):
    if False:
        while True:
            i = 10
    '\n    Builtin variance aggregator for groupby. Synonym for tc.aggregate.VARIANCE\n\n    Example: Get the rating variance of each user.\n\n    >>> sf.groupby("user",\n    ...            {\'rating_var\':tc.aggregate.VAR(\'rating\')})\n    '
    return ('__builtin__var__', [src_column])

def VARIANCE(src_column):
    if False:
        print('Hello World!')
    '\n    Builtin variance aggregator for groupby. Synonym for tc.aggregate.VAR\n\n    Example: Get the rating variance of each user.\n\n    >>> sf.groupby("user",\n    ...            {\'rating_var\':tc.aggregate.VARIANCE(\'rating\')})\n    '
    return ('__builtin__var__', [src_column])

def STD(src_column):
    if False:
        print('Hello World!')
    '\n    Builtin standard deviation aggregator for groupby. Synonym for tc.aggregate.STDV\n\n    Example: Get the rating standard deviation of each user.\n\n    >>> sf.groupby("user",\n    ...            {\'rating_std\':tc.aggregate.STD(\'rating\')})\n    '
    return ('__builtin__stdv__', [src_column])

def STDV(src_column):
    if False:
        return 10
    '\n    Builtin standard deviation aggregator for groupby. Synonym for tc.aggregate.STD\n\n    Example: Get the rating standard deviation of each user.\n\n    >>> sf.groupby("user",\n    ...            {\'rating_stdv\':tc.aggregate.STDV(\'rating\')})\n    '
    return ('__builtin__stdv__', [src_column])

def SELECT_ONE(src_column):
    if False:
        i = 10
        return i + 15
    '\n    Builtin aggregator for groupby which selects one row in the group.\n\n    Example: Get one rating row from a user.\n\n    >>> sf.groupby("user",\n    ...            {\'rating\':tc.aggregate.SELECT_ONE(\'rating\')})\n\n    If multiple columns are selected, they are guaranteed to come from the\n    same row. for instance:\n    >>> sf.groupby("user",\n    ...            {\'rating\':tc.aggregate.SELECT_ONE(\'rating\')},\n    ...            {\'item\':tc.aggregate.SELECT_ONE(\'item\')})\n\n    The selected \'rating\' and \'item\' value for each user will come from the\n    same row in the SFrame.\n    '
    return ('__builtin__select_one__', [src_column])

def CONCAT(src_column, dict_value_column=None):
    if False:
        i = 10
        return i + 15
    '\n    Builtin aggregator that combines values from one or two columns in one group\n    into either a dictionary value or list value.\n\n    If only one column is given, then the values of this column are\n    aggregated into a list.  Order is not preserved.  For example:\n\n    >>> sf.groupby(["user"],\n    ...     {"friends": tc.aggregate.CONCAT("friend")})\n\n    would form a new column "friends" containing values in column\n    "friend" aggregated into a list of friends.\n\n    If `dict_value_column` is given, then the aggregation forms a dictionary with\n    the keys taken from src_column and the values taken from `dict_value_column`.\n    For example:\n\n    >>> sf.groupby(["document"],\n    ...     {"word_count": tc.aggregate.CONCAT("word", "count")})\n\n    would aggregate words from column "word" and counts from column\n    "count" into a dictionary with keys being words and values being\n    counts.\n    '
    if dict_value_column is None:
        return ('__builtin__concat__list__', [src_column])
    else:
        return ('__builtin__concat__dict__', [src_column, dict_value_column])

def QUANTILE(src_column, *args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Builtin approximate quantile aggregator for groupby.\n    Accepts as an argument, one or more of a list of quantiles to query.\n    For instance:\n\n    To extract the median\n\n    >>> sf.groupby("user",\n    ...   {\'rating_quantiles\':tc.aggregate.QUANTILE(\'rating\', 0.5)})\n\n    To extract a few quantiles\n\n    >>> sf.groupby("user",\n    ...   {\'rating_quantiles\':tc.aggregate.QUANTILE(\'rating\', [0.25,0.5,0.75])})\n\n    Or equivalently\n\n    >>> sf.groupby("user",\n    ...     {\'rating_quantiles\':tc.aggregate.QUANTILE(\'rating\', 0.25,0.5,0.75)})\n\n    The returned quantiles are guaranteed to have 0.5% accuracy. That is to say,\n    if the requested quantile is 0.50, the resultant quantile value may be\n    between 0.495 and 0.505 of the true quantile.\n    '
    if len(args) == 1:
        quantiles = args[0]
    else:
        quantiles = list(args)
    if not _is_non_string_iterable(quantiles):
        quantiles = [quantiles]
    query = ','.join([str(i) for i in quantiles])
    return ('__builtin__quantile__[' + query + ']', [src_column])

def COUNT_DISTINCT(src_column):
    if False:
        return 10
    '\n    Builtin unique counter for groupby. Counts the number of unique values\n\n    Example: Get the number of unique ratings produced by each user.\n\n    >>> sf.groupby("user",\n    ...    {\'rating_distinct_count\':tc.aggregate.COUNT_DISTINCT(\'rating\')})\n    '
    return ('__builtin__count__distinct__', [src_column])

def DISTINCT(src_column):
    if False:
        while True:
            i = 10
    '\n    Builtin distinct values for groupby. Returns a list of distinct values.\n\n    >>> sf.groupby("user",\n    ...       {\'rating_distinct\':tc.aggregate.DISTINCT(\'rating\')})\n    '
    return ('__builtin__distinct__', [src_column])

def FREQ_COUNT(src_column):
    if False:
        while True:
            i = 10
    '\n    Builtin frequency counts for groupby. Returns a dictionary where the key is\n    the `src_column` and the value is the number of times each value occurs.\n\n    >>> sf.groupby("user",\n    ...       {\'rating_distinct\':tc.aggregate.FREQ_COUNT(\'rating\')})\n    '
    return ('__builtin__freq_count__', [src_column])