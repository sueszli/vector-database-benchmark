"""
factor.py
"""
from operator import attrgetter
from numbers import Number
from math import ceil
from textwrap import dedent
from numpy import empty_like, inf, isnan, nan, where
from scipy.stats import rankdata
from zipline.utils.compat import wraps
from zipline.errors import BadPercentileBounds, UnknownRankMethod, UnsupportedDataType
from zipline.lib.normalize import naive_grouped_rowwise_apply
from zipline.lib.rank import masked_rankdata_2d, rankdata_1d_descending
from zipline.pipeline.api_utils import restrict_to_dtype
from zipline.pipeline.classifiers import Classifier, Everything, Quantiles
from zipline.pipeline.dtypes import CLASSIFIER_DTYPES, FACTOR_DTYPES, FILTER_DTYPES
from zipline.pipeline.expression import BadBinaryOperator, COMPARISONS, is_comparison, MATH_BINOPS, method_name_for_op, NumericalExpression, NUMEXPR_MATH_FUNCS, UNARY_OPS, unary_op_name
from zipline.pipeline.filters import Filter, NumExprFilter, PercentileFilter, MaximumFilter
from zipline.pipeline.mixins import CustomTermMixin, LatestMixin, PositiveWindowLengthMixin, RestrictedDTypeMixin, SingleInputMixin
from zipline.pipeline.sentinels import NotSpecified, NotSpecifiedType
from zipline.pipeline.term import AssetExists, ComputableTerm, Term
from zipline.utils.functional import with_doc, with_name
from zipline.utils.input_validation import expect_types
from zipline.utils.math_utils import nanmax, nanmean, nanmedian, nanmin, nanstd, nansum
from zipline.utils.numpy_utils import as_column, bool_dtype, coerce_to_dtype, float64_dtype, is_missing
from zipline.utils.sharedoc import templated_docstring
_RANK_METHODS = frozenset(['average', 'min', 'max', 'dense', 'ordinal'])

def coerce_numbers_to_my_dtype(f):
    if False:
        print('Hello World!')
    '\n    A decorator for methods whose signature is f(self, other) that coerces\n    ``other`` to ``self.dtype``.\n\n    This is used to make comparison operations between numbers and `Factor`\n    instances work independently of whether the user supplies a float or\n    integer literal.\n\n    For example, if I write::\n\n        my_filter = my_factor > 3\n\n    my_factor probably has dtype float64, but 3 is an int, so we want to coerce\n    to float64 before doing the comparison.\n    '

    @wraps(f)
    def method(self, other):
        if False:
            print('Hello World!')
        if isinstance(other, Number):
            other = coerce_to_dtype(self.dtype, other)
        return f(self, other)
    return method

def binop_return_dtype(op, left, right):
    if False:
        i = 10
        return i + 15
    "\n    Compute the expected return dtype for the given binary operator.\n\n    Parameters\n    ----------\n    op : str\n        Operator symbol, (e.g. '+', '-', ...).\n    left : numpy.dtype\n        Dtype of left hand side.\n    right : numpy.dtype\n        Dtype of right hand side.\n\n    Returns\n    -------\n    outdtype : numpy.dtype\n        The dtype of the result of `left <op> right`.\n    "
    if is_comparison(op):
        if left != right:
            raise TypeError("Don't know how to compute {left} {op} {right}.\nComparisons are only supported between Factors of equal dtypes.".format(left=left, op=op, right=right))
        return bool_dtype
    elif left != float64_dtype or right != float64_dtype:
        raise TypeError("Don't know how to compute {left} {op} {right}.\nArithmetic operators are only supported between Factors of dtype 'float64'.".format(left=left.name, op=op, right=right.name))
    return float64_dtype
BINOP_DOCSTRING_TEMPLATE = '\nConstruct a :class:`~zipline.pipeline.{rtype}` computing ``self {op} other``.\n\nParameters\n----------\nother : zipline.pipeline.Factor, float\n    Right-hand side of the expression.\n\nReturns\n-------\n{ret}\n'
BINOP_RETURN_FILTER = 'filter : zipline.pipeline.Filter\n    Filter computing ``self {op} other`` with the outputs of ``self`` and\n    ``other``.\n'
BINOP_RETURN_FACTOR = 'factor : zipline.pipeline.Factor\n    Factor computing ``self {op} other`` with outputs of ``self`` and\n    ``other``.\n'

def binary_operator(op):
    if False:
        for i in range(10):
            print('nop')
    '\n    Factory function for making binary operator methods on a Factor subclass.\n\n    Returns a function, "binary_operator" suitable for implementing functions\n    like __add__.\n    '
    commuted_method_getter = attrgetter(method_name_for_op(op, commute=True))
    is_compare = is_comparison(op)
    if is_compare:
        ret_doc = BINOP_RETURN_FILTER.format(op=op)
        rtype = 'Filter'
    else:
        ret_doc = BINOP_RETURN_FACTOR.format(op=op)
        rtype = 'Factor'
    docstring = BINOP_DOCSTRING_TEMPLATE.format(op=op, ret=ret_doc, rtype=rtype)

    @with_doc(docstring)
    @with_name(method_name_for_op(op))
    @coerce_numbers_to_my_dtype
    def binary_operator(self, other):
        if False:
            for i in range(10):
                print('nop')
        return_type = NumExprFilter if is_compare else NumExprFactor
        if isinstance(self, NumExprFactor):
            (self_expr, other_expr, new_inputs) = self.build_binary_op(op, other)
            return return_type('({left}) {op} ({right})'.format(left=self_expr, op=op, right=other_expr), new_inputs, dtype=binop_return_dtype(op, self.dtype, other.dtype))
        elif isinstance(other, NumExprFactor):
            return commuted_method_getter(other)(self)
        elif isinstance(other, Term):
            if self is other:
                return return_type('x_0 {op} x_0'.format(op=op), (self,), dtype=binop_return_dtype(op, self.dtype, other.dtype))
            return return_type('x_0 {op} x_1'.format(op=op), (self, other), dtype=binop_return_dtype(op, self.dtype, other.dtype))
        elif isinstance(other, Number):
            return return_type('x_0 {op} ({constant})'.format(op=op, constant=other), binds=(self,), dtype=binop_return_dtype(op, self.dtype, other.dtype))
        raise BadBinaryOperator(op, self, other)
    return binary_operator

def reflected_binary_operator(op):
    if False:
        i = 10
        return i + 15
    '\n    Factory function for making binary operator methods on a Factor.\n\n    Returns a function, "reflected_binary_operator" suitable for implementing\n    functions like __radd__.\n    '
    assert not is_comparison(op)

    @with_name(method_name_for_op(op, commute=True))
    @coerce_numbers_to_my_dtype
    def reflected_binary_operator(self, other):
        if False:
            return 10
        if isinstance(self, NumericalExpression):
            (self_expr, other_expr, new_inputs) = self.build_binary_op(op, other)
            return NumExprFactor('({left}) {op} ({right})'.format(left=other_expr, right=self_expr, op=op), new_inputs, dtype=binop_return_dtype(op, other.dtype, self.dtype))
        elif isinstance(other, Number):
            return NumExprFactor('{constant} {op} x_0'.format(op=op, constant=other), binds=(self,), dtype=binop_return_dtype(op, other.dtype, self.dtype))
        raise BadBinaryOperator(op, other, self)
    return reflected_binary_operator

def unary_operator(op):
    if False:
        while True:
            i = 10
    '\n    Factory function for making unary operator methods for Factors.\n    '
    valid_ops = {'-'}
    if op not in valid_ops:
        raise ValueError('Invalid unary operator %s.' % op)

    @with_doc("Unary Operator: '%s'" % op)
    @with_name(unary_op_name(op))
    def unary_operator(self):
        if False:
            while True:
                i = 10
        if self.dtype != float64_dtype:
            raise TypeError("Can't apply unary operator {op!r} to instance of {typename!r} with dtype {dtypename!r}.\n{op!r} is only supported for Factors of dtype 'float64'.".format(op=op, typename=type(self).__name__, dtypename=self.dtype.name))
        if isinstance(self, NumericalExpression):
            return NumExprFactor('{op}({expr})'.format(op=op, expr=self._expr), self.inputs, dtype=float64_dtype)
        else:
            return NumExprFactor('{op}x_0'.format(op=op), (self,), dtype=float64_dtype)
    return unary_operator

def function_application(func):
    if False:
        for i in range(10):
            print('nop')
    '\n    Factory function for producing function application methods for Factor\n    subclasses.\n    '
    if func not in NUMEXPR_MATH_FUNCS:
        raise ValueError("Unsupported mathematical function '%s'" % func)
    docstring = dedent('        Construct a Factor that computes ``{}()`` on each output of ``self``.\n\n        Returns\n        -------\n        factor : zipline.pipeline.Factor\n        '.format(func))

    @with_doc(docstring)
    @with_name(func)
    def mathfunc(self):
        if False:
            while True:
                i = 10
        if isinstance(self, NumericalExpression):
            return NumExprFactor('{func}({expr})'.format(func=func, expr=self._expr), self.inputs, dtype=float64_dtype)
        else:
            return NumExprFactor('{func}(x_0)'.format(func=func), (self,), dtype=float64_dtype)
    return mathfunc
if_not_float64_tell_caller_to_use_isnull = restrict_to_dtype(dtype=float64_dtype, message_template='{method_name}() was called on a factor of dtype {received_dtype}.\n{method_name}() is only defined for dtype {expected_dtype}.To filter missing data, use isnull() or notnull().')
float64_only = restrict_to_dtype(dtype=float64_dtype, message_template='{method_name}() is only defined on Factors of dtype {expected_dtype}, but it was called on a Factor of dtype {received_dtype}.')
CORRELATION_METHOD_NOTE = dedent('    This method can only be called on expressions which are deemed safe for use\n    as inputs to windowed :class:`~zipline.pipeline.Factor` objects. Examples\n    of such expressions include This includes\n    :class:`~zipline.pipeline.data.BoundColumn`\n    :class:`~zipline.pipeline.factors.Returns` and any factors created from\n    :meth:`~zipline.pipeline.Factor.rank` or\n    :meth:`~zipline.pipeline.Factor.zscore`.\n    ')

class summary_funcs(object):
    """Namespace of functions meant to be used with DailySummary.
    """

    @staticmethod
    def mean(a, missing_value):
        if False:
            print('Hello World!')
        return nanmean(a, axis=1)

    @staticmethod
    def stddev(a, missing_value):
        if False:
            for i in range(10):
                print('nop')
        return nanstd(a, axis=1)

    @staticmethod
    def max(a, missing_value):
        if False:
            i = 10
            return i + 15
        return nanmax(a, axis=1)

    @staticmethod
    def min(a, missing_value):
        if False:
            print('Hello World!')
        return nanmin(a, axis=1)

    @staticmethod
    def median(a, missing_value):
        if False:
            for i in range(10):
                print('nop')
        return nanmedian(a, axis=1)

    @staticmethod
    def sum(a, missing_value):
        if False:
            return 10
        return nansum(a, axis=1)

    @staticmethod
    def notnull_count(a, missing_value):
        if False:
            for i in range(10):
                print('nop')
        return (~is_missing(a, missing_value)).sum(axis=1)
    names = {k for k in locals() if not k.startswith('_')}

def summary_method(name):
    if False:
        i = 10
        return i + 15
    func = getattr(summary_funcs, name)

    @expect_types(mask=(Filter, NotSpecifiedType))
    @float64_only
    def f(self, mask=NotSpecified):
        if False:
            while True:
                i = 10
        'Create a 1-dimensional factor computing the {} of self, each day.\n\n        Parameters\n        ----------\n        mask : zipline.pipeline.Filter, optional\n           A Filter representing assets to consider when computing results.\n           If supplied, we ignore asset/date pairs where ``mask`` produces\n           ``False``.\n\n        Returns\n        -------\n        result : zipline.pipeline.Factor\n        '
        return DailySummary(func, self, mask=mask, dtype=self.dtype)
    f.__name__ = func.__name__
    f.__doc__ = f.__doc__.format(f.__name__)
    return f

class Factor(RestrictedDTypeMixin, ComputableTerm):
    """
    Pipeline API expression producing a numerical or date-valued output.

    Factors are the most commonly-used Pipeline term, representing the result
    of any computation producing a numerical result.

    Factors can be combined, both with other Factors and with scalar values,
    via any of the builtin mathematical operators (``+``, ``-``, ``*``, etc).

    This makes it easy to write complex expressions that combine multiple
    Factors. For example, constructing a Factor that computes the average of
    two other Factors is simply::

        >>> f1 = SomeFactor(...)  # doctest: +SKIP
        >>> f2 = SomeOtherFactor(...)  # doctest: +SKIP
        >>> average = (f1 + f2) / 2.0  # doctest: +SKIP

    Factors can also be converted into :class:`zipline.pipeline.Filter` objects
    via comparison operators: (``<``, ``<=``, ``!=``, ``eq``, ``>``, ``>=``).

    There are many natural operators defined on Factors besides the basic
    numerical operators. These include methods for identifying missing or
    extreme-valued outputs (:meth:`isnull`, :meth:`notnull`, :meth:`isnan`,
    :meth:`notnan`), methods for normalizing outputs (:meth:`rank`,
    :meth:`demean`, :meth:`zscore`), and methods for constructing Filters based
    on rank-order properties of results (:meth:`top`, :meth:`bottom`,
    :meth:`percentile_between`).
    """
    ALLOWED_DTYPES = FACTOR_DTYPES
    clsdict = locals()
    clsdict.update({method_name_for_op(op): binary_operator(op) for op in MATH_BINOPS.union(COMPARISONS - {'=='})})
    clsdict.update({method_name_for_op(op, commute=True): reflected_binary_operator(op) for op in MATH_BINOPS})
    clsdict.update({unary_op_name(op): unary_operator(op) for op in UNARY_OPS})
    clsdict.update({funcname: function_application(funcname) for funcname in NUMEXPR_MATH_FUNCS})
    __truediv__ = clsdict['__div__']
    __rtruediv__ = clsdict['__rdiv__']
    clsdict.update({name: summary_method(name) for name in summary_funcs.names})
    del clsdict
    eq = binary_operator('==')

    @expect_types(mask=(Filter, NotSpecifiedType), groupby=(Classifier, NotSpecifiedType))
    @float64_only
    def demean(self, mask=NotSpecified, groupby=NotSpecified):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a Factor that computes ``self`` and subtracts the mean from\n        row of the result.\n\n        If ``mask`` is supplied, ignore values where ``mask`` returns False\n        when computing row means, and output NaN anywhere the mask is False.\n\n        If ``groupby`` is supplied, compute by partitioning each row based on\n        the values produced by ``groupby``, de-meaning the partitioned arrays,\n        and stitching the sub-results back together.\n\n        Parameters\n        ----------\n        mask : zipline.pipeline.Filter, optional\n            A Filter defining values to ignore when computing means.\n        groupby : zipline.pipeline.Classifier, optional\n            A classifier defining partitions over which to compute means.\n\n        Examples\n        --------\n        Let ``f`` be a Factor which would produce the following output::\n\n                         AAPL   MSFT    MCD     BK\n            2017-03-13    1.0    2.0    3.0    4.0\n            2017-03-14    1.5    2.5    3.5    1.0\n            2017-03-15    2.0    3.0    4.0    1.5\n            2017-03-16    2.5    3.5    1.0    2.0\n\n        Let ``c`` be a Classifier producing the following output::\n\n                         AAPL   MSFT    MCD     BK\n            2017-03-13      1      1      2      2\n            2017-03-14      1      1      2      2\n            2017-03-15      1      1      2      2\n            2017-03-16      1      1      2      2\n\n        Let ``m`` be a Filter producing the following output::\n\n                         AAPL   MSFT    MCD     BK\n            2017-03-13  False   True   True   True\n            2017-03-14   True  False   True   True\n            2017-03-15   True   True  False   True\n            2017-03-16   True   True   True  False\n\n        Then ``f.demean()`` will subtract the mean from each row produced by\n        ``f``.\n\n        ::\n\n                         AAPL   MSFT    MCD     BK\n            2017-03-13 -1.500 -0.500  0.500  1.500\n            2017-03-14 -0.625  0.375  1.375 -1.125\n            2017-03-15 -0.625  0.375  1.375 -1.125\n            2017-03-16  0.250  1.250 -1.250 -0.250\n\n        ``f.demean(mask=m)`` will subtract the mean from each row, but means\n        will be calculated ignoring values on the diagonal, and NaNs will\n        written to the diagonal in the output. Diagonal values are ignored\n        because they are the locations where the mask ``m`` produced False.\n\n        ::\n\n                         AAPL   MSFT    MCD     BK\n            2017-03-13    NaN -1.000  0.000  1.000\n            2017-03-14 -0.500    NaN  1.500 -1.000\n            2017-03-15 -0.166  0.833    NaN -0.666\n            2017-03-16  0.166  1.166 -1.333    NaN\n\n        ``f.demean(groupby=c)`` will subtract the group-mean of AAPL/MSFT and\n        MCD/BK from their respective entries.  The AAPL/MSFT are grouped\n        together because both assets always produce 1 in the output of the\n        classifier ``c``.  Similarly, MCD/BK are grouped together because they\n        always produce 2.\n\n        ::\n\n                         AAPL   MSFT    MCD     BK\n            2017-03-13 -0.500  0.500 -0.500  0.500\n            2017-03-14 -0.500  0.500  1.250 -1.250\n            2017-03-15 -0.500  0.500  1.250 -1.250\n            2017-03-16 -0.500  0.500 -0.500  0.500\n\n        ``f.demean(mask=m, groupby=c)`` will also subtract the group-mean of\n        AAPL/MSFT and MCD/BK, but means will be calculated ignoring values on\n        the diagonal , and NaNs will be written to the diagonal in the output.\n\n        ::\n\n                         AAPL   MSFT    MCD     BK\n            2017-03-13    NaN  0.000 -0.500  0.500\n            2017-03-14  0.000    NaN  1.250 -1.250\n            2017-03-15 -0.500  0.500    NaN  0.000\n            2017-03-16 -0.500  0.500  0.000    NaN\n\n        Notes\n        -----\n        Mean is sensitive to the magnitudes of outliers. When working with\n        factor that can potentially produce large outliers, it is often useful\n        to use the ``mask`` parameter to discard values at the extremes of the\n        distribution::\n\n            >>> base = MyFactor(...)  # doctest: +SKIP\n            >>> normalized = base.demean(\n            ...     mask=base.percentile_between(1, 99),\n            ... )  # doctest: +SKIP\n\n        ``demean()`` is only supported on Factors of dtype float64.\n\n        See Also\n        --------\n        :meth:`pandas.DataFrame.groupby`\n        '
        return GroupedRowTransform(transform=demean, transform_args=(), factor=self, groupby=groupby, dtype=self.dtype, missing_value=self.missing_value, window_safe=self.window_safe, mask=mask)

    @expect_types(mask=(Filter, NotSpecifiedType), groupby=(Classifier, NotSpecifiedType))
    @float64_only
    def zscore(self, mask=NotSpecified, groupby=NotSpecified):
        if False:
            return 10
        "\n        Construct a Factor that Z-Scores each day's results.\n\n        The Z-Score of a row is defined as::\n\n            (row - row.mean()) / row.stddev()\n\n        If ``mask`` is supplied, ignore values where ``mask`` returns False\n        when computing row means and standard deviations, and output NaN\n        anywhere the mask is False.\n\n        If ``groupby`` is supplied, compute by partitioning each row based on\n        the values produced by ``groupby``, z-scoring the partitioned arrays,\n        and stitching the sub-results back together.\n\n        Parameters\n        ----------\n        mask : zipline.pipeline.Filter, optional\n            A Filter defining values to ignore when Z-Scoring.\n        groupby : zipline.pipeline.Classifier, optional\n            A classifier defining partitions over which to compute Z-Scores.\n\n        Returns\n        -------\n        zscored : zipline.pipeline.Factor\n            A Factor producing that z-scores the output of self.\n\n        Notes\n        -----\n        Mean and standard deviation are sensitive to the magnitudes of\n        outliers. When working with factor that can potentially produce large\n        outliers, it is often useful to use the ``mask`` parameter to discard\n        values at the extremes of the distribution::\n\n            >>> base = MyFactor(...)  # doctest: +SKIP\n            >>> normalized = base.zscore(\n            ...    mask=base.percentile_between(1, 99),\n            ... )  # doctest: +SKIP\n\n        ``zscore()`` is only supported on Factors of dtype float64.\n\n        Examples\n        --------\n        See :meth:`~zipline.pipeline.Factor.demean` for an in-depth\n        example of the semantics for ``mask`` and ``groupby``.\n\n        See Also\n        --------\n        :meth:`pandas.DataFrame.groupby`\n        "
        return GroupedRowTransform(transform=zscore, transform_args=(), factor=self, groupby=groupby, dtype=self.dtype, missing_value=self.missing_value, mask=mask, window_safe=True)

    def rank(self, method='ordinal', ascending=True, mask=NotSpecified, groupby=NotSpecified):
        if False:
            for i in range(10):
                print('nop')
        "\n        Construct a new Factor representing the sorted rank of each column\n        within each row.\n\n        Parameters\n        ----------\n        method : str, {'ordinal', 'min', 'max', 'dense', 'average'}\n            The method used to assign ranks to tied elements. See\n            `scipy.stats.rankdata` for a full description of the semantics for\n            each ranking method. Default is 'ordinal'.\n        ascending : bool, optional\n            Whether to return sorted rank in ascending or descending order.\n            Default is True.\n        mask : zipline.pipeline.Filter, optional\n            A Filter representing assets to consider when computing ranks.\n            If mask is supplied, ranks are computed ignoring any asset/date\n            pairs for which `mask` produces a value of False.\n        groupby : zipline.pipeline.Classifier, optional\n            A classifier defining partitions over which to perform ranking.\n\n        Returns\n        -------\n        ranks : zipline.pipeline.Factor\n            A new factor that will compute the ranking of the data produced by\n            `self`.\n\n        Notes\n        -----\n        The default value for `method` is different from the default for\n        `scipy.stats.rankdata`.  See that function's documentation for a full\n        description of the valid inputs to `method`.\n\n        Missing or non-existent data on a given day will cause an asset to be\n        given a rank of NaN for that day.\n\n        See Also\n        --------\n        :func:`scipy.stats.rankdata`\n        "
        if groupby is NotSpecified:
            return Rank(self, method=method, ascending=ascending, mask=mask)
        return GroupedRowTransform(transform=rankdata if ascending else rankdata_1d_descending, transform_args=(method,), factor=self, groupby=groupby, dtype=float64_dtype, missing_value=nan, mask=mask, window_safe=True)

    @expect_types(target=Term, correlation_length=int, mask=(Filter, NotSpecifiedType))
    @templated_docstring(CORRELATION_METHOD_NOTE=CORRELATION_METHOD_NOTE)
    def pearsonr(self, target, correlation_length, mask=NotSpecified):
        if False:
            while True:
                i = 10
        "\n        Construct a new Factor that computes rolling pearson correlation\n        coefficients between ``target`` and the columns of ``self``.\n\n        Parameters\n        ----------\n        target : zipline.pipeline.Term\n            The term used to compute correlations against each column of data\n            produced by `self`. This may be a Factor, a BoundColumn or a Slice.\n            If `target` is two-dimensional, correlations are computed\n            asset-wise.\n        correlation_length : int\n            Length of the lookback window over which to compute each\n            correlation coefficient.\n        mask : zipline.pipeline.Filter, optional\n            A Filter describing which assets should have their correlation with\n            the target slice computed each day.\n\n        Returns\n        -------\n        correlations : zipline.pipeline.Factor\n            A new Factor that will compute correlations between ``target`` and\n            the columns of ``self``.\n\n        Notes\n        -----\n        {CORRELATION_METHOD_NOTE}\n\n        Examples\n        --------\n        Suppose we want to create a factor that computes the correlation\n        between AAPL's 10-day returns and the 10-day returns of all other\n        assets, computing each correlation over 30 days. This can be achieved\n        by doing the following::\n\n            returns = Returns(window_length=10)\n            returns_slice = returns[sid(24)]\n            aapl_correlations = returns.pearsonr(\n                target=returns_slice, correlation_length=30,\n            )\n\n        This is equivalent to doing::\n\n            aapl_correlations = RollingPearsonOfReturns(\n                target=sid(24), returns_length=10, correlation_length=30,\n            )\n\n        See Also\n        --------\n        :func:`scipy.stats.pearsonr`\n        :class:`zipline.pipeline.factors.RollingPearsonOfReturns`\n        :meth:`Factor.spearmanr`\n        "
        from .statistical import RollingPearson
        return RollingPearson(base_factor=self, target=target, correlation_length=correlation_length, mask=mask)

    @expect_types(target=Term, correlation_length=int, mask=(Filter, NotSpecifiedType))
    @templated_docstring(CORRELATION_METHOD_NOTE=CORRELATION_METHOD_NOTE)
    def spearmanr(self, target, correlation_length, mask=NotSpecified):
        if False:
            while True:
                i = 10
        "\n        Construct a new Factor that computes rolling spearman rank correlation\n        coefficients between ``target`` and the columns of ``self``.\n\n        Parameters\n        ----------\n        target : zipline.pipeline.Term\n            The term used to compute correlations against each column of data\n            produced by `self`. This may be a Factor, a BoundColumn or a Slice.\n            If `target` is two-dimensional, correlations are computed\n            asset-wise.\n        correlation_length : int\n            Length of the lookback window over which to compute each\n            correlation coefficient.\n        mask : zipline.pipeline.Filter, optional\n            A Filter describing which assets should have their correlation with\n            the target slice computed each day.\n\n        Returns\n        -------\n        correlations : zipline.pipeline.Factor\n            A new Factor that will compute correlations between ``target`` and\n            the columns of ``self``.\n\n        Notes\n        -----\n        {CORRELATION_METHOD_NOTE}\n\n        Examples\n        --------\n        Suppose we want to create a factor that computes the correlation\n        between AAPL's 10-day returns and the 10-day returns of all other\n        assets, computing each correlation over 30 days. This can be achieved\n        by doing the following::\n\n            returns = Returns(window_length=10)\n            returns_slice = returns[sid(24)]\n            aapl_correlations = returns.spearmanr(\n                target=returns_slice, correlation_length=30,\n            )\n\n        This is equivalent to doing::\n\n            aapl_correlations = RollingSpearmanOfReturns(\n                target=sid(24), returns_length=10, correlation_length=30,\n            )\n\n        See Also\n        --------\n        :func:`scipy.stats.spearmanr`\n        :meth:`Factor.pearsonr`\n        "
        from .statistical import RollingSpearman
        return RollingSpearman(base_factor=self, target=target, correlation_length=correlation_length, mask=mask)

    @expect_types(target=Term, regression_length=int, mask=(Filter, NotSpecifiedType))
    @templated_docstring(CORRELATION_METHOD_NOTE=CORRELATION_METHOD_NOTE)
    def linear_regression(self, target, regression_length, mask=NotSpecified):
        if False:
            for i in range(10):
                print('nop')
        "\n        Construct a new Factor that performs an ordinary least-squares\n        regression predicting the columns of `self` from `target`.\n\n        Parameters\n        ----------\n        target : zipline.pipeline.Term\n            The term to use as the predictor/independent variable in each\n            regression. This may be a Factor, a BoundColumn or a Slice. If\n            `target` is two-dimensional, regressions are computed asset-wise.\n        regression_length : int\n            Length of the lookback window over which to compute each\n            regression.\n        mask : zipline.pipeline.Filter, optional\n            A Filter describing which assets should be regressed with the\n            target slice each day.\n\n        Returns\n        -------\n        regressions : zipline.pipeline.Factor\n            A new Factor that will compute linear regressions of `target`\n            against the columns of `self`.\n\n        Notes\n        -----\n        {CORRELATION_METHOD_NOTE}\n\n        Examples\n        --------\n        Suppose we want to create a factor that regresses AAPL's 10-day returns\n        against the 10-day returns of all other assets, computing each\n        regression over 30 days. This can be achieved by doing the following::\n\n            returns = Returns(window_length=10)\n            returns_slice = returns[sid(24)]\n            aapl_regressions = returns.linear_regression(\n                target=returns_slice, regression_length=30,\n            )\n\n        This is equivalent to doing::\n\n            aapl_regressions = RollingLinearRegressionOfReturns(\n                target=sid(24), returns_length=10, regression_length=30,\n            )\n\n        See Also\n        --------\n        :func:`scipy.stats.linregress`\n        "
        from .statistical import RollingLinearRegression
        return RollingLinearRegression(dependent=self, independent=target, regression_length=regression_length, mask=mask)

    @expect_types(min_percentile=(int, float), max_percentile=(int, float), mask=(Filter, NotSpecifiedType), groupby=(Classifier, NotSpecifiedType))
    @float64_only
    def winsorize(self, min_percentile, max_percentile, mask=NotSpecified, groupby=NotSpecified):
        if False:
            for i in range(10):
                print('nop')
        "\n        Construct a new factor that winsorizes the result of this factor.\n\n        Winsorizing changes values ranked less than the minimum percentile to\n        the value at the minimum percentile. Similarly, values ranking above\n        the maximum percentile are changed to the value at the maximum\n        percentile.\n\n        Winsorizing is useful for limiting the impact of extreme data points\n        without completely removing those points.\n\n        If ``mask`` is supplied, ignore values where ``mask`` returns False\n        when computing percentile cutoffs, and output NaN anywhere the mask is\n        False.\n\n        If ``groupby`` is supplied, winsorization is applied separately\n        separately to each group defined by ``groupby``.\n\n        Parameters\n        ----------\n        min_percentile: float, int\n            Entries with values at or below this percentile will be replaced\n            with the (len(input) * min_percentile)th lowest value. If low\n            values should not be clipped, use 0.\n        max_percentile: float, int\n            Entries with values at or above this percentile will be replaced\n            with the (len(input) * max_percentile)th lowest value. If high\n            values should not be clipped, use 1.\n        mask : zipline.pipeline.Filter, optional\n            A Filter defining values to ignore when winsorizing.\n        groupby : zipline.pipeline.Classifier, optional\n            A classifier defining partitions over which to winsorize.\n\n        Returns\n        -------\n        winsorized : zipline.pipeline.Factor\n            A Factor producing a winsorized version of self.\n\n        Examples\n        --------\n        .. code-block:: python\n\n            price = USEquityPricing.close.latest\n            columns={\n                'PRICE': price,\n                'WINSOR_1: price.winsorize(\n                    min_percentile=0.25, max_percentile=0.75\n                ),\n                'WINSOR_2': price.winsorize(\n                    min_percentile=0.50, max_percentile=1.0\n                ),\n                'WINSOR_3': price.winsorize(\n                    min_percentile=0.0, max_percentile=0.5\n                ),\n\n            }\n\n        Given a pipeline with columns, defined above, the result for a\n        given day could look like:\n\n        ::\n\n                    'PRICE' 'WINSOR_1' 'WINSOR_2' 'WINSOR_3'\n            Asset_1    1        2          4          3\n            Asset_2    2        2          4          3\n            Asset_3    3        3          4          3\n            Asset_4    4        4          4          4\n            Asset_5    5        5          5          4\n            Asset_6    6        5          5          4\n\n        See Also\n        --------\n        :func:`scipy.stats.mstats.winsorize`\n        :meth:`pandas.DataFrame.groupby`\n        "
        if not 0.0 <= min_percentile < max_percentile <= 1.0:
            raise BadPercentileBounds(min_percentile=min_percentile, max_percentile=max_percentile, upper_bound=1.0)
        return GroupedRowTransform(transform=winsorize, transform_args=(min_percentile, max_percentile), factor=self, groupby=groupby, dtype=self.dtype, missing_value=self.missing_value, mask=mask, window_safe=self.window_safe)

    @expect_types(bins=int, mask=(Filter, NotSpecifiedType))
    def quantiles(self, bins, mask=NotSpecified):
        if False:
            print('Hello World!')
        '\n        Construct a Classifier computing quantiles of the output of ``self``.\n\n        Every non-NaN data point the output is labelled with an integer value\n        from 0 to (bins - 1). NaNs are labelled with -1.\n\n        If ``mask`` is supplied, ignore data points in locations for which\n        ``mask`` produces False, and emit a label of -1 at those locations.\n\n        Parameters\n        ----------\n        bins : int\n            Number of bins labels to compute.\n        mask : zipline.pipeline.Filter, optional\n            Mask of values to ignore when computing quantiles.\n\n        Returns\n        -------\n        quantiles : zipline.pipeline.Classifier\n            A classifier producing integer labels ranging from 0 to (bins - 1).\n        '
        if mask is NotSpecified:
            mask = self.mask
        return Quantiles(inputs=(self,), bins=bins, mask=mask)

    @expect_types(mask=(Filter, NotSpecifiedType))
    def quartiles(self, mask=NotSpecified):
        if False:
            i = 10
            return i + 15
        '\n        Construct a Classifier computing quartiles over the output of ``self``.\n\n        Every non-NaN data point the output is labelled with a value of either\n        0, 1, 2, or 3, corresponding to the first, second, third, or fourth\n        quartile over each row.  NaN data points are labelled with -1.\n\n        If ``mask`` is supplied, ignore data points in locations for which\n        ``mask`` produces False, and emit a label of -1 at those locations.\n\n        Parameters\n        ----------\n        mask : zipline.pipeline.Filter, optional\n            Mask of values to ignore when computing quartiles.\n\n        Returns\n        -------\n        quartiles : zipline.pipeline.Classifier\n            A classifier producing integer labels ranging from 0 to 3.\n        '
        return self.quantiles(bins=4, mask=mask)

    @expect_types(mask=(Filter, NotSpecifiedType))
    def quintiles(self, mask=NotSpecified):
        if False:
            return 10
        '\n        Construct a Classifier computing quintile labels on ``self``.\n\n        Every non-NaN data point the output is labelled with a value of either\n        0, 1, 2, or 3, 4, corresonding to quintiles over each row.  NaN data\n        points are labelled with -1.\n\n        If ``mask`` is supplied, ignore data points in locations for which\n        ``mask`` produces False, and emit a label of -1 at those locations.\n\n        Parameters\n        ----------\n        mask : zipline.pipeline.Filter, optional\n            Mask of values to ignore when computing quintiles.\n\n        Returns\n        -------\n        quintiles : zipline.pipeline.Classifier\n            A classifier producing integer labels ranging from 0 to 4.\n        '
        return self.quantiles(bins=5, mask=mask)

    @expect_types(mask=(Filter, NotSpecifiedType))
    def deciles(self, mask=NotSpecified):
        if False:
            print('Hello World!')
        '\n        Construct a Classifier computing decile labels on ``self``.\n\n        Every non-NaN data point the output is labelled with a value from 0 to\n        9 corresonding to deciles over each row.  NaN data points are labelled\n        with -1.\n\n        If ``mask`` is supplied, ignore data points in locations for which\n        ``mask`` produces False, and emit a label of -1 at those locations.\n\n        Parameters\n        ----------\n        mask : zipline.pipeline.Filter, optional\n            Mask of values to ignore when computing deciles.\n\n        Returns\n        -------\n        deciles : zipline.pipeline.Classifier\n            A classifier producing integer labels ranging from 0 to 9.\n        '
        return self.quantiles(bins=10, mask=mask)

    def top(self, N, mask=NotSpecified, groupby=NotSpecified):
        if False:
            while True:
                i = 10
        '\n        Construct a Filter matching the top N asset values of self each day.\n\n        If ``groupby`` is supplied, returns a Filter matching the top N asset\n        values for each group.\n\n        Parameters\n        ----------\n        N : int\n            Number of assets passing the returned filter each day.\n        mask : zipline.pipeline.Filter, optional\n            A Filter representing assets to consider when computing ranks.\n            If mask is supplied, top values are computed ignoring any\n            asset/date pairs for which `mask` produces a value of False.\n        groupby : zipline.pipeline.Classifier, optional\n            A classifier defining partitions over which to perform ranking.\n\n        Returns\n        -------\n        filter : zipline.pipeline.Filter\n        '
        if N == 1:
            return self._maximum(mask=mask, groupby=groupby)
        return self.rank(ascending=False, mask=mask, groupby=groupby) <= N

    def bottom(self, N, mask=NotSpecified, groupby=NotSpecified):
        if False:
            print('Hello World!')
        '\n        Construct a Filter matching the bottom N asset values of self each day.\n\n        If ``groupby`` is supplied, returns a Filter matching the bottom N\n        asset values **for each group** defined by ``groupby``.\n\n        Parameters\n        ----------\n        N : int\n            Number of assets passing the returned filter each day.\n        mask : zipline.pipeline.Filter, optional\n            A Filter representing assets to consider when computing ranks.\n            If mask is supplied, bottom values are computed ignoring any\n            asset/date pairs for which `mask` produces a value of False.\n        groupby : zipline.pipeline.Classifier, optional\n            A classifier defining partitions over which to perform ranking.\n\n        Returns\n        -------\n        filter : zipline.pipeline.Filter\n        '
        return self.rank(ascending=True, mask=mask, groupby=groupby) <= N

    def _maximum(self, mask=NotSpecified, groupby=NotSpecified):
        if False:
            while True:
                i = 10
        return MaximumFilter(self, groupby=groupby, mask=mask)

    def percentile_between(self, min_percentile, max_percentile, mask=NotSpecified):
        if False:
            while True:
                i = 10
        '\n        Construct a Filter matching values of self that fall within the range\n        defined by ``min_percentile`` and ``max_percentile``.\n\n        Parameters\n        ----------\n        min_percentile : float [0.0, 100.0]\n            Return True for assets falling above this percentile in the data.\n        max_percentile : float [0.0, 100.0]\n            Return True for assets falling below this percentile in the data.\n        mask : zipline.pipeline.Filter, optional\n            A Filter representing assets to consider when percentile\n            calculating thresholds.  If mask is supplied, percentile cutoffs\n            are computed each day using only assets for which ``mask`` returns\n            True.  Assets for which ``mask`` produces False will produce False\n            in the output of this Factor as well.\n\n        Returns\n        -------\n        out : zipline.pipeline.Filter\n            A new filter that will compute the specified percentile-range mask.\n        '
        return PercentileFilter(self, min_percentile=min_percentile, max_percentile=max_percentile, mask=mask)

    @if_not_float64_tell_caller_to_use_isnull
    def isnan(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A Filter producing True for all values where this Factor is NaN.\n\n        Returns\n        -------\n        nanfilter : zipline.pipeline.Filter\n        '
        return self != self

    @if_not_float64_tell_caller_to_use_isnull
    def notnan(self):
        if False:
            while True:
                i = 10
        '\n        A Filter producing True for values where this Factor is not NaN.\n\n        Returns\n        -------\n        nanfilter : zipline.pipeline.Filter\n        '
        return ~self.isnan()

    @if_not_float64_tell_caller_to_use_isnull
    def isfinite(self):
        if False:
            print('Hello World!')
        '\n        A Filter producing True for values where this Factor is anything but\n        NaN, inf, or -inf.\n        '
        return (-inf < self) & (self < inf)

    def clip(self, min_bound, max_bound, mask=NotSpecified):
        if False:
            while True:
                i = 10
        '\n        Clip (limit) the values in a factor.\n\n        Given an interval, values outside the interval are clipped to the\n        interval edges. For example, if an interval of ``[0, 1]`` is specified,\n        values smaller than 0 become 0, and values larger than 1 become 1.\n\n        Parameters\n        ----------\n        min_bound : float\n            The minimum value to use.\n        max_bound : float\n            The maximum value to use.\n        mask : zipline.pipeline.Filter, optional\n            A Filter representing assets to consider when clipping.\n\n        Notes\n        -----\n        To only clip values on one side, ``-np.inf` and ``np.inf`` may be\n        passed.  For example, to only clip the maximum value but not clip a\n        minimum value:\n\n        .. code-block:: python\n\n           factor.clip(min_bound=-np.inf, max_bound=user_provided_max)\n\n        See Also\n        --------\n        numpy.clip\n        '
        from .basic import Clip
        return Clip(inputs=[self], min_bound=min_bound, max_bound=max_bound)

    @classmethod
    def _principal_computable_term_type(cls):
        if False:
            return 10
        return Factor

class NumExprFactor(NumericalExpression, Factor):
    """
    Factor computed from a numexpr expression.

    Parameters
    ----------
    expr : string
       A string suitable for passing to numexpr.  All variables in 'expr'
       should be of the form "x_i", where i is the index of the corresponding
       factor input in 'binds'.
    binds : tuple
       A tuple of factors to use as inputs.

    Notes
    -----
    NumExprFactors are constructed by numerical operators like `+` and `-`.
    Users should rarely need to construct a NumExprFactor directly.
    """
    pass

class GroupedRowTransform(Factor):
    """
    A Factor that transforms an input factor by applying a row-wise
    shape-preserving transformation on classifier-defined groups of that
    Factor.

    This is most often useful for normalization operators like ``zscore`` or
    ``demean`` or for performing ranking using ``rank``.

    Parameters
    ----------
    transform : function[ndarray[ndim=1] -> ndarray[ndim=1]]
        Function to apply over each row group.
    factor : zipline.pipeline.Factor
        The factor providing baseline data to transform.
    mask : zipline.pipeline.Filter
        Mask of entries to ignore when calculating transforms.
    groupby : zipline.pipeline.Classifier
        Classifier partitioning ``factor`` into groups to use when calculating
        means.
    transform_args : tuple[hashable]
        Additional positional arguments to forward to ``transform``.

    Notes
    -----
    Users should rarely construct instances of this factor directly.  Instead,
    they should construct instances via factor normalization methods like
    ``zscore`` and ``demean`` or using ``rank`` with ``groupby``.

    See Also
    --------
    zipline.pipeline.Factor.zscore
    zipline.pipeline.Factor.demean
    zipline.pipeline.Factor.rank
    """
    window_length = 0

    def __new__(cls, transform, transform_args, factor, groupby, dtype, missing_value, mask, **kwargs):
        if False:
            print('Hello World!')
        if mask is NotSpecified:
            mask = factor.mask
        else:
            mask = mask & factor.mask
        if groupby is NotSpecified:
            groupby = Everything(mask=mask)
        return super(GroupedRowTransform, cls).__new__(GroupedRowTransform, transform=transform, transform_args=transform_args, inputs=(factor, groupby), missing_value=missing_value, mask=mask, dtype=dtype, **kwargs)

    def _init(self, transform, transform_args, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._transform = transform
        self._transform_args = transform_args
        return super(GroupedRowTransform, self)._init(*args, **kwargs)

    @classmethod
    def _static_identity(cls, transform, transform_args, *args, **kwargs):
        if False:
            while True:
                i = 10
        return (super(GroupedRowTransform, cls)._static_identity(*args, **kwargs), transform, transform_args)

    def _compute(self, arrays, dates, assets, mask):
        if False:
            print('Hello World!')
        data = arrays[0]
        (group_labels, null_label) = self.inputs[1]._to_integral(arrays[1])
        group_labels = where(mask, group_labels, null_label)
        return where(group_labels != null_label, naive_grouped_rowwise_apply(data=data, group_labels=group_labels, func=self._transform, func_args=self._transform_args, out=empty_like(data, dtype=self.dtype)), self.missing_value)

    @property
    def transform_name(self):
        if False:
            print('Hello World!')
        return self._transform.__name__

    def graph_repr(self):
        if False:
            return 10
        'Short repr to use when rendering Pipeline graphs.'
        return type(self).__name__ + '(%r)' % self.transform_name

class Rank(SingleInputMixin, Factor):
    """
    A Factor representing the row-wise rank data of another Factor.

    Parameters
    ----------
    factor : zipline.pipeline.Factor
        The factor on which to compute ranks.
    method : str, {'average', 'min', 'max', 'dense', 'ordinal'}
        The method used to assign ranks to tied elements.  See
        `scipy.stats.rankdata` for a full description of the semantics for each
        ranking method.

    See Also
    --------
    :func:`scipy.stats.rankdata`
    :class:`Factor.rank`

    Notes
    -----
    Most users should call Factor.rank rather than directly construct an
    instance of this class.
    """
    window_length = 0
    dtype = float64_dtype
    window_safe = True

    def __new__(cls, factor, method, ascending, mask):
        if False:
            print('Hello World!')
        return super(Rank, cls).__new__(cls, inputs=(factor,), method=method, ascending=ascending, mask=mask)

    def _init(self, method, ascending, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        self._method = method
        self._ascending = ascending
        return super(Rank, self)._init(*args, **kwargs)

    @classmethod
    def _static_identity(cls, method, ascending, *args, **kwargs):
        if False:
            while True:
                i = 10
        return (super(Rank, cls)._static_identity(*args, **kwargs), method, ascending)

    def _validate(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Verify that the stored rank method is valid.\n        '
        if self._method not in _RANK_METHODS:
            raise UnknownRankMethod(method=self._method, choices=set(_RANK_METHODS))
        return super(Rank, self)._validate()

    def _compute(self, arrays, dates, assets, mask):
        if False:
            print('Hello World!')
        '\n        For each row in the input, compute a like-shaped array of per-row\n        ranks.\n        '
        return masked_rankdata_2d(arrays[0], mask, self.inputs[0].missing_value, self._method, self._ascending)

    def __repr__(self):
        if False:
            while True:
                i = 10
        if self.mask is AssetExists():
            mask_info = ''
        else:
            mask_info = ', mask={}'.format(self.mask.recursive_repr())
        return "{type}({input_}, method='{method}'{mask_info})".format(type=type(self).__name__, input_=self.inputs[0].recursive_repr(), method=self._method, mask_info=mask_info)

    def graph_repr(self):
        if False:
            while True:
                i = 10
        return 'Rank:\\l  method: {!r}\\l  mask: {}\\l'.format(self._method, type(self.mask).__name__)

class CustomFactor(PositiveWindowLengthMixin, CustomTermMixin, Factor):
    '''
    Base class for user-defined Factors.

    Parameters
    ----------
    inputs : iterable, optional
        An iterable of `BoundColumn` instances (e.g. USEquityPricing.close),
        describing the data to load and pass to `self.compute`.  If this
        argument is not passed to the CustomFactor constructor, we look for a
        class-level attribute named `inputs`.
    outputs : iterable[str], optional
        An iterable of strings which represent the names of each output this
        factor should compute and return. If this argument is not passed to the
        CustomFactor constructor, we look for a class-level attribute named
        `outputs`.
    window_length : int, optional
        Number of rows to pass for each input.  If this argument is not passed
        to the CustomFactor constructor, we look for a class-level attribute
        named `window_length`.
    mask : zipline.pipeline.Filter, optional
        A Filter describing the assets on which we should compute each day.
        Each call to ``CustomFactor.compute`` will only receive assets for
        which ``mask`` produced True on the day for which compute is being
        called.

    Notes
    -----
    Users implementing their own Factors should subclass CustomFactor and
    implement a method named `compute` with the following signature:

    .. code-block:: python

        def compute(self, today, assets, out, *inputs):
           ...

    On each simulation date, ``compute`` will be called with the current date,
    an array of sids, an output array, and an input array for each expression
    passed as inputs to the CustomFactor constructor.

    The specific types of the values passed to `compute` are as follows::

        today : np.datetime64[ns]
            Row label for the last row of all arrays passed as `inputs`.
        assets : np.array[int64, ndim=1]
            Column labels for `out` and`inputs`.
        out : np.array[self.dtype, ndim=1]
            Output array of the same shape as `assets`.  `compute` should write
            its desired return values into `out`. If multiple outputs are
            specified, `compute` should write its desired return values into
            `out.<output_name>` for each output name in `self.outputs`.
        *inputs : tuple of np.array
            Raw data arrays corresponding to the values of `self.inputs`.

    ``compute`` functions should expect to be passed NaN values for dates on
    which no data was available for an asset.  This may include dates on which
    an asset did not yet exist.

    For example, if a CustomFactor requires 10 rows of close price data, and
    asset A started trading on Monday June 2nd, 2014, then on Tuesday, June
    3rd, 2014, the column of input data for asset A will have 9 leading NaNs
    for the preceding days on which data was not yet available.

    Examples
    --------

    A CustomFactor with pre-declared defaults:

    .. code-block:: python

        class TenDayRange(CustomFactor):
            """
            Computes the difference between the highest high in the last 10
            days and the lowest low.

            Pre-declares high and low as default inputs and `window_length` as
            10.
            """

            inputs = [USEquityPricing.high, USEquityPricing.low]
            window_length = 10

            def compute(self, today, assets, out, highs, lows):
                from numpy import nanmin, nanmax

                highest_highs = nanmax(highs, axis=0)
                lowest_lows = nanmin(lows, axis=0)
                out[:] = highest_highs - lowest_lows


        # Doesn't require passing inputs or window_length because they're
        # pre-declared as defaults for the TenDayRange class.
        ten_day_range = TenDayRange()

    A CustomFactor without defaults:

    .. code-block:: python

        class MedianValue(CustomFactor):
            """
            Computes the median value of an arbitrary single input over an
            arbitrary window..

            Does not declare any defaults, so values for `window_length` and
            `inputs` must be passed explicitly on every construction.
            """

            def compute(self, today, assets, out, data):
                from numpy import nanmedian
                out[:] = data.nanmedian(data, axis=0)

        # Values for `inputs` and `window_length` must be passed explicitly to
        # MedianValue.
        median_close10 = MedianValue([USEquityPricing.close], window_length=10)
        median_low15 = MedianValue([USEquityPricing.low], window_length=15)

    A CustomFactor with multiple outputs:

    .. code-block:: python

        class MultipleOutputs(CustomFactor):
            inputs = [USEquityPricing.close]
            outputs = ['alpha', 'beta']
            window_length = N

            def compute(self, today, assets, out, close):
                computed_alpha, computed_beta = some_function(close)
                out.alpha[:] = computed_alpha
                out.beta[:] = computed_beta

        # Each output is returned as its own Factor upon instantiation.
        alpha, beta = MultipleOutputs()

        # Equivalently, we can create a single factor instance and access each
        # output as an attribute of that instance.
        multiple_outputs = MultipleOutputs()
        alpha = multiple_outputs.alpha
        beta = multiple_outputs.beta

    Note: If a CustomFactor has multiple outputs, all outputs must have the
    same dtype. For instance, in the example above, if alpha is a float then
    beta must also be a float.
    '''
    dtype = float64_dtype

    def _validate(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            super(CustomFactor, self)._validate()
        except UnsupportedDataType:
            if self.dtype in CLASSIFIER_DTYPES:
                raise UnsupportedDataType(typename=type(self).__name__, dtype=self.dtype, hint='Did you mean to create a CustomClassifier?')
            elif self.dtype in FILTER_DTYPES:
                raise UnsupportedDataType(typename=type(self).__name__, dtype=self.dtype, hint='Did you mean to create a CustomFilter?')
            raise

    def __getattribute__(self, name):
        if False:
            i = 10
            return i + 15
        outputs = object.__getattribute__(self, 'outputs')
        if outputs is NotSpecified:
            return super(CustomFactor, self).__getattribute__(name)
        elif name in outputs:
            return RecarrayField(factor=self, attribute=name)
        else:
            try:
                return super(CustomFactor, self).__getattribute__(name)
            except AttributeError:
                raise AttributeError('Instance of {factor} has no output named {attr!r}. Possible choices are: {choices}.'.format(factor=type(self).__name__, attr=name, choices=self.outputs))

    def __iter__(self):
        if False:
            return 10
        if self.outputs is NotSpecified:
            raise ValueError('{factor} does not have multiple outputs.'.format(factor=type(self).__name__))
        return (RecarrayField(self, attr) for attr in self.outputs)

class RecarrayField(SingleInputMixin, Factor):
    """
    A single field from a multi-output factor.
    """

    def __new__(cls, factor, attribute):
        if False:
            print('Hello World!')
        return super(RecarrayField, cls).__new__(cls, attribute=attribute, inputs=[factor], window_length=0, mask=factor.mask, dtype=factor.dtype, missing_value=factor.missing_value, window_safe=factor.window_safe)

    def _init(self, attribute, *args, **kwargs):
        if False:
            while True:
                i = 10
        self._attribute = attribute
        return super(RecarrayField, self)._init(*args, **kwargs)

    @classmethod
    def _static_identity(cls, attribute, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return (super(RecarrayField, cls)._static_identity(*args, **kwargs), attribute)

    def _compute(self, windows, dates, assets, mask):
        if False:
            return 10
        return windows[0][self._attribute]

    def graph_repr(self):
        if False:
            return 10
        return '{}.{}'.format(self.inputs[0].recursive_repr(), self._attribute)

class Latest(LatestMixin, CustomFactor):
    """
    Factor producing the most recently-known value of `inputs[0]` on each day.

    The `.latest` attribute of DataSet columns returns an instance of this
    Factor.
    """
    window_length = 1

    def compute(self, today, assets, out, data):
        if False:
            i = 10
            return i + 15
        out[:] = data[-1]

class DailySummary(SingleInputMixin, Factor):
    """1D Factor that computes a summary statistic across all assets.
    """
    ndim = 1
    window_length = 0
    params = ('func',)

    def __new__(cls, func, input_, mask, dtype):
        if False:
            i = 10
            return i + 15
        if dtype != float64_dtype:
            raise AssertionError('DailySummary only supports float64 dtype, got {}'.format(dtype))
        return super(DailySummary, cls).__new__(cls, inputs=[input_], dtype=dtype, missing_value=nan, window_safe=input_.window_safe, func=func, mask=mask)

    def _compute(self, arrays, dates, assets, mask):
        if False:
            while True:
                i = 10
        func = self.params['func']
        data = arrays[0]
        data[~mask] = nan
        if not isnan(self.inputs[0].missing_value):
            data[data == self.inputs[0].missing_value] = nan
        return as_column(func(data, self.inputs[0].missing_value))

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '{}.{}()'.format(self.inputs[0].recursive_repr(), self.params['func'].__name__)
    graph_repr = recursive_repr = __repr__

def demean(row):
    if False:
        i = 10
        return i + 15
    return row - nanmean(row)

def zscore(row):
    if False:
        return 10
    return (row - nanmean(row)) / nanstd(row)

def winsorize(row, min_percentile, max_percentile):
    if False:
        return 10
    '\n    This implementation is based on scipy.stats.mstats.winsorize\n    '
    a = row.copy()
    nan_count = isnan(row).sum()
    nonnan_count = a.size - nan_count
    idx = a.argsort()
    if min_percentile > 0:
        lower_cutoff = int(min_percentile * nonnan_count)
        a[idx[:lower_cutoff]] = a[idx[lower_cutoff]]
    if max_percentile < 1:
        upper_cutoff = int(ceil(nonnan_count * max_percentile))
        if upper_cutoff < nonnan_count:
            start_of_nans = -nan_count if nan_count else None
            a[idx[upper_cutoff:start_of_nans]] = a[idx[upper_cutoff - 1]]
    return a