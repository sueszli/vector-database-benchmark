"""
Methods for analyzing two-way contingency tables (i.e. frequency
tables for observations that are cross-classified with respect to two
categorical variables).

The main classes are:

  * Table : implements methods that can be applied to any two-way
  contingency table.

  * SquareTable : implements methods that can be applied to a square
  two-way contingency table.

  * Table2x2 : implements methods that can be applied to a 2x2
  contingency table.

  * StratifiedTable : implements methods that can be applied to a
  collection of 2x2 contingency tables.

Also contains functions for conducting McNemar's test and Cochran's q
test.

Note that the inference procedures may depend on how the data were
sampled.  In general the observed units are independent and
identically distributed.
"""
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels import iolib
from statsmodels.tools import sm_exceptions
from statsmodels.tools.decorators import cache_readonly

def _make_df_square(table):
    if False:
        for i in range(10):
            print('nop')
    '\n    Reindex a pandas DataFrame so that it becomes square, meaning that\n    the row and column indices contain the same values, in the same\n    order.  The row and column index are extended to achieve this.\n    '
    if not isinstance(table, pd.DataFrame):
        return table
    if not table.index.equals(table.columns):
        ix = list(set(table.index) | set(table.columns))
        ix.sort()
        table = table.reindex(index=ix, columns=ix, fill_value=0)
    table = table.reindex(table.columns)
    return table

class _Bunch:

    def __repr__(self):
        if False:
            return 10
        return '<bunch containing results, print to see contents>'

    def __str__(self):
        if False:
            while True:
                i = 10
        ky = [k for (k, _) in self.__dict__.items()]
        ky.sort()
        m = max([len(k) for k in ky])
        tab = []
        f = '{:' + str(m) + '}   {}'
        for k in ky:
            tab.append(f.format(k, self.__dict__[k]))
        return '\n'.join(tab)

class Table:
    """
    A two-way contingency table.

    Parameters
    ----------
    table : array_like
        A contingency table.
    shift_zeros : bool
        If True and any cell count is zero, add 0.5 to all values
        in the table.

    Attributes
    ----------
    table_orig : array_like
        The original table is cached as `table_orig`.

    See Also
    --------
    statsmodels.graphics.mosaicplot.mosaic
    scipy.stats.chi2_contingency

    Notes
    -----
    The inference procedures used here are all based on a sampling
    model in which the units are independent and identically
    distributed, with each unit being classified with respect to two
    categorical variables.

    References
    ----------
    Definitions of residuals:
        https://onlinecourses.science.psu.edu/stat504/node/86
    """

    def __init__(self, table, shift_zeros=True):
        if False:
            while True:
                i = 10
        self.table_orig = table
        self.table = np.asarray(table, dtype=np.float64)
        if shift_zeros and self.table.min() == 0:
            self.table[self.table == 0] = 0.5

    def __str__(self):
        if False:
            return 10
        s = 'A %dx%d contingency table with counts:\n' % tuple(self.table.shape)
        s += np.array_str(self.table)
        return s

    @classmethod
    def from_data(cls, data, shift_zeros=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a Table object from data.\n\n        Parameters\n        ----------\n        data : array_like\n            The raw data, from which a contingency table is constructed\n            using the first two columns.\n        shift_zeros : bool\n            If True and any cell count is zero, add 0.5 to all values\n            in the table.\n\n        Returns\n        -------\n        A Table instance.\n        '
        if isinstance(data, pd.DataFrame):
            table = pd.crosstab(data.iloc[:, 0], data.iloc[:, 1])
        else:
            table = pd.crosstab(data[:, 0], data[:, 1])
        return cls(table, shift_zeros)

    def test_nominal_association(self):
        if False:
            print('Hello World!')
        '\n        Assess independence for nominal factors.\n\n        Assessment of independence between rows and columns using\n        chi^2 testing.  The rows and columns are treated as nominal\n        (unordered) categorical variables.\n\n        Returns\n        -------\n        A bunch containing the following attributes:\n\n        statistic : float\n            The chi^2 test statistic.\n        df : int\n            The degrees of freedom of the reference distribution\n        pvalue : float\n            The p-value for the test.\n        '
        statistic = np.asarray(self.chi2_contribs).sum()
        df = np.prod(np.asarray(self.table.shape) - 1)
        pvalue = 1 - stats.chi2.cdf(statistic, df)
        b = _Bunch()
        b.statistic = statistic
        b.df = df
        b.pvalue = pvalue
        return b

    def test_ordinal_association(self, row_scores=None, col_scores=None):
        if False:
            return 10
        "\n        Assess independence between two ordinal variables.\n\n        This is the 'linear by linear' association test, which uses\n        weights or scores to target the test to have more power\n        against ordered alternatives.\n\n        Parameters\n        ----------\n        row_scores : array_like\n            An array of numeric row scores\n        col_scores : array_like\n            An array of numeric column scores\n\n        Returns\n        -------\n        A bunch with the following attributes:\n\n        statistic : float\n            The test statistic.\n        null_mean : float\n            The expected value of the test statistic under the null\n            hypothesis.\n        null_sd : float\n            The standard deviation of the test statistic under the\n            null hypothesis.\n        zscore : float\n            The Z-score for the test statistic.\n        pvalue : float\n            The p-value for the test.\n\n        Notes\n        -----\n        The scores define the trend to which the test is most sensitive.\n\n        Using the default row and column scores gives the\n        Cochran-Armitage trend test.\n        "
        if row_scores is None:
            row_scores = np.arange(self.table.shape[0])
        if col_scores is None:
            col_scores = np.arange(self.table.shape[1])
        if len(row_scores) != self.table.shape[0]:
            msg = 'The length of `row_scores` must match the first ' + 'dimension of `table`.'
            raise ValueError(msg)
        if len(col_scores) != self.table.shape[1]:
            msg = 'The length of `col_scores` must match the second ' + 'dimension of `table`.'
            raise ValueError(msg)
        statistic = np.dot(row_scores, np.dot(self.table, col_scores))
        n_obs = self.table.sum()
        rtot = self.table.sum(1)
        um = np.dot(row_scores, rtot)
        u2m = np.dot(row_scores ** 2, rtot)
        ctot = self.table.sum(0)
        vn = np.dot(col_scores, ctot)
        v2n = np.dot(col_scores ** 2, ctot)
        e_stat = um * vn / n_obs
        v_stat = (u2m - um ** 2 / n_obs) * (v2n - vn ** 2 / n_obs) / (n_obs - 1)
        sd_stat = np.sqrt(v_stat)
        zscore = (statistic - e_stat) / sd_stat
        pvalue = 2 * stats.norm.cdf(-np.abs(zscore))
        b = _Bunch()
        b.statistic = statistic
        b.null_mean = e_stat
        b.null_sd = sd_stat
        b.zscore = zscore
        b.pvalue = pvalue
        return b

    @cache_readonly
    def marginal_probabilities(self):
        if False:
            i = 10
            return i + 15
        '\n        Estimate marginal probability distributions for the rows and columns.\n\n        Returns\n        -------\n        row : ndarray\n            Marginal row probabilities\n        col : ndarray\n            Marginal column probabilities\n        '
        n = self.table.sum()
        row = self.table.sum(1) / n
        col = self.table.sum(0) / n
        if isinstance(self.table_orig, pd.DataFrame):
            row = pd.Series(row, self.table_orig.index)
            col = pd.Series(col, self.table_orig.columns)
        return (row, col)

    @cache_readonly
    def independence_probabilities(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns fitted joint probabilities under independence.\n\n        The returned table is outer(row, column), where row and\n        column are the estimated marginal distributions\n        of the rows and columns.\n        '
        (row, col) = self.marginal_probabilities
        itab = np.outer(row, col)
        if isinstance(self.table_orig, pd.DataFrame):
            itab = pd.DataFrame(itab, self.table_orig.index, self.table_orig.columns)
        return itab

    @cache_readonly
    def fittedvalues(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns fitted cell counts under independence.\n\n        The returned cell counts are estimates under a model\n        where the rows and columns of the table are independent.\n        '
        probs = self.independence_probabilities
        fit = self.table.sum() * probs
        return fit

    @cache_readonly
    def resid_pearson(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns Pearson residuals.\n\n        The Pearson residuals are calculated under a model where\n        the rows and columns of the table are independent.\n        '
        fit = self.fittedvalues
        resids = (self.table - fit) / np.sqrt(fit)
        return resids

    @cache_readonly
    def standardized_resids(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns standardized residuals under independence.\n        '
        (row, col) = self.marginal_probabilities
        sresids = self.resid_pearson / np.sqrt(np.outer(1 - row, 1 - col))
        return sresids

    @cache_readonly
    def chi2_contribs(self):
        if False:
            return 10
        '\n        Returns the contributions to the chi^2 statistic for independence.\n\n        The returned table contains the contribution of each cell to the chi^2\n        test statistic for the null hypothesis that the rows and columns\n        are independent.\n        '
        return self.resid_pearson ** 2

    @cache_readonly
    def local_log_oddsratios(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns local log odds ratios.\n\n        The local log odds ratios are the log odds ratios\n        calculated for contiguous 2x2 sub-tables.\n        '
        ta = self.table.copy()
        a = ta[0:-1, 0:-1]
        b = ta[0:-1, 1:]
        c = ta[1:, 0:-1]
        d = ta[1:, 1:]
        tab = np.log(a) + np.log(d) - np.log(b) - np.log(c)
        rslt = np.empty(self.table.shape, np.float64)
        rslt *= np.nan
        rslt[0:-1, 0:-1] = tab
        if isinstance(self.table_orig, pd.DataFrame):
            rslt = pd.DataFrame(rslt, index=self.table_orig.index, columns=self.table_orig.columns)
        return rslt

    @cache_readonly
    def local_oddsratios(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns local odds ratios.\n\n        See documentation for local_log_oddsratios.\n        '
        return np.exp(self.local_log_oddsratios)

    @cache_readonly
    def cumulative_log_oddsratios(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns cumulative log odds ratios.\n\n        The cumulative log odds ratios for a contingency table\n        with ordered rows and columns are calculated by collapsing\n        all cells to the left/right and above/below a given point,\n        to obtain a 2x2 table from which a log odds ratio can be\n        calculated.\n        '
        ta = self.table.cumsum(0).cumsum(1)
        a = ta[0:-1, 0:-1]
        b = ta[0:-1, -1:] - a
        c = ta[-1:, 0:-1] - a
        d = ta[-1, -1] - (a + b + c)
        tab = np.log(a) + np.log(d) - np.log(b) - np.log(c)
        rslt = np.empty(self.table.shape, np.float64)
        rslt *= np.nan
        rslt[0:-1, 0:-1] = tab
        if isinstance(self.table_orig, pd.DataFrame):
            rslt = pd.DataFrame(rslt, index=self.table_orig.index, columns=self.table_orig.columns)
        return rslt

    @cache_readonly
    def cumulative_oddsratios(self):
        if False:
            while True:
                i = 10
        '\n        Returns the cumulative odds ratios for a contingency table.\n\n        See documentation for cumulative_log_oddsratio.\n        '
        return np.exp(self.cumulative_log_oddsratios)

class SquareTable(Table):
    """
    Methods for analyzing a square contingency table.

    Parameters
    ----------
    table : array_like
        A square contingency table, or DataFrame that is converted
        to a square form.
    shift_zeros : bool
        If True and any cell count is zero, add 0.5 to all values
        in the table.

    Notes
    -----
    These methods should only be used when the rows and columns of the
    table have the same categories.  If `table` is provided as a
    Pandas DataFrame, the row and column indices will be extended to
    create a square table, inserting zeros where a row or column is
    missing.  Otherwise the table should be provided in a square form,
    with the (implicit) row and column categories appearing in the
    same order.
    """

    def __init__(self, table, shift_zeros=True):
        if False:
            print('Hello World!')
        table = _make_df_square(table)
        (k1, k2) = table.shape
        if k1 != k2:
            raise ValueError('table must be square')
        super(SquareTable, self).__init__(table, shift_zeros)

    def symmetry(self, method='bowker'):
        if False:
            print('Hello World!')
        '\n        Test for symmetry of a joint distribution.\n\n        This procedure tests the null hypothesis that the joint\n        distribution is symmetric around the main diagonal, that is\n\n        .. math::\n\n        p_{i, j} = p_{j, i}  for all i, j\n\n        Returns\n        -------\n        Bunch\n            A bunch with attributes\n\n            * statistic : float\n                chisquare test statistic\n            * p-value : float\n                p-value of the test statistic based on chisquare distribution\n            * df : int\n                degrees of freedom of the chisquare distribution\n\n        Notes\n        -----\n        The implementation is based on the SAS documentation. R includes\n        it in `mcnemar.test` if the table is not 2 by 2.  However a more\n        direct generalization of the McNemar test to larger tables is\n        provided by the homogeneity test (TableSymmetry.homogeneity).\n\n        The p-value is based on the chi-square distribution which requires\n        that the sample size is not very small to be a good approximation\n        of the true distribution. For 2x2 contingency tables the exact\n        distribution can be obtained with `mcnemar`\n\n        See Also\n        --------\n        mcnemar\n        homogeneity\n        '
        if method.lower() != 'bowker':
            raise ValueError("method for symmetry testing must be 'bowker'")
        k = self.table.shape[0]
        upp_idx = np.triu_indices(k, 1)
        tril = self.table.T[upp_idx]
        triu = self.table[upp_idx]
        statistic = ((tril - triu) ** 2 / (tril + triu + 1e-20)).sum()
        df = k * (k - 1) / 2.0
        pvalue = stats.chi2.sf(statistic, df)
        b = _Bunch()
        b.statistic = statistic
        b.pvalue = pvalue
        b.df = df
        return b

    def homogeneity(self, method='stuart_maxwell'):
        if False:
            for i in range(10):
                print('nop')
        "\n        Compare row and column marginal distributions.\n\n        Parameters\n        ----------\n        method : str\n            Either 'stuart_maxwell' or 'bhapkar', leading to two different\n            estimates of the covariance matrix for the estimated\n            difference between the row margins and the column margins.\n\n        Returns\n        -------\n        Bunch\n            A bunch with attributes:\n\n            * statistic : float\n                The chi^2 test statistic\n            * pvalue : float\n                The p-value of the test statistic\n            * df : int\n                The degrees of freedom of the reference distribution\n\n        Notes\n        -----\n        For a 2x2 table this is equivalent to McNemar's test.  More\n        generally the procedure tests the null hypothesis that the\n        marginal distribution of the row factor is equal to the\n        marginal distribution of the column factor.  For this to be\n        meaningful, the two factors must have the same sample space\n        (i.e. the same categories).\n        "
        if self.table.shape[0] < 1:
            raise ValueError('table is empty')
        elif self.table.shape[0] == 1:
            b = _Bunch()
            b.statistic = 0
            b.pvalue = 1
            b.df = 0
            return b
        method = method.lower()
        if method not in ['bhapkar', 'stuart_maxwell']:
            raise ValueError("method '%s' for homogeneity not known" % method)
        n_obs = self.table.sum()
        pr = self.table.astype(np.float64) / n_obs
        row = pr.sum(1)[0:-1]
        col = pr.sum(0)[0:-1]
        pr = pr[0:-1, 0:-1]
        d = col - row
        df = pr.shape[0]
        if method == 'bhapkar':
            vmat = -(pr + pr.T) - np.outer(d, d)
            dv = col + row - 2 * np.diag(pr) - d ** 2
            np.fill_diagonal(vmat, dv)
        elif method == 'stuart_maxwell':
            vmat = -(pr + pr.T)
            dv = row + col - 2 * np.diag(pr)
            np.fill_diagonal(vmat, dv)
        try:
            statistic = n_obs * np.dot(d, np.linalg.solve(vmat, d))
        except np.linalg.LinAlgError:
            warnings.warn('Unable to invert covariance matrix', sm_exceptions.SingularMatrixWarning)
            b = _Bunch()
            b.statistic = np.nan
            b.pvalue = np.nan
            b.df = df
            return b
        pvalue = 1 - stats.chi2.cdf(statistic, df)
        b = _Bunch()
        b.statistic = statistic
        b.pvalue = pvalue
        b.df = df
        return b

    def summary(self, alpha=0.05, float_format='%.3f'):
        if False:
            for i in range(10):
                print('nop')
        "\n        Produce a summary of the analysis.\n\n        Parameters\n        ----------\n        alpha : float\n            `1 - alpha` is the nominal coverage probability of the interval.\n        float_format : str\n            Used to format numeric values in the table.\n        method : str\n            The method for producing the confidence interval.  Currently\n            must be 'normal' which uses the normal approximation.\n        "
        fmt = float_format
        headers = ['Statistic', 'P-value', 'DF']
        stubs = ['Symmetry', 'Homogeneity']
        sy = self.symmetry()
        hm = self.homogeneity()
        data = [[fmt % sy.statistic, fmt % sy.pvalue, '%d' % sy.df], [fmt % hm.statistic, fmt % hm.pvalue, '%d' % hm.df]]
        tab = iolib.SimpleTable(data, headers, stubs, data_aligns='r', table_dec_above='')
        return tab

class Table2x2(SquareTable):
    """
    Analyses that can be performed on a 2x2 contingency table.

    Parameters
    ----------
    table : array_like
        A 2x2 contingency table
    shift_zeros : bool
        If true, 0.5 is added to all cells of the table if any cell is
        equal to zero.

    Notes
    -----
    The inference procedures used here are all based on a sampling
    model in which the units are independent and identically
    distributed, with each unit being classified with respect to two
    categorical variables.

    Note that for the risk ratio, the analysis is not symmetric with
    respect to the rows and columns of the contingency table.  The two
    rows define population subgroups, column 0 is the number of
    'events', and column 1 is the number of 'non-events'.
    """

    def __init__(self, table, shift_zeros=True):
        if False:
            i = 10
            return i + 15
        if type(table) is list:
            table = np.asarray(table)
        if table.ndim != 2 or table.shape[0] != 2 or table.shape[1] != 2:
            raise ValueError('Table2x2 takes a 2x2 table as input.')
        super(Table2x2, self).__init__(table, shift_zeros)

    @classmethod
    def from_data(cls, data, shift_zeros=True):
        if False:
            i = 10
            return i + 15
        '\n        Construct a Table object from data.\n\n        Parameters\n        ----------\n        data : array_like\n            The raw data, the first column defines the rows and the\n            second column defines the columns.\n        shift_zeros : bool\n            If True, and if there are any zeros in the contingency\n            table, add 0.5 to all four cells of the table.\n        '
        if isinstance(data, pd.DataFrame):
            table = pd.crosstab(data.iloc[:, 0], data.iloc[:, 1])
        else:
            table = pd.crosstab(data[:, 0], data[:, 1])
        return cls(table, shift_zeros)

    @cache_readonly
    def log_oddsratio(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the log odds ratio for a 2x2 table.\n        '
        f = self.table.flatten()
        return np.dot(np.log(f), np.r_[1, -1, -1, 1])

    @cache_readonly
    def oddsratio(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the odds ratio for a 2x2 table.\n        '
        return self.table[0, 0] * self.table[1, 1] / (self.table[0, 1] * self.table[1, 0])

    @cache_readonly
    def log_oddsratio_se(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the standard error for the log odds ratio.\n        '
        return np.sqrt(np.sum(1 / self.table))

    def oddsratio_pvalue(self, null=1):
        if False:
            return 10
        '\n        P-value for a hypothesis test about the odds ratio.\n\n        Parameters\n        ----------\n        null : float\n            The null value of the odds ratio.\n        '
        return self.log_oddsratio_pvalue(np.log(null))

    def log_oddsratio_pvalue(self, null=0):
        if False:
            i = 10
            return i + 15
        '\n        P-value for a hypothesis test about the log odds ratio.\n\n        Parameters\n        ----------\n        null : float\n            The null value of the log odds ratio.\n        '
        zscore = (self.log_oddsratio - null) / self.log_oddsratio_se
        pvalue = 2 * stats.norm.cdf(-np.abs(zscore))
        return pvalue

    def log_oddsratio_confint(self, alpha=0.05, method='normal'):
        if False:
            print('Hello World!')
        "\n        A confidence level for the log odds ratio.\n\n        Parameters\n        ----------\n        alpha : float\n            `1 - alpha` is the nominal coverage probability of the\n            confidence interval.\n        method : str\n            The method for producing the confidence interval.  Currently\n            must be 'normal' which uses the normal approximation.\n        "
        f = -stats.norm.ppf(alpha / 2)
        lor = self.log_oddsratio
        se = self.log_oddsratio_se
        lcb = lor - f * se
        ucb = lor + f * se
        return (lcb, ucb)

    def oddsratio_confint(self, alpha=0.05, method='normal'):
        if False:
            for i in range(10):
                print('nop')
        "\n        A confidence interval for the odds ratio.\n\n        Parameters\n        ----------\n        alpha : float\n            `1 - alpha` is the nominal coverage probability of the\n            confidence interval.\n        method : str\n            The method for producing the confidence interval.  Currently\n            must be 'normal' which uses the normal approximation.\n        "
        (lcb, ucb) = self.log_oddsratio_confint(alpha, method=method)
        return (np.exp(lcb), np.exp(ucb))

    @cache_readonly
    def riskratio(self):
        if False:
            return 10
        '\n        Returns the risk ratio for a 2x2 table.\n\n        The risk ratio is calculated with respect to the rows.\n        '
        p = self.table[:, 0] / self.table.sum(1)
        return p[0] / p[1]

    @cache_readonly
    def log_riskratio(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the log of the risk ratio.\n        '
        return np.log(self.riskratio)

    @cache_readonly
    def log_riskratio_se(self):
        if False:
            return 10
        '\n        Returns the standard error of the log of the risk ratio.\n        '
        n = self.table.sum(1)
        p = self.table[:, 0] / n
        va = np.sum((1 - p) / (n * p))
        return np.sqrt(va)

    def riskratio_pvalue(self, null=1):
        if False:
            while True:
                i = 10
        '\n        p-value for a hypothesis test about the risk ratio.\n\n        Parameters\n        ----------\n        null : float\n            The null value of the risk ratio.\n        '
        return self.log_riskratio_pvalue(np.log(null))

    def log_riskratio_pvalue(self, null=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        p-value for a hypothesis test about the log risk ratio.\n\n        Parameters\n        ----------\n        null : float\n            The null value of the log risk ratio.\n        '
        zscore = (self.log_riskratio - null) / self.log_riskratio_se
        pvalue = 2 * stats.norm.cdf(-np.abs(zscore))
        return pvalue

    def log_riskratio_confint(self, alpha=0.05, method='normal'):
        if False:
            while True:
                i = 10
        "\n        A confidence interval for the log risk ratio.\n\n        Parameters\n        ----------\n        alpha : float\n            `1 - alpha` is the nominal coverage probability of the\n            confidence interval.\n        method : str\n            The method for producing the confidence interval.  Currently\n            must be 'normal' which uses the normal approximation.\n        "
        f = -stats.norm.ppf(alpha / 2)
        lrr = self.log_riskratio
        se = self.log_riskratio_se
        lcb = lrr - f * se
        ucb = lrr + f * se
        return (lcb, ucb)

    def riskratio_confint(self, alpha=0.05, method='normal'):
        if False:
            return 10
        "\n        A confidence interval for the risk ratio.\n\n        Parameters\n        ----------\n        alpha : float\n            `1 - alpha` is the nominal coverage probability of the\n            confidence interval.\n        method : str\n            The method for producing the confidence interval.  Currently\n            must be 'normal' which uses the normal approximation.\n        "
        (lcb, ucb) = self.log_riskratio_confint(alpha, method=method)
        return (np.exp(lcb), np.exp(ucb))

    def summary(self, alpha=0.05, float_format='%.3f', method='normal'):
        if False:
            i = 10
            return i + 15
        "\n        Summarizes results for a 2x2 table analysis.\n\n        Parameters\n        ----------\n        alpha : float\n            `1 - alpha` is the nominal coverage probability of the confidence\n            intervals.\n        float_format : str\n            Used to format the numeric values in the table.\n        method : str\n            The method for producing the confidence interval.  Currently\n            must be 'normal' which uses the normal approximation.\n        "

        def fmt(x):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(x, str):
                return x
            return float_format % x
        headers = ['Estimate', 'SE', 'LCB', 'UCB', 'p-value']
        stubs = ['Odds ratio', 'Log odds ratio', 'Risk ratio', 'Log risk ratio']
        (lcb1, ucb1) = self.oddsratio_confint(alpha, method)
        (lcb2, ucb2) = self.log_oddsratio_confint(alpha, method)
        (lcb3, ucb3) = self.riskratio_confint(alpha, method)
        (lcb4, ucb4) = self.log_riskratio_confint(alpha, method)
        data = [[fmt(x) for x in [self.oddsratio, '', lcb1, ucb1, self.oddsratio_pvalue()]], [fmt(x) for x in [self.log_oddsratio, self.log_oddsratio_se, lcb2, ucb2, self.oddsratio_pvalue()]], [fmt(x) for x in [self.riskratio, '', lcb3, ucb3, self.riskratio_pvalue()]], [fmt(x) for x in [self.log_riskratio, self.log_riskratio_se, lcb4, ucb4, self.riskratio_pvalue()]]]
        tab = iolib.SimpleTable(data, headers, stubs, data_aligns='r', table_dec_above='')
        return tab

class StratifiedTable:
    """
    Analyses for a collection of 2x2 contingency tables.

    Such a collection may arise by stratifying a single 2x2 table with
    respect to another factor.  This class implements the
    'Cochran-Mantel-Haenszel' and 'Breslow-Day' procedures for
    analyzing collections of 2x2 contingency tables.

    Parameters
    ----------
    tables : list or ndarray
        Either a list containing several 2x2 contingency tables, or
        a 2x2xk ndarray in which each slice along the third axis is a
        2x2 contingency table.

    Notes
    -----
    This results are based on a sampling model in which the units are
    independent both within and between strata.
    """

    def __init__(self, tables, shift_zeros=False):
        if False:
            i = 10
            return i + 15
        if isinstance(tables, np.ndarray):
            sp = tables.shape
            if len(sp) != 3 or sp[0] != 2 or sp[1] != 2:
                raise ValueError('If an ndarray, argument must be 2x2xn')
            table = tables * 1.0
        else:
            if any([np.asarray(x).shape != (2, 2) for x in tables]):
                m = 'If `tables` is a list, all of its elements should be 2x2'
                raise ValueError(m)
            table = np.dstack(tables).astype(np.float64)
        if shift_zeros:
            zx = (table == 0).sum(0).sum(0)
            ix = np.flatnonzero(zx > 0)
            if len(ix) > 0:
                table = table.copy()
                table[:, :, ix] += 0.5
        self.table = table
        self._cache = {}
        self._apb = table[0, 0, :] + table[0, 1, :]
        self._apc = table[0, 0, :] + table[1, 0, :]
        self._bpd = table[0, 1, :] + table[1, 1, :]
        self._cpd = table[1, 0, :] + table[1, 1, :]
        self._ad = table[0, 0, :] * table[1, 1, :]
        self._bc = table[0, 1, :] * table[1, 0, :]
        self._apd = table[0, 0, :] + table[1, 1, :]
        self._dma = table[1, 1, :] - table[0, 0, :]
        self._n = table.sum(0).sum(0)

    @classmethod
    def from_data(cls, var1, var2, strata, data):
        if False:
            i = 10
            return i + 15
        '\n        Construct a StratifiedTable object from data.\n\n        Parameters\n        ----------\n        var1 : int or string\n            The column index or name of `data` specifying the variable\n            defining the rows of the contingency table.  The variable\n            must have only two distinct values.\n        var2 : int or string\n            The column index or name of `data` specifying the variable\n            defining the columns of the contingency table.  The variable\n            must have only two distinct values.\n        strata : int or string\n            The column index or name of `data` specifying the variable\n            defining the strata.\n        data : array_like\n            The raw data.  A cross-table for analysis is constructed\n            from the first two columns.\n\n        Returns\n        -------\n        StratifiedTable\n        '
        if not isinstance(data, pd.DataFrame):
            data1 = pd.DataFrame(index=np.arange(data.shape[0]), columns=[var1, var2, strata])
            data1[data1.columns[var1]] = data[:, var1]
            data1[data1.columns[var2]] = data[:, var2]
            data1[data1.columns[strata]] = data[:, strata]
        else:
            data1 = data[[var1, var2, strata]]
        gb = data1.groupby(strata).groups
        tables = []
        for g in gb:
            ii = gb[g]
            tab = pd.crosstab(data1.loc[ii, var1], data1.loc[ii, var2])
            if (tab.shape != np.r_[2, 2]).any():
                msg = 'Invalid table dimensions'
                raise ValueError(msg)
            tables.append(np.asarray(tab))
        return cls(tables)

    def test_null_odds(self, correction=False):
        if False:
            while True:
                i = 10
        "\n        Test that all tables have odds ratio equal to 1.\n\n        This is the 'Mantel-Haenszel' test.\n\n        Parameters\n        ----------\n        correction : bool\n            If True, use the continuity correction when calculating the\n            test statistic.\n\n        Returns\n        -------\n        Bunch\n            A bunch containing the chi^2 test statistic and p-value.\n        "
        statistic = np.sum(self.table[0, 0, :] - self._apb * self._apc / self._n)
        statistic = np.abs(statistic)
        if correction:
            statistic -= 0.5
        statistic = statistic ** 2
        denom = self._apb * self._apc * self._bpd * self._cpd
        denom /= self._n ** 2 * (self._n - 1)
        denom = np.sum(denom)
        statistic /= denom
        pvalue = 1 - stats.chi2.cdf(statistic, 1)
        b = _Bunch()
        b.statistic = statistic
        b.pvalue = pvalue
        return b

    @cache_readonly
    def oddsratio_pooled(self):
        if False:
            i = 10
            return i + 15
        '\n        The pooled odds ratio.\n\n        The value is an estimate of a common odds ratio across all of the\n        stratified tables.\n        '
        odds_ratio = np.sum(self._ad / self._n) / np.sum(self._bc / self._n)
        return odds_ratio

    @cache_readonly
    def logodds_pooled(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the logarithm of the pooled odds ratio.\n\n        See oddsratio_pooled for more information.\n        '
        return np.log(self.oddsratio_pooled)

    @cache_readonly
    def riskratio_pooled(self):
        if False:
            while True:
                i = 10
        '\n        Estimate of the pooled risk ratio.\n        '
        acd = self.table[0, 0, :] * self._cpd
        cab = self.table[1, 0, :] * self._apb
        rr = np.sum(acd / self._n) / np.sum(cab / self._n)
        return rr

    @cache_readonly
    def logodds_pooled_se(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Estimated standard error of the pooled log odds ratio\n\n        References\n        ----------\n        J. Robins, N. Breslow, S. Greenland. "Estimators of the\n        Mantel-Haenszel Variance Consistent in Both Sparse Data and\n        Large-Strata Limiting Models." Biometrics 42, no. 2 (1986): 311-23.\n        '
        adns = np.sum(self._ad / self._n)
        bcns = np.sum(self._bc / self._n)
        lor_va = np.sum(self._apd * self._ad / self._n ** 2) / adns ** 2
        mid = self._apd * self._bc / self._n ** 2
        mid += (1 - self._apd / self._n) * self._ad / self._n
        mid = np.sum(mid)
        mid /= adns * bcns
        lor_va += mid
        lor_va += np.sum((1 - self._apd / self._n) * self._bc / self._n) / bcns ** 2
        lor_va /= 2
        lor_se = np.sqrt(lor_va)
        return lor_se

    def logodds_pooled_confint(self, alpha=0.05, method='normal'):
        if False:
            while True:
                i = 10
        "\n        A confidence interval for the pooled log odds ratio.\n\n        Parameters\n        ----------\n        alpha : float\n            `1 - alpha` is the nominal coverage probability of the\n            interval.\n        method : str\n            The method for producing the confidence interval.  Currently\n            must be 'normal' which uses the normal approximation.\n\n        Returns\n        -------\n        lcb : float\n            The lower confidence limit.\n        ucb : float\n            The upper confidence limit.\n        "
        lor = np.log(self.oddsratio_pooled)
        lor_se = self.logodds_pooled_se
        f = -stats.norm.ppf(alpha / 2)
        lcb = lor - f * lor_se
        ucb = lor + f * lor_se
        return (lcb, ucb)

    def oddsratio_pooled_confint(self, alpha=0.05, method='normal'):
        if False:
            while True:
                i = 10
        "\n        A confidence interval for the pooled odds ratio.\n\n        Parameters\n        ----------\n        alpha : float\n            `1 - alpha` is the nominal coverage probability of the\n            interval.\n        method : str\n            The method for producing the confidence interval.  Currently\n            must be 'normal' which uses the normal approximation.\n\n        Returns\n        -------\n        lcb : float\n            The lower confidence limit.\n        ucb : float\n            The upper confidence limit.\n        "
        (lcb, ucb) = self.logodds_pooled_confint(alpha, method=method)
        lcb = np.exp(lcb)
        ucb = np.exp(ucb)
        return (lcb, ucb)

    def test_equal_odds(self, adjust=False):
        if False:
            i = 10
            return i + 15
        "\n        Test that all odds ratios are identical.\n\n        This is the 'Breslow-Day' testing procedure.\n\n        Parameters\n        ----------\n        adjust : bool\n            Use the 'Tarone' adjustment to achieve the chi^2\n            asymptotic distribution.\n\n        Returns\n        -------\n        A bunch containing the following attributes:\n\n        statistic : float\n            The chi^2 test statistic.\n        p-value : float\n            The p-value for the test.\n        "
        table = self.table
        r = self.oddsratio_pooled
        a = 1 - r
        b = r * (self._apb + self._apc) + self._dma
        c = -r * self._apb * self._apc
        dr = np.sqrt(b ** 2 - 4 * a * c)
        e11 = (-b + dr) / (2 * a)
        v11 = 1 / e11 + 1 / (self._apc - e11) + 1 / (self._apb - e11) + 1 / (self._dma + e11)
        v11 = 1 / v11
        statistic = np.sum((table[0, 0, :] - e11) ** 2 / v11)
        if adjust:
            adj = table[0, 0, :].sum() - e11.sum()
            adj = adj ** 2
            adj /= np.sum(v11)
            statistic -= adj
        pvalue = 1 - stats.chi2.cdf(statistic, table.shape[2] - 1)
        b = _Bunch()
        b.statistic = statistic
        b.pvalue = pvalue
        return b

    def summary(self, alpha=0.05, float_format='%.3f', method='normal'):
        if False:
            i = 10
            return i + 15
        "\n        A summary of all the main results.\n\n        Parameters\n        ----------\n        alpha : float\n            `1 - alpha` is the nominal coverage probability of the\n            confidence intervals.\n        float_format : str\n            Used for formatting numeric values in the summary.\n        method : str\n            The method for producing the confidence interval.  Currently\n            must be 'normal' which uses the normal approximation.\n        "

        def fmt(x):
            if False:
                i = 10
                return i + 15
            if isinstance(x, str):
                return x
            return float_format % x
        (co_lcb, co_ucb) = self.oddsratio_pooled_confint(alpha=alpha, method=method)
        (clo_lcb, clo_ucb) = self.logodds_pooled_confint(alpha=alpha, method=method)
        headers = ['Estimate', 'LCB', 'UCB']
        stubs = ['Pooled odds', 'Pooled log odds', 'Pooled risk ratio', '']
        data = [[fmt(x) for x in [self.oddsratio_pooled, co_lcb, co_ucb]], [fmt(x) for x in [self.logodds_pooled, clo_lcb, clo_ucb]], [fmt(x) for x in [self.riskratio_pooled, '', '']], ['', '', '']]
        tab1 = iolib.SimpleTable(data, headers, stubs, data_aligns='r', table_dec_above='')
        headers = ['Statistic', 'P-value', '']
        stubs = ['Test of OR=1', 'Test constant OR']
        rslt1 = self.test_null_odds()
        rslt2 = self.test_equal_odds()
        data = [[fmt(x) for x in [rslt1.statistic, rslt1.pvalue, '']], [fmt(x) for x in [rslt2.statistic, rslt2.pvalue, '']]]
        tab2 = iolib.SimpleTable(data, headers, stubs, data_aligns='r')
        tab1.extend(tab2)
        headers = ['', '', '']
        stubs = ['Number of tables', 'Min n', 'Max n', 'Avg n', 'Total n']
        ss = self.table.sum(0).sum(0)
        data = [['%d' % self.table.shape[2], '', ''], ['%d' % min(ss), '', ''], ['%d' % max(ss), '', ''], ['%.0f' % np.mean(ss), '', ''], ['%d' % sum(ss), '', '', '']]
        tab3 = iolib.SimpleTable(data, headers, stubs, data_aligns='r')
        tab1.extend(tab3)
        return tab1

def mcnemar(table, exact=True, correction=True):
    if False:
        return 10
    "\n    McNemar test of homogeneity.\n\n    Parameters\n    ----------\n    table : array_like\n        A square contingency table.\n    exact : bool\n        If exact is true, then the binomial distribution will be used.\n        If exact is false, then the chisquare distribution will be\n        used, which is the approximation to the distribution of the\n        test statistic for large sample sizes.\n    correction : bool\n        If true, then a continuity correction is used for the chisquare\n        distribution (if exact is false.)\n\n    Returns\n    -------\n    A bunch with attributes:\n\n    statistic : float or int, array\n        The test statistic is the chisquare statistic if exact is\n        false. If the exact binomial distribution is used, then this\n        contains the min(n1, n2), where n1, n2 are cases that are zero\n        in one sample but one in the other sample.\n    pvalue : float or array\n        p-value of the null hypothesis of equal marginal distributions.\n\n    Notes\n    -----\n    This is a special case of Cochran's Q test, and of the homogeneity\n    test. The results when the chisquare distribution is used are\n    identical, except for continuity correction.\n    "
    table = _make_df_square(table)
    table = np.asarray(table, dtype=np.float64)
    (n1, n2) = (table[0, 1], table[1, 0])
    if exact:
        statistic = np.minimum(n1, n2)
        int_sum = int(n1 + n2)
        if int_sum != n1 + n2:
            raise ValueError('exact can only be used with tables containing integers.')
        pvalue = stats.binom.cdf(statistic, int_sum, 0.5) * 2
        pvalue = np.minimum(pvalue, 1)
    else:
        corr = int(correction)
        statistic = (np.abs(n1 - n2) - corr) ** 2 / (1.0 * (n1 + n2))
        df = 1
        pvalue = stats.chi2.sf(statistic, df)
    b = _Bunch()
    b.statistic = statistic
    b.pvalue = pvalue
    return b

def cochrans_q(x, return_object=True):
    if False:
        while True:
            i = 10
    "\n    Cochran's Q test for identical binomial proportions.\n\n    Parameters\n    ----------\n    x : array_like, 2d (N, k)\n        data with N cases and k variables\n    return_object : bool\n        Return values as bunch instead of as individual values.\n\n    Returns\n    -------\n    Returns a bunch containing the following attributes, or the\n    individual values according to the value of `return_object`.\n\n    statistic : float\n       test statistic\n    pvalue : float\n       pvalue from the chisquare distribution\n\n    Notes\n    -----\n    Cochran's Q is a k-sample extension of the McNemar test. If there\n    are only two groups, then Cochran's Q test and the McNemar test\n    are equivalent.\n\n    The procedure tests that the probability of success is the same\n    for every group.  The alternative hypothesis is that at least two\n    groups have a different probability of success.\n\n    In Wikipedia terminology, rows are blocks and columns are\n    treatments.  The number of rows N, should be large for the\n    chisquare distribution to be a good approximation.\n\n    The Null hypothesis of the test is that all treatments have the\n    same effect.\n\n    References\n    ----------\n    https://en.wikipedia.org/wiki/Cochran_test\n    SAS Manual for NPAR TESTS\n    "
    x = np.asarray(x, dtype=np.float64)
    gruni = np.unique(x)
    (N, k) = x.shape
    count_row_success = (x == gruni[-1]).sum(1, float)
    count_col_success = (x == gruni[-1]).sum(0, float)
    count_row_ss = count_row_success.sum()
    count_col_ss = count_col_success.sum()
    assert count_row_ss == count_col_ss
    q_stat = (k - 1) * (k * np.sum(count_col_success ** 2) - count_col_ss ** 2) / (k * count_row_ss - np.sum(count_row_success ** 2))
    df = k - 1
    pvalue = stats.chi2.sf(q_stat, df)
    if return_object:
        b = _Bunch()
        b.statistic = q_stat
        b.df = df
        b.pvalue = pvalue
        return b
    return (q_stat, pvalue, df)