import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal
from statsmodels.iolib.summary2 import summary_col
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS

class TestSummaryLatex:

    def test_summarycol(self):
        if False:
            print('Hello World!')
        desired = '\n\\begin{table}\n\\caption{}\n\\label{}\n\\begin{center}\n\\begin{tabular}{lll}\n\\hline\n               & y I      & y II      \\\\\n\\hline\nconst          & 7.7500   & 12.4231   \\\\\n               & (1.1058) & (3.1872)  \\\\\nx1             & -0.7500  & -1.5769   \\\\\n               & (0.2368) & (0.6826)  \\\\\nR-squared      & 0.7697   & 0.6401    \\\\\nR-squared Adj. & 0.6930   & 0.5202    \\\\\n\\hline\n\\end{tabular}\n\\end{center}\n\\end{table}\n\\bigskip\nStandard errors in parentheses.\n'
        x = [1, 5, 7, 3, 5]
        x = add_constant(x)
        y1 = [6, 4, 2, 7, 4]
        y2 = [8, 5, 0, 12, 4]
        reg1 = OLS(y1, x).fit()
        reg2 = OLS(y2, x).fit()
        actual = summary_col([reg1, reg2]).as_latex()
        actual = '\n%s\n' % actual
        assert_equal(desired, actual)

    def test_summarycol_float_format(self):
        if False:
            for i in range(10):
                print('nop')
        desired = '\n==========================\n                y I   y II\n--------------------------\nconst          7.7   12.4 \n               (1.1) (3.2)\nx1             -0.7  -1.6 \n               (0.2) (0.7)\nR-squared      0.8   0.6  \nR-squared Adj. 0.7   0.5  \n==========================\nStandard errors in\nparentheses.\n'
        x = [1, 5, 7, 3, 5]
        x = add_constant(x)
        y1 = [6, 4, 2, 7, 4]
        y2 = [8, 5, 0, 12, 4]
        reg1 = OLS(y1, x).fit()
        reg2 = OLS(y2, x).fit()
        actual = summary_col([reg1, reg2], float_format='%0.1f').as_text()
        actual = '%s\n' % actual
        assert_equal(actual, desired)
        starred = summary_col([reg1, reg2], stars=True, float_format='%0.1f')
        assert '7.7***' in str(starred)
        assert '12.4**' in str(starred)
        assert '12.4***' not in str(starred)

    def test_summarycol_drop_omitted(self):
        if False:
            return 10
        x = [1, 5, 7, 3, 5]
        x = add_constant(x)
        x2 = np.concatenate([x, np.array([[3], [9], [-1], [4], [0]])], 1)
        y1 = [6, 4, 2, 7, 4]
        y2 = [8, 5, 0, 12, 4]
        reg1 = OLS(y1, x).fit()
        reg2 = OLS(y2, x2).fit()
        actual = summary_col([reg1, reg2], regressor_order=['const', 'x1'], drop_omitted=True)
        assert 'x2' not in str(actual)
        actual = summary_col([reg1, reg2], regressor_order=['x1'], drop_omitted=False)
        assert 'const' in str(actual)
        assert 'x2' in str(actual)

    def test_summary_col_ordering_preserved(self):
        if False:
            while True:
                i = 10
        x = [1, 5, 7, 3, 5]
        x = add_constant(x)
        x2 = np.concatenate([x, np.array([[3], [9], [-1], [4], [0]])], 1)
        x2 = pd.DataFrame(x2, columns=['const', 'b', 'a'])
        y1 = [6, 4, 2, 7, 4]
        y2 = [8, 5, 0, 12, 4]
        reg1 = OLS(y1, x2).fit()
        reg2 = OLS(y2, x2).fit()
        info_dict = {'R2': lambda x: '{:.3f}'.format(int(x.rsquared)), 'N': lambda x: '{0:d}'.format(int(x.nobs))}
        original = actual = summary_col([reg1, reg2], float_format='%0.4f')
        actual = summary_col([reg1, reg2], regressor_order=['a', 'b'], float_format='%0.4f', info_dict=info_dict)
        variables = ('const', 'b', 'a')
        for line in str(original).split('\n'):
            for variable in variables:
                if line.startswith(variable):
                    assert line in str(actual)

    def test__repr_latex_(self):
        if False:
            print('Hello World!')
        desired = '\n\\begin{table}\n\\caption{}\n\\label{}\n\\begin{center}\n\\begin{tabular}{lll}\n\\hline\n               & y I      & y II      \\\\\n\\hline\nconst          & 7.7500   & 12.4231   \\\\\n               & (1.1058) & (3.1872)  \\\\\nx1             & -0.7500  & -1.5769   \\\\\n               & (0.2368) & (0.6826)  \\\\\nR-squared      & 0.7697   & 0.6401    \\\\\nR-squared Adj. & 0.6930   & 0.5202    \\\\\n\\hline\n\\end{tabular}\n\\end{center}\n\\end{table}\n\\bigskip\nStandard errors in parentheses.\n'
        x = [1, 5, 7, 3, 5]
        x = add_constant(x)
        y1 = [6, 4, 2, 7, 4]
        y2 = [8, 5, 0, 12, 4]
        reg1 = OLS(y1, x).fit()
        reg2 = OLS(y2, x).fit()
        actual = summary_col([reg1, reg2])._repr_latex_()
        actual = '\n%s\n' % actual
        assert_equal(actual, desired)

    def test_OLSsummary(self):
        if False:
            print('Hello World!')
        x = [1, 5, 7, 3, 5]
        x = add_constant(x)
        y1 = [6, 4, 2, 7, 4]
        reg1 = OLS(y1, x).fit()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            actual = reg1.summary().as_latex()
        string_to_find = '\\end{tabular}\n\\begin{tabular}'
        result = string_to_find in actual
        assert result is True

def test_ols_summary_rsquared_label():
    if False:
        for i in range(10):
            print('nop')
    x = [1, 5, 7, 3, 5, 2, 5, 3]
    y = [6, 4, 2, 7, 4, 9, 10, 2]
    reg_with_constant = OLS(y, add_constant(x)).fit()
    r2_str = 'R-squared:'
    with pytest.warns(UserWarning):
        assert r2_str in str(reg_with_constant.summary2())
    with pytest.warns(UserWarning):
        assert r2_str in str(reg_with_constant.summary())
    reg_without_constant = OLS(y, x, hasconst=False).fit()
    r2_str = 'R-squared (uncentered):'
    with pytest.warns(UserWarning):
        assert r2_str in str(reg_without_constant.summary2())
    with pytest.warns(UserWarning):
        assert r2_str in str(reg_without_constant.summary())

class TestSummaryLabels:
    """
    Test that the labels are correctly set in the summary table"""

    @classmethod
    def setup_class(cls):
        if False:
            print('Hello World!')
        y = [1, 1, 4, 2] * 4
        x = add_constant([1, 2, 3, 4] * 4)
        cls.mod = OLS(endog=y, exog=x).fit()

    def test_summary_col_r2(self):
        if False:
            print('Hello World!')
        table = summary_col(results=self.mod, include_r2=True)
        assert 'R-squared  ' in str(table)
        assert 'R-squared Adj.' in str(table)

    def test_absence_of_r2(self):
        if False:
            return 10
        table = summary_col(results=self.mod, include_r2=False)
        assert 'R-squared' not in str(table)
        assert 'R-squared Adj.' not in str(table)