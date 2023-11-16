import unittest
import pandas as pd
from pyspark import pandas as ps
from pyspark.testing.pandasutils import ComparisonTestBase
from pyspark.testing.sqlutils import SQLTestUtils

class GroupbySplitApplyMixin:

    @property
    def pdf(self):
        if False:
            i = 10
            return i + 15
        return pd.DataFrame({'A': [1, 2, 1, 2], 'B': [3.1, 4.1, 4.1, 3.1], 'C': ['a', 'b', 'b', 'a'], 'D': [True, False, False, True]})

    @property
    def psdf(self):
        if False:
            return 10
        return ps.from_pandas(self.pdf)

    def test_split_apply_combine_on_series(self):
        if False:
            i = 10
            return i + 15
        pdf = pd.DataFrame({'a': [1, 2, 6, 4, 4, 6, 4, 3, 7], 'b': [4, 2, 7, 3, 3, 1, 1, 1, 2], 'c': [4, 2, 7, 3, None, 1, 1, 1, 2]}, index=[0, 1, 3, 5, 6, 8, 9, 9, 9])
        psdf = ps.from_pandas(pdf)
        funcs = [((True, False), ['sum', 'min', 'max', 'count', 'first', 'last']), ((True, True), ['mean']), ((False, False), ['var', 'std', 'skew'])]
        funcs = [(check_exact, almost, f) for ((check_exact, almost), fs) in funcs for f in fs]
        for as_index in [True, False]:
            if as_index:

                def sort(df):
                    if False:
                        i = 10
                        return i + 15
                    return df.sort_index()
            else:

                def sort(df):
                    if False:
                        i = 10
                        return i + 15
                    return df.sort_values(list(df.columns)).reset_index(drop=True)
            for (check_exact, almost, func) in funcs:
                for (kkey, pkey) in [('b', 'b'), (psdf.b, pdf.b)]:
                    with self.subTest(as_index=as_index, func=func, key=pkey):
                        if as_index is True or func != 'std':
                            self.assert_eq(sort(getattr(psdf.groupby(kkey, as_index=as_index).a, func)()), sort(getattr(pdf.groupby(pkey, as_index=as_index).a, func)()), check_exact=check_exact, almost=almost)
                            self.assert_eq(sort(getattr(psdf.groupby(kkey, as_index=as_index), func)()), sort(getattr(pdf.groupby(pkey, as_index=as_index), func)()), check_exact=check_exact, almost=almost)
                        else:
                            self.assert_eq(sort(getattr(psdf.groupby(kkey, as_index=as_index).a, func)()), sort(pdf.groupby(pkey, as_index=True).a.std().reset_index()), check_exact=check_exact, almost=almost)
                            self.assert_eq(sort(getattr(psdf.groupby(kkey, as_index=as_index), func)()), sort(pdf.groupby(pkey, as_index=True).std().reset_index()), check_exact=check_exact, almost=almost)
                for (kkey, pkey) in [(psdf.b + 1, pdf.b + 1), (psdf.copy().b, pdf.copy().b)]:
                    with self.subTest(as_index=as_index, func=func, key=pkey):
                        self.assert_eq(sort(getattr(psdf.groupby(kkey, as_index=as_index).a, func)()), sort(getattr(pdf.groupby(pkey, as_index=as_index).a, func)()), check_exact=check_exact, almost=almost)
                        self.assert_eq(sort(getattr(psdf.groupby(kkey, as_index=as_index), func)()), sort(getattr(pdf.groupby(pkey, as_index=as_index), func)()), check_exact=check_exact, almost=almost)
            for (check_exact, almost, func) in funcs:
                for i in [0, 4, 7]:
                    with self.subTest(as_index=as_index, func=func, i=i):
                        self.assert_eq(sort(getattr(psdf.groupby(psdf.b > i, as_index=as_index).a, func)()), sort(getattr(pdf.groupby(pdf.b > i, as_index=as_index).a, func)()), check_exact=check_exact, almost=almost)
                        self.assert_eq(sort(getattr(psdf.groupby(psdf.b > i, as_index=as_index), func)()), sort(getattr(pdf.groupby(pdf.b > i, as_index=as_index), func)()), check_exact=check_exact, almost=almost)
        for (check_exact, almost, func) in funcs:
            for (kkey, pkey) in [(psdf.b, pdf.b), (psdf.b + 1, pdf.b + 1), (psdf.copy().b, pdf.copy().b), (psdf.b.rename(), pdf.b.rename())]:
                with self.subTest(func=func, key=pkey):
                    self.assert_eq(getattr(psdf.a.groupby(kkey), func)().sort_index(), getattr(pdf.a.groupby(pkey), func)().sort_index(), check_exact=check_exact, almost=almost)
                    self.assert_eq(getattr((psdf.a + 1).groupby(kkey), func)().sort_index(), getattr((pdf.a + 1).groupby(pkey), func)().sort_index(), check_exact=check_exact, almost=almost)
                    self.assert_eq(getattr((psdf.b + 1).groupby(kkey), func)().sort_index(), getattr((pdf.b + 1).groupby(pkey), func)().sort_index(), check_exact=check_exact, almost=almost)
                    self.assert_eq(getattr(psdf.a.rename().groupby(kkey), func)().sort_index(), getattr(pdf.a.rename().groupby(pkey), func)().sort_index(), check_exact=check_exact, almost=almost)

class GroupbySplitApplyTests(GroupbySplitApplyMixin, ComparisonTestBase, SQLTestUtils):
    pass
if __name__ == '__main__':
    from pyspark.pandas.tests.groupby.test_split_apply import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)