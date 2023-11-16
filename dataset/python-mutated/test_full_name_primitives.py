import numpy as np
import pandas as pd
from featuretools.primitives import FullNameToFirstName, FullNameToLastName, FullNameToTitle
from featuretools.tests.primitive_tests.utils import PrimitiveTestBase, find_applicable_primitives, valid_dfs

class TestFullNameToFirstName(PrimitiveTestBase):
    primitive = FullNameToFirstName

    def test_urls(self):
        if False:
            for i in range(10):
                print('nop')
        primitive_func = self.primitive().get_function()
        names = pd.Series(['Spector, Mr. Woolf', 'Oliva y Ocana, Dona. Fermina', 'Saether, Mr. Simon Sivertsen', 'Ware, Mr. Frederick', 'Peter, Master. Michael J'])
        answer = pd.Series(['Woolf', 'Oliva', 'Simon', 'Frederick', 'Michael'])
        pd.testing.assert_series_equal(primitive_func(names), answer, check_names=False)

    def test_no_title(self):
        if False:
            print('Hello World!')
        primitive_func = self.primitive().get_function()
        names = pd.Series(['Peter, Michael J', 'James Masters', 'Kate Elizabeth Brown-Jones'])
        answer = pd.Series(['Michael', 'James', 'Kate'], dtype=object)
        pd.testing.assert_series_equal(primitive_func(names), answer, check_names=False)

    def test_empty_string(self):
        if False:
            i = 10
            return i + 15
        primitive_func = self.primitive().get_function()
        names = pd.Series(['Peter, Michael J', '', 'Kate Elizabeth Brown-Jones'])
        answer = pd.Series(['Michael', np.nan, 'Kate'], dtype=object)
        pd.testing.assert_series_equal(primitive_func(names), answer, check_names=False)

    def test_single_name(self):
        if False:
            print('Hello World!')
        primitive_func = self.primitive().get_function()
        names = pd.Series(['Peter, Michael J', 'James', 'Kate Elizabeth Brown-Jones'])
        answer = pd.Series(['Michael', 'James', 'Kate'], dtype=object)
        pd.testing.assert_series_equal(primitive_func(names), answer, check_names=False)

    def test_nan(self):
        if False:
            i = 10
            return i + 15
        primitive_func = self.primitive().get_function()
        names = pd.Series(['Mr. James Brown', np.nan, None])
        answer = pd.Series(['James', np.nan, np.nan])
        pd.testing.assert_series_equal(primitive_func(names), answer, check_names=False)

    def test_with_featuretools(self, pd_es):
        if False:
            return 10
        (transform, aggregation) = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        transform.append(primitive_instance)
        valid_dfs(pd_es, aggregation, transform, self.primitive)

class TestFullNameToLastName(PrimitiveTestBase):
    primitive = FullNameToLastName

    def test_urls(self):
        if False:
            print('Hello World!')
        primitive_func = self.primitive().get_function()
        names = pd.Series(['Spector, Mr. Woolf', 'Oliva y Ocana, Dona. Fermina', 'Saether, Mr. Simon Sivertsen', 'Ware, Mr. Frederick', 'Peter, Master. Michael J'])
        answer = pd.Series(['Spector', 'Oliva y Ocana', 'Saether', 'Ware', 'Peter'])
        pd.testing.assert_series_equal(primitive_func(names), answer, check_names=False)

    def test_no_title(self):
        if False:
            return 10
        primitive_func = self.primitive().get_function()
        names = pd.Series(['Peter, Michael J', 'James Masters', 'Kate Elizabeth Brown-Jones'])
        answer = pd.Series(['Peter', 'Masters', 'Brown-Jones'], dtype=object)
        pd.testing.assert_series_equal(primitive_func(names), answer, check_names=False)

    def test_empty_string(self):
        if False:
            for i in range(10):
                print('nop')
        primitive_func = self.primitive().get_function()
        names = pd.Series(['Peter, Michael J', '', 'Kate Elizabeth Brown-Jones'])
        answer = pd.Series(['Peter', np.nan, 'Brown-Jones'], dtype=object)
        pd.testing.assert_series_equal(primitive_func(names), answer, check_names=False)

    def test_single_name(self):
        if False:
            print('Hello World!')
        primitive_func = self.primitive().get_function()
        names = pd.Series(['Peter, Michael J', 'James', 'Kate Elizabeth Brown-Jones'])
        answer = pd.Series(['Peter', np.nan, 'Brown-Jones'], dtype=object)
        pd.testing.assert_series_equal(primitive_func(names), answer, check_names=False)

    def test_nan(self):
        if False:
            i = 10
            return i + 15
        primitive_func = self.primitive().get_function()
        names = pd.Series(['Mr. James Brown', np.nan, None])
        answer = pd.Series(['Brown', np.nan, np.nan])
        pd.testing.assert_series_equal(primitive_func(names), answer, check_names=False)

    def test_with_featuretools(self, pd_es):
        if False:
            print('Hello World!')
        (transform, aggregation) = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        transform.append(primitive_instance)
        valid_dfs(pd_es, aggregation, transform, self.primitive)

class TestFullNameToTitle(PrimitiveTestBase):
    primitive = FullNameToTitle

    def test_urls(self):
        if False:
            print('Hello World!')
        primitive_func = self.primitive().get_function()
        names = pd.Series(['Spector, Mr. Woolf', 'Oliva y Ocana, Dona. Fermina', 'Saether, Mr. Simon Sivertsen', 'Ware, Mr. Frederick', 'Peter, Master. Michael J', 'Mr. Brown'])
        answer = pd.Series(['Mr', 'Dona', 'Mr', 'Mr', 'Master', 'Mr'])
        pd.testing.assert_series_equal(primitive_func(names), answer, check_names=False)

    def test_no_title(self):
        if False:
            i = 10
            return i + 15
        primitive_func = self.primitive().get_function()
        names = pd.Series(['Peter, Michael J', 'James Master.', 'Mrs Brown', ''])
        answer = pd.Series([np.nan, np.nan, np.nan, np.nan], dtype=object)
        pd.testing.assert_series_equal(primitive_func(names), answer, check_names=False)

    def test_nan(self):
        if False:
            for i in range(10):
                print('nop')
        primitive_func = self.primitive().get_function()
        names = pd.Series(['Mr. Brown', np.nan, None])
        answer = pd.Series(['Mr', np.nan, np.nan])
        pd.testing.assert_series_equal(primitive_func(names), answer, check_names=False)

    def test_with_featuretools(self, pd_es):
        if False:
            i = 10
            return i + 15
        (transform, aggregation) = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        transform.append(primitive_instance)
        valid_dfs(pd_es, aggregation, transform, self.primitive)