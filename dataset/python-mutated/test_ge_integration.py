import sys
from unittest.mock import Mock
import pandas as pd
import pytest
from ydata_profiling import ProfileReport

@pytest.fixture
def df():
    if False:
        return 10
    return pd.DataFrame({'num': [1, 2, 3, 4, 5]})

@pytest.fixture(scope='function')
def mod():
    if False:
        i = 10
        return i + 15
    mod = Mock()
    sys.modules['great_expectations'] = mod
    return mod

@pytest.fixture(scope='function')
def context():
    if False:
        while True:
            i = 10
    return Mock()

def test_to_expectation_suite_raises(df):
    if False:
        print('Hello World!')
    report = ProfileReport(df)
    with pytest.raises(ImportError):
        report.to_expectation_suite()

def test_to_expectations_suite_context_save_and_build_data_docs(mod, context, df):
    if False:
        for i in range(10):
            print('nop')
    report = ProfileReport(df)
    _ = report.to_expectation_suite(data_context=context, save_suite=True, run_validation=False, build_data_docs=True)
    mod.data_context.DataContext.assert_not_called()
    mod.dataset.PandasDataset.assert_called_once()
    context.create_expectation_suite.assert_called_once()
    context.save_expectation_suite.assert_called_once()
    context.build_data_docs.assert_called_once()
    context.open_data_docs.assert_called_once()

def test_to_expectations_suite_context_no_save_and_build_data_docs(mod, context, df):
    if False:
        return 10
    report = ProfileReport(df)
    _ = report.to_expectation_suite(data_context=context, save_suite=False, run_validation=False, build_data_docs=True)
    mod.data_context.DataContext.assert_not_called()
    mod.dataset.PandasDataset.assert_called_once()
    context.create_expectation_suite.assert_called_once()
    context.save_expectation_suite.assert_called_once()
    context.build_data_docs.assert_called_once()
    context.open_data_docs.assert_called_once()

def test_to_expectations_suite_context_no_save_and_no_build_data_docs(mod, context, df):
    if False:
        i = 10
        return i + 15
    report = ProfileReport(df)
    _ = report.to_expectation_suite(data_context=context, save_suite=False, run_validation=False, build_data_docs=False)
    mod.data_context.DataContext.assert_not_called()
    mod.dataset.PandasDataset.assert_called_once()
    context.create_expectation_suite.assert_called_once()
    context.save_expectation_suite.assert_not_called()
    context.build_data_docs.assert_not_called()
    context.open_data_docs.assert_not_called()

def test_to_expectations_suite_title(context, df):
    if False:
        return 10
    report = ProfileReport(df, title='Expectations Dataset')
    _ = report.to_expectation_suite(suite_name=None, data_context=context, run_validation=False)
    context.create_expectation_suite.assert_called_once_with('expectations-dataset', overwrite_existing=True)

def test_to_expectation_suite_no_context(mod, df):
    if False:
        i = 10
        return i + 15
    report = ProfileReport(df)
    _ = report.to_expectation_suite(data_context=None, save_suite=False, run_validation=False, build_data_docs=False)
    mod.data_context.DataContext.assert_called_once()