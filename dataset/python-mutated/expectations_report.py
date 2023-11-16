from typing import Any, Optional
import pandas as pd
from visions import VisionsTypeset
from ydata_profiling.config import Settings
from ydata_profiling.model import BaseDescription, expectation_algorithms
from ydata_profiling.model.handler import Handler
from ydata_profiling.utils.dataframe import slugify

class ExpectationHandler(Handler):
    """Default handler"""

    def __init__(self, typeset: VisionsTypeset, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        mapping = {'Unsupported': [expectation_algorithms.generic_expectations], 'Text': [expectation_algorithms.categorical_expectations], 'Categorical': [expectation_algorithms.categorical_expectations], 'Boolean': [expectation_algorithms.categorical_expectations], 'Numeric': [expectation_algorithms.numeric_expectations], 'URL': [expectation_algorithms.url_expectations], 'File': [expectation_algorithms.file_expectations], 'Path': [expectation_algorithms.path_expectations], 'DateTime': [expectation_algorithms.datetime_expectations], 'Image': [expectation_algorithms.image_expectations]}
        super().__init__(mapping, typeset, *args, **kwargs)

class ExpectationsReport:
    config: Settings
    df: Optional[pd.DataFrame] = None

    @property
    def typeset(self) -> Optional[VisionsTypeset]:
        if False:
            return 10
        return None

    def to_expectation_suite(self, suite_name: Optional[str]=None, data_context: Optional[Any]=None, save_suite: bool=True, run_validation: bool=True, build_data_docs: bool=True, handler: Optional[Handler]=None) -> Any:
        if False:
            return 10
        '\n        All parameters default to True to make it easier to access the full functionality of Great Expectations out of\n        the box.\n        Args:\n            suite_name: The name of your expectation suite\n            data_context: A user-specified data context\n            save_suite: Boolean to determine whether to save the suite to .json as part of the method\n            run_validation: Boolean to determine whether to run validation as part of the method\n            build_data_docs: Boolean to determine whether to build data docs, save the .html file, and open data docs in\n                your browser\n            handler: The handler to use for building expectation\n\n        Returns:\n            An ExpectationSuite\n        '
        try:
            import great_expectations as ge
        except ImportError as ex:
            raise ImportError('Please install great expectations before using the expectation functionality') from ex
        if suite_name is None:
            suite_name = slugify(self.config.title)
        if handler is None:
            handler = ExpectationHandler(self.typeset)
        if not data_context:
            data_context = ge.data_context.DataContext()
        suite = data_context.create_expectation_suite(suite_name, overwrite_existing=True)
        batch = ge.dataset.PandasDataset(self.df, expectation_suite=suite)
        summary: BaseDescription = self.get_description()
        for (name, variable_summary) in summary.variables.items():
            handler.handle(variable_summary['type'], name, variable_summary, batch)
        suite = batch.get_expectation_suite()
        validation_result_identifier = None
        if run_validation:
            batch = ge.dataset.PandasDataset(self.df, expectation_suite=suite)
            results = data_context.run_validation_operator('action_list_operator', assets_to_validate=[batch])
            validation_result_identifier = results.list_validation_result_identifiers()[0]
        if save_suite or build_data_docs:
            data_context.save_expectation_suite(suite)
        if build_data_docs:
            data_context.build_data_docs()
            data_context.open_data_docs(validation_result_identifier)
        return batch.get_expectation_suite()