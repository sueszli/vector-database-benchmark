from robot.errors import DataError
from robot.utils import get_error_details, Importer, is_string, split_args_from_name_or_path, type_name
from .visitor import SuiteVisitor

class ModelModifier(SuiteVisitor):

    def __init__(self, visitors, empty_suite_ok, logger):
        if False:
            return 10
        self._log_error = logger.error
        self._empty_suite_ok = empty_suite_ok
        self._visitors = list(self._yield_visitors(visitors, logger))

    def visit_suite(self, suite):
        if False:
            return 10
        for visitor in self._visitors:
            try:
                suite.visit(visitor)
            except Exception:
                (message, details) = get_error_details()
                self._log_error(f"Executing model modifier '{type_name(visitor)}' failed: {message}\n{details}")
        if not (suite.has_tests or self._empty_suite_ok):
            raise DataError(f"Suite '{suite.name}' contains no tests after model modifiers.")

    def _yield_visitors(self, visitors, logger):
        if False:
            for i in range(10):
                print('nop')
        importer = Importer('model modifier', logger=logger)
        for visitor in visitors:
            if is_string(visitor):
                (name, args) = split_args_from_name_or_path(visitor)
                try:
                    yield importer.import_class_or_module(name, args)
                except DataError as err:
                    logger.error(err.message)
            else:
                yield visitor