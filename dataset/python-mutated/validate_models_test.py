import tempfile
from pathlib import Path
from typing import Iterable
import testslide
from ... import error
from ...tests import setup
from .. import daemon_query, validate_models

class ValidateModelsTest(testslide.TestCase):

    def test_parse_response(self) -> None:
        if False:
            return 10

        def assert_parsed(payload: object, expected: Iterable[error.ModelVerificationError]) -> None:
            if False:
                return 10
            self.assertEqual(validate_models.parse_validation_errors_response(payload), list(expected))

        def assert_not_parsed(payload: object) -> None:
            if False:
                for i in range(10):
                    print('nop')
            with self.assertRaises(daemon_query.InvalidQueryResponse):
                validate_models.parse_validation_errors_response(payload)
        assert_not_parsed(42)
        assert_not_parsed('derp')
        assert_not_parsed({})
        assert_not_parsed({'no_response': 42})
        assert_not_parsed({'response': 42})
        assert_not_parsed({'response': {'errors': 42}})
        assert_parsed({'response': {}}, expected=[])
        assert_parsed({'response': {'errors': []}}, expected=[])
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root).resolve()
            with setup.switch_working_directory(root_path):
                assert_parsed({'response': {'errors': [{'line': 1, 'column': 1, 'stop_line': 2, 'stop_column': 2, 'path': str(root_path / 'test.py'), 'description': 'Some description', 'code': 1001}, {'line': 3, 'column': 3, 'stop_line': 4, 'stop_column': 4, 'path': None, 'description': 'Some description', 'code': 1001}]}}, expected=[error.ModelVerificationError(line=3, column=3, stop_line=4, stop_column=4, path=None, description='Some description', code=1001), error.ModelVerificationError(line=1, column=1, stop_line=2, stop_column=2, path=Path('test.py'), description='Some description', code=1001)])