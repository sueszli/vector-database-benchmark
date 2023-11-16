import unittest
from unittest.mock import MagicMock, patch
from ... import errors
from ...repository import Repository
from ..command import ErrorSource, ErrorSuppressingCommand
from ..fixme import Fixme
repository = Repository()

class FixmeTest(unittest.TestCase):

    def test_run(self) -> None:
        if False:
            while True:
                i = 10
        arguments = MagicMock()
        arguments.error_source = ErrorSource.STDIN
        mock_errors = MagicMock()
        with patch.object(errors.Errors, 'from_stdin', return_value=mock_errors) as errors_from_stdin, patch.object(ErrorSuppressingCommand, '_apply_suppressions') as apply_suppressions:
            Fixme.from_arguments(arguments, repository).run()
            errors_from_stdin.assert_called_once()
            apply_suppressions.assert_called_once_with(mock_errors)
        arguments.error_source = ErrorSource.GENERATE
        arguments.lint = False
        with patch.object(Fixme, '_generate_errors', return_value=mock_errors) as generate_errors, patch.object(ErrorSuppressingCommand, '_apply_suppressions') as apply_suppressions:
            Fixme.from_arguments(arguments, repository).run()
            generate_errors.assert_called_once()
            apply_suppressions.assert_called_once_with(mock_errors)