import contextlib
import io
import unittest
from unittest.mock import patch
from streamlit.error_util import _print_rich_exception, handle_uncaught_app_exception
from tests import testutil

class ErrorUtilTest(unittest.TestCase):

    @patch('streamlit.exception')
    @patch('streamlit.error')
    def test_uncaught_exception_show_details(self, mock_st_error, mock_st_exception):
        if False:
            i = 10
            return i + 15
        'If client.showErrorDetails is true, uncaught app errors print\n        to the frontend.'
        with testutil.patch_config_options({'client.showErrorDetails': True}):
            exc = RuntimeError('boom!')
            handle_uncaught_app_exception(exc)
            mock_st_error.assert_not_called()
            mock_st_exception.assert_called_once_with(exc)

    @patch('streamlit.exception')
    @patch('streamlit.error')
    def test_uncaught_exception_no_details(self, mock_st_error, mock_st_exception):
        if False:
            i = 10
            return i + 15
        'If client.showErrorDetails is false, uncaught app errors are logged,\n        and a generic error message is printed to the frontend.'
        with testutil.patch_config_options({'client.showErrorDetails': False}):
            exc = RuntimeError('boom!')
            handle_uncaught_app_exception(exc)
            mock_st_error.assert_not_called()
            mock_st_exception.assert_called_once()

    def test_handle_print_rich_exception(self):
        if False:
            return 10
        'Test if the print rich exception method is working fine.'
        with io.StringIO() as buf:
            with contextlib.redirect_stdout(buf):
                _print_rich_exception(Exception('boom!'))
            captured_output = buf.getvalue()
            assert 'Exception:' in captured_output
            assert 'boom!' in captured_output

    def test_handle_uncaught_app_exception_with_rich(self):
        if False:
            i = 10
            return i + 15
        'Test if the exception is logged with rich enabled and disabled.'
        exc = Exception('boom!')
        with testutil.patch_config_options({'logger.enableRich': True}):
            with io.StringIO() as buf:
                with contextlib.redirect_stdout(buf):
                    handle_uncaught_app_exception(exc)
                captured_output = buf.getvalue()
                assert 'Exception:' in captured_output
                assert 'boom!' in captured_output
                assert 'Uncaught app exception' not in captured_output
        with testutil.patch_config_options({'logger.enableRich': False}):
            with io.StringIO() as buf:
                with contextlib.redirect_stdout(buf):
                    handle_uncaught_app_exception(exc)
                captured_output = buf.getvalue()
                assert 'Exception:' not in captured_output
                assert 'boom!' not in captured_output