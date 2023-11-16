import unittest
from unittest.mock import Mock, patch
from streamlit.deprecation_util import deprecate_func_name, deprecate_obj_name, show_deprecation_warning
from tests.testutil import patch_config_options

class DeprecationUtilTest(unittest.TestCase):

    @patch('streamlit.deprecation_util._LOGGER')
    @patch('streamlit.warning')
    def test_show_deprecation_warning(self, mock_warning: Mock, mock_logger: Mock):
        if False:
            i = 10
            return i + 15
        'show_deprecation_warning logs warnings always, and prints to the browser only\n        if config.client.showErrorDetails is True.\n        '
        message = "We regret the bother, but it's been fated:\nthe function you called is DEPRECATED."
        with patch_config_options({'client.showErrorDetails': True}):
            show_deprecation_warning(message)
            mock_logger.warning.assert_called_once_with(message)
            mock_warning.assert_called_once_with(message)
        mock_logger.reset_mock()
        mock_warning.reset_mock()
        with patch_config_options({'client.showErrorDetails': False}):
            show_deprecation_warning(message)
            mock_logger.warning.assert_called_once_with(message)
            mock_warning.assert_not_called()

    @patch('streamlit.deprecation_util.show_deprecation_warning')
    def test_deprecate_func_name(self, mock_show_warning: Mock):
        if False:
            while True:
                i = 10

        def multiply(a, b):
            if False:
                print('Hello World!')
            return a * b
        beta_multiply = deprecate_func_name(multiply, 'beta_multiply', '1980-01-01')
        self.assertEqual(beta_multiply(3, 2), 6)
        expected_warning = 'Please replace `st.beta_multiply` with `st.multiply`.\n\n`st.beta_multiply` will be removed after 1980-01-01.'
        mock_show_warning.assert_called_once_with(expected_warning)

    @patch('streamlit.deprecation_util.show_deprecation_warning')
    def test_deprecate_func_name_with_override(self, mock_show_warning: Mock):
        if False:
            print('Hello World!')

        def multiply(a, b):
            if False:
                i = 10
                return i + 15
            return a * b
        beta_multiply = deprecate_func_name(multiply, 'beta_multiply', '1980-01-01', name_override='mul')
        self.assertEqual(beta_multiply(3, 2), 6)
        expected_warning = 'Please replace `st.beta_multiply` with `st.mul`.\n\n`st.beta_multiply` will be removed after 1980-01-01.'
        mock_show_warning.assert_called_once_with(expected_warning)

    @patch('streamlit.deprecation_util.show_deprecation_warning')
    def test_deprecate_obj_name(self, mock_show_warning: Mock):
        if False:
            return 10
        'Test that we override dunder methods.'

        class DictClass(dict):
            pass
        beta_dict = deprecate_obj_name(DictClass(), 'beta_dict', 'my_dict', '1980-01-01')
        beta_dict['foo'] = 'bar'
        self.assertEqual(beta_dict['foo'], 'bar')
        self.assertEqual(len(beta_dict), 1)
        self.assertEqual(list(beta_dict), ['foo'])
        expected_warning = 'Please replace `st.beta_dict` with `st.my_dict`.\n\n`st.beta_dict` will be removed after 1980-01-01.'
        mock_show_warning.assert_called_once_with(expected_warning)

    @patch('streamlit.deprecation_util.show_deprecation_warning')
    def test_deprecate_obj_name_no_st_prefix(self, mock_show_warning: Mock):
        if False:
            for i in range(10):
                print('nop')

        class DictClass(dict):
            pass
        beta_dict = deprecate_obj_name(DictClass(), 'beta_dict', 'my_dict', '1980-01-01', include_st_prefix=False)
        beta_dict['foo'] = 'bar'
        self.assertEqual(beta_dict['foo'], 'bar')
        self.assertEqual(len(beta_dict), 1)
        self.assertEqual(list(beta_dict), ['foo'])
        expected_warning = 'Please replace `beta_dict` with `my_dict`.\n\n`beta_dict` will be removed after 1980-01-01.'
        mock_show_warning.assert_called_once_with(expected_warning)