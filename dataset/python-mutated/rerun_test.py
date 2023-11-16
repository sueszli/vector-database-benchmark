from unittest.mock import Mock, patch
import streamlit as st

@patch('streamlit.commands.execution_control._LOGGER.warning')
def test_deprecation_warnings(logger_mock: Mock):
    if False:
        print('Hello World!')
    st.experimental_rerun()
    logger_mock.assert_called_once()
    msg = logger_mock.call_args.args[0]
    assert 'will be removed' in msg
    logger_mock.reset_mock()
    st.rerun()
    logger_mock.assert_not_called()