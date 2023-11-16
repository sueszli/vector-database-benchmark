from __future__ import annotations
import logging
from unittest.mock import patch
import pytest
from airflow.utils.log.colored_log import CustomTTYColoredFormatter
pytestmark = pytest.mark.db_test

@patch('airflow.utils.log.timezone_aware.TimezoneAware.formatTime')
def test_format_time_uses_tz_aware(mock_fmt):
    if False:
        i = 10
        return i + 15
    logger = logging.getLogger('test_format_time')
    h = logging.StreamHandler()
    h.setFormatter(CustomTTYColoredFormatter())
    logger.addHandler(h)
    logger.info('hi')
    mock_fmt.assert_called()