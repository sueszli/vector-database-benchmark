from __future__ import annotations
import re
from unittest import mock
import pytest
from airflow.www.extensions import init_views
from tests.test_utils.config import conf_vars
pytestmark = pytest.mark.db_test

class TestInitApiExperimental:

    @conf_vars({('api', 'enable_experimental_api'): 'true'})
    def test_should_raise_deprecation_warning_when_enabled(self):
        if False:
            return 10
        app = mock.MagicMock()
        with pytest.warns(DeprecationWarning, match=re.escape('The experimental REST API is deprecated.')):
            init_views.init_api_experimental(app)

    @conf_vars({('api', 'enable_experimental_api'): 'false'})
    def test_should_not_raise_deprecation_warning_when_disabled(self):
        if False:
            i = 10
            return i + 15
        app = mock.MagicMock()
        with pytest.warns(None) as warnings:
            init_views.init_api_experimental(app)
        assert len(warnings) == 0