from __future__ import annotations
import pytest
from airflow.utils.module_loading import import_string

class TestModuleImport:

    def test_import_string(self):
        if False:
            return 10
        cls = import_string('airflow.utils.module_loading.import_string')
        assert cls == import_string
        with pytest.raises(ImportError):
            import_string('no_dots_in_path')
        msg = 'Module "airflow.utils" does not define a "nonexistent" attribute'
        with pytest.raises(ImportError, match=msg):
            import_string('airflow.utils.nonexistent')