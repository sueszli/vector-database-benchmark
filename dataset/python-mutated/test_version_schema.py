from __future__ import annotations
import pytest
from airflow.api_connexion.endpoints.version_endpoint import VersionInfo
from airflow.api_connexion.schemas.version_schema import version_info_schema

class TestVersionInfoSchema:

    @pytest.mark.parametrize('git_commit', ['GIT_COMMIT', None])
    def test_serialize(self, git_commit):
        if False:
            return 10
        version_info = VersionInfo('VERSION', git_commit)
        current_data = version_info_schema.dump(version_info)
        expected_result = {'version': 'VERSION', 'git_version': git_commit}
        assert expected_result == current_data