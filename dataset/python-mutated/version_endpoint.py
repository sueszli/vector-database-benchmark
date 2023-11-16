from __future__ import annotations
from typing import TYPE_CHECKING, NamedTuple
import airflow
from airflow.api_connexion.schemas.version_schema import version_info_schema
from airflow.utils.platform import get_airflow_git_version
if TYPE_CHECKING:
    from airflow.api_connexion.types import APIResponse

class VersionInfo(NamedTuple):
    """Version information."""
    version: str
    git_version: str | None

def get_version() -> APIResponse:
    if False:
        print('Hello World!')
    'Get version information.'
    airflow_version = airflow.__version__
    git_version = get_airflow_git_version()
    version_info = VersionInfo(version=airflow_version, git_version=git_version)
    return version_info_schema.dump(version_info)