"""Objects relating to sourcing connections from environment variables."""
from __future__ import annotations
import os
import warnings
from airflow.exceptions import RemovedInAirflow3Warning
from airflow.secrets import BaseSecretsBackend
CONN_ENV_PREFIX = 'AIRFLOW_CONN_'
VAR_ENV_PREFIX = 'AIRFLOW_VAR_'

class EnvironmentVariablesBackend(BaseSecretsBackend):
    """Retrieves Connection object and Variable from environment variable."""

    def get_conn_uri(self, conn_id: str) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Return URI representation of Connection conn_id.\n\n        :param conn_id: the connection id\n\n        :return: deserialized Connection\n        '
        warnings.warn('This method is deprecated. Please use `airflow.secrets.environment_variables.EnvironmentVariablesBackend.get_conn_value`.', RemovedInAirflow3Warning, stacklevel=2)
        return self.get_conn_value(conn_id)

    def get_conn_value(self, conn_id: str) -> str | None:
        if False:
            for i in range(10):
                print('nop')
        return os.environ.get(CONN_ENV_PREFIX + conn_id.upper())

    def get_variable(self, key: str) -> str | None:
        if False:
            print('Hello World!')
        '\n        Get Airflow Variable from Environment Variable.\n\n        :param key: Variable Key\n        :return: Variable Value\n        '
        return os.environ.get(VAR_ENV_PREFIX + key.upper())