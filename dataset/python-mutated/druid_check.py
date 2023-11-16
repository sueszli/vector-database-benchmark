from __future__ import annotations
import warnings
from airflow.exceptions import AirflowProviderDeprecationWarning
from airflow.providers.common.sql.operators.sql import SQLCheckOperator

class DruidCheckOperator(SQLCheckOperator):
    """
    This class is deprecated.

    Please use :class:`airflow.providers.common.sql.operators.sql.SQLCheckOperator`.
    """

    def __init__(self, druid_broker_conn_id: str='druid_broker_default', **kwargs):
        if False:
            i = 10
            return i + 15
        warnings.warn('This class is deprecated.\n            Please use `airflow.providers.common.sql.operators.sql.SQLCheckOperator`.', AirflowProviderDeprecationWarning, stacklevel=2)
        super().__init__(conn_id=druid_broker_conn_id, **kwargs)