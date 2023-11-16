"""Objects relating to sourcing connections from metastore database."""
from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
from sqlalchemy import select
from airflow.api_internal.internal_api_call import internal_api_call
from airflow.exceptions import RemovedInAirflow3Warning
from airflow.secrets import BaseSecretsBackend
from airflow.utils.session import NEW_SESSION, provide_session
if TYPE_CHECKING:
    from sqlalchemy.orm import Session
    from airflow.models.connection import Connection

class MetastoreBackend(BaseSecretsBackend):
    """Retrieves Connection object and Variable from airflow metastore database."""

    @provide_session
    def get_connection(self, conn_id: str, session: Session=NEW_SESSION) -> Connection | None:
        if False:
            for i in range(10):
                print('nop')
        return MetastoreBackend._fetch_connection(conn_id, session=session)

    @provide_session
    def get_connections(self, conn_id: str, session: Session=NEW_SESSION) -> list[Connection]:
        if False:
            return 10
        warnings.warn('This method is deprecated. Please use `airflow.secrets.metastore.MetastoreBackend.get_connection`.', RemovedInAirflow3Warning, stacklevel=3)
        conn = self.get_connection(conn_id=conn_id, session=session)
        if conn:
            return [conn]
        return []

    @provide_session
    def get_variable(self, key: str, session: Session=NEW_SESSION) -> str | None:
        if False:
            print('Hello World!')
        '\n        Get Airflow Variable from Metadata DB.\n\n        :param key: Variable Key\n        :return: Variable Value\n        '
        return MetastoreBackend._fetch_variable(key=key, session=session)

    @staticmethod
    @internal_api_call
    @provide_session
    def _fetch_connection(conn_id: str, session: Session=NEW_SESSION) -> Connection | None:
        if False:
            return 10
        from airflow.models.connection import Connection
        conn = session.scalar(select(Connection).where(Connection.conn_id == conn_id).limit(1))
        session.expunge_all()
        return conn

    @staticmethod
    @internal_api_call
    @provide_session
    def _fetch_variable(key: str, session: Session=NEW_SESSION) -> str | None:
        if False:
            return 10
        from airflow.models.variable import Variable
        var_value = session.scalar(select(Variable).where(Variable.key == key).limit(1))
        session.expunge_all()
        if var_value:
            return var_value.val
        return None