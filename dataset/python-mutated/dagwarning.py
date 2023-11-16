from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING
from sqlalchemy import Column, ForeignKeyConstraint, String, Text, delete, false, select
from airflow.api_internal.internal_api_call import internal_api_call
from airflow.models.base import Base, StringID
from airflow.utils import timezone
from airflow.utils.retries import retry_db_transaction
from airflow.utils.session import NEW_SESSION, provide_session
from airflow.utils.sqlalchemy import UtcDateTime
if TYPE_CHECKING:
    from sqlalchemy.orm import Session

class DagWarning(Base):
    """
    A table to store DAG warnings.

    DAG warnings are problems that don't rise to the level of failing the DAG parse
    but which users should nonetheless be warned about.  These warnings are recorded
    when parsing DAG and displayed on the Webserver in a flash message.
    """
    dag_id = Column(StringID(), primary_key=True)
    warning_type = Column(String(50), primary_key=True)
    message = Column(Text, nullable=False)
    timestamp = Column(UtcDateTime, nullable=False, default=timezone.utcnow)
    __tablename__ = 'dag_warning'
    __table_args__ = (ForeignKeyConstraint(('dag_id',), ['dag.dag_id'], name='dcw_dag_id_fkey', ondelete='CASCADE'),)

    def __init__(self, dag_id: str, error_type: str, message: str, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.dag_id = dag_id
        self.warning_type = DagWarningType(error_type).value
        self.message = message

    def __eq__(self, other) -> bool:
        if False:
            while True:
                i = 10
        return self.dag_id == other.dag_id and self.warning_type == other.warning_type

    def __hash__(self) -> int:
        if False:
            while True:
                i = 10
        return hash((self.dag_id, self.warning_type))

    @classmethod
    @internal_api_call
    @provide_session
    def purge_inactive_dag_warnings(cls, session: Session=NEW_SESSION) -> None:
        if False:
            print('Hello World!')
        '\n        Deactivate DagWarning records for inactive dags.\n\n        :return: None\n        '
        cls._purge_inactive_dag_warnings_with_retry(session)

    @classmethod
    @retry_db_transaction
    def _purge_inactive_dag_warnings_with_retry(cls, session: Session) -> None:
        if False:
            print('Hello World!')
        from airflow.models.dag import DagModel
        if session.get_bind().dialect.name == 'sqlite':
            dag_ids_stmt = select(DagModel.dag_id).where(DagModel.is_active == false())
            query = delete(cls).where(cls.dag_id.in_(dag_ids_stmt.scalar_subquery()))
        else:
            query = delete(cls).where(cls.dag_id == DagModel.dag_id, DagModel.is_active == false())
        session.execute(query.execution_options(synchronize_session=False))
        session.commit()

class DagWarningType(str, Enum):
    """
    Enum for DAG warning types.

    This is the set of allowable values for the ``warning_type`` field
    in the DagWarning model.
    """
    NONEXISTENT_POOL = 'non-existent pool'