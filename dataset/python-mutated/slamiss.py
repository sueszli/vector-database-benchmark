from __future__ import annotations
from sqlalchemy import Boolean, Column, Index, String, Text
from airflow.models.base import COLLATION_ARGS, ID_LEN, Base
from airflow.utils.sqlalchemy import UtcDateTime

class SlaMiss(Base):
    """
    Model that stores a history of the SLA that have been missed.

    It is used to keep track of SLA failures over time and to avoid double triggering alert emails.
    """
    __tablename__ = 'sla_miss'
    task_id = Column(String(ID_LEN, **COLLATION_ARGS), primary_key=True)
    dag_id = Column(String(ID_LEN, **COLLATION_ARGS), primary_key=True)
    execution_date = Column(UtcDateTime, primary_key=True)
    email_sent = Column(Boolean, default=False)
    timestamp = Column(UtcDateTime)
    description = Column(Text)
    notification_sent = Column(Boolean, default=False)
    __table_args__ = (Index('sm_dag', dag_id, unique=False),)

    def __repr__(self):
        if False:
            print('Hello World!')
        return str((self.dag_id, self.task_id, self.execution_date.isoformat()))