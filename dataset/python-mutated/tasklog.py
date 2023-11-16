from __future__ import annotations
from sqlalchemy import Column, Integer, Text
from airflow.models.base import Base
from airflow.utils import timezone
from airflow.utils.sqlalchemy import UtcDateTime

class LogTemplate(Base):
    """Changes to ``log_filename_template`` and ``elasticsearch_id``.

    This table is automatically populated when Airflow starts up, to store the
    config's value if it does not match the last row in the table.
    """
    __tablename__ = 'log_template'
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(Text, nullable=False)
    elasticsearch_id = Column(Text, nullable=False)
    created_at = Column(UtcDateTime, nullable=False, default=timezone.utcnow)

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        attrs = ', '.join((f'{k}={getattr(self, k)}' for k in ('filename', 'elasticsearch_id')))
        return f'LogTemplate({attrs})'