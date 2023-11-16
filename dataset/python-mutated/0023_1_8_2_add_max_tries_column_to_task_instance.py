"""Add ``max_tries`` column to ``task_instance``

Revision ID: cc1e65623dc7
Revises: 127d2bf2dfa7
Create Date: 2017-06-19 16:53:12.851141

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from sqlalchemy import Column, Integer, String, inspect
from sqlalchemy.orm import declarative_base
from airflow import settings
from airflow.models import DagBag
revision = 'cc1e65623dc7'
down_revision = '127d2bf2dfa7'
branch_labels = None
depends_on = None
airflow_version = '1.8.2'
Base = declarative_base()
BATCH_SIZE = 5000

class TaskInstance(Base):
    """Task Instance class."""
    __tablename__ = 'task_instance'
    task_id = Column(String(), primary_key=True)
    dag_id = Column(String(), primary_key=True)
    execution_date = Column(sa.DateTime, primary_key=True)
    max_tries = Column(Integer)
    try_number = Column(Integer, default=0)

def upgrade():
    if False:
        return 10
    op.add_column('task_instance', sa.Column('max_tries', sa.Integer, server_default='-1'))
    connection = op.get_bind()
    inspector = inspect(connection)
    tables = inspector.get_table_names()
    if 'task_instance' in tables:
        sessionmaker = sa.orm.sessionmaker()
        session = sessionmaker(bind=connection)
        if not bool(session.query(TaskInstance).first()):
            return
        dagbag = DagBag(settings.DAGS_FOLDER)
        query = session.query(sa.func.count(TaskInstance.max_tries)).filter(TaskInstance.max_tries == -1)
        while query.scalar():
            tis = session.query(TaskInstance).filter(TaskInstance.max_tries == -1).limit(BATCH_SIZE).all()
            for ti in tis:
                dag = dagbag.get_dag(ti.dag_id)
                if not dag or not dag.has_task(ti.task_id):
                    ti.max_tries = ti.try_number
                else:
                    task = dag.get_task(ti.task_id)
                    if task.retries:
                        ti.max_tries = task.retries
                    else:
                        ti.max_tries = ti.try_number
                session.merge(ti)
            session.commit()
        session.commit()

def downgrade():
    if False:
        print('Hello World!')
    engine = settings.engine
    connection = op.get_bind()
    if engine.dialect.has_table(connection, 'task_instance'):
        sessionmaker = sa.orm.sessionmaker()
        session = sessionmaker(bind=connection)
        dagbag = DagBag(settings.DAGS_FOLDER)
        query = session.query(sa.func.count(TaskInstance.max_tries)).filter(TaskInstance.max_tries != -1)
        while query.scalar():
            tis = session.query(TaskInstance).filter(TaskInstance.max_tries != -1).limit(BATCH_SIZE).all()
            for ti in tis:
                dag = dagbag.get_dag(ti.dag_id)
                if not dag or not dag.has_task(ti.task_id):
                    ti.try_number = 0
                else:
                    task = dag.get_task(ti.task_id)
                    ti.try_number = max(0, task.retries - (ti.max_tries - ti.try_number))
                ti.max_tries = -1
                session.merge(ti)
            session.commit()
        session.commit()
    op.drop_column('task_instance', 'max_tries')