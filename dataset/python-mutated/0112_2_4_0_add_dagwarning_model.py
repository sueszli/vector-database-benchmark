"""Add DagWarning model

Revision ID: 424117c37d18
Revises: 3c94c427fdf6
Create Date: 2022-04-27 15:57:36.736743
"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from airflow.migrations.db_types import TIMESTAMP, StringID
revision = '424117c37d18'
down_revision = 'f5fcbda3e651'
branch_labels = None
depends_on = None
airflow_version = '2.4.0'

def upgrade():
    if False:
        while True:
            i = 10
    'Apply Add DagWarning model'
    op.create_table('dag_warning', sa.Column('dag_id', StringID(), primary_key=True), sa.Column('warning_type', sa.String(length=50), primary_key=True), sa.Column('message', sa.Text(), nullable=False), sa.Column('timestamp', TIMESTAMP, nullable=False), sa.ForeignKeyConstraint(('dag_id',), ['dag.dag_id'], name='dcw_dag_id_fkey', ondelete='CASCADE'))

def downgrade():
    if False:
        i = 10
        return i + 15
    'Unapply Add DagWarning model'
    op.drop_table('dag_warning')