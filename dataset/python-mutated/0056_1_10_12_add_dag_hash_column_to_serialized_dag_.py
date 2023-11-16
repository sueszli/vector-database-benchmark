"""Add ``dag_hash`` Column to ``serialized_dag`` table

Revision ID: da3f683c3a5a
Revises: a66efa278eea
Create Date: 2020-08-07 20:52:09.178296

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = 'da3f683c3a5a'
down_revision = 'a66efa278eea'
branch_labels = None
depends_on = None
airflow_version = '1.10.12'

def upgrade():
    if False:
        print('Hello World!')
    'Apply Add ``dag_hash`` Column to ``serialized_dag`` table'
    op.add_column('serialized_dag', sa.Column('dag_hash', sa.String(32), nullable=False, server_default='Hash not calculated yet'))

def downgrade():
    if False:
        print('Hello World!')
    'Unapply Add ``dag_hash`` Column to ``serialized_dag`` table'
    op.drop_column('serialized_dag', 'dag_hash')