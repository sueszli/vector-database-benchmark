"""Add ``DagTags`` table

Revision ID: 7939bcff74ba
Revises: fe461863935f
Create Date: 2020-01-07 19:39:01.247442

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from airflow.migrations.db_types import StringID
revision = '7939bcff74ba'
down_revision = 'fe461863935f'
branch_labels = None
depends_on = None
airflow_version = '1.10.8'

def upgrade():
    if False:
        print('Hello World!')
    'Apply Add ``DagTags`` table'
    op.create_table('dag_tag', sa.Column('name', sa.String(length=100), nullable=False), sa.Column('dag_id', StringID(), nullable=False), sa.ForeignKeyConstraint(['dag_id'], ['dag.dag_id']), sa.PrimaryKeyConstraint('name', 'dag_id'))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    'Unapply Add ``DagTags`` table'
    op.drop_table('dag_tag')