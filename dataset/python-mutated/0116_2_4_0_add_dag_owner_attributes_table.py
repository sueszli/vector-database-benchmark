"""add dag_owner_attributes table

Revision ID: 1486deb605b4
Revises: f4ff391becb5
Create Date: 2022-08-04 16:59:45.406589

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from airflow.migrations.db_types import StringID
revision = '1486deb605b4'
down_revision = 'f4ff391becb5'
branch_labels = None
depends_on = None
airflow_version = '2.4.0'

def upgrade():
    if False:
        while True:
            i = 10
    'Apply Add ``DagOwnerAttributes`` table'
    op.create_table('dag_owner_attributes', sa.Column('dag_id', StringID(), nullable=False), sa.Column('owner', sa.String(length=500), nullable=False), sa.Column('link', sa.String(length=500), nullable=False), sa.ForeignKeyConstraint(['dag_id'], ['dag.dag_id'], ondelete='CASCADE'), sa.PrimaryKeyConstraint('dag_id', 'owner'))

def downgrade():
    if False:
        print('Hello World!')
    'Unapply Add Dataset model'
    op.drop_table('dag_owner_attributes')