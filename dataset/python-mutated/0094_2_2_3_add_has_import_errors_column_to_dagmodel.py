"""Add has_import_errors column to DagModel

Revision ID: be2bfac3da23
Revises: 7b2661a43ba3
Create Date: 2021-11-04 20:33:11.009547

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = 'be2bfac3da23'
down_revision = '7b2661a43ba3'
branch_labels = None
depends_on = None
airflow_version = '2.2.3'

def upgrade():
    if False:
        i = 10
        return i + 15
    'Apply Add has_import_errors column to DagModel'
    op.add_column('dag', sa.Column('has_import_errors', sa.Boolean(), server_default='0'))

def downgrade():
    if False:
        while True:
            i = 10
    'Unapply Add has_import_errors column to DagModel'
    with op.batch_alter_table('dag') as batch_op:
        batch_op.drop_column('has_import_errors', mssql_drop_default=True)