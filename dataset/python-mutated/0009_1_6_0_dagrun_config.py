"""Add ``conf`` column in ``dag_run`` table

Revision ID: 40e67319e3a9
Revises: 2e541a1dcfed
Create Date: 2015-10-29 08:36:31.726728

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = '40e67319e3a9'
down_revision = '2e541a1dcfed'
branch_labels = None
depends_on = None
airflow_version = '1.6.0'

def upgrade():
    if False:
        print('Hello World!')
    op.add_column('dag_run', sa.Column('conf', sa.PickleType(), nullable=True))

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_column('dag_run', 'conf')