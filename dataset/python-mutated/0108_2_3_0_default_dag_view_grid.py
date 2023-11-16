"""Update dag.default_view to grid.

Revision ID: b1b348e02d07
Revises: 75d5ed6c2b43
Create Date: 2022-04-19 17:25:00.872220

"""
from __future__ import annotations
from alembic import op
from sqlalchemy import String
from sqlalchemy.sql import column, table
revision = 'b1b348e02d07'
down_revision = '75d5ed6c2b43'
branch_labels = None
depends_on = '75d5ed6c2b43'
airflow_version = '2.3.0'
dag = table('dag', column('default_view', String))

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.execute(dag.update().where(dag.c.default_view == op.inline_literal('tree')).values({'default_view': op.inline_literal('grid')}))

def downgrade():
    if False:
        return 10
    op.execute(dag.update().where(dag.c.default_view == op.inline_literal('grid')).values({'default_view': op.inline_literal('tree')}))