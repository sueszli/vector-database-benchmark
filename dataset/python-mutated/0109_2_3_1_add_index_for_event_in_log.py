"""Add index for ``event`` column in ``log`` table.

Revision ID: 1de7bc13c950
Revises: b1b348e02d07
Create Date: 2022-05-10 18:18:43.484829

"""
from __future__ import annotations
from alembic import op
revision = '1de7bc13c950'
down_revision = 'b1b348e02d07'
branch_labels = None
depends_on = None
airflow_version = '2.3.1'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    'Apply Add index for ``event`` column in ``log`` table.'
    op.create_index('idx_log_event', 'log', ['event'], unique=False)

def downgrade():
    if False:
        while True:
            i = 10
    'Unapply Add index for ``event`` column in ``log`` table.'
    op.drop_index('idx_log_event', table_name='log')