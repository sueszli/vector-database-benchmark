"""Add queued by Job ID to TI

Revision ID: b247b1e3d1ed
Revises: e38be357a868
Create Date: 2020-09-04 11:53:00.978882

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = 'b247b1e3d1ed'
down_revision = 'e38be357a868'
branch_labels = None
depends_on = None
airflow_version = '2.0.0'

def upgrade():
    if False:
        print('Hello World!')
    'Apply Add queued by Job ID to TI'
    with op.batch_alter_table('task_instance') as batch_op:
        batch_op.add_column(sa.Column('queued_by_job_id', sa.Integer(), nullable=True))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    'Unapply Add queued by Job ID to TI'
    with op.batch_alter_table('task_instance') as batch_op:
        batch_op.drop_column('queued_by_job_id')