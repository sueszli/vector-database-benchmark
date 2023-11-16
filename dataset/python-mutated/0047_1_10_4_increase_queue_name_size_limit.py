"""Increase queue name size limit

Revision ID: 004c1210f153
Revises: 939bb1e647c8
Create Date: 2019-06-07 07:46:04.262275

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = '004c1210f153'
down_revision = '939bb1e647c8'
branch_labels = None
depends_on = None
airflow_version = '1.10.4'

def upgrade():
    if False:
        return 10
    '\n    Increase column size from 50 to 256 characters, closing AIRFLOW-4737 caused\n    by broker backends that might use unusually large queue names.\n    '
    with op.batch_alter_table('task_instance') as batch_op:
        batch_op.alter_column('queue', type_=sa.String(256))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    'Revert column size from 256 to 50 characters, might result in data loss.'
    with op.batch_alter_table('task_instance') as batch_op:
        batch_op.alter_column('queue', type_=sa.String(50))