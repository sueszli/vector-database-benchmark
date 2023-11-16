"""Change ``task_instance.task_duration`` type to ``FLOAT``

Revision ID: 2e541a1dcfed
Revises: 1b38cef5b76e
Create Date: 2015-10-28 20:38:41.266143

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql
revision = '2e541a1dcfed'
down_revision = '1b38cef5b76e'
branch_labels = None
depends_on = None
airflow_version = '1.6.0'

def upgrade():
    if False:
        i = 10
        return i + 15
    with op.batch_alter_table('task_instance') as batch_op:
        batch_op.alter_column('duration', existing_type=mysql.INTEGER(display_width=11), type_=sa.Float(), existing_nullable=True)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    pass