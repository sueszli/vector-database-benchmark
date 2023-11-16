"""Remove path, path_no_int, and ref from logs

Revision ID: 811494c0cc23
Revises: 8ee129739cf9
Create Date: 2020-12-03 16:21:06.771684

"""
revision = '811494c0cc23'
down_revision = '8ee129739cf9'
import sqlalchemy as sa
from alembic import op
from superset.migrations.shared import utils

def upgrade():
    if False:
        print('Hello World!')
    with op.batch_alter_table('logs') as batch_op:
        if utils.table_has_column('logs', 'path'):
            batch_op.drop_column('path')
        if utils.table_has_column('logs', 'path_no_int'):
            batch_op.drop_column('path_no_int')
        if utils.table_has_column('logs', 'ref'):
            batch_op.drop_column('ref')

def downgrade():
    if False:
        print('Hello World!')
    pass