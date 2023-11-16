"""Add path to logs

Revision ID: a8173232b786
Revises: 49b5a32daba5
Create Date: 2020-11-15 16:08:24.580764

"""
revision = 'a8173232b786'
down_revision = '49b5a32daba5'
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import mysql
from superset.migrations.shared import utils

def upgrade():
    if False:
        return 10
    pass

def downgrade():
    if False:
        while True:
            i = 10
    with op.batch_alter_table('logs') as batch_op:
        if utils.table_has_column('logs', 'path'):
            batch_op.drop_column('path')
        if utils.table_has_column('logs', 'path_no_int'):
            batch_op.drop_column('path_no_int')
        if utils.table_has_column('logs', 'ref'):
            batch_op.drop_column('ref')