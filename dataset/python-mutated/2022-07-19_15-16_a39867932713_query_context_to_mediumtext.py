"""query_context_to_mediumtext

Revision ID: a39867932713
Revises: 06e1e70058c7
Create Date: 2022-07-19 15:16:06.091961

"""
from alembic import op
from sqlalchemy.dialects.mysql.base import MySQLDialect
revision = 'a39867932713'
down_revision = '06e1e70058c7'

def upgrade():
    if False:
        print('Hello World!')
    if isinstance(op.get_bind().dialect, MySQLDialect):
        op.execute('ALTER TABLE slices MODIFY params MEDIUMTEXT')
        op.execute('ALTER TABLE slices MODIFY query_context MEDIUMTEXT')

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    pass