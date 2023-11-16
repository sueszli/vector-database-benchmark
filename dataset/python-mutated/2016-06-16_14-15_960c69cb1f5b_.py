"""add dttm_format related fields in table_columns

Revision ID: 960c69cb1f5b
Revises: d8bc074f7aad
Create Date: 2016-06-16 14:15:19.573183

"""
revision = '960c69cb1f5b'
down_revision = '27ae655e4247'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.add_column('table_columns', sa.Column('python_date_format', sa.String(length=255), nullable=True))
    op.add_column('table_columns', sa.Column('database_expression', sa.String(length=255), nullable=True))

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_column('table_columns', 'python_date_format')
    op.drop_column('table_columns', 'database_expression')