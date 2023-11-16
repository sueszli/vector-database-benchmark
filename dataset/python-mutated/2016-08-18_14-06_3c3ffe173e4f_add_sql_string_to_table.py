"""add_sql_string_to_table

Revision ID: 3c3ffe173e4f
Revises: ad82a75afd82
Create Date: 2016-08-18 14:06:28.784699

"""
revision = '3c3ffe173e4f'
down_revision = 'ad82a75afd82'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        while True:
            i = 10
    op.add_column('tables', sa.Column('sql', sa.Text(), nullable=True))

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_column('tables', 'sql')