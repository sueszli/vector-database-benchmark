"""empty message

Revision ID: 1e2841a4128
Revises: 5a7bad26f2a7
Create Date: 2015-10-05 22:11:00.537054

"""
revision = '1e2841a4128'
down_revision = '5a7bad26f2a7'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.add_column('table_columns', sa.Column('expression', sa.Text(), nullable=True))

def downgrade():
    if False:
        return 10
    op.drop_column('table_columns', 'expression')