"""database options for sql lab

Revision ID: 41f6a59a61f2
Revises: 3c3ffe173e4f
Create Date: 2016-08-31 10:26:37.969107

"""
import sqlalchemy as sa
from alembic import op
revision = '41f6a59a61f2'
down_revision = '3c3ffe173e4f'

def upgrade():
    if False:
        print('Hello World!')
    op.add_column('dbs', sa.Column('allow_ctas', sa.Boolean(), nullable=True))
    op.add_column('dbs', sa.Column('expose_in_sqllab', sa.Boolean(), nullable=True))
    op.add_column('dbs', sa.Column('force_ctas_schema', sa.String(length=250), nullable=True))

def downgrade():
    if False:
        i = 10
        return i + 15
    op.drop_column('dbs', 'force_ctas_schema')
    op.drop_column('dbs', 'expose_in_sqllab')
    op.drop_column('dbs', 'allow_ctas')