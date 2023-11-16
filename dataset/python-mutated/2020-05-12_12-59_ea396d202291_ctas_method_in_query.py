"""Add ctas_method to the Query object

Revision ID: ea396d202291
Revises: e557699a813e
Create Date: 2020-05-12 12:59:26.583276

"""
revision = 'ea396d202291'
down_revision = 'e557699a813e'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        print('Hello World!')
    op.add_column('query', sa.Column('ctas_method', sa.String(length=16), nullable=True))
    op.add_column('dbs', sa.Column('allow_cvas', sa.Boolean(), nullable=True))

def downgrade():
    if False:
        return 10
    op.drop_column('query', 'ctas_method')
    op.drop_column('dbs', 'allow_cvas')