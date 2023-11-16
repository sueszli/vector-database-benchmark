"""materialize perms

Revision ID: e46f2d27a08e
Revises: c611f2b591b8
Create Date: 2016-11-14 15:23:32.594898

"""
revision = 'e46f2d27a08e'
down_revision = 'c611f2b591b8'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        i = 10
        return i + 15
    op.add_column('datasources', sa.Column('perm', sa.String(length=1000), nullable=True))
    op.add_column('dbs', sa.Column('perm', sa.String(length=1000), nullable=True))
    op.add_column('tables', sa.Column('perm', sa.String(length=1000), nullable=True))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_column('tables', 'perm')
    op.drop_column('datasources', 'perm')
    op.drop_column('dbs', 'perm')