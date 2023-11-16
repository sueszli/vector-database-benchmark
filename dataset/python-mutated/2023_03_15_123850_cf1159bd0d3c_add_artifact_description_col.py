"""Add description col to artifact table

Revision ID: cf1159bd0d3c
Revises: f3df94dca3cc
Create Date: 2023-03-15 12:38:50.049225

"""
import sqlalchemy as sa
from alembic import op
revision = 'cf1159bd0d3c'
down_revision = 'f3df94dca3cc'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    op.execute('PRAGMA foreign_keys=OFF')
    with op.batch_alter_table('artifact', schema=None) as batch_op:
        batch_op.add_column(sa.Column('description', sa.String(), nullable=True))
    op.execute('PRAGMA foreign_keys=ON')

def downgrade():
    if False:
        print('Hello World!')
    op.execute('PRAGMA foreign_keys=OFF')
    with op.batch_alter_table('artifact', schema=None) as batch_op:
        batch_op.drop_column('description')
    op.execute('PRAGMA foreign_keys=ON')