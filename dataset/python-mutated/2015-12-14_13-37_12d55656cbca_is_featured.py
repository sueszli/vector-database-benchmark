"""is_featured

Revision ID: 12d55656cbca
Revises: 55179c7f25c7
Create Date: 2015-12-14 13:37:17.374852

"""
revision = '12d55656cbca'
down_revision = '55179c7f25c7'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.add_column('tables', sa.Column('is_featured', sa.Boolean(), nullable=True))

def downgrade():
    if False:
        i = 10
        return i + 15
    op.drop_column('tables', 'is_featured')