"""sqla_descr

Revision ID: 55179c7f25c7
Revises: 315b3f4da9b0
Create Date: 2015-12-13 08:38:43.704145

"""
revision = '55179c7f25c7'
down_revision = '315b3f4da9b0'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        i = 10
        return i + 15
    op.add_column('tables', sa.Column('description', sa.Text(), nullable=True))

def downgrade():
    if False:
        return 10
    op.drop_column('tables', 'description')