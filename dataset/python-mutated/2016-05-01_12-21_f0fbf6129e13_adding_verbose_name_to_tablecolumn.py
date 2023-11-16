"""Adding verbose_name to tablecolumn

Revision ID: f0fbf6129e13
Revises: c3a8f8611885
Create Date: 2016-05-01 12:21:18.331191

"""
revision = 'f0fbf6129e13'
down_revision = 'c3a8f8611885'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        return 10
    op.add_column('table_columns', sa.Column('verbose_name', sa.String(length=1024), nullable=True))

def downgrade():
    if False:
        print('Hello World!')
    op.drop_column('table_columns', 'verbose_name')