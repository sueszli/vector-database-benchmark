"""query.start_running_time

Revision ID: a6c18f869a4e
Revises: 979c03af3341
Create Date: 2017-03-28 11:28:41.387182

"""
import sqlalchemy as sa
from alembic import op
revision = 'a6c18f869a4e'
down_revision = '979c03af3341'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.add_column('query', sa.Column('start_running_time', sa.Numeric(precision=20, scale=6), nullable=True))

def downgrade():
    if False:
        i = 10
        return i + 15
    op.drop_column('query', 'start_running_time')