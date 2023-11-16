"""empty message

Revision ID: 43df8de3a5f4
Revises: 7dbf98566af7
Create Date: 2016-01-18 23:43:16.073483

"""
revision = '43df8de3a5f4'
down_revision = '7dbf98566af7'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.add_column('dashboards', sa.Column('json_metadata', sa.Text(), nullable=True))

def downgrade():
    if False:
        i = 10
        return i + 15
    op.drop_column('dashboards', 'json_metadata')