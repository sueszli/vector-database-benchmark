"""log dt

Revision ID: 1d2ddd543133
Revises: d2424a248d63
Create Date: 2016-03-25 14:35:44.642576

"""
revision = '1d2ddd543133'
down_revision = 'd2424a248d63'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        print('Hello World!')
    op.add_column('logs', sa.Column('dt', sa.Date(), nullable=True))

def downgrade():
    if False:
        return 10
    op.drop_column('logs', 'dt')