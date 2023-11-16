"""log_this_plus

Revision ID: 525c854f0005
Revises: e46f2d27a08e
Create Date: 2016-12-13 16:19:02.239322

"""
revision = '525c854f0005'
down_revision = 'e46f2d27a08e'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        i = 10
        return i + 15
    op.add_column('logs', sa.Column('duration_ms', sa.Integer(), nullable=True))
    op.add_column('logs', sa.Column('referrer', sa.String(length=1024), nullable=True))

def downgrade():
    if False:
        return 10
    op.drop_column('logs', 'referrer')
    op.drop_column('logs', 'duration_ms')