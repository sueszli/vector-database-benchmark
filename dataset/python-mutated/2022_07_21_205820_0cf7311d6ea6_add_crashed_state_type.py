"""Add Crashed state type

Revision ID: 0cf7311d6ea6
Revises: bb4dc90d3e29
Create Date: 2022-07-21 20:58:20.807489

"""
from alembic import op
revision = '0cf7311d6ea6'
down_revision = 'bb4dc90d3e29'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    op.execute("ALTER TYPE state_type ADD VALUE IF NOT EXISTS 'CRASHED';")

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    pass