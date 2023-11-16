"""add pause_reason column to workers table

Revision ID: 064
Revises: 063

"""
import sqlalchemy as sa
from alembic import op
revision = '064'
down_revision = '063'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.add_column('workers', sa.Column('pause_reason', sa.Text, nullable=True))

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_column('workers', 'pause_reason')