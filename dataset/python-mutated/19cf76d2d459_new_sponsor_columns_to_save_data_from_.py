"""
New Sponsor columns to save data from pythondotorg API

Revision ID: 19cf76d2d459
Revises: 29a8901a4635
Create Date: 2022-02-13 14:31:18.366248
"""
import sqlalchemy as sa
from alembic import op
revision = '19cf76d2d459'
down_revision = '29a8901a4635'

def upgrade():
    if False:
        while True:
            i = 10
    op.add_column('sponsors', sa.Column('origin', sa.String(), nullable=True))
    op.add_column('sponsors', sa.Column('level_name', sa.String(), nullable=True))
    op.add_column('sponsors', sa.Column('level_order', sa.Integer(), nullable=True))
    op.add_column('sponsors', sa.Column('slug', sa.String(), nullable=True))

def downgrade():
    if False:
        return 10
    op.drop_column('sponsors', 'slug')
    op.drop_column('sponsors', 'level_order')
    op.drop_column('sponsors', 'level_name')
    op.drop_column('sponsors', 'origin')