"""Add edited_time to Comments

Revision ID: cf7bf6d0e6bd
Revises: 500117641608
Create Date: 2017-10-28 15:32:12.687378

"""
from alembic import op
import sqlalchemy as sa
revision = 'cf7bf6d0e6bd'
down_revision = '500117641608'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    op.add_column('nyaa_comments', sa.Column('edited_time', sa.DateTime(), nullable=True))
    op.add_column('sukebei_comments', sa.Column('edited_time', sa.DateTime(), nullable=True))

def downgrade():
    if False:
        i = 10
        return i + 15
    op.drop_column('sukebei_comments', 'edited_time')
    op.drop_column('nyaa_comments', 'edited_time')