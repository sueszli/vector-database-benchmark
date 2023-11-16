"""add exec info to saved queries

Revision ID: 585b0b1a7b18
Revises: af30ca79208f
Create Date: 2020-10-20 17:28:22.857694

"""
revision = '585b0b1a7b18'
down_revision = 'af30ca79208f'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        return 10
    op.add_column('saved_query', sa.Column('last_run', sa.DateTime(), nullable=True))
    op.add_column('saved_query', sa.Column('rows', sa.Integer(), nullable=True))

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_column('saved_query', 'rows')
    op.drop_column('saved_query', 'last_run')