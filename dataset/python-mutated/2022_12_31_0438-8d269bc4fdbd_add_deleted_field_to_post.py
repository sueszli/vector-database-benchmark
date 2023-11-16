"""add deleted field to post

Revision ID: 8d269bc4fdbd
Revises: abb47e9d145a
Create Date: 2022-12-31 04:38:41.799206

"""
import sqlalchemy as sa
from alembic import op
revision = '8d269bc4fdbd'
down_revision = 'abb47e9d145a'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        i = 10
        return i + 15
    op.add_column('message', sa.Column('deleted', sa.Boolean(), server_default=sa.text('false'), nullable=False))

def downgrade() -> None:
    if False:
        while True:
            i = 10
    op.drop_column('message', 'deleted')