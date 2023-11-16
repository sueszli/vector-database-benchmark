"""add filter colume in webhooks

Revision ID: 40affbf3022b
Revises: 5d5f801f28e7
Create Date: 2023-08-28 12:30:35.171176

"""
from alembic import op
import sqlalchemy as sa
revision = '40affbf3022b'
down_revision = '5d5f801f28e7'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        i = 10
        return i + 15
    op.add_column('webhooks', sa.Column('filters', sa.JSON(), nullable=True))

def downgrade() -> None:
    if False:
        return 10
    op.drop_column('webhooks', 'filters')