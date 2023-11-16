"""Add preferences to user

Revision ID: 1a737bdee53b
Revises: 847ea6602d6f
Create Date: 2023-04-03 11:03:27.102494

"""
from alembic import op
import sqlalchemy as sa
revision = '1a737bdee53b'
down_revision = '847ea6602d6f'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        while True:
            i = 10
    op.add_column('user', sa.Column('preferences', sa.JSON(), nullable=True))

def downgrade() -> None:
    if False:
        while True:
            i = 10
    op.drop_column('user', 'preferences')