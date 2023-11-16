"""Created date

Revision ID: aac6b2f66006
Revises: 35bdc1a08bb8
Create Date: 2023-01-08 21:28:27.342729

"""
import sqlalchemy as sa
from alembic import op
revision = 'aac6b2f66006'
down_revision = '35bdc1a08bb8'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        return 10
    op.add_column('message_embedding', sa.Column('created_date', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP'), nullable=False))

def downgrade() -> None:
    if False:
        i = 10
        return i + 15
    op.drop_column('message_embedding', 'created_date')