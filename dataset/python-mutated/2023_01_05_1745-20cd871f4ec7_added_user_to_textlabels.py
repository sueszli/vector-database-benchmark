"""Added user to TextLabels

Revision ID: 20cd871f4ec7
Revises: d4161e384f83
Create Date: 2023-01-05 17:45:15.696468

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
revision = '20cd871f4ec7'
down_revision = '3b0adfadbef9'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        return 10
    op.add_column('text_labels', sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False))
    op.create_foreign_key(None, 'text_labels', 'user', ['user_id'], ['id'])

def downgrade() -> None:
    if False:
        return 10
    op.drop_constraint(None, 'text_labels', type_='foreignkey')
    op.drop_column('text_labels', 'user_id')