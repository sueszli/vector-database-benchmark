"""added used plugin to message

Revision ID: 5b4211625a9f
Revises: ea19bbc743f9
Create Date: 2023-05-01 22:53:16.297495

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
revision = '5b4211625a9f'
down_revision = 'ea19bbc743f9'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        return 10
    op.add_column('message', sa.Column('used_plugin', postgresql.JSONB(astext_type=sa.Text()), nullable=True))

def downgrade() -> None:
    if False:
        print('Hello World!')
    op.drop_column('message', 'used_plugin')