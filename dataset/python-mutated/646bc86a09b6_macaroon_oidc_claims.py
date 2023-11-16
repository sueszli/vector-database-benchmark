"""
macaroon oidc claims

Revision ID: 646bc86a09b6
Revises: 60e6b0dd0f47
Create Date: 2023-06-01 16:50:32.765849
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
revision = '646bc86a09b6'
down_revision = '60e6b0dd0f47'

def upgrade():
    if False:
        i = 10
        return i + 15
    op.add_column('macaroons', sa.Column('additional', postgresql.JSONB(astext_type=sa.Text()), nullable=True))

def downgrade():
    if False:
        print('Hello World!')
    op.drop_column('macaroons', 'additional')