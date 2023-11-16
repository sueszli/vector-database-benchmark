"""Add user details JSON column.

Revision ID: e7f8a917aa8e
Revises: 71477dadd6ef
Create Date: 2018-11-08 16:12:17.023569

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
revision = 'e7f8a917aa8e'
down_revision = '640888ce445d'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    op.add_column('users', sa.Column('details', postgresql.JSON(astext_type=sa.Text()), server_default='{}', nullable=True))

def downgrade():
    if False:
        print('Hello World!')
    op.drop_column('users', 'details')