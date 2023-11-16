"""empty message

Revision ID: 0ec979123ba4
Revises: e5c7a4e2df4d
Create Date: 2020-12-23 21:35:32.766354

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
revision = '0ec979123ba4'
down_revision = 'e5c7a4e2df4d'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        print('Hello World!')
    op.add_column('dashboards', sa.Column('options', postgresql.JSON(astext_type=sa.Text()), server_default='{}', nullable=False))

def downgrade():
    if False:
        i = 10
        return i + 15
    op.drop_column('dashboards', 'options')