"""Add status column to Project table.

Revision ID: 5a915f45eb6f
Revises: b7562da0bb32
Create Date: 2021-01-13 14:56:12.547255

"""
import sqlalchemy as sa
from alembic import op
revision = '5a915f45eb6f'
down_revision = 'b7562da0bb32'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        return 10
    op.add_column('project', sa.Column('status', sa.String(length=15), server_default=sa.text("'READY'"), nullable=False))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_column('project', 'status')