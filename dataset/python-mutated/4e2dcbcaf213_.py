"""Make Webhook columns nullable

Revision ID: 4e2dcbcaf213
Revises: dfe5c529e6da
Create Date: 2022-05-05 13:21:37.621793

"""
import sqlalchemy as sa
from alembic import op
revision = '4e2dcbcaf213'
down_revision = 'dfe5c529e6da'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        return 10
    op.alter_column('subscribers', 'url', existing_type=sa.VARCHAR(), nullable=True)
    op.alter_column('subscribers', 'name', existing_type=sa.VARCHAR(length=100), nullable=True)
    op.alter_column('subscribers', 'verify_ssl', existing_type=sa.BOOLEAN(), nullable=True)
    op.alter_column('subscribers', 'secret', existing_type=sa.VARCHAR(), nullable=True)
    op.alter_column('subscribers', 'content_type', existing_type=sa.VARCHAR(length=50), nullable=True)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.alter_column('subscribers', 'content_type', existing_type=sa.VARCHAR(length=50), nullable=False)
    op.alter_column('subscribers', 'secret', existing_type=sa.VARCHAR(), nullable=False)
    op.alter_column('subscribers', 'verify_ssl', existing_type=sa.BOOLEAN(), nullable=False)
    op.alter_column('subscribers', 'name', existing_type=sa.VARCHAR(length=100), nullable=False)
    op.alter_column('subscribers', 'url', existing_type=sa.VARCHAR(), nullable=False)