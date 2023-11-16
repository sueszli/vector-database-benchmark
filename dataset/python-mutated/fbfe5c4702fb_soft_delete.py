"""soft delete

Revision ID: fbfe5c4702fb
Revises: 1ecf8222220d
Create Date: 2021-08-17 21:34:41.024743

"""
from alembic import op
import sqlalchemy as sa
revision = 'fbfe5c4702fb'
down_revision = '1ecf8222220d'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    op.add_column('experiment', sa.Column('is_archived', sa.Boolean(), nullable=True))
    op.add_column('tag', sa.Column('is_archived', sa.Boolean(), nullable=True))

def downgrade():
    if False:
        i = 10
        return i + 15
    op.drop_column('tag', 'is_archived')
    op.drop_column('experiment', 'is_archived')