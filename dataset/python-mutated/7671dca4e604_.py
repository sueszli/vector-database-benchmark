"""empty message

Revision ID: 7671dca4e604
Revises: d1eae8b9893e
Create Date: 2017-11-22 22:20:25.166045

"""
from alembic import op
import sqlalchemy as sa
revision = '7671dca4e604'
down_revision = 'd1eae8b9893e'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        return 10
    op.add_column('users', sa.Column('profile_image_url', sa.String(), nullable=True, server_default=None))

def downgrade():
    if False:
        i = 10
        return i + 15
    op.drop_column('users', 'profile_image_url')