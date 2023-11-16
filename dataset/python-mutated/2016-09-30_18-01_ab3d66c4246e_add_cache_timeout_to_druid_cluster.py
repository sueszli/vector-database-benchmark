"""add_cache_timeout_to_druid_cluster

Revision ID: ab3d66c4246e
Revises: eca4694defa7
Create Date: 2016-09-30 18:01:30.579760

"""
revision = 'ab3d66c4246e'
down_revision = 'eca4694defa7'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        i = 10
        return i + 15
    op.add_column('clusters', sa.Column('cache_timeout', sa.Integer(), nullable=True))

def downgrade():
    if False:
        return 10
    op.drop_column('clusters', 'cache_timeout')