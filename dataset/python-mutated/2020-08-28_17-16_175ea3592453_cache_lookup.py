"""Add cache to datasource lookup table.

Revision ID: 175ea3592453
Revises: f80a3b88324b
Create Date: 2020-08-28 17:16:57.379425

"""
revision = '175ea3592453'
down_revision = 'f80a3b88324b'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        i = 10
        return i + 15
    op.create_table('cache_keys', sa.Column('id', sa.Integer(), nullable=False), sa.Column('cache_key', sa.String(256), nullable=False), sa.Column('cache_timeout', sa.Integer(), nullable=True), sa.Column('datasource_uid', sa.String(64), nullable=False), sa.Column('created_on', sa.DateTime(), nullable=True), sa.PrimaryKeyConstraint('id'))
    op.create_index(op.f('ix_cache_keys_datasource_uid'), 'cache_keys', ['datasource_uid'], unique=False)

def downgrade():
    if False:
        return 10
    op.drop_index(op.f('ix_cache_keys_datasource_uid'), table_name='cache_keys')
    op.drop_table('cache_keys')