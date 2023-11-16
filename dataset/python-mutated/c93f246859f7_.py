"""Add watcher config table

Revision ID: c93f246859f7
Revises: 6d2354fb841c
Create Date: 2016-09-29 16:13:29.946116

"""
revision = 'c93f246859f7'
down_revision = '6d2354fb841c'
from alembic import op
import sqlalchemy as sa

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.create_table('watcher_config', sa.Column('id', sa.Integer(), nullable=False), sa.Column('index', sa.String(length=80), nullable=True), sa.Column('interval', sa.Integer(), nullable=False), sa.Column('active', sa.Boolean(), nullable=False), sa.PrimaryKeyConstraint('id'), sa.UniqueConstraint('index'))

def downgrade():
    if False:
        i = 10
        return i + 15
    op.drop_table('watcher_config')