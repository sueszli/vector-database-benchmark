"""Add Report table

Revision ID: 7f064e009cab
Revises: 2bceb2cb4d7c
Create Date: 2017-05-29 16:50:28.720980

"""
from alembic import op
import sqlalchemy as sa
revision = '7f064e009cab'
down_revision = '2bceb2cb4d7c'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.create_table('nyaa_reports', sa.Column('id', sa.Integer(), nullable=False), sa.Column('created_time', sa.DateTime(), nullable=True), sa.Column('reason', sa.String(length=255), nullable=False), sa.Column('status', sa.Integer(), nullable=False), sa.Column('torrent_id', sa.Integer(), nullable=False), sa.Column('user_id', sa.Integer(), nullable=True), sa.ForeignKeyConstraint(['torrent_id'], ['nyaa_torrents.id'], ondelete='CASCADE'), sa.ForeignKeyConstraint(['user_id'], ['users.id']), sa.PrimaryKeyConstraint('id'))
    op.create_table('sukebei_reports', sa.Column('id', sa.Integer(), nullable=False), sa.Column('created_time', sa.DateTime(), nullable=True), sa.Column('reason', sa.String(length=255), nullable=False), sa.Column('status', sa.Integer(), nullable=False), sa.Column('torrent_id', sa.Integer(), nullable=False), sa.Column('user_id', sa.Integer(), nullable=True), sa.ForeignKeyConstraint(['torrent_id'], ['sukebei_torrents.id'], ondelete='CASCADE'), sa.ForeignKeyConstraint(['user_id'], ['users.id']), sa.PrimaryKeyConstraint('id'))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_table('sukebei_reports')
    op.drop_table('nyaa_reports')