"""adding log model

Revision ID: 315b3f4da9b0
Revises: 1a48a5411020
Create Date: 2015-12-04 11:16:58.226984

"""
revision = '315b3f4da9b0'
down_revision = '1a48a5411020'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        i = 10
        return i + 15
    op.create_table('logs', sa.Column('id', sa.Integer(), nullable=False), sa.Column('action', sa.String(length=512), nullable=True), sa.Column('user_id', sa.Integer(), nullable=True), sa.Column('json', sa.Text(), nullable=True), sa.Column('dttm', sa.DateTime(), nullable=True), sa.ForeignKeyConstraint(['user_id'], ['ab_user.id']), sa.PrimaryKeyConstraint('id'))

def downgrade():
    if False:
        i = 10
        return i + 15
    op.drop_table('logs')