"""followers

Revision ID: 2356a38169ea
Revises: 288cd3dc5a8
Create Date: 2013-12-31 16:10:34.500006

"""
revision = '2356a38169ea'
down_revision = '288cd3dc5a8'
from alembic import op
import sqlalchemy as sa

def upgrade():
    if False:
        i = 10
        return i + 15
    op.create_table('follows', sa.Column('follower_id', sa.Integer(), nullable=False), sa.Column('followed_id', sa.Integer(), nullable=False), sa.Column('timestamp', sa.DateTime(), nullable=True), sa.ForeignKeyConstraint(['followed_id'], ['users.id']), sa.ForeignKeyConstraint(['follower_id'], ['users.id']), sa.PrimaryKeyConstraint('follower_id', 'followed_id'))

def downgrade():
    if False:
        return 10
    op.drop_table('follows')