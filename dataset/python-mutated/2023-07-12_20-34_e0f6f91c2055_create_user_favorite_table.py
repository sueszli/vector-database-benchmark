"""create_user_favorite_table

Revision ID: e0f6f91c2055
Revises: bf646a0c1501
Create Date: 2023-07-12 20:34:57.553981

"""
revision = 'e0f6f91c2055'
down_revision = 'bf646a0c1501'
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.create_table('user_favorite_tag', sa.Column('user_id', sa.Integer(), nullable=False), sa.Column('tag_id', sa.Integer(), nullable=False), sa.ForeignKeyConstraint(['tag_id'], ['tag.id']), sa.ForeignKeyConstraint(['user_id'], ['ab_user.id']))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_table('user_favorite_tag')