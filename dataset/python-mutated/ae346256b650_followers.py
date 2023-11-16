"""followers

Revision ID: ae346256b650
Revises: 37f06a334dbf
Create Date: 2017-09-17 15:41:30.211082

"""
from alembic import op
import sqlalchemy as sa
revision = 'ae346256b650'
down_revision = '37f06a334dbf'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    op.create_table('followers', sa.Column('follower_id', sa.Integer(), nullable=True), sa.Column('followed_id', sa.Integer(), nullable=True), sa.ForeignKeyConstraint(['followed_id'], ['user.id']), sa.ForeignKeyConstraint(['follower_id'], ['user.id']))

def downgrade():
    if False:
        return 10
    op.drop_table('followers')