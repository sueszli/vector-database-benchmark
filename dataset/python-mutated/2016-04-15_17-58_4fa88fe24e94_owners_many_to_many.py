"""owners_many_to_many

Revision ID: 4fa88fe24e94
Revises: b4456560d4f3
Create Date: 2016-04-15 17:58:33.842012

"""
revision = '4fa88fe24e94'
down_revision = 'b4456560d4f3'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        i = 10
        return i + 15
    op.create_table('dashboard_user', sa.Column('id', sa.Integer(), nullable=False), sa.Column('user_id', sa.Integer(), nullable=True), sa.Column('dashboard_id', sa.Integer(), nullable=True), sa.ForeignKeyConstraint(['dashboard_id'], ['dashboards.id']), sa.ForeignKeyConstraint(['user_id'], ['ab_user.id']), sa.PrimaryKeyConstraint('id'))
    op.create_table('slice_user', sa.Column('id', sa.Integer(), nullable=False), sa.Column('user_id', sa.Integer(), nullable=True), sa.Column('slice_id', sa.Integer(), nullable=True), sa.ForeignKeyConstraint(['slice_id'], ['slices.id']), sa.ForeignKeyConstraint(['user_id'], ['ab_user.id']), sa.PrimaryKeyConstraint('id'))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_table('slice_user')
    op.drop_table('dashboard_user')