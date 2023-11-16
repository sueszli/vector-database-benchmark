"""add_role_level_security

Revision ID: 0a6f12f60c73
Revises: 3325d4caccc8
Create Date: 2019-12-04 17:07:54.390805

"""
revision = '0a6f12f60c73'
down_revision = '3325d4caccc8'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        print('Hello World!')
    op.create_table('row_level_security_filters', sa.Column('created_on', sa.DateTime(), nullable=True), sa.Column('changed_on', sa.DateTime(), nullable=True), sa.Column('id', sa.Integer(), nullable=False), sa.Column('table_id', sa.Integer(), nullable=False), sa.Column('clause', sa.Text(), nullable=False), sa.Column('created_by_fk', sa.Integer(), nullable=True), sa.Column('changed_by_fk', sa.Integer(), nullable=True), sa.ForeignKeyConstraint(['changed_by_fk'], ['ab_user.id']), sa.ForeignKeyConstraint(['created_by_fk'], ['ab_user.id']), sa.ForeignKeyConstraint(['table_id'], ['tables.id']), sa.PrimaryKeyConstraint('id'))
    op.create_table('rls_filter_roles', sa.Column('id', sa.Integer(), nullable=False), sa.Column('role_id', sa.Integer(), nullable=False), sa.Column('rls_filter_id', sa.Integer(), nullable=True), sa.ForeignKeyConstraint(['rls_filter_id'], ['row_level_security_filters.id']), sa.ForeignKeyConstraint(['role_id'], ['ab_role.id']), sa.PrimaryKeyConstraint('id'))

def downgrade():
    if False:
        print('Hello World!')
    op.drop_table('rls_filter_roles')
    op.drop_table('row_level_security_filters')