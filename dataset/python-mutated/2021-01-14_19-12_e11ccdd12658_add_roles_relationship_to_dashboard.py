"""add roles relationship to dashboard

Revision ID: e11ccdd12658
Revises: 260bf0649a77
Create Date: 2021-01-14 19:12:43.406230
"""
revision = 'e11ccdd12658'
down_revision = '260bf0649a77'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        print('Hello World!')
    op.create_table('dashboard_roles', sa.Column('id', sa.Integer(), nullable=False), sa.Column('role_id', sa.Integer(), nullable=False), sa.Column('dashboard_id', sa.Integer(), nullable=True), sa.ForeignKeyConstraint(['dashboard_id'], ['dashboards.id']), sa.ForeignKeyConstraint(['role_id'], ['ab_role.id']), sa.PrimaryKeyConstraint('id'))

def downgrade():
    if False:
        print('Hello World!')
    op.drop_table('dashboard_roles')