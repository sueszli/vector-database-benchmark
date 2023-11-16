"""Increase size of name column in ab_view_menu

Revision ID: cefabc8f7d38
Revises: 6c7537a6004a
Create Date: 2018-12-13 15:38:36.772750

"""
revision = 'cefabc8f7d38'
down_revision = '6c7537a6004a'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        i = 10
        return i + 15
    with op.batch_alter_table('ab_view_menu') as batch_op:
        batch_op.alter_column('name', existing_type=sa.String(length=100), existing_nullable=False, type_=sa.String(length=255), nullable=False)

def downgrade():
    if False:
        print('Hello World!')
    with op.batch_alter_table('ab_view_menu') as batch_op:
        batch_op.alter_column('name', existing_type=sa.String(length=255), existing_nullable=False, type_=sa.String(length=100), nullable=False)