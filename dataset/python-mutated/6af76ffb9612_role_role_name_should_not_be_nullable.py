"""
Role.role_name should not be nullable

Revision ID: 6af76ffb9612
Revises: aaa60e8ea12e
Create Date: 2020-03-28 01:20:30.453875
"""
import sqlalchemy as sa
from alembic import op
revision = '6af76ffb9612'
down_revision = 'aaa60e8ea12e'

def upgrade():
    if False:
        print('Hello World!')
    op.alter_column('roles', 'role_name', existing_type=sa.TEXT(), nullable=False)

def downgrade():
    if False:
        while True:
            i = 10
    op.alter_column('roles', 'role_name', existing_type=sa.TEXT(), nullable=True)