"""Increase ab_user.email field size

Revision ID: ec54aca4c8a2
Revises: 9f4a086c2676
Create Date: 2023-08-22 11:09:48.577457

"""
revision = 'ec54aca4c8a2'
down_revision = '9f4a086c2676'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        print('Hello World!')
    with op.batch_alter_table('ab_user') as batch_op:
        batch_op.alter_column('email', existing_type=sa.VARCHAR(length=64), type_=sa.String(length=320), nullable=False)

def downgrade():
    if False:
        return 10
    with op.batch_alter_table('ab_user') as batch_op:
        batch_op.alter_column('email', existing_type=sa.VARCHAR(length=320), type_=sa.String(length=64), nullable=False)