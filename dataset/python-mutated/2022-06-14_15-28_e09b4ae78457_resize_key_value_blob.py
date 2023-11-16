"""Resize key_value blob

Revision ID: e09b4ae78457
Revises: e786798587de
Create Date: 2022-06-14 15:28:42.746349

"""
revision = 'e09b4ae78457'
down_revision = 'e786798587de'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        return 10
    with op.batch_alter_table('key_value', schema=None) as batch_op:
        batch_op.alter_column('value', existing_nullable=False, existing_type=sa.LargeBinary(), type_=sa.LargeBinary(length=2 ** 24 - 1))

def downgrade():
    if False:
        print('Hello World!')
    with op.batch_alter_table('key_value', schema=None) as batch_op:
        batch_op.alter_column('value', existing_nullable=False, existing_type=sa.LargeBinary(length=2 ** 24 - 1), type_=sa.LargeBinary())