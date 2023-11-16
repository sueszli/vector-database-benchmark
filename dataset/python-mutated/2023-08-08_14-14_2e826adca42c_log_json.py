"""Fix schema for log

Revision ID: 2e826adca42c
Revises: 0769ef90fddd
Create Date: 2023-08-08 14:14:23.381364

"""
import sqlalchemy as sa
from alembic import op
from superset.utils.core import MediumText
revision = '2e826adca42c'
down_revision = '0769ef90fddd'

def upgrade():
    if False:
        while True:
            i = 10
    with op.batch_alter_table('logs') as batch_op:
        batch_op.alter_column('json', existing_type=sa.Text(), type_=MediumText(), existing_nullable=True)

def downgrade():
    if False:
        while True:
            i = 10
    with op.batch_alter_table('logs') as batch_op:
        batch_op.alter_column('json', existing_type=MediumText(), type_=sa.Text(), existing_nullable=True)