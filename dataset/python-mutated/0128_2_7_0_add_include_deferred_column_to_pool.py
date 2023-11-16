"""add include_deferred column to pool

Revision ID: 405de8318b3a
Revises: 788397e78828
Create Date: 2023-07-20 04:22:21.007342

"""
import sqlalchemy as sa
from alembic import op
revision = '405de8318b3a'
down_revision = '788397e78828'
branch_labels = None
depends_on = None
airflow_version = '2.7.0'

def upgrade():
    if False:
        while True:
            i = 10
    'Apply add include_deferred column to pool'
    with op.batch_alter_table('slot_pool') as batch_op:
        batch_op.add_column(sa.Column('include_deferred', sa.Boolean))
    op.execute(sa.text(f'UPDATE slot_pool SET include_deferred = {sa.false().compile(op.get_bind())}'))
    with op.batch_alter_table('slot_pool') as batch_op:
        batch_op.alter_column('include_deferred', existing_type=sa.Boolean, nullable=False)

def downgrade():
    if False:
        i = 10
        return i + 15
    'Unapply add include_deferred column to pool'
    with op.batch_alter_table('slot_pool') as batch_op:
        batch_op.drop_column('include_deferred')