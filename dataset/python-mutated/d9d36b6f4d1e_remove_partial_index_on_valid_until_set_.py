"""remove partial_index on valid_until, set it to nullable=false, add message filter fields

Revision ID: d9d36b6f4d1e
Revises: 2e24fc7536e8
Create Date: 2022-03-16 17:31:28.408112

"""
from datetime import datetime
import sqlalchemy as sa
from alembic import op
revision = 'd9d36b6f4d1e'
down_revision = '2e24fc7536e8'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        while True:
            i = 10
    op.execute(sa.text('DROP INDEX IF EXISTS ix_one_active_instance_config'))
    op.execute(sa.text('UPDATE OR IGNORE instance_config SET valid_until=:epoch WHERE valid_until IS NULL;').bindparams(epoch=datetime.fromtimestamp(0)))
    with op.batch_alter_table('instance_config', schema=None) as batch_op:
        batch_op.add_column(sa.Column('initial_message_min_len', sa.Integer(), nullable=False, server_default='0'))
        batch_op.add_column(sa.Column('reject_message_with_codename', sa.Boolean(), nullable=False, server_default='0'))
        batch_op.alter_column('valid_until', existing_type=sa.DATETIME(), nullable=False)
    op.execute(sa.text('DROP INDEX IF EXISTS ix_one_active_instance_config'))

def downgrade() -> None:
    if False:
        i = 10
        return i + 15
    with op.batch_alter_table('instance_config', schema=None) as batch_op:
        batch_op.alter_column('valid_until', existing_type=sa.DATETIME(), nullable=True)
        batch_op.drop_column('reject_message_with_codename')
        batch_op.drop_column('initial_message_min_len')
    op.execute(sa.text('UPDATE OR IGNORE instance_config SET valid_until = NULL WHERE valid_until=:epoch;').bindparams(epoch=datetime.fromtimestamp(0)))
    op.execute(sa.text('CREATE UNIQUE INDEX IF NOT EXISTS ix_one_active_instance_config ON instance_config (valid_until IS NULL) WHERE valid_until IS NULL'))