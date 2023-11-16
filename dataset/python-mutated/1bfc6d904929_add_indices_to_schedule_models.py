"""Add indices to schedule models.

Revision ID: 1bfc6d904929
Revises: 66e67039b8a2
Create Date: 2023-05-23 15:59:05.328359

"""
from alembic import op
import sqlalchemy as sa
revision = '1bfc6d904929'
down_revision = '66e67039b8a2'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        print('Hello World!')
    with op.batch_alter_table('block_run', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_block_run_pipeline_run_id'), ['pipeline_run_id'], unique=False)
    with op.batch_alter_table('pipeline_run', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_pipeline_run_backfill_id'), ['backfill_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_pipeline_run_pipeline_schedule_id'), ['pipeline_schedule_id'], unique=False)

def downgrade() -> None:
    if False:
        return 10
    with op.batch_alter_table('pipeline_run', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_pipeline_run_pipeline_schedule_id'))
        batch_op.drop_index(batch_op.f('ix_pipeline_run_backfill_id'))
    with op.batch_alter_table('block_run', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_block_run_pipeline_run_id'))