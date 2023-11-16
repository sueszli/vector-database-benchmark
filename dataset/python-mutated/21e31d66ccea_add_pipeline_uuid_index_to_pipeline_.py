"""Add pipeline_uuid index to pipeline_schedule.

Revision ID: 21e31d66ccea
Revises: 1a737bdee53b
Create Date: 2023-04-30 17:35:47.150966

"""
from alembic import op
import sqlalchemy as sa
revision = '21e31d66ccea'
down_revision = '1a737bdee53b'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        i = 10
        return i + 15
    with op.batch_alter_table('pipeline_schedule', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_pipeline_schedule_pipeline_uuid'), ['pipeline_uuid'], unique=False)

def downgrade() -> None:
    if False:
        i = 10
        return i + 15
    with op.batch_alter_table('pipeline_schedule', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_pipeline_schedule_pipeline_uuid'))