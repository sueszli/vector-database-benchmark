"""Add global_data_product_uuid to PipelineSchedule

Revision ID: 66928ca5b0d7
Revises: e3593cc2191e
Create Date: 2023-08-01 23:59:59.520041

"""
from alembic import op
import sqlalchemy as sa
revision = '66928ca5b0d7'
down_revision = 'e3593cc2191e'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    with op.batch_alter_table('pipeline_schedule', schema=None) as batch_op:
        batch_op.add_column(sa.Column('global_data_product_uuid', sa.String(length=255), nullable=True))
        batch_op.create_index(batch_op.f('ix_pipeline_schedule_global_data_product_uuid'), ['global_data_product_uuid'], unique=False)

def downgrade() -> None:
    if False:
        return 10
    with op.batch_alter_table('pipeline_schedule', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_pipeline_schedule_global_data_product_uuid'))
        batch_op.drop_column('global_data_product_uuid')