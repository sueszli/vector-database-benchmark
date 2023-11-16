"""Add work queue name to runs

Revision ID: 575634b7acd4
Revises: 296e2665785f
Create Date: 2022-08-07 13:41:38.128173

"""
import sqlalchemy as sa
from alembic import op
revision = '575634b7acd4'
down_revision = '296e2665785f'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    with op.batch_alter_table('deployment', schema=None) as batch_op:
        batch_op.add_column(sa.Column('work_queue_name', sa.String(), nullable=True))
        batch_op.create_index(batch_op.f('ix_deployment__work_queue_name'), ['work_queue_name'], unique=False)
    with op.batch_alter_table('flow_run', schema=None) as batch_op:
        batch_op.add_column(sa.Column('work_queue_name', sa.String(), nullable=True))
        batch_op.create_index(batch_op.f('ix_flow_run__work_queue_name'), ['work_queue_name'], unique=False)
    with op.batch_alter_table('work_queue', schema=None) as batch_op:
        batch_op.alter_column('filter', nullable=True, server_default=None)

def downgrade():
    if False:
        print('Hello World!')
    op.execute("\n        UPDATE work_queue\n        SET filter = '{}'\n        WHERE filter IS NULL;\n        ")
    with op.batch_alter_table('work_queue', schema=None) as batch_op:
        batch_op.alter_column('filter', nullable=False, server_default=sa.text("'{}'"))
    with op.batch_alter_table('flow_run', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_flow_run__work_queue_name'))
        batch_op.drop_column('work_queue_name')
    with op.batch_alter_table('deployment', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_deployment__work_queue_name'))
        batch_op.drop_column('work_queue_name')