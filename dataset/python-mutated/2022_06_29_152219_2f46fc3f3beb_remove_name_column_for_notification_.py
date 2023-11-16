"""Remove name column for notification policies

Revision ID: 2f46fc3f3beb
Revises: 7296741dff68
Create Date: 2022-06-29 15:22:19.213787

"""
import sqlalchemy as sa
from alembic import op
revision = '2f46fc3f3beb'
down_revision = '7296741dff68'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        print('Hello World!')
    with op.batch_alter_table('flow_run_notification_policy', schema=None) as batch_op:
        batch_op.drop_index('ix_flow_run_notification_policy__name')
        batch_op.drop_column('name')

def downgrade():
    if False:
        return 10
    with op.batch_alter_table('flow_run_notification_policy', schema=None) as batch_op:
        batch_op.add_column(sa.Column('name', sa.VARCHAR(), nullable=False))
        batch_op.create_index('ix_flow_run_notification_policy__name', ['name'], unique=False)