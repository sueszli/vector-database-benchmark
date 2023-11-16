"""Rename run alerts to run notifications

Revision ID: cdcb4018dd0e
Revises: 724e6dcc6b5d
Create Date: 2022-05-30 11:25:49.621795

"""
from alembic import op
revision = 'cdcb4018dd0e'
down_revision = '2fe6fe6ca16e'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    op.rename_table('flow_run_alert_policy', 'flow_run_notification_policy')
    op.execute('ALTER INDEX pk_flow_run_alert RENAME TO pk_flow_run_notification_policy;')
    op.execute('ALTER INDEX ix_flow_run_alert_policy__name RENAME TO ix_flow_run_notification_policy__name;')
    op.execute('ALTER INDEX ix_flow_run_alert_policy__updated RENAME TO ix_flow_run_notification_policy__updated;')
    op.rename_table('flow_run_alert_queue', 'flow_run_notification_queue')
    op.execute('ALTER INDEX pk_flow_run_alert_queue RENAME TO pk_flow_run_notification_queue;')
    op.execute('ALTER INDEX ix_flow_run_alert_queue__updated RENAME TO ix_flow_run_notification_queue__updated;')
    op.alter_column('flow_run_notification_queue', 'flow_run_alert_policy_id', new_column_name='flow_run_notification_policy_id')

def downgrade():
    if False:
        return 10
    op.rename_table('flow_run_notification_queue', 'flow_run_alert_queue')
    op.execute('ALTER INDEX pk_flow_run_notification_queue RENAME TO pk_flow_run_alert_queue;')
    op.execute('ALTER INDEX ix_flow_run_notification_queue__updated RENAME TO ix_flow_run_alert_queue__updated;')
    op.alter_column('flow_run_alert_queue', 'flow_run_notification_policy_id', new_column_name='flow_run_alert_policy_id')
    op.rename_table('flow_run_notification_policy', 'flow_run_alert_policy')
    op.execute('ALTER INDEX pk_flow_run_notification_policy RENAME TO pk_flow_run_alert;')
    op.execute('ALTER INDEX ix_flow_run_notification_policy__name RENAME TO ix_flow_run_alert_policy__name;')
    op.execute('ALTER INDEX ix_flow_run_notification_policy__updated RENAME TO ix_flow_run_alert_policy__updated;')