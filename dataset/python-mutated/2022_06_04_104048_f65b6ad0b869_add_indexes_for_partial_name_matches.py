"""Add indexes for partial name matches

Revision ID: f65b6ad0b869
Revises: d76326ed0d06
Create Date: 2022-06-04 10:40:48.710626

"""
from alembic import op
revision = 'f65b6ad0b869'
down_revision = 'd76326ed0d06'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        return 10
    op.execute('\n        CREATE INDEX ix_flow_name_case_insensitive on flow (name COLLATE NOCASE);\n        ')
    op.execute('\n        CREATE INDEX ix_flow_run_name_case_insensitive on flow_run (name COLLATE NOCASE);\n        ')
    op.execute('\n        CREATE INDEX ix_task_run_name_case_insensitive on task_run (name COLLATE NOCASE);\n        ')
    op.execute('\n        CREATE INDEX ix_deployment_name_case_insensitive on deployment (name COLLATE NOCASE);\n        ')

def downgrade():
    if False:
        print('Hello World!')
    op.execute('\n        DROP INDEX ix_flow_name_case_insensitive;\n        ')
    op.execute('\n        DROP INDEX ix_flow_run_name_case_insensitive;\n        ')
    op.execute('\n        DROP INDEX ix_task_run_name_case_insensitive;\n        ')
    op.execute('\n        DROP INDEX ix_deployment_name_case_insensitive;\n        ')