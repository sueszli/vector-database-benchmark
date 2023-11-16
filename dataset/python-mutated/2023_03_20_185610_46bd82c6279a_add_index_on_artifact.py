"""Add index on artifact

Revision ID: 46bd82c6279a
Revises: d20618ce678e
Create Date: 2023-03-20 18:56:10.725419

"""
from alembic import op
revision = '46bd82c6279a'
down_revision = 'd20618ce678e'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.execute('\n        CREATE INDEX \n        IF NOT EXISTS\n        ix_artifact__key_created_desc\n        ON artifact (key, created DESC)\n        INCLUDE (id, updated, type, task_run_id, flow_run_id)\n    ')

def downgrade():
    if False:
        print('Hello World!')
    op.execute('\n        DROP INDEX \n        IF EXISTS\n        ix_artifact__key_created_desc\n    ')