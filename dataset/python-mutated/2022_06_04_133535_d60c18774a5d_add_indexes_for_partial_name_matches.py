"""Add indexes for partial name matches

Revision ID: 77ebcc9cf355
Revises: cdcb4018dd0e
Create Date: 2022-06-04 10:40:48.710626

"""
from alembic import op
revision = '77ebcc9cf355'
down_revision = 'cdcb4018dd0e'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.execute('CREATE EXTENSION IF NOT EXISTS pg_trgm;')
    with op.get_context().autocommit_block():
        op.execute('\n            CREATE INDEX CONCURRENTLY \n            trgm_ix_flow_name \n            ON flow USING gin (name gin_trgm_ops);\n            ')
        op.execute('\n            CREATE INDEX CONCURRENTLY \n            trgm_ix_flow_run_name \n            ON flow_run USING gin (name gin_trgm_ops);\n            ')
        op.execute('\n            CREATE INDEX CONCURRENTLY \n            trgm_ix_task_run_name \n            ON task_run USING gin (name gin_trgm_ops);\n            ')
        op.execute('\n            CREATE INDEX CONCURRENTLY \n            trgm_ix_deployment_name \n            ON deployment USING gin (name gin_trgm_ops);\n            ')

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    with op.get_context().autocommit_block():
        op.execute('\n            DROP INDEX CONCURRENTLY trgm_ix_flow_name;\n            ')
        op.execute('\n            DROP INDEX CONCURRENTLY trgm_ix_flow_run_name;\n            ')
        op.execute('\n            DROP INDEX CONCURRENTLY trgm_ix_task_run_name;\n            ')
        op.execute('\n            DROP INDEX CONCURRENTLY trgm_ix_deployment_name;\n            ')
    op.execute('DROP EXTENSION IF EXISTS pg_trgm;')