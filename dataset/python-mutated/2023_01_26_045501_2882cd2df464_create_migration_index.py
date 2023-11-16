"""Adds a helper index for the artifact data migration

Revision ID: 2882cd2df464
Revises: 2882cd2df463
Create Date: 2023-01-26 04:55:01.358638

"""
import sqlalchemy as sa
from alembic import op
revision = '2882cd2df464'
down_revision = '2882cd2df463'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    with op.batch_alter_table('flow_run_state', schema=None) as batch_op:
        batch_op.add_column(sa.Column('has_data', sa.Boolean))
        batch_op.create_index(batch_op.f('ix_flow_run_state__has_data'), ['has_data'], unique=False)
    with op.batch_alter_table('task_run_state', schema=None) as batch_op:
        batch_op.add_column(sa.Column('has_data', sa.Boolean))
        batch_op.create_index(batch_op.f('ix_task_run_state__has_data'), ['has_data'], unique=False)

    def populate_flow_has_data_in_batches(batch_size):
        if False:
            return 10
        return f"\n            UPDATE flow_run_state\n            SET has_data = (data IS NOT NULL AND data != 'null')\n            WHERE flow_run_state.id in (SELECT id FROM flow_run_state WHERE (has_data IS NULL) LIMIT {batch_size});\n        "

    def populate_task_has_data_in_batches(batch_size):
        if False:
            i = 10
            return i + 15
        return f"\n            UPDATE task_run_state\n            SET has_data = (data IS NOT NULL AND data != 'null')\n            WHERE task_run_state.id in (SELECT id FROM task_run_state WHERE (has_data IS NULL) LIMIT {batch_size});\n        "
    migration_statements = [populate_flow_has_data_in_batches, populate_task_has_data_in_batches]
    with op.get_context().autocommit_block():
        conn = op.get_bind()
        for query in migration_statements:
            batch_size = 500
            while True:
                sql_stmt = sa.text(query(batch_size))
                result = conn.execute(sql_stmt)
                if result.rowcount < batch_size:
                    break

def downgrade():
    if False:
        i = 10
        return i + 15
    with op.batch_alter_table('task_run_state', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_task_run_state__has_data'))
        batch_op.drop_column('has_data')
    with op.batch_alter_table('flow_run_state', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_flow_run_state__has_data'))
        batch_op.drop_column('has_data')