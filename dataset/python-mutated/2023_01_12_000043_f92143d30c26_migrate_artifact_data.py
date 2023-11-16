"""Migrates state data to the artifact table

Revision ID: f92143d30c26
Revises: f92143d30c25
Create Date: 2023-01-12 00:00:43.488367

"""
import sqlalchemy as sa
from alembic import op
revision = 'f92143d30c26'
down_revision = 'f92143d30c25'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    op.execute('PRAGMA foreign_keys=OFF')

    def update_task_run_artifact_data_in_batches(batch_size, offset):
        if False:
            print('Hello World!')
        return f'\n            INSERT INTO artifact (task_run_state_id, task_run_id, data)\n            SELECT id, task_run_id, data\n            FROM task_run_state\n            WHERE has_data IS TRUE\n            ORDER by id\n            LIMIT {batch_size} OFFSET {offset};\n        '

    def update_task_run_state_from_artifact_id_in_batches(batch_size, offset):
        if False:
            return 10
        return f'\n            UPDATE task_run_state\n            SET result_artifact_id = (SELECT id FROM artifact WHERE task_run_state.id = task_run_state_id)\n            WHERE task_run_state.id in (SELECT id FROM task_run_state WHERE (has_data IS TRUE) AND (result_artifact_id IS NULL) LIMIT {batch_size});\n        '

    def update_flow_run_artifact_data_in_batches(batch_size, offset):
        if False:
            return 10
        return f'\n            INSERT INTO artifact (flow_run_state_id, flow_run_id, data)\n            SELECT id, flow_run_id, data\n            FROM flow_run_state\n            WHERE has_data IS TRUE\n            ORDER by id\n            LIMIT {batch_size} OFFSET {offset};\n        '

    def update_flow_run_state_from_artifact_id_in_batches(batch_size, offset):
        if False:
            print('Hello World!')
        return f'\n            UPDATE flow_run_state\n            SET result_artifact_id = (SELECT id FROM artifact WHERE flow_run_state.id = flow_run_state_id)\n            WHERE flow_run_state.id in (SELECT id FROM flow_run_state WHERE (has_data IS TRUE) AND (result_artifact_id IS NULL) LIMIT {batch_size});\n        '
    data_migration_queries = [update_task_run_artifact_data_in_batches, update_task_run_state_from_artifact_id_in_batches, update_flow_run_artifact_data_in_batches, update_flow_run_state_from_artifact_id_in_batches]
    with op.get_context().autocommit_block():
        conn = op.get_bind()
        for query in data_migration_queries:
            batch_size = 500
            offset = 0
            while True:
                sql_stmt = sa.text(query(batch_size, offset))
                result = conn.execute(sql_stmt)
                if result.rowcount <= 0:
                    break
                offset += batch_size

def downgrade():
    if False:
        while True:
            i = 10

    def nullify_artifact_ref_from_flow_run_state_in_batches(batch_size):
        if False:
            return 10
        return f'\n            UPDATE flow_run_state\n            SET result_artifact_id = NULL\n            WHERE flow_run_state.id in (SELECT id FROM flow_run_state WHERE result_artifact_id IS NOT NULL LIMIT {batch_size});\n        '

    def nullify_artifact_ref_from_task_run_state_in_batches(batch_size):
        if False:
            for i in range(10):
                print('nop')
        return f'\n            UPDATE task_run_state\n            SET result_artifact_id = NULL\n            WHERE task_run_state.id in (SELECT id FROM task_run_state WHERE result_artifact_id IS NOT NULL LIMIT {batch_size});\n        '

    def delete_artifacts_in_batches(batch_size):
        if False:
            print('Hello World!')
        return f'\n            DELETE FROM artifact\n            WHERE artifact.id IN (SELECT id FROM artifact LIMIT {batch_size});\n        '
    data_migration_queries = [delete_artifacts_in_batches, nullify_artifact_ref_from_flow_run_state_in_batches, nullify_artifact_ref_from_task_run_state_in_batches]
    with op.get_context().autocommit_block():
        conn = op.get_bind()
        for query in data_migration_queries:
            batch_size = 500
            while True:
                sql_stmt = sa.text(query(batch_size))
                result = conn.execute(sql_stmt)
                if result.rowcount <= 0:
                    break