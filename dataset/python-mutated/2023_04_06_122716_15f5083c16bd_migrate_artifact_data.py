"""Migrate artifact data to artifact_collection table

Revision ID: 15f5083c16bd
Revises: 310dda75f561
Create Date: 2023-04-06 12:27:16.676260

"""
import sqlalchemy as sa
from alembic import op
revision = '15f5083c16bd'
down_revision = '310dda75f561'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    '\n    A data-only migration that populates flow_run_id, task_run_id, type, description, and metadata_ columns\n    for artifact_collection table.\n    '
    batch_size = 500
    offset = 0
    update_artifact_collection_table = '\n        WITH artifact_collection_cte AS (\n            SELECT * FROM artifact_collection WHERE id = :id FOR UPDATE\n        )\n        UPDATE artifact_collection\n        SET data = artifact.data,\n            description = artifact.description,\n            flow_run_id = artifact.flow_run_id,\n            task_run_id = artifact.task_run_id,\n            type = artifact.type,\n            metadata_ = artifact.metadata_\n        FROM artifact, artifact_collection_cte\n        WHERE artifact_collection.latest_id = artifact.id\n        AND artifact.id = artifact_collection_cte.latest_id;\n    '
    with op.get_context().autocommit_block():
        conn = op.get_bind()
        while True:
            select_artifact_collection_cte = f'\n                SELECT * from artifact_collection ORDER BY id LIMIT {batch_size} OFFSET {offset} FOR UPDATE;\n            '
            selected_artifact_collections = conn.execute(sa.text(select_artifact_collection_cte)).fetchall()
            if not selected_artifact_collections:
                break
            for row in selected_artifact_collections:
                id_to_update = row[0]
                conn.execute(sa.text(update_artifact_collection_table), {'id': id_to_update})
                offset += batch_size

def downgrade():
    if False:
        return 10
    '\n    Data-only migration, no action needed.\n    '