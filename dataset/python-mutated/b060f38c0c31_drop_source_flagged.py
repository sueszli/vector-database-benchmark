"""drop Source.flagged

Revision ID: b060f38c0c31
Revises: 92fba0be98e9
Create Date: 2021-05-10 18:15:56.071880

"""
import sqlalchemy as sa
from alembic import op
revision = 'b060f38c0c31'
down_revision = '92fba0be98e9'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        return 10
    with op.batch_alter_table('sources', schema=None) as batch_op:
        batch_op.drop_column('flagged')

def downgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    conn = op.get_bind()
    conn.execute('PRAGMA legacy_alter_table=ON')
    op.rename_table('sources', 'sources_tmp')
    conn.execute(sa.text('\n            CREATE TABLE "sources" (\n                id INTEGER NOT NULL,\n                uuid VARCHAR(36) NOT NULL,\n                filesystem_id VARCHAR(96),\n                journalist_designation VARCHAR(255) NOT NULL,\n                last_updated DATETIME,\n                pending BOOLEAN,\n                interaction_count INTEGER NOT NULL,\n                deleted_at DATETIME,\n                flagged BOOLEAN,\n                PRIMARY KEY (id),\n                CHECK (pending IN (0, 1)),\n                CHECK (flagged IN (0, 1)),\n                UNIQUE (filesystem_id),\n                UNIQUE (uuid)\n            )\n            '))
    conn.execute('\n        INSERT INTO sources (\n            id, uuid, filesystem_id, journalist_designation,\n            last_updated, pending, interaction_count, deleted_at\n        ) SELECT\n            id, uuid, filesystem_id, journalist_designation,\n            last_updated, pending, interaction_count, deleted_at\n        FROM sources_tmp;\n        ')
    op.drop_table('sources_tmp')