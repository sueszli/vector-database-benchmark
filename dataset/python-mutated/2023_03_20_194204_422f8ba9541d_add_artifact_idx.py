"""Add index for querying artifacts

Revision ID: 422f8ba9541d
Revises: b9aafc3ab936
Create Date: 2023-03-20 19:42:04.862363

"""
from alembic import op
revision = '422f8ba9541d'
down_revision = 'b9aafc3ab936'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.execute('PRAGMA foreign_keys=OFF')
    op.execute('\n        CREATE INDEX IF NOT EXISTS\n        ix_artifact__key_created_desc\n        ON artifact_collection (key, created DESC)\n        ')
    op.execute('PRAGMA foreign_keys=ON')

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.execute('PRAGMA foreign_keys=OFF')
    op.execute('\n        DROP INDEX IF EXISTS\n        ix_artifact__key_created_desc\n        ')
    op.execute('PRAGMA foreign_keys=ON')