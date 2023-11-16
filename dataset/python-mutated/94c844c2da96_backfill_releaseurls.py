"""
Backfill ReleaseURLs

Revision ID: 94c844c2da96
Revises: 7a8c380cefa4
Create Date: 2022-06-10 23:54:30.955026
"""
from alembic import op
revision = '94c844c2da96'
down_revision = '7a8c380cefa4'

def upgrade():
    if False:
        while True:
            i = 10
    op.create_check_constraint('release_urls_valid_name', 'release_urls', 'char_length(name) BETWEEN 1 AND 32')
    op.execute("\n        INSERT INTO release_urls (release_id, name, url)\n            SELECT release_id,\n                (regexp_match(specifier, '^([^,]+)\\s*,\\s*(.*)$'))[1],\n                (regexp_match(specifier, '^([^,]+)\\s*,\\s*(.*)$'))[2]\n            FROM release_dependencies\n            WHERE release_dependencies.kind = 8\n            ON CONFLICT ON CONSTRAINT release_urls_release_id_name_key\n            DO NOTHING;\n        ")

def downgrade():
    if False:
        while True:
            i = 10
    pass