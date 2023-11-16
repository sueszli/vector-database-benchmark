"""
Disallow multiple sdists for a release

Revision ID: f449e5bff5a5
Revises: f404a67e0370
Create Date: 2016-12-17 17:10:31.252165
"""
import sqlalchemy as sa
from alembic import op
revision = 'f449e5bff5a5'
down_revision = 'f404a67e0370'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.add_column('release_files', sa.Column('allow_multiple_sdist', sa.Boolean(), nullable=True))
    op.execute(" DO $$\n            DECLARE\n                row record;\n            BEGIN\n                FOR row IN SELECT name, version, COUNT(*) as sdist_count\n                            FROM release_files\n                            WHERE packagetype = 'sdist'\n                            GROUP BY name, version\n                            HAVING COUNT(*) > 1\n                LOOP\n                    UPDATE release_files\n                    SET allow_multiple_sdist = true\n                    FROM (\n                        SELECT id\n                        FROM release_files\n                        WHERE name = row.name\n                            AND version = row.version\n                            AND packagetype = 'sdist'\n                        ORDER BY upload_time\n                        LIMIT (row.sdist_count - 1)\n                    ) s\n                    WHERE release_files.id = s.id;\n                END LOOP;\n            END $$;\n        ")
    op.execute(' UPDATE release_files\n            SET allow_multiple_sdist = false\n            WHERE allow_multiple_sdist IS NULL\n        ')
    op.alter_column('release_files', 'allow_multiple_sdist', nullable=False, server_default=sa.text('false'))
    op.create_index('release_files_single_sdist', 'release_files', ['name', 'version', 'packagetype'], unique=True, postgresql_where=sa.text("packagetype = 'sdist' AND allow_multiple_sdist = false"))

def downgrade():
    if False:
        print('Hello World!')
    op.drop_index('release_files_single_sdist', table_name='release_files')
    op.drop_column('release_files', 'allow_multiple_sdist')