"""
Disable legacy file types unless a project has used them previously

Revision ID: f404a67e0370
Revises: b8fda0d7fbb5
Create Date: 2016-12-17 02:58:55.328035
"""
from alembic import op
revision = 'f404a67e0370'
down_revision = 'b8fda0d7fbb5'

def upgrade():
    if False:
        i = 10
        return i + 15
    op.execute(" UPDATE packages\n            SET allow_legacy_files = 'f'\n            WHERE name NOT IN (\n                SELECT DISTINCT ON (packages.name) packages.name\n                FROM packages, release_files\n                WHERE packages.name = release_files.name\n                    AND (\n                        filename !~* '.+?\\.(tar\\.gz|zip|whl|egg)$'\n                        OR packagetype NOT IN (\n                            'sdist',\n                            'bdist_wheel',\n                            'bdist_egg'\n                        )\n                    )\n            )\n        ")

def downgrade():
    if False:
        print('Hello World!')
    raise RuntimeError('Order No. 227 - Ни шагу назад!')