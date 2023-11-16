"""
Update release_file on insert

The release.requires_python and release_files.requires_python can be out of
sync if a file is uploaded after the release has been created. It's
requires_python field will stay empty.

We need to :
 - Fix potentially existing mistakes
 - Create a trigger that ensure consistency

Revision ID: 99291f0fe9c2
Revises: e7b09b5c089d
Create Date: 2016-12-02 00:58:53.109880
"""
from alembic import op
revision = '99291f0fe9c2'
down_revision = 'e7b09b5c089d'

def upgrade():
    if False:
        print('Hello World!')
    op.execute(' UPDATE release_files\n            SET requires_python = releases.requires_python\n            FROM releases\n            WHERE\n                release_files.name=releases.name\n                AND release_files.version=releases.version;\n        ')
    op.execute(' CREATE TRIGGER release_files_requires_python\n              AFTER INSERT ON release_files\n              FOR EACH ROW\n                  EXECUTE PROCEDURE update_release_files_requires_python();\n        ')

def downgrade():
    if False:
        print('Hello World!')
    op.execute('DROP TRIGGER release_files_requires_python ON release_files')