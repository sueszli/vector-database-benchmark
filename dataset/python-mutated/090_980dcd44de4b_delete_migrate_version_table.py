"""delete migrate version table

Revision ID: 980dcd44de4b
Revises: 23c92480926e
Create Date: 2019-05-09 13:39:44.097930

"""
from alembic import op
revision = u'980dcd44de4b'
down_revision = u'23c92480926e'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    u'Drop version table, created by sqlalchemy-migrate.\n\n    There is a chance, that we are initializing a new instance and\n    there is no `migrate_version` table, so DO NOT remove `IF EXISTS`\n    clause.\n    '
    op.execute(u'DROP TABLE IF EXISTS migrate_version')

def downgrade():
    if False:
        i = 10
        return i + 15
    u"We aren't going to recreate `migrate_version` here.\n\n    There is a chance, that this table even never was created for\n    target database. This migration tries to seamlessly upgrade\n    existing instance from usage of sqlalchemy-migrate to alembic. And\n    we don't want to downgrade to sqlalchemy-migrate back again.\n    "
    pass