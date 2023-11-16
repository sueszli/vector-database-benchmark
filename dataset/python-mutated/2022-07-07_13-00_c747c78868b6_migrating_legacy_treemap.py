"""Migrating legacy TreeMap

Revision ID: c747c78868b6
Revises: cdcf3d64daf4
Create Date: 2022-06-30 22:04:17.686635

"""
from alembic import op
from sqlalchemy.dialects.mysql.base import MySQLDialect
from superset import db
from superset.migrations.shared.migrate_viz import MigrateTreeMap
revision = 'c747c78868b6'
down_revision = 'cdcf3d64daf4'

def upgrade():
    if False:
        i = 10
        return i + 15
    bind = op.get_bind()
    if isinstance(bind.dialect, MySQLDialect):
        op.execute('ALTER TABLE slices MODIFY params MEDIUMTEXT')
        op.execute('ALTER TABLE slices MODIFY query_context MEDIUMTEXT')
    session = db.Session(bind=bind)
    MigrateTreeMap.upgrade(session)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    bind = op.get_bind()
    session = db.Session(bind=bind)
    MigrateTreeMap.downgrade(session)