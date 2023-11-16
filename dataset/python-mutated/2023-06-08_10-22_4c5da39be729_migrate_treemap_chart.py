"""migrate_treemap_chart

Revision ID: 4c5da39be729
Revises: 9ba2ce3086e5
Create Date: 2023-06-08 10:22:23.192064

"""
from alembic import op
from sqlalchemy.dialects.mysql.base import MySQLDialect
from superset import db
from superset.migrations.shared.migrate_viz import MigrateTreeMap
revision = '4c5da39be729'
down_revision = '9ba2ce3086e5'

def upgrade():
    if False:
        while True:
            i = 10
    bind = op.get_bind()
    if isinstance(bind.dialect, MySQLDialect):
        op.execute('ALTER TABLE slices MODIFY params MEDIUMTEXT')
        op.execute('ALTER TABLE slices MODIFY query_context MEDIUMTEXT')
    session = db.Session(bind=bind)
    MigrateTreeMap.upgrade(session)

def downgrade():
    if False:
        i = 10
        return i + 15
    bind = op.get_bind()
    session = db.Session(bind=bind)
    MigrateTreeMap.downgrade(session)