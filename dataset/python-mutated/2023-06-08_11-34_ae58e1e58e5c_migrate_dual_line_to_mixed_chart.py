"""migrate_dual_line_to_mixed_chart

Revision ID: ae58e1e58e5c
Revises: 4c5da39be729
Create Date: 2023-06-08 11:34:36.241939

"""
from alembic import op
from superset import db
revision = 'ae58e1e58e5c'
down_revision = '4c5da39be729'
from superset.migrations.shared.migrate_viz.processors import MigrateDualLine

def upgrade():
    if False:
        print('Hello World!')
    bind = op.get_bind()
    session = db.Session(bind=bind)
    MigrateDualLine.upgrade(session)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    bind = op.get_bind()
    session = db.Session(bind=bind)
    MigrateDualLine.downgrade(session)