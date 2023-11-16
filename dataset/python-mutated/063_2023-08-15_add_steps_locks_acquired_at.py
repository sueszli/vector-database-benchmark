"""add locks_acquired_at column to steps table

Revision ID: 063
Revises: 062

"""
import sqlalchemy as sa
from alembic import op
from buildbot.util import sautils
revision = '063'
down_revision = '062'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        print('Hello World!')
    op.add_column('steps', sa.Column('locks_acquired_at', sa.Integer, nullable=True))
    metadata = sa.MetaData()
    steps_tbl = sautils.Table('steps', metadata, sa.Column('started_at', sa.Integer), sa.Column('locks_acquired_at', sa.Integer))
    op.execute(steps_tbl.update(values={steps_tbl.c.locks_acquired_at: steps_tbl.c.started_at}))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_column('steps', 'locks_acquired_at')