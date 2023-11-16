"""Add ``timetable_description`` column to DagModel for UI.

Revision ID: 786e3737b18f
Revises: 5e3ec427fdd3
Create Date: 2021-10-15 13:33:04.754052

"""
from __future__ import annotations
import sqlalchemy as sa
from alembic import op
revision = '786e3737b18f'
down_revision = '5e3ec427fdd3'
branch_labels = None
depends_on = None
airflow_version = '2.3.0'

def upgrade():
    if False:
        i = 10
        return i + 15
    'Apply Add ``timetable_description`` column to DagModel for UI.'
    with op.batch_alter_table('dag', schema=None) as batch_op:
        batch_op.add_column(sa.Column('timetable_description', sa.String(length=1000), nullable=True))

def downgrade():
    if False:
        return 10
    'Unapply Add ``timetable_description`` column to DagModel for UI.'
    is_sqlite = bool(op.get_bind().dialect.name == 'sqlite')
    if is_sqlite:
        op.execute('PRAGMA foreign_keys=off')
    with op.batch_alter_table('dag') as batch_op:
        batch_op.drop_column('timetable_description')
    if is_sqlite:
        op.execute('PRAGMA foreign_keys=on')