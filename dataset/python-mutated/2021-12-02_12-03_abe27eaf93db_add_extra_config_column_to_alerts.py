"""add_extra_config_column_to_alerts

Revision ID: abe27eaf93db
Revises: 0ca9e5f1dacd
Create Date: 2021-12-02 12:03:20.691171

"""
revision = 'abe27eaf93db'
down_revision = '0ca9e5f1dacd'
import sqlalchemy as sa
from alembic import op
from sqlalchemy import String
from sqlalchemy.sql import column, table
report_schedule = table('report_schedule', column('extra', String))

def upgrade():
    if False:
        while True:
            i = 10
    bind = op.get_bind()
    with op.batch_alter_table('report_schedule') as batch_op:
        batch_op.add_column(sa.Column('extra', sa.Text(), nullable=True, default='{}'))
    bind.execute(report_schedule.update().values({'extra': '{}'}))
    with op.batch_alter_table('report_schedule') as batch_op:
        batch_op.alter_column('extra', existing_type=sa.Text(), nullable=False)

def downgrade():
    if False:
        print('Hello World!')
    op.drop_column('report_schedule', 'extra')