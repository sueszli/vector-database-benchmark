"""reports alter crontab size

Revision ID: ab104a954a8f
Revises: 5daced1f0e76
Create Date: 2020-12-15 09:07:24.730545

"""
revision = 'ab104a954a8f'
down_revision = 'e37912a26567'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        return 10
    with op.batch_alter_table('report_schedule') as batch_op:
        batch_op.alter_column('crontab', existing_type=sa.VARCHAR(length=50), type_=sa.VARCHAR(length=1000), existing_nullable=False)

def downgrade():
    if False:
        i = 10
        return i + 15
    with op.batch_alter_table('report_schedule') as batch_op:
        batch_op.alter_column('crontab', existing_type=sa.VARCHAR(length=1000), type_=sa.VARCHAR(length=50), existing_nullable=False)