"""add_creation_method_to_reports_model

Revision ID: 3317e9248280
Revises: 453530256cea
Create Date: 2021-07-14 10:31:38.610095

"""
revision = '3317e9248280'
down_revision = '453530256cea'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        while True:
            i = 10
    with op.batch_alter_table('report_schedule') as batch_op:
        batch_op.add_column(sa.Column('creation_method', sa.VARCHAR(255), server_default='alerts_reports'))
        batch_op.create_index(op.f('ix_creation_method'), ['creation_method'], unique=False)

def downgrade():
    if False:
        print('Hello World!')
    with op.batch_alter_table('report_schedule') as batch_op:
        batch_op.drop_index('ix_creation_method')
        batch_op.drop_column('creation_method')