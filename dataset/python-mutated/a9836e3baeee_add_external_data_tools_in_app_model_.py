"""add external_data_tools in app model config

Revision ID: a9836e3baeee
Revises: 968fff4c0ab9
Create Date: 2023-11-02 04:04:57.609485

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
revision = 'a9836e3baeee'
down_revision = '968fff4c0ab9'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        while True:
            i = 10
    with op.batch_alter_table('app_model_configs', schema=None) as batch_op:
        batch_op.add_column(sa.Column('external_data_tools', sa.Text(), nullable=True))

def downgrade():
    if False:
        print('Hello World!')
    with op.batch_alter_table('app_model_configs', schema=None) as batch_op:
        batch_op.drop_column('external_data_tools')