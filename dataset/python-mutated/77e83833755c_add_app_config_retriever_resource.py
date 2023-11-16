"""add_app_config_retriever_resource

Revision ID: 77e83833755c
Revises: 6dcb43972bdc
Create Date: 2023-09-06 17:26:40.311927

"""
from alembic import op
import sqlalchemy as sa
revision = '77e83833755c'
down_revision = '6dcb43972bdc'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    with op.batch_alter_table('app_model_configs', schema=None) as batch_op:
        batch_op.add_column(sa.Column('retriever_resource', sa.Text(), nullable=True))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    with op.batch_alter_table('app_model_configs', schema=None) as batch_op:
        batch_op.drop_column('retriever_resource')