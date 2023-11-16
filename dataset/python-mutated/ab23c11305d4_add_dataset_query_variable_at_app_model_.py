"""add dataset query variable at app model configs.

Revision ID: ab23c11305d4
Revises: 6e2cfb077b04
Create Date: 2023-09-26 12:22:59.044088

"""
from alembic import op
import sqlalchemy as sa
revision = 'ab23c11305d4'
down_revision = '6e2cfb077b04'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    with op.batch_alter_table('app_model_configs', schema=None) as batch_op:
        batch_op.add_column(sa.Column('dataset_query_variable', sa.String(length=255), nullable=True))

def downgrade():
    if False:
        print('Hello World!')
    with op.batch_alter_table('app_model_configs', schema=None) as batch_op:
        batch_op.drop_column('dataset_query_variable')