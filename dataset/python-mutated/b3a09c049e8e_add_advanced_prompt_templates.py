"""add advanced prompt templates

Revision ID: b3a09c049e8e
Revises: 2e9819ca5b28
Create Date: 2023-10-10 15:23:23.395420

"""
from alembic import op
import sqlalchemy as sa
revision = 'b3a09c049e8e'
down_revision = '2e9819ca5b28'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    with op.batch_alter_table('app_model_configs', schema=None) as batch_op:
        batch_op.add_column(sa.Column('prompt_type', sa.String(length=255), nullable=False, server_default='simple'))
        batch_op.add_column(sa.Column('chat_prompt_config', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('completion_prompt_config', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('dataset_configs', sa.Text(), nullable=True))

def downgrade():
    if False:
        while True:
            i = 10
    with op.batch_alter_table('app_model_configs', schema=None) as batch_op:
        batch_op.drop_column('dataset_configs')
        batch_op.drop_column('completion_prompt_config')
        batch_op.drop_column('chat_prompt_config')
        batch_op.drop_column('prompt_type')