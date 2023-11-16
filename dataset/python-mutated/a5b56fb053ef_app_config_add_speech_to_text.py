"""app config add speech_to_text

Revision ID: a5b56fb053ef
Revises: d3d503a3471c
Create Date: 2023-07-06 17:55:20.894149

"""
from alembic import op
import sqlalchemy as sa
revision = 'a5b56fb053ef'
down_revision = 'd3d503a3471c'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        i = 10
        return i + 15
    with op.batch_alter_table('app_model_configs', schema=None) as batch_op:
        batch_op.add_column(sa.Column('speech_to_text', sa.Text(), nullable=True))

def downgrade():
    if False:
        print('Hello World!')
    with op.batch_alter_table('app_model_configs', schema=None) as batch_op:
        batch_op.drop_column('speech_to_text')