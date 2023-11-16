"""update_tag_model_w_description

Revision ID: 240d23c7f86f
Revises: 8e5b0fb85b9a
Create Date: 2023-06-29 18:38:30.033529

"""
revision = '240d23c7f86f'
down_revision = '8e5b0fb85b9a'
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

def upgrade():
    if False:
        print('Hello World!')
    op.add_column('tag', sa.Column('description', sa.Text(), nullable=True))

def downgrade():
    if False:
        i = 10
        return i + 15
    op.drop_column('tag', 'description')