"""
Create banner model

Revision ID: 10825786b3df
Revises: 590c513f1c74
Create Date: 2021-06-22 18:16:50.425481
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
revision = '10825786b3df'
down_revision = '590c513f1c74'

def upgrade():
    if False:
        i = 10
        return i + 15
    op.create_table('banners', sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False), sa.Column('name', sa.String(), nullable=False), sa.Column('text', sa.Text(), nullable=False), sa.Column('link_url', sa.Text(), nullable=False), sa.Column('link_label', sa.String(), nullable=False), sa.Column('fa_icon', sa.String(), nullable=False), sa.Column('active', sa.Boolean(), nullable=False), sa.Column('end', sa.Date(), nullable=False), sa.PrimaryKeyConstraint('id'))

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_table('banners')