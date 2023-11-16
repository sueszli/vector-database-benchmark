"""
Sponsor model

Revision ID: d0c22553b338
Revises: 69b928240b2f
Create Date: 2021-05-26 18:23:27.021443
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
revision = 'd0c22553b338'
down_revision = '69b928240b2f'

def upgrade():
    if False:
        return 10
    op.create_table('sponsors', sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False), sa.Column('name', sa.String(), nullable=False), sa.Column('service', sa.String(), nullable=True), sa.Column('activity_markdown', sa.Text(), nullable=True), sa.Column('link_url', sa.Text(), nullable=False), sa.Column('color_logo_url', sa.Text(), nullable=False), sa.Column('white_logo_url', sa.Text(), nullable=True), sa.Column('is_active', sa.Boolean(), nullable=False), sa.Column('footer', sa.Boolean(), nullable=False), sa.Column('psf_sponsor', sa.Boolean(), nullable=False), sa.Column('infra_sponsor', sa.Boolean(), nullable=False), sa.Column('one_time', sa.Boolean(), nullable=False), sa.Column('sidebar', sa.Boolean(), nullable=False), sa.PrimaryKeyConstraint('id'))

def downgrade():
    if False:
        print('Hello World!')
    op.drop_table('sponsors')