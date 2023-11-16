"""services

Revision ID: af4cbdb2d13c
Revises: eeb276e51423
Create Date: 2016-07-28 16:16:38.245348

"""
revision = 'af4cbdb2d13c'
down_revision = 'eeb276e51423'
branch_labels = None
depends_on = None
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        return 10
    op.add_column('api_tokens', sa.Column('service_id', sa.Integer))

def downgrade():
    if False:
        print('Hello World!')
    op.drop_column('api_tokens', 'service_id')