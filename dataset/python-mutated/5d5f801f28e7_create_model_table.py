"""create model table

Revision ID: 5d5f801f28e7
Revises: 520aa6776347
Create Date: 2023-08-07 05:36:29.791610

"""
from alembic import op
import sqlalchemy as sa
revision = '5d5f801f28e7'
down_revision = 'be1d922bf2ad'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        return 10
    op.create_table('models', sa.Column('id', sa.Integer(), nullable=False), sa.Column('model_name', sa.String(), nullable=False), sa.Column('description', sa.String(), nullable=True), sa.Column('end_point', sa.String(), nullable=False), sa.Column('model_provider_id', sa.Integer(), nullable=False), sa.Column('token_limit', sa.Integer(), nullable=False), sa.Column('type', sa.String(), nullable=False), sa.Column('version', sa.String(), nullable=False), sa.Column('org_id', sa.Integer(), nullable=False), sa.Column('model_features', sa.String(), nullable=False), sa.Column('created_at', sa.DateTime(), nullable=True), sa.Column('updated_at', sa.DateTime(), nullable=True), sa.PrimaryKeyConstraint('id'))

def downgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    op.drop_table('models')