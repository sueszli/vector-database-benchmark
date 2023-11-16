"""added resources

Revision ID: a91808a89623
Revises: 44b0d6f2d1b3
Create Date: 2023-06-01 07:00:33.982485

"""
from alembic import op
import sqlalchemy as sa
revision = 'a91808a89623'
down_revision = '44b0d6f2d1b3'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        return 10
    op.create_table('resources', sa.Column('created_at', sa.DateTime(), nullable=True), sa.Column('updated_at', sa.DateTime(), nullable=True), sa.Column('id', sa.Integer(), nullable=False), sa.Column('name', sa.String(), nullable=True), sa.Column('storage_type', sa.String(), nullable=True), sa.Column('path', sa.String(), nullable=True), sa.Column('size', sa.Integer(), nullable=True), sa.Column('type', sa.String(), nullable=True), sa.Column('channel', sa.String(), nullable=True), sa.Column('project_id', sa.Integer(), nullable=True), sa.PrimaryKeyConstraint('id'))
    op.add_column('agent_execution_feeds', sa.Column('extra_info', sa.String(), nullable=True))
    op.add_column('agent_executions', sa.Column('name', sa.String(), nullable=True))

def downgrade() -> None:
    if False:
        i = 10
        return i + 15
    op.drop_column('agent_executions', 'name')
    op.drop_column('agent_execution_feeds', 'extra_info')
    op.drop_table('resources')