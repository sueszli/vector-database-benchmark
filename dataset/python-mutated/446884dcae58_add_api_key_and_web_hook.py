"""add api_key and web_hook

Revision ID: 446884dcae58
Revises: 71e3980d55f5
Create Date: 2023-07-29 10:55:21.714245

"""
from alembic import op
import sqlalchemy as sa
revision = '446884dcae58'
down_revision = '2fbd6472112c'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        print('Hello World!')
    op.create_table('api_keys', sa.Column('id', sa.Integer(), nullable=False), sa.Column('org_id', sa.Integer(), nullable=True), sa.Column('name', sa.String(), nullable=True), sa.Column('key', sa.String(), nullable=True), sa.Column('created_at', sa.DateTime(), nullable=True), sa.Column('updated_at', sa.DateTime(), nullable=True), sa.Column('is_expired', sa.Boolean(), nullable=True, default=False), sa.PrimaryKeyConstraint('id'))
    op.create_table('webhooks', sa.Column('id', sa.Integer(), nullable=False), sa.Column('name', sa.String(), nullable=True), sa.Column('org_id', sa.Integer(), nullable=True), sa.Column('url', sa.String(), nullable=True), sa.Column('headers', sa.JSON(), nullable=True), sa.Column('created_at', sa.DateTime(), nullable=True), sa.Column('updated_at', sa.DateTime(), nullable=True), sa.Column('is_deleted', sa.Boolean(), nullable=True), sa.PrimaryKeyConstraint('id'))
    op.create_table('webhook_events', sa.Column('id', sa.Integer(), nullable=False), sa.Column('agent_id', sa.Integer(), nullable=True), sa.Column('run_id', sa.Integer(), nullable=True), sa.Column('event', sa.String(), nullable=True), sa.Column('status', sa.String(), nullable=True), sa.Column('errors', sa.Text(), nullable=True), sa.Column('created_at', sa.DateTime(), nullable=True), sa.Column('updated_at', sa.DateTime(), nullable=True), sa.PrimaryKeyConstraint('id'))

def downgrade() -> None:
    if False:
        i = 10
        return i + 15
    op.drop_table('webhooks')
    op.drop_table('api_keys')
    op.drop_table('webhook_events')