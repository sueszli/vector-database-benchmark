"""Make subscription subscriber_uuid, event_type, type not nullable.

Revision ID: d0eac6764a55
Revises: 5acfad9426b4
Create Date: 2022-04-26 13:48:24.744397

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
revision = 'd0eac6764a55'
down_revision = '5acfad9426b4'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.alter_column('subscriptions', 'subscriber_uuid', existing_type=postgresql.UUID(), nullable=False)
    op.alter_column('subscriptions', 'event_type', existing_type=sa.VARCHAR(length=50), nullable=False)
    op.alter_column('subscriptions', 'type', existing_type=sa.VARCHAR(length=50), nullable=False)

def downgrade():
    if False:
        print('Hello World!')
    op.alter_column('subscriptions', 'type', existing_type=sa.VARCHAR(length=50), nullable=True)
    op.alter_column('subscriptions', 'event_type', existing_type=sa.VARCHAR(length=50), nullable=True)
    op.alter_column('subscriptions', 'subscriber_uuid', existing_type=postgresql.UUID(), nullable=True)