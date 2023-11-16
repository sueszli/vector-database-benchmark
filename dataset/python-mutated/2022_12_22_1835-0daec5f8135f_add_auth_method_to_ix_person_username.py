"""add_auth_method_to_ix_person_username

Revision ID: 0daec5f8135f
Revises: 6368515778c5
Create Date: 2022-12-22 18:35:59.609013

"""
import sqlalchemy as sa
from alembic import op
revision = '0daec5f8135f'
down_revision = '6368515778c5'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        while True:
            i = 10
    op.drop_index('ix_person_username', table_name='person')
    op.create_index('ix_person_username', 'person', ['api_client_id', 'username', 'auth_method'], unique=True)

def downgrade() -> None:
    if False:
        i = 10
        return i + 15
    op.drop_index('ix_person_username', table_name='person')
    op.create_index('ix_person_username', 'person', ['api_client_id', 'username'], unique=False)