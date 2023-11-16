"""
Make users optional with Macaroons

Revision ID: 43bf0b6badcb
Revises: ef0a77c48089
Create Date: 2022-04-19 14:57:54.765006
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
revision = '43bf0b6badcb'
down_revision = 'ef0a77c48089'

def upgrade():
    if False:
        return 10
    op.alter_column('macaroons', 'user_id', existing_type=postgresql.UUID(), nullable=True)
    op.add_column('macaroons', sa.Column('oidc_provider_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_index(op.f('ix_macaroons_oidc_provider_id'), 'macaroons', ['oidc_provider_id'], unique=False)
    op.create_foreign_key(None, 'macaroons', 'oidc_providers', ['oidc_provider_id'], ['id'])
    op.alter_column('journals', 'submitted_by', existing_type=postgresql.CITEXT(), nullable=True)
    op.create_check_constraint('_user_xor_oidc_provider_macaroon', table_name='macaroons', condition='(user_id::text IS NULL) <> (oidc_provider_id::text IS NULL)')

def downgrade():
    if False:
        print('Hello World!')
    op.alter_column('macaroons', 'user_id', existing_type=postgresql.UUID(), nullable=False)
    op.alter_column('journals', 'submitted_by', existing_type=postgresql.CITEXT(), nullable=False)