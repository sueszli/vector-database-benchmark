"""
update_name_catalog_to_allow_null_org

Revision ID: d1c00b634ac8
Revises: ad71523546f9
Create Date: 2022-05-11 17:20:56.596019
"""
from alembic import op
from sqlalchemy.dialects import postgresql
revision = 'd1c00b634ac8'
down_revision = 'ad71523546f9'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.alter_column('organization_name_catalog', 'organization_id', existing_type=postgresql.UUID(), nullable=True)
    op.create_index(op.f('ix_organization_name_catalog_normalized_name'), 'organization_name_catalog', ['normalized_name'], unique=False)
    op.create_index(op.f('ix_organization_name_catalog_organization_id'), 'organization_name_catalog', ['organization_id'], unique=False)
    op.drop_constraint('organization_name_catalog_organization_id_fkey', 'organization_name_catalog', type_='foreignkey')

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.create_foreign_key('organization_name_catalog_organization_id_fkey', 'organization_name_catalog', 'organizations', ['organization_id'], ['id'], onupdate='CASCADE', ondelete='CASCADE')
    op.drop_index(op.f('ix_organization_name_catalog_organization_id'), table_name='organization_name_catalog')
    op.drop_index(op.f('ix_organization_name_catalog_normalized_name'), table_name='organization_name_catalog')
    op.alter_column('organization_name_catalog', 'organization_id', existing_type=postgresql.UUID(), nullable=False)