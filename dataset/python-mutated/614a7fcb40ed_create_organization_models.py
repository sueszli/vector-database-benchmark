"""
Create Organization models

Revision ID: 614a7fcb40ed
Revises: 5e02c4f9f95c
Create Date: 2022-04-13 17:23:17.396325
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
revision = '614a7fcb40ed'
down_revision = '5e02c4f9f95c'

def upgrade():
    if False:
        while True:
            i = 10
    op.create_table('organizations', sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False), sa.Column('name', sa.Text(), nullable=False), sa.Column('display_name', sa.Text(), nullable=False), sa.Column('orgtype', sa.Text(), nullable=False), sa.Column('link_url', sa.Text(), nullable=False), sa.Column('description', sa.Text(), nullable=False), sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.sql.false()), sa.Column('is_approved', sa.Boolean(), nullable=True), sa.Column('created', sa.DateTime(), server_default=sa.text('now()'), nullable=False), sa.Column('date_approved', sa.DateTime(), nullable=True), sa.CheckConstraint("name ~* '^([A-Z0-9]|[A-Z0-9][A-Z0-9._-]*[A-Z0-9])$'::text", name='organizations_valid_name'), sa.PrimaryKeyConstraint('id'))
    op.create_index(op.f('ix_organizations_created'), 'organizations', ['created'], unique=False)
    op.create_table('organization_invitations', sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False), sa.Column('invite_status', sa.Text(), nullable=False), sa.Column('token', sa.Text(), nullable=False), sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False), sa.Column('organization_id', postgresql.UUID(as_uuid=True), nullable=False), sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], onupdate='CASCADE', ondelete='CASCADE'), sa.ForeignKeyConstraint(['user_id'], ['users.id'], onupdate='CASCADE', ondelete='CASCADE'), sa.PrimaryKeyConstraint('id'), sa.UniqueConstraint('user_id', 'organization_id', name='_organization_invitations_user_organization_uc'))
    op.create_index(op.f('ix_organization_invitations_organization_id'), 'organization_invitations', ['organization_id'], unique=False)
    op.create_index(op.f('ix_organization_invitations_user_id'), 'organization_invitations', ['user_id'], unique=False)
    op.create_index('organization_invitations_user_id_idx', 'organization_invitations', ['user_id'], unique=False)
    op.create_table('organization_name_catalog', sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False), sa.Column('normalized_name', sa.Text(), nullable=False), sa.Column('organization_id', postgresql.UUID(as_uuid=True), nullable=False), sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], onupdate='CASCADE', ondelete='CASCADE'), sa.PrimaryKeyConstraint('id'), sa.UniqueConstraint('normalized_name', 'organization_id', name='_organization_name_catalog_normalized_name_organization_uc'))
    op.create_index('organization_name_catalog_normalized_name_idx', 'organization_name_catalog', ['normalized_name'], unique=False)
    op.create_index('organization_name_catalog_organization_id_idx', 'organization_name_catalog', ['organization_id'], unique=False)
    op.create_table('organization_project', sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False), sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.sql.false()), sa.Column('organization_id', postgresql.UUID(as_uuid=True), nullable=False), sa.Column('project_id', postgresql.UUID(as_uuid=True), nullable=False), sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], onupdate='CASCADE', ondelete='CASCADE'), sa.ForeignKeyConstraint(['project_id'], ['projects.id'], onupdate='CASCADE', ondelete='CASCADE'), sa.PrimaryKeyConstraint('id'), sa.UniqueConstraint('organization_id', 'project_id', name='_organization_project_organization_project_uc'))
    op.create_index('organization_project_organization_id_idx', 'organization_project', ['organization_id'], unique=False)
    op.create_index('organization_project_project_id_idx', 'organization_project', ['project_id'], unique=False)
    op.create_table('organization_roles', sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False), sa.Column('role_name', sa.Text(), nullable=False), sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False), sa.Column('organization_id', postgresql.UUID(as_uuid=True), nullable=False), sa.ForeignKeyConstraint(['organization_id'], ['organizations.id'], onupdate='CASCADE', ondelete='CASCADE'), sa.ForeignKeyConstraint(['user_id'], ['users.id'], onupdate='CASCADE', ondelete='CASCADE'), sa.PrimaryKeyConstraint('id'), sa.UniqueConstraint('user_id', 'organization_id', name='_organization_roles_user_organization_uc'))
    op.create_index('organization_roles_organization_id_idx', 'organization_roles', ['organization_id'], unique=False)
    op.create_index('organization_roles_user_id_idx', 'organization_roles', ['user_id'], unique=False)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_index('organization_roles_user_id_idx', table_name='organization_roles')
    op.drop_index('organization_roles_organization_id_idx', table_name='organization_roles')
    op.drop_table('organization_roles')
    op.drop_index('organization_project_project_id_idx', table_name='organization_project')
    op.drop_index('organization_project_organization_id_idx', table_name='organization_project')
    op.drop_table('organization_project')
    op.drop_index('organization_name_catalog_organization_id_idx', table_name='organization_name_catalog')
    op.drop_index('organization_name_catalog_name_idx', table_name='organization_name_catalog')
    op.drop_table('organization_name_catalog')
    op.drop_index('organization_invitations_user_id_idx', table_name='organization_invitations')
    op.drop_index(op.f('ix_organization_invitations_user_id'), table_name='organization_invitations')
    op.drop_index(op.f('ix_organization_invitations_organization_id'), table_name='organization_invitations')
    op.drop_table('organization_invitations')
    op.drop_index(op.f('ix_organizations_created'), table_name='organizations')
    op.drop_table('organizations')