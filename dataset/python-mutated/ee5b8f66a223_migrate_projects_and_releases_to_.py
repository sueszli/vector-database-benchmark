"""
migrate projects and releases to surrogate primary_key

Revision ID: ee5b8f66a223
Revises: e82c3a017d60
Create Date: 2018-10-27 16:31:38.859484
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
revision = 'ee5b8f66a223'
down_revision = 'eeb23d9b4d00'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.add_column('packages', sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False))
    op.add_column('releases', sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False))
    op.add_column('roles', sa.Column('project_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column('releases', sa.Column('project_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column('release_files', sa.Column('release_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column('release_dependencies', sa.Column('release_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column('release_classifiers', sa.Column('release_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column('warehouse_admin_squat', sa.Column('squattee_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.add_column('warehouse_admin_squat', sa.Column('squatter_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.execute(' UPDATE releases\n            SET project_id = packages.id\n            FROM packages\n            WHERE releases.name = packages.name\n        ')
    op.execute(' UPDATE roles\n            SET project_id = packages.id\n            FROM packages\n            WHERE\n                packages.name = roles.package_name\n        ')
    op.execute(' UPDATE release_files\n            SET release_id = releases.id\n            FROM releases\n            WHERE\n                release_files.name = releases.name\n                AND release_files.version = releases.version\n        ')
    op.execute(' DELETE FROM release_dependencies\n            WHERE\n                name IS NULL AND version IS NULL\n        ')
    op.execute(' UPDATE release_dependencies\n            SET release_id = releases.id\n            FROM releases\n            WHERE\n                release_dependencies.name = releases.name\n                AND release_dependencies.version = releases.version\n        ')
    op.execute(' UPDATE release_classifiers\n            SET release_id = releases.id\n            FROM releases\n            WHERE\n                release_classifiers.name = releases.name\n                AND release_classifiers.version = releases.version\n        ')
    op.execute(' UPDATE warehouse_admin_squat\n            SET squattee_id = packages.id\n            FROM packages\n            WHERE\n                packages.name = warehouse_admin_squat.squattee_name\n        ')
    op.execute(' UPDATE warehouse_admin_squat\n            SET squatter_id = packages.id\n            FROM packages\n            WHERE\n                packages.name = warehouse_admin_squat.squatter_name\n        ')
    op.execute("DELETE FROM roles WHERE role_name = 'Admin'")
    op.alter_column('roles', 'project_id', nullable=False)
    op.alter_column('releases', 'project_id', nullable=False)
    op.alter_column('release_files', 'release_id', nullable=False)
    op.alter_column('release_dependencies', 'release_id', nullable=False)
    op.alter_column('release_classifiers', 'release_id', nullable=False)
    op.alter_column('warehouse_admin_squat', 'squattee_id', nullable=False)
    op.alter_column('warehouse_admin_squat', 'squatter_id', nullable=False)
    op.drop_constraint('release_classifiers_name_fkey', 'release_classifiers', type_='foreignkey')
    op.drop_constraint('release_dependencies_name_fkey', 'release_dependencies', type_='foreignkey')
    op.drop_constraint('release_files_name_fkey', 'release_files', type_='foreignkey')
    op.drop_constraint('releases_name_fkey', 'releases', type_='foreignkey')
    op.drop_constraint('warehouse_admin_squat_squattee_name_fkey', 'warehouse_admin_squat', type_='foreignkey')
    op.drop_constraint('warehouse_admin_squat_squatter_name_fkey', 'warehouse_admin_squat', type_='foreignkey')
    op.execute('ALTER TABLE packages DROP CONSTRAINT packages_pkey CASCADE')
    op.create_primary_key(None, 'packages', ['id'])
    op.create_index('release_normalized_name_version_idx', 'releases', [sa.text('normalize_pep426_name(name)'), 'version'], unique=True)
    op.execute('ALTER TABLE releases DROP CONSTRAINT releases_pkey CASCADE')
    op.create_primary_key(None, 'releases', ['id'])
    op.create_foreign_key(None, 'releases', 'packages', ['project_id'], ['id'], onupdate='CASCADE', ondelete='CASCADE')
    op.create_foreign_key(None, 'roles', 'packages', ['project_id'], ['id'], onupdate='CASCADE', ondelete='CASCADE')
    op.create_foreign_key(None, 'release_files', 'releases', ['release_id'], ['id'], onupdate='CASCADE', ondelete='CASCADE')
    op.create_foreign_key(None, 'release_dependencies', 'releases', ['release_id'], ['id'], onupdate='CASCADE', ondelete='CASCADE')
    op.create_foreign_key(None, 'release_classifiers', 'releases', ['release_id'], ['id'], onupdate='CASCADE', ondelete='CASCADE')
    op.create_foreign_key(None, 'warehouse_admin_squat', 'packages', ['squattee_id'], ['id'], onupdate='CASCADE', ondelete='CASCADE')
    op.create_foreign_key(None, 'warehouse_admin_squat', 'packages', ['squatter_id'], ['id'], onupdate='CASCADE', ondelete='CASCADE')
    op.drop_index('rel_dep_name_version_kind_idx', table_name='release_dependencies')
    op.create_index('release_dependencies_release_kind_idx', 'release_dependencies', ['release_id', 'kind'])
    op.drop_index('release_name_created_idx', table_name='releases')
    op.create_index('release_project_created_idx', 'releases', ['project_id', sa.text('created DESC')])
    op.drop_index('release_files_name_version_idx', table_name='release_files')
    op.drop_index('release_files_version_idx', table_name='release_files')
    op.drop_index('release_files_single_sdist', table_name='release_files')
    op.create_index('release_files_release_id_idx', 'release_files', ['release_id'])
    op.create_index('release_files_single_sdist', 'release_files', ['release_id', 'packagetype'], unique=True, postgresql_where=sa.text("packagetype = 'sdist' AND allow_multiple_sdist = false"))
    op.drop_index('rel_class_name_version_idx', table_name='release_classifiers')
    op.drop_index('rel_class_version_id_idx', table_name='release_classifiers')
    op.create_index('rel_class_release_id_idx', 'release_classifiers', ['release_id'])
    op.drop_column('roles', 'package_name')
    op.drop_column('releases', 'name')
    op.drop_column('release_files', 'name')
    op.drop_column('release_files', 'version')
    op.drop_column('release_classifiers', 'name')
    op.drop_column('release_classifiers', 'version')
    op.drop_column('release_dependencies', 'name')
    op.drop_column('release_dependencies', 'version')
    op.drop_column('warehouse_admin_squat', 'squattee_name')
    op.drop_column('warehouse_admin_squat', 'squatter_name')
    op.execute("CREATE OR REPLACE FUNCTION update_release_files_requires_python()\n            RETURNS TRIGGER AS $$\n            BEGIN\n                IF (TG_TABLE_NAME = 'releases') THEN\n                    UPDATE\n                        release_files\n                    SET\n                        requires_python = releases.requires_python\n                    FROM releases\n                    WHERE\n                        release_files.release_id = releases.id\n                            AND releases.id = NEW.id;\n                ELSEIF (TG_TABLE_NAME = 'release_files') THEN\n                    UPDATE\n                        release_files\n                    SET\n                        requires_python = releases.requires_python\n                    FROM releases\n                    WHERE\n                        release_files.release_id = releases.id\n                            AND releases.id = NEW.release_id;\n                END IF;\n\n                RETURN NULL;\n            END;\n            $$ LANGUAGE plpgsql;\n        ")

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    raise RuntimeError('Order No. 227 - Ни шагу назад!')