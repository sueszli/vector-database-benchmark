"""
Use enums instead of text

Revision ID: 111d8fc0443
Revises: 5988e3e8d2e
Create Date: 2015-03-08 22:46:46.870190
"""
from alembic import op
from sqlalchemy.dialects.postgresql import ENUM
revision = '111d8fc0443'
down_revision = '5988e3e8d2e'

def upgrade():
    if False:
        i = 10
        return i + 15
    package_type = ENUM('bdist_dmg', 'bdist_dumb', 'bdist_egg', 'bdist_msi', 'bdist_rpm', 'bdist_wheel', 'bdist_wininst', 'sdist', name='package_type', create_type=False)
    package_type.create(op.get_bind(), checkfirst=False)
    op.execute(' ALTER TABLE release_files\n                ALTER COLUMN packagetype\n                TYPE package_type\n                USING packagetype::package_type\n        ')

def downgrade():
    if False:
        return 10
    op.execute(' ALTER TABLE release_files\n                ALTER COLUMN packagetype\n                TYPE text\n        ')
    ENUM(name='package_type', create_type=False).drop(op.get_bind(), checkfirst=False)