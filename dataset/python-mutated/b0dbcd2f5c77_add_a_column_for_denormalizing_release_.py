"""
Add a column for denormalizing Release.is_prerelease

Revision ID: b0dbcd2f5c77
Revises: 8bee9c119e41
Create Date: 2022-06-27 17:19:00.117464
"""
import sqlalchemy as sa
from alembic import op
revision = 'b0dbcd2f5c77'
down_revision = '1e61006a47c2'

def upgrade():
    if False:
        print('Hello World!')
    op.add_column('releases', sa.Column('is_prerelease', sa.Boolean(), nullable=True))
    op.execute(' CREATE OR REPLACE FUNCTION maintain_releases_is_prerelease()\n            RETURNS TRIGGER AS $$\n                BEGIN\n                    NEW.is_prerelease :=  pep440_is_prerelease(NEW.version);\n                    RETURN NEW;\n                END;\n            $$\n            LANGUAGE plpgsql\n        ')
    op.execute(' CREATE TRIGGER releases_update_is_prerelease\n            BEFORE INSERT OR UPDATE OF version ON releases\n            FOR EACH ROW\n            EXECUTE PROCEDURE maintain_releases_is_prerelease()\n        ')

def downgrade():
    if False:
        while True:
            i = 10
    op.drop_column('releases', 'is_prerelease')