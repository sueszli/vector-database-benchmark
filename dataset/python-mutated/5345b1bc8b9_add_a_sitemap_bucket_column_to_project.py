"""
Add a sitemap_bucket column to Project

Revision ID: 5345b1bc8b9
Revises: 4ec0adada10
Create Date: 2015-09-06 19:56:58.188767
"""
import sqlalchemy as sa
from alembic import op
revision = '5345b1bc8b9'
down_revision = '4ec0adada10'

def upgrade():
    if False:
        print('Hello World!')
    op.add_column('packages', sa.Column('sitemap_bucket', sa.Text(), nullable=True))
    op.execute('\n        UPDATE packages\n        SET sitemap_bucket = sitemap_bucket(name)\n        WHERE sitemap_bucket IS NULL\n        ')
    op.alter_column('packages', 'sitemap_bucket', nullable=False)
    op.execute(' CREATE OR REPLACE FUNCTION maintain_project_sitemap_bucket()\n            RETURNS TRIGGER AS $$\n                BEGIN\n                    NEW.sitemap_bucket := sitemap_bucket(NEW.name);\n                    RETURN NEW;\n                END;\n            $$\n            LANGUAGE plpgsql\n        ')
    op.execute(' CREATE TRIGGER projects_update_sitemap_bucket\n            BEFORE INSERT OR UPDATE OF name ON packages\n            FOR EACH ROW\n            EXECUTE PROCEDURE maintain_project_sitemap_bucket()\n        ')

def downgrade():
    if False:
        return 10
    op.drop_column('packages', 'sitemap_bucket')