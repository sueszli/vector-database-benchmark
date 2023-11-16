"""
Add a sitemap_bucket column to User

Revision ID: 116be7c87e1
Revises: 5345b1bc8b9
Create Date: 2015-09-06 20:28:24.073366
"""
import sqlalchemy as sa
from alembic import op
revision = '116be7c87e1'
down_revision = '5345b1bc8b9'

def upgrade():
    if False:
        return 10
    op.add_column('accounts_user', sa.Column('sitemap_bucket', sa.Text(), nullable=True))
    op.execute('\n        UPDATE accounts_user\n        SET sitemap_bucket = sitemap_bucket(username)\n        WHERE sitemap_bucket IS NULL\n        ')
    op.alter_column('accounts_user', 'sitemap_bucket', nullable=False)
    op.execute(' CREATE OR REPLACE FUNCTION maintain_accounts_user_sitemap_bucket()\n            RETURNS TRIGGER AS $$\n                BEGIN\n                    NEW.sitemap_bucket := sitemap_bucket(NEW.username);\n                    RETURN NEW;\n                END;\n            $$\n            LANGUAGE plpgsql\n        ')
    op.execute(' CREATE TRIGGER accounts_user_update_sitemap_bucket\n            BEFORE INSERT OR UPDATE OF username ON accounts_user\n            FOR EACH ROW\n            EXECUTE PROCEDURE maintain_accounts_user_sitemap_bucket()\n        ')

def downgrade():
    if False:
        return 10
    op.drop_column('accounts_user', 'sitemap_bucket')