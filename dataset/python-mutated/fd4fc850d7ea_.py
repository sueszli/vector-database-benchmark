"""Convert user details to jsonb and move user profile image url into details column

Revision ID: fd4fc850d7ea
Revises: 89bc7873a3e0
Create Date: 2022-01-31 15:24:16.507888

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from redash.models import db
revision = 'fd4fc850d7ea'
down_revision = '89bc7873a3e0'
branch_labels = None
depends_on = None

def upgrade():
    if False:
        return 10
    connection = op.get_bind()
    op.alter_column('users', 'details', existing_type=postgresql.JSON(astext_type=sa.Text()), type_=postgresql.JSONB(astext_type=sa.Text()), existing_nullable=True, existing_server_default=sa.text("'{}'::jsonb"))
    update_query = '\n    update users\n    set details = details::jsonb || (\'{"profile_image_url": "\' || profile_image_url || \'"}\')::jsonb\n    where 1=1\n    '
    connection.execute(update_query)
    op.drop_column('users', 'profile_image_url')

def downgrade():
    if False:
        return 10
    connection = op.get_bind()
    op.add_column('users', sa.Column('profile_image_url', db.String(320), nullable=True))
    update_query = "\n    update users set\n    profile_image_url = details->>'profile_image_url',\n    details = details - 'profile_image_url' ;\n    "
    connection.execute(update_query)
    db.session.commit()
    op.alter_column('users', 'details', existing_type=postgresql.JSONB(astext_type=sa.Text()), type_=postgresql.JSON(astext_type=sa.Text()), existing_nullable=True, existing_server_default=sa.text("'{}'::json"))