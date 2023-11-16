"""
Add a table to hold blacklisted projects

Revision ID: b6a20b9c888d
Revises: 5b3f9e687d94
Create Date: 2017-09-15 16:24:03.201478
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
revision = 'b6a20b9c888d'
down_revision = '5b3f9e687d94'

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.create_table('blacklist', sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False), sa.Column('created', sa.DateTime(), server_default=sa.text('now()'), nullable=False), sa.Column('name', sa.Text(), nullable=False), sa.Column('blacklisted_by', postgresql.UUID(), nullable=True), sa.Column('comment', sa.Text(), server_default='', nullable=False), sa.CheckConstraint("name ~* '^([A-Z0-9]|[A-Z0-9][A-Z0-9._-]*[A-Z0-9])$'::text", name='blacklist_valid_name'), sa.ForeignKeyConstraint(['blacklisted_by'], ['accounts_user.id']), sa.PrimaryKeyConstraint('id'), sa.UniqueConstraint('name'))
    op.execute(' CREATE OR REPLACE FUNCTION ensure_normalized_blacklist()\n            RETURNS TRIGGER AS $$\n            BEGIN\n                NEW.name = normalize_pep426_name(NEW.name);\n                RETURN NEW;\n            END;\n            $$ LANGUAGE plpgsql;\n        ')
    op.execute(' CREATE TRIGGER normalize_blacklist\n            AFTER INSERT OR UPDATE OR DELETE ON blacklist\n            FOR EACH ROW EXECUTE PROCEDURE ensure_normalized_blacklist();\n        ')

def downgrade():
    if False:
        i = 10
        return i + 15
    raise RuntimeError('Order No. 227 - Ни шагу назад!')