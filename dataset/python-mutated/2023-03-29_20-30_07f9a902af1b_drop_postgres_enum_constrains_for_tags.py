"""drop postgres enum constrains for tags

Revision ID: 07f9a902af1b
Revises: b5ea9d343307
Create Date: 2023-03-29 20:30:10.214951

"""
revision = '07f9a902af1b'
down_revision = 'b5ea9d343307'
from alembic import op
from sqlalchemy.dialects import postgresql

def upgrade():
    if False:
        i = 10
        return i + 15
    conn = op.get_bind()
    if isinstance(conn.dialect, postgresql.dialect):
        conn.execute('ALTER TABLE "tagged_object" ALTER COLUMN "object_type" TYPE VARCHAR')
        conn.execute('ALTER TABLE "tag" ALTER COLUMN "type" TYPE VARCHAR')
        conn.execute('DROP TYPE IF EXISTS objecttypes')
        conn.execute('DROP TYPE IF EXISTS tagtypes')

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    pass