"""Remove limit used from query model

Revision ID: d7c1a0d6f2da
Revises: afc69274c25a
Create Date: 2019-06-04 10:12:36.675369

"""
revision = 'd7c1a0d6f2da'
down_revision = 'afc69274c25a'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        i = 10
        return i + 15
    with op.batch_alter_table('query') as batch_op:
        batch_op.drop_column('limit_used')

def downgrade():
    if False:
        print('Hello World!')
    op.add_column('query', sa.Column('limit_used', sa.BOOLEAN(), nullable=True))