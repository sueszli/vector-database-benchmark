"""update datasources

Revision ID: 937d04c16b64
Revises: d94d33dbe938
Create Date: 2018-07-20 16:08:10.195843

"""
revision = '937d04c16b64'
down_revision = 'd94d33dbe938'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        print('Hello World!')
    with op.batch_alter_table('datasources') as batch_op:
        batch_op.alter_column('datasource_name', existing_type=sa.String(255), nullable=False)

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    with op.batch_alter_table('datasources') as batch_op:
        batch_op.alter_column('datasource_name', existing_type=sa.String(255), nullable=True)