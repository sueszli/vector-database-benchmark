"""Change datatype of type in BaseColumn

Revision ID: 3ba29ecbaac5
Revises: abe27eaf93db
Create Date: 2021-11-02 17:44:51.792138

"""
revision = '3ba29ecbaac5'
down_revision = 'abe27eaf93db'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        return 10
    with op.batch_alter_table('table_columns') as batch_op:
        batch_op.alter_column('type', existing_type=sa.VARCHAR(length=32), type_=sa.TEXT())

def downgrade():
    if False:
        while True:
            i = 10
    with op.batch_alter_table('table_columns') as batch_op:
        batch_op.alter_column('type', existing_type=sa.TEXT(), type_=sa.VARCHAR(length=32))