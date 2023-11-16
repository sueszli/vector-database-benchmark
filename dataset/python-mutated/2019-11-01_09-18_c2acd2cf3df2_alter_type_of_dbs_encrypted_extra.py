"""alter type of dbs encrypted_extra


Revision ID: c2acd2cf3df2
Revises: cca2f5d568c8
Create Date: 2019-11-01 09:18:36.953603

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy_utils import EncryptedType
revision = 'c2acd2cf3df2'
down_revision = 'cca2f5d568c8'

def upgrade():
    if False:
        return 10
    with op.batch_alter_table('dbs') as batch_op:
        try:
            batch_op.alter_column('encrypted_extra', existing_type=sa.Text(), type_=sa.LargeBinary(), postgresql_using='encrypted_extra::bytea', existing_nullable=True)
        except TypeError:
            batch_op.alter_column('dbs', 'encrypted_extra', existing_type=sa.Text(), type_=sa.LargeBinary(), existing_nullable=True)

def downgrade():
    if False:
        print('Hello World!')
    with op.batch_alter_table('dbs') as batch_op:
        batch_op.alter_column('encrypted_extra', existing_type=sa.LargeBinary(), type_=sa.Text(), existing_nullable=True)