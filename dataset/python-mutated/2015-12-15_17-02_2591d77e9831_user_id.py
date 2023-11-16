"""user_id

Revision ID: 2591d77e9831
Revises: 12d55656cbca
Create Date: 2015-12-15 17:02:45.128709

"""
revision = '2591d77e9831'
down_revision = '12d55656cbca'
import sqlalchemy as sa
from alembic import op

def upgrade():
    if False:
        return 10
    with op.batch_alter_table('tables') as batch_op:
        batch_op.add_column(sa.Column('user_id', sa.Integer()))
        batch_op.create_foreign_key('user_id', 'ab_user', ['user_id'], ['id'])

def downgrade():
    if False:
        while True:
            i = 10
    with op.batch_alter_table('tables') as batch_op:
        batch_op.drop_constraint('user_id', type_='foreignkey')
        batch_op.drop_column('user_id')