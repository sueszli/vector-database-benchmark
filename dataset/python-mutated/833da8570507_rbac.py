"""
rbac changes for jupyterhub 2.0

Revision ID: 833da8570507
Revises: 4dc2d5a8c53c
Create Date: 2021-02-17 15:03:04.360368

"""
revision = '833da8570507'
down_revision = '4dc2d5a8c53c'
branch_labels = None
depends_on = None
import sqlalchemy as sa
from alembic import op
from jupyterhub import orm
naming_convention = orm.meta.naming_convention

def upgrade():
    if False:
        i = 10
        return i + 15
    for table_name in ('services', 'spawners'):
        column_name = 'oauth_client_id'
        target_table = 'oauth_clients'
        target_column = 'identifier'
        with op.batch_alter_table(table_name, schema=None) as batch_op:
            batch_op.add_column(sa.Column('oauth_client_id', sa.Unicode(length=255), nullable=True))
            batch_op.create_foreign_key(naming_convention['fk'] % dict(table_name=table_name, column_0_name=column_name, referred_table_name=target_table), target_table, [column_name], [target_column], ondelete='SET NULL')
    op.drop_table('api_tokens')
    op.drop_table('oauth_access_tokens')
    return

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    for table_name in ('services', 'spawners'):
        column_name = 'oauth_client_id'
        target_table = 'oauth_clients'
        target_column = 'identifier'
        with op.batch_alter_table(table_name, schema=None, naming_convention=orm.meta.naming_convention) as batch_op:
            batch_op.drop_constraint(naming_convention['fk'] % dict(table_name=table_name, column_0_name=column_name, referred_table_name=target_table), type_='foreignkey')
            batch_op.drop_column(column_name)
    op.drop_table('api_tokens')
    op.drop_table('api_token_role_map')
    op.drop_table('service_role_map')
    op.drop_table('user_role_map')
    op.drop_table('roles')