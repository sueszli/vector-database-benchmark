"""0.8 changes

- encrypted auth_state
- remove proxy/hub data from db

OAuth data was also added in this revision,
but no migration to do because they are entirely new tables,
which will be created on launch.

Revision ID: 3ec6993fe20c
Revises: af4cbdb2d13c
Create Date: 2017-07-28 16:44:40.413648

"""
revision = '3ec6993fe20c'
down_revision = 'af4cbdb2d13c'
branch_labels = None
depends_on = None
import logging
logger = logging.getLogger('alembic')
import sqlalchemy as sa
from alembic import op
from jupyterhub.orm import JSONDict

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_table('proxies')
    op.drop_table('hubs')
    try:
        op.drop_column('users', 'auth_state')
        if op.get_context().dialect.name == 'mysql':
            op.drop_constraint('users_ibfk_1', 'users', type_='foreignkey')
        op.drop_column('users', '_server_id')
    except sa.exc.OperationalError:
        if op.get_context().dialect.name == 'sqlite':
            logger.warning('sqlite cannot drop columns. Leaving unused old columns in place.')
        else:
            raise
    op.add_column('users', sa.Column('encrypted_auth_state', sa.types.LargeBinary))

def downgrade():
    if False:
        i = 10
        return i + 15
    engine = op.get_bind().engine
    for table in ('oauth_clients', 'oauth_codes', 'oauth_access_tokens', 'spawners'):
        if engine.has_table(table):
            op.drop_table(table)
    op.drop_column('users', 'encrypted_auth_state')
    op.add_column('users', sa.Column('auth_state', JSONDict))
    op.add_column('users', sa.Column('_server_id', sa.Integer, sa.ForeignKey('servers.id')))