"""api_token_scopes

Revision ID: 651f5419b74d
Revises: 833da8570507
Create Date: 2022-02-28 12:42:55.149046

"""
revision = '651f5419b74d'
down_revision = '833da8570507'
branch_labels = None
depends_on = None
import sqlalchemy as sa
from alembic import op
from sqlalchemy import Column, ForeignKey, Table, text
from sqlalchemy.orm import raiseload, relationship, selectinload
from sqlalchemy.orm.session import Session
from jupyterhub import orm, roles

def access_scopes(oauth_client: orm.OAuthClient, db: Session):
    if False:
        for i in range(10):
            print('nop')
    'Return scope(s) required to access an oauth client\n    This is a clone of `scopes.access_scopes` without using\n    the `orm.Service`\n    '
    scopes = set()
    if oauth_client.identifier == 'jupyterhub':
        return frozenset()
    spawner = oauth_client.spawner
    if spawner:
        scopes.add(f'access:servers!server={spawner.user.name}/{spawner.name}')
    else:
        statement = 'SELECT * FROM services WHERE oauth_client_id = :identifier'
        service = db.execute(text(statement), {'identifier': oauth_client.identifier}).fetchall()
        if len(service) > 0:
            scopes.add(f'access:services!service={service[0].name}')
    return frozenset(scopes)

def upgrade():
    if False:
        i = 10
        return i + 15
    c = op.get_bind()
    tables = sa.inspect(c.engine).get_table_names()
    if 'oauth_code_role_map' in tables:
        op.drop_table('oauth_code_role_map')
    if 'oauth_codes' in tables:
        op.add_column('oauth_codes', sa.Column('scopes', orm.JSONList(), nullable=True))
    if 'api_tokens' in tables:
        op.add_column('api_tokens', sa.Column('scopes', orm.JSONList(), nullable=True))
        if 'api_token_role_map' in tables:
            token_role_map = Table('api_token_role_map', orm.Base.metadata, Column('api_token_id', ForeignKey('api_tokens.id', ondelete='CASCADE'), primary_key=True), Column('role_id', ForeignKey('roles.id', ondelete='CASCADE'), primary_key=True), extend_existing=True)
            orm.APIToken.roles = relationship('Role', secondary='api_token_role_map')
            db = Session(bind=c)
            for token in db.query(orm.APIToken).options(selectinload(orm.APIToken.roles), raiseload('*')):
                token.scopes = list(roles.roles_to_scopes(token.roles))
            db.commit()
            op.drop_table('api_token_role_map')
    if 'oauth_clients' in tables:
        op.add_column('oauth_clients', sa.Column('allowed_scopes', orm.JSONList(), nullable=True))
        if 'oauth_client_role_map' in tables:
            client_role_map = Table('oauth_client_role_map', orm.Base.metadata, Column('oauth_client_id', ForeignKey('oauth_clients.id', ondelete='CASCADE'), primary_key=True), Column('role_id', ForeignKey('roles.id', ondelete='CASCADE'), primary_key=True), extend_existing=True)
            orm.OAuthClient.allowed_roles = relationship('Role', secondary='oauth_client_role_map')
            db = Session(bind=c)
            for oauth_client in db.query(orm.OAuthClient):
                allowed_scopes = set(roles.roles_to_scopes(oauth_client.allowed_roles))
                allowed_scopes.update(access_scopes(oauth_client, db))
                oauth_client.allowed_scopes = sorted(allowed_scopes)
            db.commit()
            op.drop_table('oauth_client_role_map')

def downgrade():
    if False:
        for i in range(10):
            print('nop')
    op.drop_table('api_tokens')
    op.drop_table('oauth_clients')
    op.drop_table('oauth_codes')