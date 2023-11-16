"""create_ssh_tunnel_credentials_tbl

Revision ID: f3c2d8ec8595
Revises: 4ce1d9b25135
Create Date: 2022-10-20 10:48:08.722861

"""
revision = 'f3c2d8ec8595'
down_revision = '4ce1d9b25135'
from uuid import uuid4
import sqlalchemy as sa
from alembic import op
from sqlalchemy_utils import UUIDType
from superset import app
from superset.extensions import encrypted_field_factory
app_config = app.config

def upgrade():
    if False:
        for i in range(10):
            print('nop')
    op.create_table('ssh_tunnels', sa.Column('created_on', sa.DateTime(), nullable=True), sa.Column('changed_on', sa.DateTime(), nullable=True), sa.Column('created_by_fk', sa.Integer(), nullable=True), sa.Column('changed_by_fk', sa.Integer(), nullable=True), sa.Column('extra_json', sa.Text(), nullable=True), sa.Column('uuid', UUIDType(binary=True), primary_key=False, default=uuid4, unique=True, index=True), sa.Column('id', sa.Integer(), primary_key=True), sa.Column('database_id', sa.INTEGER(), sa.ForeignKey('dbs.id'), unique=True, index=True), sa.Column('server_address', sa.String(256)), sa.Column('server_port', sa.INTEGER()), sa.Column('username', encrypted_field_factory.create(sa.String(256))), sa.Column('password', encrypted_field_factory.create(sa.String(256)), nullable=True), sa.Column('private_key', encrypted_field_factory.create(sa.String(1024)), nullable=True), sa.Column('private_key_password', encrypted_field_factory.create(sa.String(256)), nullable=True))

def downgrade():
    if False:
        i = 10
        return i + 15
    op.drop_table('ssh_tunnels')