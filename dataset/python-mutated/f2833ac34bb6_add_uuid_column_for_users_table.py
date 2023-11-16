"""add UUID column for users table

Revision ID: f2833ac34bb6
Revises: 6db892e17271
Create Date: 2018-08-13 18:10:19.914274

"""
import uuid
import sqlalchemy as sa
from alembic import op
revision = 'f2833ac34bb6'
down_revision = '6db892e17271'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        for i in range(10):
            print('nop')
    conn = op.get_bind()
    conn.execute('PRAGMA legacy_alter_table=ON')
    op.rename_table('journalists', 'journalists_tmp')
    op.add_column('journalists_tmp', sa.Column('uuid', sa.String(length=36)))
    journalists = conn.execute(sa.text('SELECT * FROM journalists_tmp')).fetchall()
    for journalist in journalists:
        conn.execute(sa.text('UPDATE journalists_tmp SET uuid=:journalist_uuid WHERE\n                       id=:id').bindparams(journalist_uuid=str(uuid.uuid4()), id=journalist.id))
    op.create_table('journalists', sa.Column('id', sa.Integer(), nullable=False), sa.Column('uuid', sa.String(length=36), nullable=False), sa.Column('username', sa.String(length=255), nullable=False), sa.Column('pw_salt', sa.Binary(), nullable=True), sa.Column('pw_hash', sa.Binary(), nullable=True), sa.Column('passphrase_hash', sa.String(length=256), nullable=True), sa.Column('is_admin', sa.Boolean(), nullable=True), sa.Column('otp_secret', sa.String(length=16), nullable=True), sa.Column('is_totp', sa.Boolean(), nullable=True), sa.Column('hotp_counter', sa.Integer(), nullable=True), sa.Column('last_token', sa.String(length=6), nullable=True), sa.Column('created_on', sa.DateTime(), nullable=True), sa.Column('last_access', sa.DateTime(), nullable=True), sa.PrimaryKeyConstraint('id'), sa.UniqueConstraint('username'), sa.UniqueConstraint('uuid'))
    conn.execute('\n        INSERT INTO journalists\n        SELECT id, uuid, username, pw_salt, pw_hash, passphrase_hash,\n               is_admin, otp_secret, is_totp, hotp_counter, last_token,\n               created_on, last_access\n        FROM journalists_tmp\n    ')
    op.drop_table('journalists_tmp')

def downgrade() -> None:
    if False:
        return 10
    with op.batch_alter_table('journalists', schema=None) as batch_op:
        batch_op.drop_column('uuid')