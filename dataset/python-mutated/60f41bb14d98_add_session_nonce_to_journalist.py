"""Add Session Nonce To Journalist

Revision ID: 60f41bb14d98
Revises: a9fe328b053a
Create Date: 2019-08-19 04:20:59.489516

"""
import sqlalchemy as sa
from alembic import op
revision = '60f41bb14d98'
down_revision = 'a9fe328b053a'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        print('Hello World!')
    conn = op.get_bind()
    conn.execute('PRAGMA legacy_alter_table=ON')
    op.rename_table('journalists', 'journalists_tmp')
    op.add_column('journalists_tmp', sa.Column('session_nonce', sa.Integer()))
    journalists = conn.execute(sa.text('SELECT * FROM journalists_tmp')).fetchall()
    for journalist in journalists:
        conn.execute(sa.text('UPDATE journalists_tmp SET session_nonce=0 WHERE\n                       id=:id').bindparams(id=journalist.id))
    op.create_table('journalists', sa.Column('id', sa.Integer(), nullable=False), sa.Column('uuid', sa.String(length=36), nullable=False), sa.Column('username', sa.String(length=255), nullable=False), sa.Column('first_name', sa.String(length=255), nullable=True), sa.Column('last_name', sa.String(length=255), nullable=True), sa.Column('pw_salt', sa.Binary(), nullable=True), sa.Column('pw_hash', sa.Binary(), nullable=True), sa.Column('passphrase_hash', sa.String(length=256), nullable=True), sa.Column('is_admin', sa.Boolean(), nullable=True), sa.Column('session_nonce', sa.Integer(), nullable=False), sa.Column('otp_secret', sa.String(length=16), nullable=True), sa.Column('is_totp', sa.Boolean(), nullable=True), sa.Column('hotp_counter', sa.Integer(), nullable=True), sa.Column('last_token', sa.String(length=6), nullable=True), sa.Column('created_on', sa.DateTime(), nullable=True), sa.Column('last_access', sa.DateTime(), nullable=True), sa.PrimaryKeyConstraint('id'), sa.UniqueConstraint('username'), sa.UniqueConstraint('uuid'))
    conn.execute('\n        INSERT INTO journalists\n        SELECT id, uuid, username, first_name, last_name, pw_salt, pw_hash,\n               passphrase_hash, is_admin, session_nonce, otp_secret, is_totp,\n               hotp_counter, last_token, created_on, last_access\n        FROM journalists_tmp\n    ')
    op.drop_table('journalists_tmp')

def downgrade() -> None:
    if False:
        return 10
    with op.batch_alter_table('journalists', schema=None) as batch_op:
        batch_op.drop_column('session_nonce')