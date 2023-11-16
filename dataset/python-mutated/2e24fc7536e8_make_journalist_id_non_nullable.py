"""make journalist_id non-nullable

Revision ID: 2e24fc7536e8
Revises: de00920916bf
Create Date: 2022-01-12 19:31:06.186285

"""
import uuid
import argon2
import sqlalchemy as sa
import two_factor
from alembic import op
from models import ARGON2_PARAMS
from passphrases import PassphraseGenerator
from sdconfig import SecureDropConfig
revision = '2e24fc7536e8'
down_revision = 'de00920916bf'
branch_labels = None
depends_on = None

def generate_passphrase_hash() -> str:
    if False:
        i = 10
        return i + 15
    passphrase = PassphraseGenerator.get_default().generate_passphrase()
    return argon2.PasswordHasher(**ARGON2_PARAMS).hash(passphrase)

def create_deleted() -> int:
    if False:
        i = 10
        return i + 15
    'manually insert a "deleted" journalist user.\n\n    We need to do it this way since the model will reflect the current state of\n    the schema, not what it is at the current migration step\n\n    It should be basically identical to what Journalist.get_deleted() does\n    '
    op.execute(sa.text('        INSERT INTO journalists (uuid, username, session_nonce, passphrase_hash, otp_secret)\n        VALUES (:uuid, "deleted", 0, :passphrase_hash, :otp_secret);\n        ').bindparams(uuid=str(uuid.uuid4()), passphrase_hash=generate_passphrase_hash(), otp_secret=two_factor.random_base32()))
    conn = op.get_bind()
    result = conn.execute('SELECT id FROM journalists WHERE username="deleted";').fetchall()
    return result[0][0]

def migrate_nulls() -> None:
    if False:
        while True:
            i = 10
    'migrate existing journalist_id=NULL over to deleted or delete them'
    op.execute('DELETE FROM journalist_login_attempt WHERE journalist_id IS NULL;')
    op.execute('DELETE FROM revoked_tokens WHERE journalist_id IS NULL;')
    tables = ('replies', 'seen_files', 'seen_messages', 'seen_replies')
    needs_migration = []
    conn = op.get_bind()
    for table in tables:
        result = conn.execute(f'SELECT 1 FROM {table} WHERE journalist_id IS NULL;').first()
        if result is not None:
            needs_migration.append(table)
    if not needs_migration:
        return
    deleted_id = create_deleted()
    for table in needs_migration:
        op.execute(sa.text(f'UPDATE OR IGNORE {table} SET journalist_id=:journalist_id WHERE journalist_id IS NULL;').bindparams(journalist_id=deleted_id))
        op.execute(f'DELETE FROM {table} WHERE journalist_id IS NULL')

def upgrade() -> None:
    if False:
        i = 10
        return i + 15
    try:
        SecureDropConfig.get_current()
    except ModuleNotFoundError:
        return
    migrate_nulls()
    with op.batch_alter_table('journalist_login_attempt', schema=None) as batch_op:
        batch_op.alter_column('journalist_id', existing_type=sa.INTEGER(), nullable=False)
    with op.batch_alter_table('replies', schema=None) as batch_op:
        batch_op.alter_column('journalist_id', existing_type=sa.INTEGER(), nullable=False)
    with op.batch_alter_table('revoked_tokens', schema=None) as batch_op:
        batch_op.alter_column('journalist_id', existing_type=sa.INTEGER(), nullable=False)
    with op.batch_alter_table('seen_files', schema=None) as batch_op:
        batch_op.alter_column('journalist_id', existing_type=sa.INTEGER(), nullable=False)
    with op.batch_alter_table('seen_messages', schema=None) as batch_op:
        batch_op.alter_column('journalist_id', existing_type=sa.INTEGER(), nullable=False)
    with op.batch_alter_table('seen_replies', schema=None) as batch_op:
        batch_op.alter_column('journalist_id', existing_type=sa.INTEGER(), nullable=False)

def downgrade() -> None:
    if False:
        print('Hello World!')
    with op.batch_alter_table('seen_replies', schema=None) as batch_op:
        batch_op.alter_column('journalist_id', existing_type=sa.INTEGER(), nullable=True)
    with op.batch_alter_table('seen_messages', schema=None) as batch_op:
        batch_op.alter_column('journalist_id', existing_type=sa.INTEGER(), nullable=True)
    with op.batch_alter_table('seen_files', schema=None) as batch_op:
        batch_op.alter_column('journalist_id', existing_type=sa.INTEGER(), nullable=True)
    with op.batch_alter_table('revoked_tokens', schema=None) as batch_op:
        batch_op.alter_column('journalist_id', existing_type=sa.INTEGER(), nullable=True)
    with op.batch_alter_table('replies', schema=None) as batch_op:
        batch_op.alter_column('journalist_id', existing_type=sa.INTEGER(), nullable=True)
    with op.batch_alter_table('journalist_login_attempt', schema=None) as batch_op:
        batch_op.alter_column('journalist_id', existing_type=sa.INTEGER(), nullable=True)