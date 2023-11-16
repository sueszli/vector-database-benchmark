"""PGP public keys

Revision ID: 17c559a7a685
Revises: 811334d7105f
Create Date: 2023-09-21 12:33:56.550634

"""
import traceback
import pretty_bad_protocol as gnupg
import sqlalchemy as sa
from alembic import op
from encryption import EncryptionManager
from sdconfig import SecureDropConfig
import redwood
revision = '17c559a7a685'
down_revision = '811334d7105f'
branch_labels = None
depends_on = None

def upgrade() -> None:
    if False:
        print('Hello World!')
    "\n    Migrate public keys from the GPG keyring into the SQLite database\n\n    We iterate over all the secret keys in the keyring and see if we\n    can identify the corresponding Source record. If we can, and it\n    doesn't already have key material migrated, export the key and\n    save it in the database.\n    "
    try:
        config = SecureDropConfig.get_current()
    except ModuleNotFoundError:
        return
    gpg = gnupg.GPG(binary='gpg2', homedir=str(config.GPG_KEY_DIR), options=['--pinentry-mode loopback', '--trust-model direct'])
    for keyinfo in gpg.list_keys(secret=True):
        if len(keyinfo['uids']) > 1:
            continue
        uid = keyinfo['uids'][0]
        search = EncryptionManager.SOURCE_KEY_UID_RE.search(uid)
        if not search:
            continue
        filesystem_id = search.group(2)
        conn = op.get_bind()
        result = conn.execute(sa.text('\n                SELECT pgp_public_key, pgp_fingerprint\n                FROM sources\n                WHERE filesystem_id=:filesystem_id\n                ').bindparams(filesystem_id=filesystem_id)).first()
        if result != (None, None):
            continue
        fingerprint = keyinfo['fingerprint']
        try:
            public_key = gpg.export_keys(fingerprint)
            redwood.is_valid_public_key(public_key)
        except:
            traceback.print_exc()
            continue
        op.execute(sa.text('\n                UPDATE sources\n                SET pgp_public_key=:pgp_public_key, pgp_fingerprint=:pgp_fingerprint\n                WHERE filesystem_id=:filesystem_id\n                ').bindparams(pgp_public_key=public_key, pgp_fingerprint=fingerprint, filesystem_id=filesystem_id))

def downgrade() -> None:
    if False:
        return 10
    "\n    This is a non-destructive operation, so it's not worth implementing a\n    migration from database storage to GPG.\n    "