import datetime
import logging
import os
import sqlalchemy
import tink
from .cloud_kms_env_aead import init_tink_env_aead
from .cloud_sql_connection_pool import init_db
logger = logging.getLogger(__name__)

def main() -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Connects to the database, encrypts and inserts some data.\n    '
    db_user = os.environ['DB_USER']
    db_pass = os.environ['DB_PASS']
    db_name = os.environ['DB_NAME']
    db_host = os.environ['DB_HOST']
    db_socket_dir = os.environ.get('DB_SOCKET_DIR', '/cloudsql')
    instance_connection_name = os.environ['INSTANCE_CONNECTION_NAME']
    credentials = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
    key_uri = 'gcp-kms://' + os.environ['GCP_KMS_URI']
    table_name = 'votes'
    team = 'TABS'
    email = 'hello@example.com'
    env_aead = init_tink_env_aead(key_uri, credentials)
    db = init_db(db_user, db_pass, db_name, table_name, instance_connection_name, db_socket_dir, db_host)
    encrypt_and_insert_data(db, env_aead, table_name, team, email)

def encrypt_and_insert_data(db: sqlalchemy.engine.base.Engine, env_aead: tink.aead.KmsEnvelopeAead, table_name: str, team: str, email: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Inserts a vote into the database with email address previously encrypted using\n    a KmsEnvelopeAead object.\n    '
    time_cast = datetime.datetime.now(tz=datetime.timezone.utc)
    encrypted_email = env_aead.encrypt(email.encode(), team.encode())
    if team != 'TABS' and team != 'SPACES':
        logger.error(f'Invalid team specified: {team}')
        return
    stmt = sqlalchemy.text(f'INSERT INTO {table_name} (time_cast, team, voter_email) VALUES (:time_cast, :team, :voter_email)')
    with db.connect() as conn:
        conn.execute(stmt, time_cast=time_cast, team=team, voter_email=encrypted_email)
    print(f"Vote successfully cast for '{team}' at time {time_cast}!")
if __name__ == '__main__':
    main()