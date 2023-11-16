import os
import sqlalchemy
import tink
from .cloud_kms_env_aead import init_tink_env_aead
from .cloud_sql_connection_pool import init_db
from .encrypt_and_insert_data import encrypt_and_insert_data

def main() -> None:
    if False:
        while True:
            i = 10
    '\n    Connects to the database, inserts encrypted data and retrieves encrypted data.\n    '
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
    query_and_decrypt_data(db, env_aead, table_name)

def query_and_decrypt_data(db: sqlalchemy.engine.base.Engine, env_aead: tink.aead.KmsEnvelopeAead, table_name: str) -> list[tuple[str]]:
    if False:
        return 10
    '\n    Retrieves data from the database and decrypts it using the KmsEnvelopeAead object.\n    '
    with db.connect() as conn:
        recent_votes = conn.execute(f'SELECT team, time_cast, voter_email FROM {table_name} ORDER BY time_cast DESC LIMIT 5').fetchall()
        print('Team\tEmail\tTime Cast')
        output = []
        for row in recent_votes:
            team = row[0]
            aad = team.rstrip()
            email = env_aead.decrypt(row[2], aad.encode()).decode()
            time_cast = row[1]
            print(f'{team}\t{email}\t{time_cast}')
            output.append((team, email, time_cast))
    return output
if __name__ == '__main__':
    main()