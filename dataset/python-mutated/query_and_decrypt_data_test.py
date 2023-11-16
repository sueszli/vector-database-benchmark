import os
import uuid
import pytest
import sqlalchemy
import tink
from snippets.cloud_kms_env_aead import init_tink_env_aead
from snippets.cloud_sql_connection_pool import init_db
from snippets.encrypt_and_insert_data import encrypt_and_insert_data
from snippets.query_and_decrypt_data import query_and_decrypt_data
table_name = f'votes_{uuid.uuid4().hex}'

@pytest.fixture(name='pool')
def setup_pool() -> sqlalchemy.engine.Engine:
    if False:
        for i in range(10):
            print('nop')
    try:
        db_user = os.environ['POSTGRES_USER']
        db_pass = os.environ['POSTGRES_PASSWORD']
        db_name = os.environ['POSTGRES_DATABASE']
        db_host = os.environ['POSTGRES_HOST']
    except KeyError:
        raise Exception('The following env variables must be set to run these tests:POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DATABASE, POSTGRES_HOST')
    else:
        pool = init_db(db_user=db_user, db_pass=db_pass, db_name=db_name, table_name=table_name, db_host=db_host)
        yield pool
        with pool.connect() as conn:
            conn.execute(f'DROP TABLE IF EXISTS {table_name}')

@pytest.fixture(name='env_aead')
def setup_key() -> tink.aead.KmsEnvelopeAead:
    if False:
        i = 10
        return i + 15
    credentials = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', '')
    key_uri = 'gcp-kms://' + os.environ['CLOUD_KMS_KEY']
    env_aead = init_tink_env_aead(key_uri, credentials)
    yield env_aead

def test_query_and_decrypt_data(capsys: pytest.CaptureFixture, pool: sqlalchemy.engine.Engine, env_aead: tink.aead.KmsEnvelopeAead) -> None:
    if False:
        print('Hello World!')
    encrypt_and_insert_data(pool, env_aead, table_name, 'SPACES', 'hello@example.com')
    output = query_and_decrypt_data(pool, env_aead, table_name)
    for row in output:
        if row[1] == 'hello@example.com':
            break
    else:
        pytest.fail('Failed to find vote in the decrypted data.')