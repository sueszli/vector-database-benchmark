import json
import os
from middleware import logger

def get_cred_config() -> dict[str, str]:
    if False:
        while True:
            i = 10
    'Retrieve Cloud SQL credentials stored in Secret Manager\n    or default to environment variables.\n\n    Returns:\n        A dictionary with Cloud SQL credential values\n    '
    secret = os.environ.get('CLOUD_SQL_CREDENTIALS_SECRET')
    if secret:
        return json.loads(secret)
    else:
        logger.info('CLOUD_SQL_CREDENTIALS_SECRET env var not set. Defaulting to environment variables.')
        if 'DB_USER' not in os.environ:
            raise Exception('DB_USER needs to be set.')
        if 'DB_PASSWORD' not in os.environ:
            raise Exception('DB_PASSWORD needs to be set.')
        if 'DB_NAME' not in os.environ:
            raise Exception('DB_NAME needs to be set.')
        if 'CLOUD_SQL_CONNECTION_NAME' not in os.environ:
            raise Exception('CLOUD_SQL_CONNECTION_NAME needs to be set.')
        return {'DB_USER': os.environ['DB_USER'], 'DB_PASSWORD': os.environ['DB_PASSWORD'], 'DB_NAME': os.environ['DB_NAME'], 'DB_HOST': os.environ.get('DB_HOST', None), 'CLOUD_SQL_CONNECTION_NAME': os.environ['CLOUD_SQL_CONNECTION_NAME']}