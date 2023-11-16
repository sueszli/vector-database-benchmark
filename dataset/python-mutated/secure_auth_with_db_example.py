from tinydb import TinyDB, Query
import hug
import hashlib
import logging
import os
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
db = TinyDB('db.json')
'\n  Helper Methods\n'

def hash_password(password, salt):
    if False:
        return 10
    '\n    Securely hash a password using a provided salt\n    :param password:\n    :param salt:\n    :return: Hex encoded SHA512 hash of provided password\n    '
    password = str(password).encode('utf-8')
    salt = str(salt).encode('utf-8')
    return hashlib.sha512(password + salt).hexdigest()

def gen_api_key(username):
    if False:
        i = 10
        return i + 15
    '\n    Create a random API key for a user\n    :param username:\n    :return: Hex encoded SHA512 random string\n    '
    salt = str(os.urandom(64)).encode('utf-8')
    return hash_password(username, salt)

@hug.cli()
def authenticate_user(username, password):
    if False:
        return 10
    '\n    Authenticate a username and password against our database\n    :param username:\n    :param password:\n    :return: authenticated username\n    '
    user_model = Query()
    user = db.get(user_model.username == username)
    if not user:
        logger.warning('User %s not found', username)
        return False
    if user['password'] == hash_password(password, user.get('salt')):
        return user['username']
    return False

@hug.cli()
def authenticate_key(api_key):
    if False:
        while True:
            i = 10
    '\n    Authenticate an API key against our database\n    :param api_key:\n    :return: authenticated username\n    '
    user_model = Query()
    user = db.search(user_model.api_key == api_key)[0]
    if user:
        return user['username']
    return False
'\n  API Methods start here\n'
api_key_authentication = hug.authentication.api_key(authenticate_key)
basic_authentication = hug.authentication.basic(authenticate_user)

@hug.cli()
def add_user(username, password):
    if False:
        return 10
    '\n    CLI Parameter to add a user to the database\n    :param username:\n    :param password:\n    :return: JSON status output\n    '
    user_model = Query()
    if db.search(user_model.username == username):
        return {'error': 'User {0} already exists'.format(username)}
    salt = hashlib.sha512(str(os.urandom(64)).encode('utf-8')).hexdigest()
    password = hash_password(password, salt)
    api_key = gen_api_key(username)
    user = {'username': username, 'password': password, 'salt': salt, 'api_key': api_key}
    user_id = db.insert(user)
    return {'result': 'success', 'eid': user_id, 'user_created': user}

@hug.get('/api/get_api_key', requires=basic_authentication)
def get_token(authed_user: hug.directives.user):
    if False:
        print('Hello World!')
    '\n    Get Job details\n    :param authed_user:\n    :return:\n    '
    user_model = Query()
    user = db.search(user_model.username == authed_user)[0]
    if user:
        out = {'user': user['username'], 'api_key': user['api_key']}
    else:
        out = {'error': 'User {0} does not exist'.format(authed_user)}
    return out

@hug.get(('/api/job', '/api/job/{job_id}/'), requires=api_key_authentication)
def get_job_details(job_id):
    if False:
        return 10
    '\n    Get Job details\n    :param job_id:\n    :return:\n    '
    job = {'job_id': job_id, 'details': 'Details go here'}
    return job
if __name__ == '__main__':
    add_user.interface.cli()