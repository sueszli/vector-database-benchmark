"""
Example of starting/stopping a server via the JupyterHub API

1. get user status
2. start server
3. wait for server to be ready via progress api
4. make a request to the server itself
5. stop server via API
6. wait for server to finish stopping
"""
import json
import logging
import pathlib
import time
import requests
log = logging.getLogger(__name__)

def get_token():
    if False:
        while True:
            i = 10
    'boilerplate: get token from share file.\n\n    Make sure to start jupyterhub in this directory first\n    '
    here = pathlib.Path(__file__).parent
    token_file = here.joinpath('service-token')
    log.info(f'Loading token from {token_file}')
    with token_file.open('r') as f:
        token = f.read().strip()
    return token

def make_session(token):
    if False:
        for i in range(10):
            print('nop')
    'Create a requests.Session with our service token in the Authorization header'
    session = requests.Session()
    session.headers = {'Authorization': f'token {token}'}
    return session

def event_stream(session, url):
    if False:
        return 10
    'Generator yielding events from a JSON event stream\n\n    For use with the server progress API\n    '
    r = session.get(url, stream=True)
    r.raise_for_status()
    for line in r.iter_lines():
        line = line.decode('utf8', 'replace')
        if line.startswith('data:'):
            yield json.loads(line.split(':', 1)[1])

def start_server(session, hub_url, user, server_name=''):
    if False:
        return 10
    'Start a server for a jupyterhub user\n\n    Returns the full URL for accessing the server\n    '
    user_url = f'{hub_url}/hub/api/users/{user}'
    log_name = f'{user}/{server_name}'.rstrip('/')
    r = session.get(user_url)
    r.raise_for_status()
    user_model = r.json()
    if server_name not in user_model.get('servers', {}):
        log.info(f'Starting server {log_name}')
        r = session.post(f'{user_url}/servers/{server_name}')
        r.raise_for_status()
        if r.status_code == 201:
            log.info(f'Server {log_name} is launched and ready')
        elif r.status_code == 202:
            log.info(f'Server {log_name} is launching...')
        else:
            log.warning(f'Unexpected status: {r.status_code}')
        r = session.get(user_url)
        r.raise_for_status()
        user_model = r.json()
    server = user_model['servers'][server_name]
    if server['pending']:
        status = f"pending {server['pending']}"
    elif server['ready']:
        status = 'ready'
    else:
        raise ValueError(f'Unexpected server state: {server}')
    log.info(f'Server {log_name} is {status}')
    progress_url = user_model['servers'][server_name]['progress_url']
    for event in event_stream(session, f'{hub_url}{progress_url}'):
        log.info(f"Progress {event['progress']}%: {event['message']}")
        if event.get('ready'):
            server_url = event['url']
            break
    else:
        raise ValueError(f'{log_name} never started!')
    return f'{hub_url}{server_url}'

def stop_server(session, hub_url, user, server_name=''):
    if False:
        i = 10
        return i + 15
    'Stop a server via the JupyterHub API\n\n    Returns when the server has finished stopping\n    '
    user_url = f'{hub_url}/hub/api/users/{user}'
    server_url = f'{user_url}/servers/{server_name}'
    log_name = f'{user}/{server_name}'.rstrip('/')
    log.info(f'Stopping server {log_name}')
    r = session.delete(server_url)
    if r.status_code == 404:
        log.info(f'Server {log_name} already stopped')
    r.raise_for_status()
    if r.status_code == 204:
        log.info(f'Server {log_name} stopped')
        return
    log.info(f'Server {log_name} stopping...')
    while True:
        r = session.get(user_url)
        r.raise_for_status()
        user_model = r.json()
        if server_name not in user_model.get('servers', {}):
            log.info(f'Server {log_name} stopped')
            return
        server = user_model['servers'][server_name]
        if not server['pending']:
            raise ValueError(f'Waiting for {log_name}, but no longer pending.')
        log.info(f"Server {log_name} pending: {server['pending']}")
        time.sleep(1)

def main():
    if False:
        print('Hello World!')
    'Start and stop one server\n\n    Uses test-user and hub from jupyterhub_config.py in this directory\n    '
    user = 'test-user'
    hub_url = 'http://127.0.0.1:8000'
    session = make_session(get_token())
    server_url = start_server(session, hub_url, user)
    r = session.get(f'{server_url}/api/status')
    r.raise_for_status()
    log.info(f'Server status: {r.text}')
    stop_server(session, hub_url, user)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()