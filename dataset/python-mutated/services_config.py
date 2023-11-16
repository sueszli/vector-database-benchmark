import os
SERVICES = {'default': 8000, 'static': 8001}

def init_app(app):
    if False:
        print('Hello World!')
    gae_instance = os.environ.get('GAE_INSTANCE', os.environ.get('GAE_MODULE_INSTANCE'))
    environment = 'production' if gae_instance is not None else 'development'
    app.config['SERVICE_MAP'] = map_services(environment)

def map_services(environment):
    if False:
        for i in range(10):
            print('nop')
    'Generates a map of services to correct urls for running locally\n    or when deployed.'
    url_map = {}
    for (service, local_port) in SERVICES.items():
        if environment == 'production':
            url_map[service] = production_url(service)
        if environment == 'development':
            url_map[service] = local_url(local_port)
    return url_map

def production_url(service_name):
    if False:
        return 10
    'Generates url for a service when deployed to App Engine.'
    project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
    project_url = f'{project_id}.appspot.com'
    if service_name == 'default':
        return f'https://{project_url}'
    else:
        return f'https://{service_name}-dot-{project_url}'

def local_url(port):
    if False:
        for i in range(10):
            print('nop')
    'Generates url for a service when running locally'
    return f'http://localhost:{str(port)}'