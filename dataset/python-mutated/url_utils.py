import socket
import psutil
BASE_URL = 'http://localhost'
API_ROOT_URL = '/api/v1/nni'
EXPERIMENT_API = '/experiment'
CLUSTER_METADATA_API = '/experiment/cluster-metadata'
IMPORT_DATA_API = '/experiment/import-data'
CHECK_STATUS_API = '/check-status'
TRIAL_JOBS_API = '/trial-jobs'
EXPORT_DATA_API = '/export-data'
TENSORBOARD_API = '/tensorboard'
METRIC_DATA_API = '/metric-data'

def format_url_path(path):
    if False:
        for i in range(10):
            print('nop')
    return API_ROOT_URL if path is None else f'/{path}{API_ROOT_URL}'

def set_prefix_url(prefix_path):
    if False:
        i = 10
        return i + 15
    global API_ROOT_URL
    API_ROOT_URL = format_url_path(prefix_path)

def metric_data_url(port):
    if False:
        return 10
    'get metric_data url'
    return '{0}:{1}{2}{3}'.format(BASE_URL, port, API_ROOT_URL, METRIC_DATA_API)

def check_status_url(port):
    if False:
        i = 10
        return i + 15
    'get check_status url'
    return '{0}:{1}{2}{3}'.format(BASE_URL, port, API_ROOT_URL, CHECK_STATUS_API)

def cluster_metadata_url(port):
    if False:
        return 10
    'get cluster_metadata_url'
    return '{0}:{1}{2}{3}'.format(BASE_URL, port, API_ROOT_URL, CLUSTER_METADATA_API)

def import_data_url(port):
    if False:
        for i in range(10):
            print('nop')
    'get import_data_url'
    return '{0}:{1}{2}{3}'.format(BASE_URL, port, API_ROOT_URL, IMPORT_DATA_API)

def experiment_url(port):
    if False:
        return 10
    'get experiment_url'
    return '{0}:{1}{2}{3}'.format(BASE_URL, port, API_ROOT_URL, EXPERIMENT_API)

def trial_jobs_url(port):
    if False:
        print('Hello World!')
    'get trial_jobs url'
    return '{0}:{1}{2}{3}'.format(BASE_URL, port, API_ROOT_URL, TRIAL_JOBS_API)

def trial_job_id_url(port, job_id):
    if False:
        print('Hello World!')
    'get trial_jobs with id url'
    return '{0}:{1}{2}{3}/{4}'.format(BASE_URL, port, API_ROOT_URL, TRIAL_JOBS_API, job_id)

def export_data_url(port):
    if False:
        print('Hello World!')
    'get export_data url'
    return '{0}:{1}{2}{3}'.format(BASE_URL, port, API_ROOT_URL, EXPORT_DATA_API)

def tensorboard_url(port):
    if False:
        return 10
    'get tensorboard url'
    return '{0}:{1}{2}{3}'.format(BASE_URL, port, API_ROOT_URL, TENSORBOARD_API)

def get_local_urls(port, prefix):
    if False:
        return 10
    'get urls of local machine'
    url_list = []
    for (_, info) in psutil.net_if_addrs().items():
        for addr in info:
            if socket.AddressFamily.AF_INET == addr.family:
                url_list.append('http://{0}:{1}{2}'.format(addr.address, port, prefix))
    return url_list