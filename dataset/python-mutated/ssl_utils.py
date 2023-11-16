import os

def get_ssl_filename(name):
    if False:
        for i in range(10):
            print('nop')
    root = os.path.join(os.path.dirname(__file__), '..')
    cert_dir = os.path.abspath(os.path.join(root, 'dockers', 'stunnel', 'keys'))
    if not os.path.isdir(cert_dir):
        cert_dir = os.path.abspath(os.path.join(root, '..', 'dockers', 'stunnel', 'keys'))
        if not os.path.isdir(cert_dir):
            raise IOError(f'No SSL certificates found. They should be in {cert_dir}')
    return os.path.join(cert_dir, name)