import pkg_resources
import requests
from packaging import version

def check_for_update():
    if False:
        i = 10
        return i + 15
    response = requests.get(f'https://pypi.org/pypi/open-interpreter/json')
    latest_version = response.json()['info']['version']
    current_version = pkg_resources.get_distribution('open-interpreter').version
    return version.parse(latest_version) > version.parse(current_version)