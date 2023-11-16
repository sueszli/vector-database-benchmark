from typing import Set
import requests
from nicegui import __version__
from .screen import Screen

def get_openapi_paths() -> Set[str]:
    if False:
        print('Hello World!')
    return set(requests.get(f'http://localhost:{Screen.PORT}/openapi.json', timeout=5).json()['paths'])

def test_endpoint_documentation_default(screen: Screen):
    if False:
        print('Hello World!')
    screen.open('/')
    assert get_openapi_paths() == set()

def test_endpoint_documentation_page_only(screen: Screen):
    if False:
        i = 10
        return i + 15
    screen.ui_run_kwargs['endpoint_documentation'] = 'page'
    screen.open('/')
    assert get_openapi_paths() == {'/'}

def test_endpoint_documentation_internal_only(screen: Screen):
    if False:
        return 10
    screen.ui_run_kwargs['endpoint_documentation'] = 'internal'
    screen.open('/')
    assert get_openapi_paths() == {f'/_nicegui/{__version__}/libraries/{{key}}', f'/_nicegui/{__version__}/components/{{key}}'}

def test_endpoint_documentation_all(screen: Screen):
    if False:
        while True:
            i = 10
    screen.ui_run_kwargs['endpoint_documentation'] = 'all'
    screen.open('/')
    assert get_openapi_paths() == {'/', f'/_nicegui/{__version__}/libraries/{{key}}', f'/_nicegui/{__version__}/components/{{key}}'}