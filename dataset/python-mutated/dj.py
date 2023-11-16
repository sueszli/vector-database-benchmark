from functools import WRAPPER_ASSIGNMENTS
from django.apps import apps
__all__ = ['is_installed', 'installed_apps']

def is_installed(app_name):
    if False:
        while True:
            i = 10
    return apps.is_installed(app_name)

def installed_apps():
    if False:
        for i in range(10):
            print('nop')
    return [app.name for app in apps.get_app_configs()]

def available_attrs(fn):
    if False:
        i = 10
        return i + 15
    return WRAPPER_ASSIGNMENTS