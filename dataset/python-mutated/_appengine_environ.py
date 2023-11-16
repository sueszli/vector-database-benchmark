"""
This module provides means to detect the App Engine environment.
"""
import os

def is_appengine():
    if False:
        for i in range(10):
            print('nop')
    return is_local_appengine() or is_prod_appengine()

def is_appengine_sandbox():
    if False:
        for i in range(10):
            print('nop')
    "Reports if the app is running in the first generation sandbox.\n\n    The second generation runtimes are technically still in a sandbox, but it\n    is much less restrictive, so generally you shouldn't need to check for it.\n    see https://cloud.google.com/appengine/docs/standard/runtimes\n    "
    return is_appengine() and os.environ['APPENGINE_RUNTIME'] == 'python27'

def is_local_appengine():
    if False:
        return 10
    return 'APPENGINE_RUNTIME' in os.environ and os.environ.get('SERVER_SOFTWARE', '').startswith('Development/')

def is_prod_appengine():
    if False:
        print('Hello World!')
    return 'APPENGINE_RUNTIME' in os.environ and os.environ.get('SERVER_SOFTWARE', '').startswith('Google App Engine/')

def is_prod_appengine_mvms():
    if False:
        print('Hello World!')
    'Deprecated.'
    return False