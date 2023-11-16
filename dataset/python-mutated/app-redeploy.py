"""Test app redeploy.

This file is copied over to app.py during the integration
tests to test behavior on redeploys.

"""
import os
from chalice import Chalice
app = Chalice(app_name=os.environ['APP_NAME'])

@app.route('/')
def index():
    if False:
        return 10
    return {'hello': 'world'}

@app.route('/a/b/c/d/e/f/g')
def nested_route():
    if False:
        for i in range(10):
            print('nop')
    return {'redeployed': True}

@app.route('/multimethod', methods=['GET', 'PUT'])
def multiple_methods():
    if False:
        i = 10
        return i + 15
    return {'method': app.current_request.method}

@app.route('/redeploy')
def redeploy():
    if False:
        print('Hello World!')
    return {'success': True}