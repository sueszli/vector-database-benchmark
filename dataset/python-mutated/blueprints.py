from __future__ import annotations
from flask import Blueprint, redirect, url_for
routes = Blueprint('routes', __name__)

@routes.route('/')
def index():
    if False:
        for i in range(10):
            print('nop')
    'Main Airflow page.'
    return redirect(url_for('Airflow.index'))