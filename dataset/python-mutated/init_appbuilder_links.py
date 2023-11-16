from __future__ import annotations
from airflow.configuration import conf
from airflow.security.permissions import RESOURCE_DOCS, RESOURCE_DOCS_MENU
from airflow.utils.docs import get_docs_url

def init_appbuilder_links(app):
    if False:
        while True:
            i = 10
    'Add links to the navbar.'
    appbuilder = app.appbuilder
    appbuilder.add_link(name='DAGs', href='Airflow.index')
    appbuilder.menu.menu.insert(0, appbuilder.menu.menu.pop())
    appbuilder.add_link(name='Cluster Activity', href='Airflow.cluster_activity')
    appbuilder.menu.menu.insert(1, appbuilder.menu.menu.pop())
    appbuilder.add_link(name='Datasets', href='Airflow.datasets')
    appbuilder.menu.menu.insert(2, appbuilder.menu.menu.pop())
    appbuilder.add_link(name=RESOURCE_DOCS, label='Documentation', href=get_docs_url(), category=RESOURCE_DOCS_MENU)
    appbuilder.add_link(name=RESOURCE_DOCS, label='Airflow Website', href='https://airflow.apache.org', category=RESOURCE_DOCS_MENU)
    appbuilder.add_link(name=RESOURCE_DOCS, label='GitHub Repo', href='https://github.com/apache/airflow', category=RESOURCE_DOCS_MENU)
    if conf.getboolean('webserver', 'enable_swagger_ui', fallback=True):
        appbuilder.add_link(name=RESOURCE_DOCS, label='REST API Reference (Swagger UI)', href='/api/v1./api/v1_swagger_ui_index', category=RESOURCE_DOCS_MENU)
    appbuilder.add_link(name=RESOURCE_DOCS, label='REST API Reference (Redoc)', href='RedocView.redoc', category=RESOURCE_DOCS_MENU)