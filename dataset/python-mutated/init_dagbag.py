from __future__ import annotations
import os
from airflow.models import DagBag
from airflow.settings import DAGS_FOLDER

def init_dagbag(app):
    if False:
        return 10
    '\n    Create global DagBag for webserver and API.\n\n    To access it use ``flask.current_app.dag_bag``.\n    '
    if os.environ.get('SKIP_DAGS_PARSING') == 'True':
        app.dag_bag = DagBag(os.devnull, include_examples=False)
    else:
        app.dag_bag = DagBag(DAGS_FOLDER, read_dags_from_db=True)