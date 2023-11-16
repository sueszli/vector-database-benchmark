from __future__ import annotations
import setproctitle
from airflow import settings

def post_worker_init(_):
    if False:
        i = 10
        return i + 15
    '\n    Set process title.\n\n    This is used by airflow.cli.commands.webserver_command to track the status of the worker.\n    '
    old_title = setproctitle.getproctitle()
    setproctitle.setproctitle(settings.GUNICORN_WORKER_READY_PREFIX + old_title)

def on_starting(server):
    if False:
        print('Hello World!')
    from airflow.providers_manager import ProvidersManager
    ProvidersManager().connection_form_widgets