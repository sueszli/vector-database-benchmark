from __future__ import annotations
import os
from airflow.configuration import conf
from airflow.plugins_manager import AirflowPlugin
from airflow.providers.openlineage.plugins.listener import get_openlineage_listener
from airflow.providers.openlineage.plugins.macros import lineage_parent_id, lineage_run_id

def _is_disabled() -> bool:
    if False:
        for i in range(10):
            print('nop')
    return conf.getboolean('openlineage', 'disabled', fallback=False) or os.getenv('OPENLINEAGE_DISABLED', 'false').lower() == 'true' or (conf.get('openlineage', 'transport', fallback='') == '' and conf.get('openlineage', 'config_path', fallback='') == '' and (os.getenv('OPENLINEAGE_URL', '') == '') and (os.getenv('OPENLINEAGE_CONFIG', '') == ''))

class OpenLineageProviderPlugin(AirflowPlugin):
    """
    Listener that emits numerous Events.

    OpenLineage Plugin provides listener that emits OL events on DAG start,
    complete and failure and TaskInstances start, complete and failure.
    """
    name = 'OpenLineageProviderPlugin'
    if not _is_disabled():
        macros = [lineage_run_id, lineage_parent_id]
        listeners = [get_openlineage_listener()]