from __future__ import annotations
from typing import TYPE_CHECKING
from airflow.providers.google.cloud.links.base import BaseGoogleLink
if TYPE_CHECKING:
    from airflow.utils.context import Context
BASE_LINK = 'https://console.cloud.google.com/lifesciences'
LIFESCIENCES_LIST_LINK = BASE_LINK + '/pipelines?project={project_id}'

class LifeSciencesLink(BaseGoogleLink):
    """Helper class for constructing Life Sciences List link."""
    name = 'Life Sciences'
    key = 'lifesciences_key'
    format_str = LIFESCIENCES_LIST_LINK

    @staticmethod
    def persist(context: Context, task_instance, project_id: str):
        if False:
            i = 10
            return i + 15
        task_instance.xcom_push(context=context, key=LifeSciencesLink.key, value={'project_id': project_id})