"""This module contains Google Cloud Functions links."""
from __future__ import annotations
from typing import TYPE_CHECKING
from airflow.providers.google.cloud.links.base import BaseGoogleLink
if TYPE_CHECKING:
    from airflow.models import BaseOperator
    from airflow.utils.context import Context
CLOUD_FUNCTIONS_BASE_LINK = 'https://console.cloud.google.com/functions'
CLOUD_FUNCTIONS_DETAILS_LINK = CLOUD_FUNCTIONS_BASE_LINK + '/details/{location}/{function_name}?project={project_id}'
CLOUD_FUNCTIONS_LIST_LINK = CLOUD_FUNCTIONS_BASE_LINK + '/list?project={project_id}'

class CloudFunctionsDetailsLink(BaseGoogleLink):
    """Helper class for constructing Cloud Functions Details Link."""
    name = 'Cloud Functions Details'
    key = 'cloud_functions_details'
    format_str = CLOUD_FUNCTIONS_DETAILS_LINK

    @staticmethod
    def persist(context: Context, task_instance: BaseOperator, function_name: str, location: str, project_id: str):
        if False:
            while True:
                i = 10
        task_instance.xcom_push(context, key=CloudFunctionsDetailsLink.key, value={'function_name': function_name, 'location': location, 'project_id': project_id})

class CloudFunctionsListLink(BaseGoogleLink):
    """Helper class for constructing Cloud Functions Details Link."""
    name = 'Cloud Functions List'
    key = 'cloud_functions_list'
    format_str = CLOUD_FUNCTIONS_LIST_LINK

    @staticmethod
    def persist(context: Context, task_instance: BaseOperator, project_id: str):
        if False:
            while True:
                i = 10
        task_instance.xcom_push(context, key=CloudFunctionsDetailsLink.key, value={'project_id': project_id})