"""This module contains Google Data Catalog links."""
from __future__ import annotations
from typing import TYPE_CHECKING
from airflow.providers.google.cloud.links.base import BaseGoogleLink
if TYPE_CHECKING:
    from airflow.models import BaseOperator
    from airflow.utils.context import Context
DATACATALOG_BASE_LINK = '/datacatalog'
ENTRY_GROUP_LINK = DATACATALOG_BASE_LINK + '/groups/{entry_group_id};container={project_id};location={location_id}?project={project_id}'
ENTRY_LINK = DATACATALOG_BASE_LINK + '/projects/{project_id}/locations/{location_id}/entryGroups/{entry_group_id}/entries/{entry_id}    ?project={project_id}'
TAG_TEMPLATE_LINK = DATACATALOG_BASE_LINK + '/projects/{project_id}/locations/{location_id}/tagTemplates/{tag_template_id}?project={project_id}'

class DataCatalogEntryGroupLink(BaseGoogleLink):
    """Helper class for constructing Data Catalog Entry Group Link."""
    name = 'Data Catalog Entry Group'
    key = 'data_catalog_entry_group'
    format_str = ENTRY_GROUP_LINK

    @staticmethod
    def persist(context: Context, task_instance: BaseOperator, entry_group_id: str, location_id: str, project_id: str | None):
        if False:
            for i in range(10):
                print('nop')
        task_instance.xcom_push(context, key=DataCatalogEntryGroupLink.key, value={'entry_group_id': entry_group_id, 'location_id': location_id, 'project_id': project_id})

class DataCatalogEntryLink(BaseGoogleLink):
    """Helper class for constructing Data Catalog Entry Link."""
    name = 'Data Catalog Entry'
    key = 'data_catalog_entry'
    format_str = ENTRY_LINK

    @staticmethod
    def persist(context: Context, task_instance: BaseOperator, entry_id: str, entry_group_id: str, location_id: str, project_id: str | None):
        if False:
            for i in range(10):
                print('nop')
        task_instance.xcom_push(context, key=DataCatalogEntryLink.key, value={'entry_id': entry_id, 'entry_group_id': entry_group_id, 'location_id': location_id, 'project_id': project_id})

class DataCatalogTagTemplateLink(BaseGoogleLink):
    """Helper class for constructing Data Catalog Tag Template Link."""
    name = 'Data Catalog Tag Template'
    key = 'data_catalog_tag_template'
    format_str = TAG_TEMPLATE_LINK

    @staticmethod
    def persist(context: Context, task_instance: BaseOperator, tag_template_id: str, location_id: str, project_id: str | None):
        if False:
            while True:
                i = 10
        task_instance.xcom_push(context, key=DataCatalogTagTemplateLink.key, value={'tag_template_id': tag_template_id, 'location_id': location_id, 'project_id': project_id})