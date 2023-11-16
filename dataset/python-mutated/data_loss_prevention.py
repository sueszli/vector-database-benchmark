from __future__ import annotations
from typing import TYPE_CHECKING
from airflow.providers.google.cloud.links.base import BaseGoogleLink
if TYPE_CHECKING:
    from airflow.utils.context import Context
BASE_LINK = 'https://console.cloud.google.com'
DLP_BASE_LINK = BASE_LINK + '/security/dlp'
DLP_DEIDENTIFY_TEMPLATES_LIST_LINK = DLP_BASE_LINK + '/landing/configuration/templates/deidentify?project={project_id}'
DLP_DEIDENTIFY_TEMPLATE_DETAILS_LINK = DLP_BASE_LINK + '/projects/{project_id}/locations/global/deidentifyTemplates/{template_name}?project={project_id}'
DLP_JOB_TRIGGER_LIST_LINK = DLP_BASE_LINK + '/landing/inspection/triggers?project={project_id}'
DLP_JOB_TRIGGER_DETAILS_LINK = DLP_BASE_LINK + '/projects/{project_id}/locations/global/jobTriggers/{trigger_name}?project={project_id}'
DLP_JOBS_LIST_LINK = DLP_BASE_LINK + '/landing/inspection/jobs?project={project_id}'
DLP_JOB_DETAILS_LINK = DLP_BASE_LINK + '/projects/{project_id}/locations/global/dlpJobs/{job_name}?project={project_id}'
DLP_INSPECT_TEMPLATES_LIST_LINK = DLP_BASE_LINK + '/landing/configuration/templates/inspect?project={project_id}'
DLP_INSPECT_TEMPLATE_DETAILS_LINK = DLP_BASE_LINK + '/projects/{project_id}/locations/global/inspectTemplates/{template_name}?project={project_id}'
DLP_INFO_TYPES_LIST_LINK = DLP_BASE_LINK + '/landing/configuration/infoTypes/stored?cloudshell=false&project={project_id}'
DLP_INFO_TYPE_DETAILS_LINK = DLP_BASE_LINK + '/projects/{project_id}/locations/global/storedInfoTypes/{info_type_name}?project={project_id}'
DLP_POSSIBLE_INFO_TYPES_LIST_LINK = DLP_BASE_LINK + '/landing/configuration/infoTypes/built-in?project={project_id}'

class CloudDLPDeidentifyTemplatesListLink(BaseGoogleLink):
    """Helper class for constructing Cloud Data Loss Prevention link."""
    name = 'Cloud DLP Deidentify Templates List'
    key = 'cloud_dlp_deidentify_templates_list_key'
    format_str = DLP_DEIDENTIFY_TEMPLATES_LIST_LINK

    @staticmethod
    def persist(context: Context, task_instance, project_id: str):
        if False:
            for i in range(10):
                print('nop')
        task_instance.xcom_push(context=context, key=CloudDLPDeidentifyTemplatesListLink.key, value={'project_id': project_id})

class CloudDLPDeidentifyTemplateDetailsLink(BaseGoogleLink):
    """Helper class for constructing Cloud Data Loss Prevention link."""
    name = 'Cloud DLP Deidentify Template Details'
    key = 'cloud_dlp_deidentify_template_details_key'
    format_str = DLP_DEIDENTIFY_TEMPLATE_DETAILS_LINK

    @staticmethod
    def persist(context: Context, task_instance, project_id: str, template_name: str):
        if False:
            print('Hello World!')
        task_instance.xcom_push(context=context, key=CloudDLPDeidentifyTemplateDetailsLink.key, value={'project_id': project_id, 'template_name': template_name})

class CloudDLPJobTriggersListLink(BaseGoogleLink):
    """Helper class for constructing Cloud Data Loss Prevention link."""
    name = 'Cloud DLP Job Triggers List'
    key = 'cloud_dlp_job_triggers_list_key'
    format_str = DLP_JOB_TRIGGER_LIST_LINK

    @staticmethod
    def persist(context: Context, task_instance, project_id: str):
        if False:
            print('Hello World!')
        task_instance.xcom_push(context=context, key=CloudDLPJobTriggersListLink.key, value={'project_id': project_id})

class CloudDLPJobTriggerDetailsLink(BaseGoogleLink):
    """Helper class for constructing Cloud Data Loss Prevention link."""
    name = 'Cloud DLP Job Triggers Details'
    key = 'cloud_dlp_job_trigger_details_key'
    format_str = DLP_JOB_TRIGGER_DETAILS_LINK

    @staticmethod
    def persist(context: Context, task_instance, project_id: str, trigger_name: str):
        if False:
            for i in range(10):
                print('nop')
        task_instance.xcom_push(context=context, key=CloudDLPJobTriggerDetailsLink.key, value={'project_id': project_id, 'trigger_name': trigger_name})

class CloudDLPJobsListLink(BaseGoogleLink):
    """Helper class for constructing Cloud Data Loss Prevention link."""
    name = 'Cloud DLP Jobs List'
    key = 'cloud_dlp_jobs_list_key'
    format_str = DLP_JOBS_LIST_LINK

    @staticmethod
    def persist(context: Context, task_instance, project_id: str):
        if False:
            return 10
        task_instance.xcom_push(context=context, key=CloudDLPJobsListLink.key, value={'project_id': project_id})

class CloudDLPJobDetailsLink(BaseGoogleLink):
    """Helper class for constructing Cloud Data Loss Prevention link."""
    name = 'Cloud DLP Job Details'
    key = 'cloud_dlp_job_details_key'
    format_str = DLP_JOB_DETAILS_LINK

    @staticmethod
    def persist(context: Context, task_instance, project_id: str, job_name: str):
        if False:
            while True:
                i = 10
        task_instance.xcom_push(context=context, key=CloudDLPJobDetailsLink.key, value={'project_id': project_id, 'job_name': job_name})

class CloudDLPInspectTemplatesListLink(BaseGoogleLink):
    """Helper class for constructing Cloud Data Loss Prevention link."""
    name = 'Cloud DLP Inspect Templates List'
    key = 'cloud_dlp_inspect_templates_list_key'
    format_str = DLP_INSPECT_TEMPLATES_LIST_LINK

    @staticmethod
    def persist(context: Context, task_instance, project_id: str):
        if False:
            print('Hello World!')
        task_instance.xcom_push(context=context, key=CloudDLPInspectTemplatesListLink.key, value={'project_id': project_id})

class CloudDLPInspectTemplateDetailsLink(BaseGoogleLink):
    """Helper class for constructing Cloud Data Loss Prevention link."""
    name = 'Cloud DLP Inspect Template Details'
    key = 'cloud_dlp_inspect_template_details_key'
    format_str = DLP_INSPECT_TEMPLATE_DETAILS_LINK

    @staticmethod
    def persist(context: Context, task_instance, project_id: str, template_name: str):
        if False:
            for i in range(10):
                print('nop')
        task_instance.xcom_push(context=context, key=CloudDLPInspectTemplateDetailsLink.key, value={'project_id': project_id, 'template_name': template_name})

class CloudDLPInfoTypesListLink(BaseGoogleLink):
    """Helper class for constructing Cloud Data Loss Prevention link."""
    name = 'Cloud DLP Info Types List'
    key = 'cloud_dlp_info_types_list_key'
    format_str = DLP_INFO_TYPES_LIST_LINK

    @staticmethod
    def persist(context: Context, task_instance, project_id: str):
        if False:
            print('Hello World!')
        task_instance.xcom_push(context=context, key=CloudDLPInfoTypesListLink.key, value={'project_id': project_id})

class CloudDLPInfoTypeDetailsLink(BaseGoogleLink):
    """Helper class for constructing Cloud Data Loss Prevention link."""
    name = 'Cloud DLP Info Type Details'
    key = 'cloud_dlp_info_type_details_key'
    format_str = DLP_INFO_TYPE_DETAILS_LINK

    @staticmethod
    def persist(context: Context, task_instance, project_id: str, info_type_name: str):
        if False:
            for i in range(10):
                print('nop')
        task_instance.xcom_push(context=context, key=CloudDLPInfoTypeDetailsLink.key, value={'project_id': project_id, 'info_type_name': info_type_name})

class CloudDLPPossibleInfoTypesListLink(BaseGoogleLink):
    """Helper class for constructing Cloud Data Loss Prevention link."""
    name = 'Cloud DLP Possible Info Types List'
    key = 'cloud_dlp_possible_info_types_list_key'
    format_str = DLP_POSSIBLE_INFO_TYPES_LIST_LINK

    @staticmethod
    def persist(context: Context, task_instance, project_id: str):
        if False:
            i = 10
            return i + 15
        task_instance.xcom_push(context=context, key=CloudDLPPossibleInfoTypesListLink.key, value={'project_id': project_id})