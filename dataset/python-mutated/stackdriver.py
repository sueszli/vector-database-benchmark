"""This module contains Google Stackdriver links."""
from __future__ import annotations
from typing import TYPE_CHECKING
from airflow.providers.google.cloud.links.base import BaseGoogleLink
if TYPE_CHECKING:
    from airflow.models import BaseOperator
    from airflow.utils.context import Context
STACKDRIVER_BASE_LINK = '/monitoring/alerting'
STACKDRIVER_NOTIFICATIONS_LINK = STACKDRIVER_BASE_LINK + '/notifications?project={project_id}'
STACKDRIVER_POLICIES_LINK = STACKDRIVER_BASE_LINK + '/policies?project={project_id}'

class StackdriverNotificationsLink(BaseGoogleLink):
    """Helper class for constructing Stackdriver Notifications Link."""
    name = 'Cloud Monitoring Notifications'
    key = 'stackdriver_notifications'
    format_str = STACKDRIVER_NOTIFICATIONS_LINK

    @staticmethod
    def persist(operator_instance: BaseOperator, context: Context, project_id: str | None):
        if False:
            for i in range(10):
                print('nop')
        operator_instance.xcom_push(context, key=StackdriverNotificationsLink.key, value={'project_id': project_id})

class StackdriverPoliciesLink(BaseGoogleLink):
    """Helper class for constructing Stackdriver Policies Link."""
    name = 'Cloud Monitoring Policies'
    key = 'stackdriver_policies'
    format_str = STACKDRIVER_POLICIES_LINK

    @staticmethod
    def persist(operator_instance: BaseOperator, context: Context, project_id: str | None):
        if False:
            for i in range(10):
                print('nop')
        operator_instance.xcom_push(context, key=StackdriverPoliciesLink.key, value={'project_id': project_id})