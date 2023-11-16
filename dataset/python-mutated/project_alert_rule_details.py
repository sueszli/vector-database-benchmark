from rest_framework.request import Request
from rest_framework.response import Response
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.incidents.endpoints.bases import ProjectAlertRuleEndpoint
from sentry.incidents.endpoints.organization_alert_rule_details import fetch_alert_rule, remove_alert_rule, update_alert_rule

@region_silo_endpoint
class ProjectAlertRuleDetailsEndpoint(ProjectAlertRuleEndpoint):
    publish_status = {'DELETE': ApiPublishStatus.UNKNOWN, 'GET': ApiPublishStatus.UNKNOWN, 'PUT': ApiPublishStatus.UNKNOWN}

    def get(self, request: Request, project, alert_rule) -> Response:
        if False:
            i = 10
            return i + 15
        '\n        Fetch a metric alert rule. @deprecated. Use OrganizationAlertRuleDetailsEndpoint instead.\n        ``````````````````\n        :auth: required\n        '
        return fetch_alert_rule(request, project.organization, alert_rule)

    def put(self, request: Request, project, alert_rule) -> Response:
        if False:
            for i in range(10):
                print('nop')
        '\n        Update a metric alert rule. @deprecated. Use OrganizationAlertRuleDetailsEndpoint instead.\n        ``````````````````\n        :auth: required\n        '
        return update_alert_rule(request, project.organization, alert_rule)

    def delete(self, request: Request, project, alert_rule) -> Response:
        if False:
            while True:
                i = 10
        '\n        Delete a metric alert rule. @deprecated. Use OrganizationAlertRuleDetailsEndpoint instead.\n        ``````````````````\n        :auth: required\n        '
        return remove_alert_rule(request, project.organization, alert_rule)