from rest_framework.request import Request
from rest_framework.response import Response
from sentry.api.api_publish_status import ApiPublishStatus
from sentry.api.base import region_silo_endpoint
from sentry.api.bases.organization import OrganizationEndpoint
from sentry.rules import rules

@region_silo_endpoint
class ProjectAgnosticRuleConditionsEndpoint(OrganizationEndpoint):
    publish_status = {'GET': ApiPublishStatus.UNKNOWN}

    def get(self, request: Request, organization) -> Response:
        if False:
            print('Hello World!')
        '\n        Retrieve the list of rule conditions\n        '

        def info_extractor(rule_cls):
            if False:
                for i in range(10):
                    print('nop')
            context = {'id': rule_cls.id, 'label': rule_cls.label}
            node = rule_cls(None)
            if hasattr(node, 'form_fields'):
                context['formFields'] = node.form_fields
            return context
        return Response([info_extractor(rule_cls) for (rule_type, rule_cls) in rules if rule_type.startswith('condition/')])