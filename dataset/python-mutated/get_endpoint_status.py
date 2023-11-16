from django import template
from dojo.models import Endpoint_Status
from django.db.models import Q
register = template.Library()

@register.filter(name='has_endpoints')
def has_endpoints(finding):
    if False:
        for i in range(10):
            print('nop')
    return True if finding.endpoints.all() else False

@register.filter(name='get_vulnerable_endpoints')
def get_vulnerable_endpoints(finding):
    if False:
        while True:
            i = 10
    return finding.endpoints.filter(status_endpoint__mitigated=False, status_endpoint__false_positive=False, status_endpoint__out_of_scope=False, status_endpoint__risk_accepted=False)

@register.filter(name='get_mitigated_endpoints')
def get_mitigated_endpoints(finding):
    if False:
        print('Hello World!')
    return finding.endpoints.filter(Q(status_endpoint__mitigated=True) | Q(status_endpoint__false_positive=True) | Q(status_endpoint__out_of_scope=True) | Q(status_endpoint__risk_accepted=True))

@register.filter
def endpoint_display_status(endpoint, finding):
    if False:
        while True:
            i = 10
    status = Endpoint_Status.objects.get(endpoint=endpoint, finding=finding)
    statuses = []
    if status.false_positive:
        statuses.append('False Positive')
    if status.risk_accepted:
        statuses.append('Risk Accepted')
    if status.out_of_scope:
        statuses.append('Out of Scope')
    if status.mitigated:
        statuses.append('Mitigated')
    if statuses:
        return ', '.join(statuses)
    else:
        return 'Active'

@register.filter
def endpoint_update_time(endpoint, finding):
    if False:
        print('Hello World!')
    status = Endpoint_Status.objects.get(endpoint=endpoint, finding=finding)
    return status.last_modified

@register.filter
def endpoint_date(endpoint, finding):
    if False:
        print('Hello World!')
    status = Endpoint_Status.objects.get(endpoint=endpoint, finding=finding)
    return status.date

@register.filter
def endpoint_mitigator(endpoint, finding):
    if False:
        for i in range(10):
            print('nop')
    status = Endpoint_Status.objects.get(endpoint=endpoint, finding=finding)
    return status.mitigated_by

@register.filter
def endpoint_mitigated_time(endpoint, finding):
    if False:
        i = 10
        return i + 15
    status = Endpoint_Status.objects.get(endpoint=endpoint, finding=finding)
    return status.mitigated_time