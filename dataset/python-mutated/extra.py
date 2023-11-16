from django import template
from django.conf import settings
from datetime import timedelta
register = template.Library()

@register.simple_tag
def jenkins_human_url(jobname):
    if False:
        return 10
    return '{}job/{}/'.format(settings.JENKINS_API, jobname)

@register.simple_tag
def echo_setting(setting):
    if False:
        for i in range(10):
            print('nop')
    return getattr(settings, setting, '')

@register.filter(name='format_timedelta')
def format_timedelta(delta):
    if False:
        for i in range(10):
            print('nop')
    return str(timedelta(days=delta.days, seconds=delta.seconds))

@register.filter
def for_service(objects, service):
    if False:
        i = 10
        return i + 15
    return objects.filter(service=service)