from django import template
from django.contrib.admin.templatetags.base import InclusionAdminNode
from typing import Union
register = template.Library()

@register.filter(name='split')
def split(value, separator: str=','):
    if False:
        for i in range(10):
            print('nop')
    return (value or '').split(separator)

@register.filter
def file_size(num_bytes: Union[int, float]) -> str:
    if False:
        print('Hello World!')
    for count in ['Bytes', 'KB', 'MB', 'GB']:
        if num_bytes > -1024.0 and num_bytes < 1024.0:
            return '%3.1f %s' % (num_bytes, count)
        num_bytes /= 1024.0
    return '%3.1f %s' % (num_bytes, 'TB')

def result_list(cl):
    if False:
        print('Hello World!')
    '\n    Monkey patched result\n    '
    num_sorted_fields = 0
    return {'cl': cl, 'num_sorted_fields': num_sorted_fields, 'results': cl.result_list}

@register.tag(name='snapshots_grid')
def result_list_tag(parser, token):
    if False:
        while True:
            i = 10
    return InclusionAdminNode(parser, token, func=result_list, template_name='snapshots_grid.html', takes_context=False)