from django.http import JsonResponse
from django.template.loader import render_to_string

def render_modal_workflow(request, html_template, js_template=None, template_vars=None, json_data=None):
    if False:
        print('Hello World!')
    ' "\n    Render a response consisting of an HTML chunk and a JS onload chunk\n    in the format required by the modal-workflow framework.\n    '
    if js_template:
        raise TypeError('Passing a js_template argument to render_modal_workflow is no longer supported')
    response = {}
    if html_template:
        response['html'] = render_to_string(html_template, template_vars or {}, request=request)
    if json_data:
        response.update(json_data)
    return JsonResponse(response)