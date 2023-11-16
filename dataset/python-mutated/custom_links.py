from django import template
from django.contrib.contenttypes.models import ContentType
from django.utils.safestring import mark_safe
from extras.models import CustomLink
register = template.Library()
LINK_BUTTON = '<a href="{}"{} class="btn btn-sm btn-{}">{}</a>\n'
GROUP_BUTTON = '\n<div class="dropdown">\n    <button\n        class="btn btn-sm btn-{} dropdown-toggle"\n        type="button"\n        data-bs-toggle="dropdown"\n        aria-expanded="false">\n        {}\n    </button>\n    <ul class="dropdown-menu dropdown-menu-end">\n        {}\n    </ul>\n</div>\n'
GROUP_LINK = '<li><a class="dropdown-item" href="{}"{}>{}</a></li>\n'

@register.simple_tag(takes_context=True)
def custom_links(context, obj):
    if False:
        while True:
            i = 10
    '\n    Render all applicable links for the given object.\n    '
    content_type = ContentType.objects.get_for_model(obj)
    custom_links = CustomLink.objects.filter(content_types=content_type, enabled=True)
    if not custom_links:
        return ''
    link_context = {'object': obj, 'debug': context.get('debug', False), 'request': context['request'], 'user': context['user'], 'perms': context['perms']}
    template_code = ''
    group_names = {}
    for cl in custom_links:
        if cl.group_name and cl.group_name in group_names:
            group_names[cl.group_name].append(cl)
        elif cl.group_name:
            group_names[cl.group_name] = [cl]
        else:
            try:
                rendered = cl.render(link_context)
                if rendered:
                    template_code += LINK_BUTTON.format(rendered['link'], rendered['link_target'], cl.button_class, rendered['text'])
            except Exception as e:
                template_code += f'<a class="btn btn-sm btn-outline-dark" disabled="disabled" title="{e}"><i class="mdi mdi-alert"></i> {cl.name}</a>\n'
    for (group, links) in group_names.items():
        links_rendered = []
        for cl in links:
            try:
                rendered = cl.render(link_context)
                if rendered:
                    links_rendered.append(GROUP_LINK.format(rendered['link'], rendered['link_target'], rendered['text']))
            except Exception as e:
                links_rendered.append(f'<li><a class="dropdown-item" disabled="disabled" title="{e}"><span class="text-muted"><i class="mdi mdi-alert"></i> {cl.name}</span></a></li>')
        if links_rendered:
            template_code += GROUP_BUTTON.format(links[0].button_class, group, ''.join(links_rendered))
    return mark_safe(template_code)