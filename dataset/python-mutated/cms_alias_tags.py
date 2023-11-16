from django import template
from django.utils.safestring import mark_safe
from cms.toolbar.utils import get_toolbar_from_request
from cms.utils.plugins import build_plugin_tree, downcast_plugins
register = template.Library()

@register.simple_tag(takes_context=True)
def render_alias_plugin(context, instance):
    if False:
        return 10
    request = context['request']
    toolbar = get_toolbar_from_request(request)
    renderer = toolbar.content_renderer
    source = instance.plugin or instance.alias_placeholder
    if not toolbar.edit_mode_active and source and source.page:
        can_see_content = source.page.is_published(instance.language)
    else:
        can_see_content = True
    if can_see_content and instance.plugin:
        plugins = instance.plugin.get_descendants().order_by('placeholder', 'path')
        plugins = [instance.plugin] + list(plugins)
        plugins = downcast_plugins(plugins, request=request)
        plugins = list(plugins)
        plugins[0].parent_id = None
        plugins = build_plugin_tree(plugins)
        content = renderer.render_plugin(instance=plugins[0], context=context, editable=False)
        return mark_safe(content)
    if can_see_content and instance.alias_placeholder:
        content = renderer.render_placeholder(placeholder=instance.alias_placeholder, context=context, editable=False)
        return mark_safe(content)
    return ''