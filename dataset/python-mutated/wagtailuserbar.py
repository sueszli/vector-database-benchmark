from django import template
from django.template.loader import render_to_string
from django.utils import translation
from wagtail import hooks
from wagtail.admin.userbar import AccessibilityItem, AddPageItem, AdminItem, EditPageItem, ExplorePageItem
from wagtail.models import PAGE_TEMPLATE_VAR, Page, Revision
from wagtail.users.models import UserProfile
register = template.Library()

def get_page_instance(context):
    if False:
        for i in range(10):
            print('nop')
    '\n    Given a template context, try and find a Page variable in the common\n    places. Returns None if a page can not be found.\n    '
    possible_names = [PAGE_TEMPLATE_VAR, 'self']
    for name in possible_names:
        if name in context:
            page = context[name]
            if isinstance(page, Page):
                return page

@register.simple_tag(takes_context=True)
def wagtailuserbar(context, position='bottom-right'):
    if False:
        i = 10
        return i + 15
    try:
        request = context['request']
    except KeyError:
        return ''
    try:
        user = request.user
    except AttributeError:
        return ''
    if not user.has_perm('wagtailadmin.access_admin'):
        return ''
    if getattr(request, 'in_preview_panel', False):
        return ''
    userprofile = UserProfile.get_for_user(user)
    with translation.override(userprofile.get_preferred_language()):
        page = get_page_instance(context)
        try:
            revision_id = request.revision_id
        except AttributeError:
            revision_id = None
        if page and page.id:
            if revision_id:
                revision = Revision.page_revisions.get(id=revision_id)
                items = [AdminItem(), ExplorePageItem(revision.content_object), EditPageItem(revision.content_object), AccessibilityItem()]
            else:
                items = [AdminItem(), ExplorePageItem(page), EditPageItem(page), AddPageItem(page), AccessibilityItem()]
        else:
            items = [AdminItem(), AccessibilityItem()]
        for fn in hooks.get_hooks('construct_wagtail_userbar'):
            fn(request, items)
        rendered_items = [item.render(request) for item in items]
        rendered_items = [item for item in rendered_items if item]
        return render_to_string('wagtailadmin/userbar/base.html', {'request': request, 'items': rendered_items, 'position': position, 'page': page, 'revision_id': revision_id})