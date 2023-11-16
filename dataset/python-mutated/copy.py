from django.shortcuts import redirect
from django.template.response import TemplateResponse
from django.utils.translation import gettext as _
from wagtail import hooks
from wagtail.actions.copy_page import CopyPageAction
from wagtail.actions.create_alias import CreatePageAliasAction
from wagtail.admin import messages
from wagtail.admin.auth import user_has_any_page_permission, user_passes_test
from wagtail.admin.forms.pages import CopyForm
from wagtail.admin.utils import get_valid_next_url_from_request
from wagtail.models import Page

@user_passes_test(user_has_any_page_permission)
def copy(request, page_id):
    if False:
        print('Hello World!')
    page = Page.objects.get(id=page_id)
    parent_page = page.get_parent()
    can_publish = parent_page.permissions_for_user(request.user).can_publish_subpage()
    form = CopyForm(request.POST or None, user=request.user, page=page, can_publish=can_publish)
    next_url = get_valid_next_url_from_request(request)
    for fn in hooks.get_hooks('before_copy_page'):
        result = fn(request, page)
        if hasattr(result, 'status_code'):
            return result
    if request.method == 'POST':
        parent_page = Page.objects.get(id=request.POST['new_parent_page'])
        if form.is_valid():
            if form.cleaned_data['new_parent_page']:
                parent_page = form.cleaned_data['new_parent_page']
            can_publish = parent_page.permissions_for_user(request.user).can_publish_subpage()
            keep_live = can_publish and form.cleaned_data.get('publish_copies')
            if can_publish and form.cleaned_data.get('alias'):
                action = CreatePageAliasAction(page.specific, recursive=form.cleaned_data.get('copy_subpages'), parent=parent_page, update_slug=form.cleaned_data['new_slug'], user=request.user)
                new_page = action.execute(skip_permission_checks=True)
            else:
                action = CopyPageAction(page=page, recursive=form.cleaned_data.get('copy_subpages'), to=parent_page, update_attrs={'title': form.cleaned_data['new_title'], 'slug': form.cleaned_data['new_slug']}, keep_live=keep_live, user=request.user)
                new_page = action.execute()
            if form.cleaned_data.get('copy_subpages'):
                messages.success(request, _("Page '%(page_title)s' and %(subpages_count)s subpages copied.") % {'page_title': page.specific_deferred.get_admin_display_title(), 'subpages_count': new_page.get_descendants().count()})
            else:
                messages.success(request, _("Page '%(page_title)s' copied.") % {'page_title': page.specific_deferred.get_admin_display_title()})
            for fn in hooks.get_hooks('after_copy_page'):
                result = fn(request, page, new_page)
                if hasattr(result, 'status_code'):
                    return result
            if next_url:
                return redirect(next_url)
            return redirect('wagtailadmin_explore', parent_page.id)
    return TemplateResponse(request, 'wagtailadmin/pages/copy.html', {'page': page, 'form': form, 'next': next_url})