from django.forms import Media
from wagtail import hooks
from wagtail.admin.auth import user_has_any_page_permission
from wagtail.admin.navigation import get_site_for_user
from wagtail.admin.ui.components import Component
from wagtail.models import Page, Site

class SummaryItem(Component):
    order = 100

    def __init__(self, request):
        if False:
            for i in range(10):
                print('nop')
        self.request = request

    def is_shown(self):
        if False:
            return 10
        return True

class PagesSummaryItem(SummaryItem):
    order = 100
    template_name = 'wagtailadmin/home/site_summary_pages.html'

    def get_context_data(self, parent_context):
        if False:
            for i in range(10):
                print('nop')
        site_details = get_site_for_user(self.request.user)
        root_page = site_details['root_page']
        site_name = site_details['site_name']
        if root_page:
            page_count = Page.objects.descendant_of(root_page, inclusive=True).count()
            if root_page.is_root():
                page_count -= 1
                try:
                    root_page = Site.objects.get().root_page
                except (Site.DoesNotExist, Site.MultipleObjectsReturned):
                    pass
        else:
            page_count = 0
        return {'root_page': root_page, 'total_pages': page_count, 'site_name': site_name}

    def is_shown(self):
        if False:
            while True:
                i = 10
        return user_has_any_page_permission(self.request.user)

class SiteSummaryPanel(Component):
    name = 'site_summary'
    template_name = 'wagtailadmin/home/site_summary.html'
    order = 100

    def __init__(self, request):
        if False:
            for i in range(10):
                print('nop')
        self.request = request
        summary_items = []
        for fn in hooks.get_hooks('construct_homepage_summary_items'):
            fn(request, summary_items)
        self.summary_items = [s for s in summary_items if s.is_shown()]
        self.summary_items.sort(key=lambda p: p.order)

    def get_context_data(self, parent_context):
        if False:
            for i in range(10):
                print('nop')
        context = super().get_context_data(parent_context)
        context['summary_items'] = self.summary_items
        return context

    @property
    def media(self):
        if False:
            while True:
                i = 10
        media = Media()
        for item in self.summary_items:
            media += item.media
        return media