from django.contrib.sitemaps import Sitemap as DjangoSitemap

class Sitemap(DjangoSitemap):

    def __init__(self, request=None):
        if False:
            i = 10
            return i + 15
        self.request = request

    def location(self, obj):
        if False:
            return 10
        return obj.get_full_url(self.request)

    def lastmod(self, obj):
        if False:
            for i in range(10):
                print('nop')
        return obj.last_published_at or obj.latest_revision_created_at

    def get_wagtail_site(self):
        if False:
            i = 10
            return i + 15
        from wagtail.models import Site
        site = Site.find_for_request(self.request)
        if site is None:
            return Site.objects.select_related('root_page').get(is_default_site=True)
        return site

    def items(self):
        if False:
            return 10
        return self.get_wagtail_site().root_page.get_descendants(inclusive=True).live().public().order_by('path').defer_streamfields().specific()

    def _urls(self, page, protocol, domain):
        if False:
            while True:
                i = 10
        urls = []
        last_mods = set()
        for item in self.paginator.page(page).object_list.iterator():
            url_info_items = item.get_sitemap_urls(self.request)
            for url_info in url_info_items:
                urls.append(url_info)
                last_mods.add(url_info.get('lastmod'))
        if last_mods and None not in last_mods:
            self.latest_lastmod = max(last_mods)
        return urls