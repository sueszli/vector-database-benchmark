from django.contrib.sitemaps import Sitemap
from django.db.models import Q
from django.utils import translation
from cms.models import Title
from cms.utils import get_current_site
from cms.utils.i18n import get_public_languages

def from_iterable(iterables):
    if False:
        return 10
    '\n    Backport of itertools.chain.from_iterable\n    '
    for it in iterables:
        for element in it:
            yield element

class CMSSitemap(Sitemap):
    changefreq = 'monthly'
    priority = 0.5

    def items(self):
        if False:
            i = 10
            return i + 15
        site = get_current_site()
        languages = get_public_languages(site_id=site.pk)
        all_titles = Title.objects.public().filter(Q(redirect='') | Q(redirect__isnull=True), language__in=languages, page__login_required=False, page__node__site=site).order_by('page__node__path')
        return all_titles

    def lastmod(self, title):
        if False:
            while True:
                i = 10
        modification_dates = [title.page.changed_date, title.page.publication_date]

        def plugins_for_placeholder(placeholder):
            if False:
                print('Hello World!')
            return placeholder.get_plugins()
        plugins = from_iterable(map(plugins_for_placeholder, title.page.placeholders.all()))
        plugin_modification_dates = (plugin.changed_date for plugin in plugins)
        modification_dates.extend(plugin_modification_dates)
        return max(modification_dates)

    def location(self, title):
        if False:
            for i in range(10):
                print('nop')
        translation.activate(title.language)
        url = title.page.get_absolute_url(title.language)
        translation.deactivate()
        return url