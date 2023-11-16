from django.contrib.auth import get_user_model
from django.contrib.syndication.views import Feed
from django.utils import timezone
from django.utils.feedgenerator import Rss201rev2Feed
from blog.models import Article
from djangoblog.utils import CommonMarkdown

class DjangoBlogFeed(Feed):
    feed_type = Rss201rev2Feed
    description = '大巧无工,重剑无锋.'
    title = '且听风吟 大巧无工,重剑无锋. '
    link = '/feed/'

    def author_name(self):
        if False:
            while True:
                i = 10
        return get_user_model().objects.first().nickname

    def author_link(self):
        if False:
            print('Hello World!')
        return get_user_model().objects.first().get_absolute_url()

    def items(self):
        if False:
            return 10
        return Article.objects.filter(type='a', status='p').order_by('-pub_time')[:5]

    def item_title(self, item):
        if False:
            for i in range(10):
                print('nop')
        return item.title

    def item_description(self, item):
        if False:
            print('Hello World!')
        return CommonMarkdown.get_markdown(item.body)

    def feed_copyright(self):
        if False:
            return 10
        now = timezone.now()
        return 'Copyright© {year} 且听风吟'.format(year=now.year)

    def item_link(self, item):
        if False:
            for i in range(10):
                print('nop')
        return item.get_absolute_url()

    def item_guid(self, item):
        if False:
            return 10
        return