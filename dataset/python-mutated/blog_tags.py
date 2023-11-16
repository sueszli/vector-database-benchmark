import hashlib
import logging
import random
import urllib
from django import template
from django.conf import settings
from django.db.models import Q
from django.shortcuts import get_object_or_404
from django.template.defaultfilters import stringfilter
from django.templatetags.static import static
from django.urls import reverse
from django.utils.safestring import mark_safe
from blog.models import Article, Category, Tag, Links, SideBar, LinkShowType
from comments.models import Comment
from djangoblog.utils import CommonMarkdown, sanitize_html
from djangoblog.utils import cache
from djangoblog.utils import get_current_site
from oauth.models import OAuthUser
logger = logging.getLogger(__name__)
register = template.Library()

@register.simple_tag
def timeformat(data):
    if False:
        i = 10
        return i + 15
    try:
        return data.strftime(settings.TIME_FORMAT)
    except Exception as e:
        logger.error(e)
        return ''

@register.simple_tag
def datetimeformat(data):
    if False:
        return 10
    try:
        return data.strftime(settings.DATE_TIME_FORMAT)
    except Exception as e:
        logger.error(e)
        return ''

@register.filter()
@stringfilter
def custom_markdown(content):
    if False:
        return 10
    return mark_safe(CommonMarkdown.get_markdown(content))

@register.simple_tag
def get_markdown_toc(content):
    if False:
        i = 10
        return i + 15
    from djangoblog.utils import CommonMarkdown
    (body, toc) = CommonMarkdown.get_markdown_with_toc(content)
    return mark_safe(toc)

@register.filter()
@stringfilter
def comment_markdown(content):
    if False:
        while True:
            i = 10
    content = CommonMarkdown.get_markdown(content)
    return mark_safe(sanitize_html(content))

@register.filter(is_safe=True)
@stringfilter
def truncatechars_content(content):
    if False:
        print('Hello World!')
    '\n    获得文章内容的摘要\n    :param content:\n    :return:\n    '
    from django.template.defaultfilters import truncatechars_html
    from djangoblog.utils import get_blog_setting
    blogsetting = get_blog_setting()
    return truncatechars_html(content, blogsetting.article_sub_length)

@register.filter(is_safe=True)
@stringfilter
def truncate(content):
    if False:
        while True:
            i = 10
    from django.utils.html import strip_tags
    return strip_tags(content)[:150]

@register.inclusion_tag('blog/tags/breadcrumb.html')
def load_breadcrumb(article):
    if False:
        i = 10
        return i + 15
    '\n    获得文章面包屑\n    :param article:\n    :return:\n    '
    names = article.get_category_tree()
    from djangoblog.utils import get_blog_setting
    blogsetting = get_blog_setting()
    site = get_current_site().domain
    names.append((blogsetting.site_name, '/'))
    names = names[::-1]
    return {'names': names, 'title': article.title, 'count': len(names) + 1}

@register.inclusion_tag('blog/tags/article_tag_list.html')
def load_articletags(article):
    if False:
        print('Hello World!')
    '\n    文章标签\n    :param article:\n    :return:\n    '
    tags = article.tags.all()
    tags_list = []
    for tag in tags:
        url = tag.get_absolute_url()
        count = tag.get_article_count()
        tags_list.append((url, count, tag, random.choice(settings.BOOTSTRAP_COLOR_TYPES)))
    return {'article_tags_list': tags_list}

@register.inclusion_tag('blog/tags/sidebar.html')
def load_sidebar(user, linktype):
    if False:
        return 10
    '\n    加载侧边栏\n    :return:\n    '
    value = cache.get('sidebar' + linktype)
    if value:
        value['user'] = user
        return value
    else:
        logger.info('load sidebar')
        from djangoblog.utils import get_blog_setting
        blogsetting = get_blog_setting()
        recent_articles = Article.objects.filter(status='p')[:blogsetting.sidebar_article_count]
        sidebar_categorys = Category.objects.all()
        extra_sidebars = SideBar.objects.filter(is_enable=True).order_by('sequence')
        most_read_articles = Article.objects.filter(status='p').order_by('-views')[:blogsetting.sidebar_article_count]
        dates = Article.objects.datetimes('creation_time', 'month', order='DESC')
        links = Links.objects.filter(is_enable=True).filter(Q(show_type=str(linktype)) | Q(show_type=LinkShowType.A))
        commment_list = Comment.objects.filter(is_enable=True).order_by('-id')[:blogsetting.sidebar_comment_count]
        increment = 5
        tags = Tag.objects.all()
        sidebar_tags = None
        if tags and len(tags) > 0:
            s = [t for t in [(t, t.get_article_count()) for t in tags] if t[1]]
            count = sum([t[1] for t in s])
            dd = 1 if count == 0 or not len(tags) else count / len(tags)
            import random
            sidebar_tags = list(map(lambda x: (x[0], x[1], x[1] / dd * increment + 10), s))
            random.shuffle(sidebar_tags)
        value = {'recent_articles': recent_articles, 'sidebar_categorys': sidebar_categorys, 'most_read_articles': most_read_articles, 'article_dates': dates, 'sidebar_comments': commment_list, 'sidabar_links': links, 'show_google_adsense': blogsetting.show_google_adsense, 'google_adsense_codes': blogsetting.google_adsense_codes, 'open_site_comment': blogsetting.open_site_comment, 'show_gongan_code': blogsetting.show_gongan_code, 'sidebar_tags': sidebar_tags, 'extra_sidebars': extra_sidebars}
        cache.set('sidebar' + linktype, value, 60 * 60 * 60 * 3)
        logger.info('set sidebar cache.key:{key}'.format(key='sidebar' + linktype))
        value['user'] = user
        return value

@register.inclusion_tag('blog/tags/article_meta_info.html')
def load_article_metas(article, user):
    if False:
        i = 10
        return i + 15
    '\n    获得文章meta信息\n    :param article:\n    :return:\n    '
    return {'article': article, 'user': user}

@register.inclusion_tag('blog/tags/article_pagination.html')
def load_pagination_info(page_obj, page_type, tag_name):
    if False:
        while True:
            i = 10
    previous_url = ''
    next_url = ''
    if page_type == '':
        if page_obj.has_next():
            next_number = page_obj.next_page_number()
            next_url = reverse('blog:index_page', kwargs={'page': next_number})
        if page_obj.has_previous():
            previous_number = page_obj.previous_page_number()
            previous_url = reverse('blog:index_page', kwargs={'page': previous_number})
    if page_type == '分类标签归档':
        tag = get_object_or_404(Tag, name=tag_name)
        if page_obj.has_next():
            next_number = page_obj.next_page_number()
            next_url = reverse('blog:tag_detail_page', kwargs={'page': next_number, 'tag_name': tag.slug})
        if page_obj.has_previous():
            previous_number = page_obj.previous_page_number()
            previous_url = reverse('blog:tag_detail_page', kwargs={'page': previous_number, 'tag_name': tag.slug})
    if page_type == '作者文章归档':
        if page_obj.has_next():
            next_number = page_obj.next_page_number()
            next_url = reverse('blog:author_detail_page', kwargs={'page': next_number, 'author_name': tag_name})
        if page_obj.has_previous():
            previous_number = page_obj.previous_page_number()
            previous_url = reverse('blog:author_detail_page', kwargs={'page': previous_number, 'author_name': tag_name})
    if page_type == '分类目录归档':
        category = get_object_or_404(Category, name=tag_name)
        if page_obj.has_next():
            next_number = page_obj.next_page_number()
            next_url = reverse('blog:category_detail_page', kwargs={'page': next_number, 'category_name': category.slug})
        if page_obj.has_previous():
            previous_number = page_obj.previous_page_number()
            previous_url = reverse('blog:category_detail_page', kwargs={'page': previous_number, 'category_name': category.slug})
    return {'previous_url': previous_url, 'next_url': next_url, 'page_obj': page_obj}

@register.inclusion_tag('blog/tags/article_info.html')
def load_article_detail(article, isindex, user):
    if False:
        for i in range(10):
            print('nop')
    '\n    加载文章详情\n    :param article:\n    :param isindex:是否列表页，若是列表页只显示摘要\n    :return:\n    '
    from djangoblog.utils import get_blog_setting
    blogsetting = get_blog_setting()
    return {'article': article, 'isindex': isindex, 'user': user, 'open_site_comment': blogsetting.open_site_comment}

@register.filter
def gravatar_url(email, size=40):
    if False:
        while True:
            i = 10
    '获得gravatar头像'
    cachekey = 'gravatat/' + email
    url = cache.get(cachekey)
    if url:
        return url
    else:
        usermodels = OAuthUser.objects.filter(email=email)
        if usermodels:
            o = list(filter(lambda x: x.picture is not None, usermodels))
            if o:
                return o[0].picture
        email = email.encode('utf-8')
        default = static('blog/img/avatar.png')
        url = 'https://www.gravatar.com/avatar/%s?%s' % (hashlib.md5(email.lower()).hexdigest(), urllib.parse.urlencode({'d': default, 's': str(size)}))
        cache.set(cachekey, url, 60 * 60 * 10)
        logger.info('set gravatar cache.key:{key}'.format(key=cachekey))
        return url

@register.filter
def gravatar(email, size=40):
    if False:
        i = 10
        return i + 15
    '获得gravatar头像'
    url = gravatar_url(email, size)
    return mark_safe('<img src="%s" height="%d" width="%d">' % (url, size, size))

@register.simple_tag
def query(qs, **kwargs):
    if False:
        i = 10
        return i + 15
    ' template tag which allows queryset filtering. Usage:\n          {% query books author=author as mybooks %}\n          {% for book in mybooks %}\n            ...\n          {% endfor %}\n    '
    return qs.filter(**kwargs)

@register.filter
def addstr(arg1, arg2):
    if False:
        print('Hello World!')
    'concatenate arg1 & arg2'
    return str(arg1) + str(arg2)