from django.utils.html import escape
from wagtail.models import Page
from wagtail.rich_text import LinkHandler

class PageLinkHandler(LinkHandler):
    identifier = 'page'

    @staticmethod
    def get_model():
        if False:
            i = 10
            return i + 15
        return Page

    @classmethod
    def get_instance(cls, attrs):
        if False:
            for i in range(10):
                print('nop')
        return super().get_instance(attrs).specific

    @classmethod
    def expand_db_attributes(cls, attrs):
        if False:
            while True:
                i = 10
        try:
            page = cls.get_instance(attrs)
            return '<a href="%s">' % escape(page.localized.specific.url)
        except Page.DoesNotExist:
            return '<a>'

    @classmethod
    def extract_references(self, attrs):
        if False:
            while True:
                i = 10
        yield (Page, attrs['id'], '', '')