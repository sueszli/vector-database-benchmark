from django.template import Context, Template
from django.test import RequestFactory, TestCase
from django.utils.html import format_html
from wagtail.admin.ui.tables import BaseColumn, Column, Table, TitleColumn
from wagtail.models import Page, Site

class TestTable(TestCase):
    fixtures = ['test.json']

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.rf = RequestFactory()

    def render_component(self, obj):
        if False:
            i = 10
            return i + 15
        request = self.rf.get('/')
        template = Template('{% load wagtailadmin_tags %}{% component obj %}')
        return template.render(Context({'request': request, 'obj': obj}))

    def test_table_render(self):
        if False:
            print('Hello World!')
        data = [{'first_name': 'Paul', 'last_name': 'Simon'}, {'first_name': 'Art', 'last_name': 'Garfunkel'}]
        table = Table([Column('first_name'), Column('last_name')], data)
        html = self.render_component(table)
        self.assertHTMLEqual(html, '\n            <table class="listing">\n                <thead>\n                    <tr><th>First name</th><th>Last name</th></tr>\n                </thead>\n                <tbody>\n                    <tr><td>Paul</td><td>Simon</td></tr>\n                    <tr><td>Art</td><td>Garfunkel</td></tr>\n                </tbody>\n            </table>\n        ')

    def test_table_render_with_width(self):
        if False:
            for i in range(10):
                print('nop')
        data = [{'first_name': 'Paul', 'last_name': 'Simon'}, {'first_name': 'Art', 'last_name': 'Garfunkel'}]
        table = Table([Column('first_name'), Column('last_name', width='75%')], data)
        html = self.render_component(table)
        self.assertHTMLEqual(html, '\n            <table class="listing">\n                <col />\n                <col width="75%" />\n                <thead>\n                    <tr><th>First name</th><th>Last name</th></tr>\n                </thead>\n                <tbody>\n                    <tr><td>Paul</td><td>Simon</td></tr>\n                    <tr><td>Art</td><td>Garfunkel</td></tr>\n                </tbody>\n            </table>\n        ')

    def test_title_column(self):
        if False:
            i = 10
            return i + 15
        root_page = Page.objects.filter(depth=2).first()
        blog = Site.objects.create(hostname='blog.example.com', site_name='My blog', root_page=root_page)
        gallery = Site.objects.create(hostname='gallery.example.com', site_name='My gallery', root_page=root_page)
        data = [blog, gallery]
        table = Table([TitleColumn('hostname', url_name='wagtailsites:edit', link_classname='choose-site', link_attrs={'data-chooser': 'yes'}), Column('site_name', label='Site name')], data)
        html = self.render_component(table)
        self.assertHTMLEqual(html, '\n            <table class="listing">\n                <thead>\n                    <tr><th>Hostname</th><th>Site name</th></tr>\n                </thead>\n                <tbody>\n                    <tr>\n                        <td class="title">\n                            <div class="title-wrapper">\n                                <a href="/admin/sites/edit/%d/" class="choose-site" data-chooser="yes">blog.example.com</a>\n                            </div>\n                        </td>\n                        <td>My blog</td>\n                    </tr>\n                    <tr>\n                        <td class="title">\n                            <div class="title-wrapper">\n                                <a href="/admin/sites/edit/%d/" class="choose-site" data-chooser="yes">gallery.example.com</a>\n                            </div>\n                        </td>\n                        <td>My gallery</td>\n                    </tr>\n                </tbody>\n            </table>\n        ' % (blog.pk, gallery.pk))

    def test_column_media(self):
        if False:
            print('Hello World!')

        class FancyColumn(Column):

            class Media:
                js = ['js/gradient-fill.js']
        data = [{'first_name': 'Paul', 'last_name': 'Simon'}, {'first_name': 'Art', 'last_name': 'Garfunkel'}]
        table = Table([FancyColumn('first_name'), Column('last_name')], data)
        self.assertIn('src="/static/js/gradient-fill.js"', str(table.media['js']))

    def test_row_classname(self):
        if False:
            print('Hello World!')

        class SiteTable(Table):

            def get_row_classname(self, instance):
                if False:
                    i = 10
                    return i + 15
                return 'default-site' if instance.is_default_site else ''
        root_page = Page.objects.filter(depth=2).first()
        blog = Site.objects.create(hostname='blog.example.com', site_name='My blog', root_page=root_page, is_default_site=True)
        gallery = Site.objects.create(hostname='gallery.example.com', site_name='My gallery', root_page=root_page)
        data = [blog, gallery]
        table = SiteTable([Column('hostname'), Column('site_name', label='Site name')], data)
        html = self.render_component(table)
        self.assertHTMLEqual(html, '\n            <table class="listing">\n                <thead>\n                    <tr><th>Hostname</th><th>Site name</th></tr>\n                </thead>\n                <tbody>\n                    <tr class="default-site">\n                        <td>blog.example.com</td>\n                        <td>My blog</td>\n                    </tr>\n                    <tr>\n                        <td>gallery.example.com</td>\n                        <td>My gallery</td>\n                    </tr>\n                </tbody>\n            </table>\n        ')

    def test_row_attrs(self):
        if False:
            return 10

        class SiteTable(Table):

            def get_row_attrs(self, instance):
                if False:
                    i = 10
                    return i + 15
                attrs = super().get_row_attrs(instance)
                attrs['data-id'] = instance.pk
                return attrs
        root_page = Page.objects.filter(depth=2).first()
        blog = Site.objects.create(hostname='blog.example.com', site_name='My blog', root_page=root_page, is_default_site=True)
        gallery = Site.objects.create(hostname='gallery.example.com', site_name='My gallery', root_page=root_page)
        data = [blog, gallery]
        table = SiteTable([Column('hostname'), Column('site_name', label='Site name')], data)
        html = self.render_component(table)
        self.assertHTMLEqual(html, f'\n            <table class="listing">\n                <thead>\n                    <tr><th>Hostname</th><th>Site name</th></tr>\n                </thead>\n                <tbody>\n                    <tr data-id="{blog.pk}">\n                        <td>blog.example.com</td>\n                        <td>My blog</td>\n                    </tr>\n                    <tr data-id="{gallery.pk}">\n                        <td>gallery.example.com</td>\n                        <td>My gallery</td>\n                    </tr>\n                </tbody>\n            </table>\n        ')

    def test_table_and_row_in_context(self):
        if False:
            print('Hello World!')
        data = [{'first_name': 'Paul', 'last_name': 'Simon'}, {'first_name': 'Art', 'last_name': 'Garfunkel'}]

        class CounterColumn(BaseColumn):

            def render_cell_html(self, instance, parent_context):
                if False:
                    i = 10
                    return i + 15
                context = self.get_cell_context_data(instance, parent_context)
                return format_html('<td>{} of {}</td>', context['row'].index + 1, context['table'].row_count)
        table = Table([CounterColumn('index'), Column('first_name'), Column('last_name')], data)
        html = self.render_component(table)
        self.assertHTMLEqual(html, '\n            <table class="listing">\n                <thead>\n                    <tr><th>Index</th><th>First name</th><th>Last name</th></tr>\n                </thead>\n                <tbody>\n                    <tr><td>1 of 2</td><td>Paul</td><td>Simon</td></tr>\n                    <tr><td>2 of 2</td><td>Art</td><td>Garfunkel</td></tr>\n                </tbody>\n            </table>\n        ')