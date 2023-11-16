from django.template import engines
from django.template.loader import render_to_string
from django.test import TestCase
from django.utils.safestring import mark_safe
from wagtail import __version__, blocks
from wagtail.coreutils import get_dummy_request
from wagtail.models import Page, Site
from wagtail.test.testapp.blocks import SectionBlock

class TestCoreGlobalsAndFilters(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.engine = engines['jinja2']

    def render(self, string, context=None, request_context=True):
        if False:
            i = 10
            return i + 15
        if context is None:
            context = {}
        if request_context:
            site = Site.objects.get(is_default_site=True)
            context['request'] = get_dummy_request(site=site)
        template = self.engine.from_string(string)
        return template.render(context)

    def test_richtext(self):
        if False:
            print('Hello World!')
        richtext = '<p>Merry <a linktype="page" id="2">Christmas</a>!</p>'
        self.assertEqual(self.render('{{ text|richtext }}', {'text': richtext}), '<p>Merry <a href="/">Christmas</a>!</p>')

    def test_pageurl(self):
        if False:
            i = 10
            return i + 15
        page = Page.objects.get(pk=2)
        self.assertEqual(self.render('{{ pageurl(page) }}', {'page': page}), page.url)

    def test_fullpageurl(self):
        if False:
            i = 10
            return i + 15
        page = Page.objects.get(pk=2)
        self.assertEqual(self.render('{{ fullpageurl(page) }}', {'page': page}), page.full_url)

    def test_slugurl(self):
        if False:
            i = 10
            return i + 15
        page = Page.objects.get(pk=2)
        self.assertEqual(self.render('{{ slugurl(page.slug) }}', {'page': page}), page.url)

    def test_bad_slugurl(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.render('{{ slugurl("bad-slug-doesnt-exist") }}', {}), 'None')

    def test_wagtail_site(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.render('{{ wagtail_site().hostname }}'), 'localhost')

    def test_wagtail_version(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.render('{{ wagtail_version() }}'), __version__)

class TestJinjaEscaping(TestCase):
    fixtures = ['test.json']

    def test_block_render_result_is_safe(self):
        if False:
            return 10
        "\n        Ensure that any results of template rendering in block.render are marked safe\n        so that they don't get double-escaped when inserted into a parent template (#2541)\n        "
        stream_block = blocks.StreamBlock([('paragraph', blocks.CharBlock(template='tests/jinja2/paragraph.html'))])
        stream_value = stream_block.to_python([{'type': 'paragraph', 'value': 'hello world'}])
        result = render_to_string('tests/jinja2/stream.html', {'value': stream_value})
        self.assertIn('<p>hello world</p>', result)

    def test_rich_text_is_safe(self):
        if False:
            i = 10
            return i + 15
        "\n        Ensure that RichText values are marked safe\n        so that they don't get double-escaped when inserted into a parent template (#2542)\n        "
        stream_block = blocks.StreamBlock([('paragraph', blocks.RichTextBlock(template='tests/jinja2/rich_text.html'))])
        stream_value = stream_block.to_python([{'type': 'paragraph', 'value': '<p>Merry <a linktype="page" id="4">Christmas</a>!</p>'}])
        result = render_to_string('tests/jinja2/stream.html', {'value': stream_value})
        self.assertIn('<p>Merry <a href="/events/christmas/">Christmas</a>!</p>', result)

class TestIncludeBlockTag(TestCase):

    def test_include_block_tag_with_boundblock(self):
        if False:
            return 10
        "\n        The include_block tag should be able to render a BoundBlock's template\n        while keeping the parent template's context\n        "
        block = blocks.CharBlock(template='tests/jinja2/heading_block.html')
        bound_block = block.bind('bonjour')
        result = render_to_string('tests/jinja2/include_block_test.html', {'test_block': bound_block, 'language': 'fr'})
        self.assertIn('<body><h1 lang="fr">bonjour</h1></body>', result)

    def test_include_block_tag_with_structvalue(self):
        if False:
            print('Hello World!')
        "\n        The include_block tag should be able to render a StructValue's template\n        while keeping the parent template's context\n        "
        block = SectionBlock()
        struct_value = block.to_python({'title': 'Bonjour', 'body': 'monde <i>italique</i>'})
        result = render_to_string('tests/jinja2/include_block_test.html', {'test_block': struct_value, 'language': 'fr'})
        self.assertIn('<body><h1 lang="fr">Bonjour</h1>monde <i>italique</i></body>', result)

    def test_include_block_tag_with_streamvalue(self):
        if False:
            print('Hello World!')
        "\n        The include_block tag should be able to render a StreamValue's template\n        while keeping the parent template's context\n        "
        block = blocks.StreamBlock([('heading', blocks.CharBlock(template='tests/jinja2/heading_block.html')), ('paragraph', blocks.CharBlock())], template='tests/jinja2/stream_with_language.html')
        stream_value = block.to_python([{'type': 'heading', 'value': 'Bonjour'}])
        result = render_to_string('tests/jinja2/include_block_test.html', {'test_block': stream_value, 'language': 'fr'})
        self.assertIn('<div class="heading" lang="fr"><h1 lang="fr">Bonjour</h1></div>', result)

    def test_include_block_tag_with_plain_value(self):
        if False:
            i = 10
            return i + 15
        '\n        The include_block tag should be able to render a value without a render_as_block method\n        by just rendering it as a string\n        '
        result = render_to_string('tests/jinja2/include_block_test.html', {'test_block': 42})
        self.assertIn('<body>42</body>', result)

    def test_include_block_tag_with_filtered_value(self):
        if False:
            return 10
        '\n        The block parameter on include_block tag should support complex values including filters,\n        e.g. {% include_block foo|default:123 %}\n        '
        block = blocks.CharBlock(template='tests/jinja2/heading_block.html')
        bound_block = block.bind('bonjour')
        result = render_to_string('tests/jinja2/include_block_test_with_filter.html', {'test_block': bound_block, 'language': 'fr'})
        self.assertIn('<body><h1 lang="fr">bonjour</h1></body>', result)
        result = render_to_string('tests/jinja2/include_block_test_with_filter.html', {'test_block': None, 'language': 'fr'})
        self.assertIn('<body>999</body>', result)

    def test_include_block_tag_with_additional_variable(self):
        if False:
            i = 10
            return i + 15
        '\n        The include_block tag should be able to pass local variables from parent context to the\n        child context\n        '
        block = blocks.CharBlock(template='tests/blocks/heading_block.html')
        bound_block = block.bind('bonjour')
        result = render_to_string('tests/jinja2/include_block_tag_with_additional_variable.html', {'test_block': bound_block})
        self.assertIn('<body><h1 class="important">bonjour</h1></body>', result)

    def test_include_block_html_escaping(self):
        if False:
            return 10
        '\n        Output of include_block should be escaped as per Django autoescaping rules\n        '
        block = blocks.CharBlock()
        bound_block = block.bind(block.to_python('some <em>evil</em> HTML'))
        result = render_to_string('tests/jinja2/include_block_test.html', {'test_block': bound_block})
        self.assertIn('<body>some &lt;em&gt;evil&lt;/em&gt; HTML</body>', result)
        result = render_to_string('tests/blocks/include_block_autoescape_off_test.html', {'test_block': bound_block})
        self.assertIn('<body>some <em>evil</em> HTML</body>', result)
        result = render_to_string('tests/jinja2/include_block_test.html', {'test_block': 'some <em>evil</em> HTML'})
        self.assertIn('<body>some &lt;em&gt;evil&lt;/em&gt; HTML</body>', result)
        result = render_to_string('tests/jinja2/include_block_autoescape_off_test.html', {'test_block': 'some <em>evil</em> HTML'})
        self.assertIn('<body>some <em>evil</em> HTML</body>', result)
        block = blocks.RawHTMLBlock()
        bound_block = block.bind(block.to_python('some <em>evil</em> HTML'))
        result = render_to_string('tests/jinja2/include_block_test.html', {'test_block': bound_block})
        self.assertIn('<body>some <em>evil</em> HTML</body>', result)
        result = render_to_string('tests/jinja2/include_block_test.html', {'test_block': mark_safe('some <em>evil</em> HTML')})
        self.assertIn('<body>some <em>evil</em> HTML</body>', result)