import json
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from wagtail.admin.tests.test_contentstate import content_state_equal
from wagtail.models import PAGE_MODEL_CLASSES, Page, Site
from wagtail.test.dummy_external_storage import DummyExternalStorage
from wagtail.test.testapp.models import BusinessChild, BusinessIndex, BusinessNowherePage, BusinessSubIndex, EventIndex, EventPage, SectionedRichTextPage, SimpleChildPage, SimplePage, SimpleParentPage, StreamPage
from wagtail.test.utils import WagtailPageTests, WagtailTestUtils
from wagtail.test.utils.form_data import inline_formset, nested_form_data, rich_text, streamfield

class TestAssertTagInHTML(WagtailTestUtils, TestCase):

    def test_assert_tag_in_html(self):
        if False:
            for i in range(10):
                print('nop')
        haystack = '<ul>\n            <li class="normal">hugh</li>\n            <li class="normal">pugh</li>\n            <li class="really important" lang="en"><em>barney</em> mcgrew</li>\n        </ul>'
        self.assertTagInHTML('<li lang="en" class="important really">', haystack)
        self.assertTagInHTML('<li class="normal">', haystack, count=2)
        with self.assertRaises(AssertionError):
            self.assertTagInHTML('<div lang="en" class="important really">', haystack)
        with self.assertRaises(AssertionError):
            self.assertTagInHTML('<li lang="en" class="important really">', haystack, count=2)
        with self.assertRaises(AssertionError):
            self.assertTagInHTML('<li lang="en" class="important">', haystack)
        with self.assertRaises(AssertionError):
            self.assertTagInHTML('<li lang="en" class="important really" data-extra="boom">', haystack)

    def test_assert_tag_in_html_with_extra_attrs(self):
        if False:
            return 10
        haystack = '<ul>\n            <li class="normal">hugh</li>\n            <li class="normal">pugh</li>\n            <li class="really important" lang="en"><em>barney</em> mcgrew</li>\n        </ul>'
        self.assertTagInHTML('<li class="important really">', haystack, allow_extra_attrs=True)
        self.assertTagInHTML('<li>', haystack, count=3, allow_extra_attrs=True)
        with self.assertRaises(AssertionError):
            self.assertTagInHTML('<li class="normal" lang="en">', haystack, allow_extra_attrs=True)
        with self.assertRaises(AssertionError):
            self.assertTagInHTML('<li class="important really">', haystack, count=2, allow_extra_attrs=True)

    def test_assert_tag_in_template_script(self):
        if False:
            return 10
        haystack = '<html>\n            <script type="text/template">\n                <p class="really important">first template block</p>\n            </script>\n            <script type="text/template">\n                <p class="really important">second template block</p>\n            </script>\n            <p class="normal">not in a script tag</p>\n        </html>'
        self.assertTagInTemplateScript('<p class="important really">', haystack)
        self.assertTagInTemplateScript('<p class="important really">', haystack, count=2)
        with self.assertRaises(AssertionError):
            self.assertTagInTemplateScript('<p class="normal">', haystack)

class TestWagtailPageTests(WagtailPageTests):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        site = Site.objects.get(is_default_site=True)
        self.root = site.root_page.specific

    def test_assert_can_create_at(self):
        if False:
            print('Hello World!')
        self.assertCanCreateAt(EventIndex, EventPage)
        self.assertCanCreateAt(Page, EventIndex)
        self.assertCanNotCreateAt(SimplePage, BusinessChild)
        with self.assertRaises(AssertionError):
            self.assertCanCreateAt(SimplePage, BusinessChild)
        with self.assertRaises(AssertionError):
            self.assertCanNotCreateAt(EventIndex, EventPage)

    def test_assert_can_create(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertFalse(EventIndex.objects.exists())
        self.assertCanCreate(self.root, EventIndex, {'title': 'Event Index', 'intro': '{"entityMap": {},"blocks": [\n                {"inlineStyleRanges": [], "text": "Event intro", "depth": 0, "type": "unstyled", "key": "00000", "entityRanges": []}\n            ]}'})
        self.assertTrue(EventIndex.objects.exists())
        self.assertTrue(EventIndex.objects.get().live)
        self.assertCanCreate(self.root, StreamPage, {'title': 'Flierp', 'body-0-type': 'text', 'body-0-value': 'Dit is onze mooie text', 'body-0-order': '0', 'body-0-deleted': '', 'body-1-type': 'rich_text', 'body-1-value': '{"entityMap": {},"blocks": [\n                {"inlineStyleRanges": [], "text": "Dit is onze mooie text in een ferrari", "depth": 0, "type": "unstyled", "key": "00000", "entityRanges": []}\n            ]}', 'body-1-order': '1', 'body-1-deleted': '', 'body-2-type': 'product', 'body-2-value-name': 'pegs', 'body-2-value-price': 'a pound', 'body-2-order': '2', 'body-2-deleted': '', 'body-count': '3'})
        self.assertCanCreate(self.root, SectionedRichTextPage, {'title': 'Fight Club', 'sections-TOTAL_FORMS': '2', 'sections-INITIAL_FORMS': '0', 'sections-MIN_NUM_FORMS': '0', 'sections-MAX_NUM_FORMS': '1000', 'sections-0-body': '{"entityMap": {},"blocks": [\n                {"inlineStyleRanges": [], "text": "Rule 1: You do not talk about Fight Club", "depth": 0, "type": "unstyled", "key": "00000", "entityRanges": []}\n            ]}', 'sections-0-ORDER': '0', 'sections-0-DELETE': '', 'sections-1-body': '{"entityMap": {},"blocks": [\n                {"inlineStyleRanges": [], "text": "Rule 2: You DO NOT talk about Fight Club", "depth": 0, "type": "unstyled", "key": "00000", "entityRanges": []}\n            ]}', 'sections-1-ORDER': '0', 'sections-1-DELETE': ''})

    def test_assert_can_create_for_page_without_publish(self):
        if False:
            i = 10
            return i + 15
        self.assertCanCreate(self.root, SimplePage, {'title': 'Simple Lorem Page', 'content': 'Lorem ipsum dolor sit amet'}, publish=False)
        created_page = Page.objects.get(title='Simple Lorem Page')
        self.assertFalse(created_page.live)

    def test_assert_can_create_with_form_helpers(self):
        if False:
            return 10
        self.assertFalse(EventIndex.objects.exists())
        self.assertCanCreate(self.root, EventIndex, nested_form_data({'title': 'Event Index', 'intro': rich_text('<p>Event intro</p>')}))
        self.assertTrue(EventIndex.objects.exists())
        self.assertCanCreate(self.root, StreamPage, nested_form_data({'title': 'Flierp', 'body': streamfield([('text', 'Dit is onze mooie text'), ('rich_text', rich_text('<p>Dit is onze mooie text in een ferrari</p>')), ('product', {'name': 'pegs', 'price': 'a pound'})])}))
        self.assertCanCreate(self.root, SectionedRichTextPage, nested_form_data({'title': 'Fight Club', 'sections': inline_formset([{'body': rich_text('<p>Rule 1: You do not talk about Fight Club</p>')}, {'body': rich_text('<p>Rule 2: You DO NOT talk about Fight Club</p>')}])}))

    def test_assert_can_create_subpage_rules(self):
        if False:
            i = 10
            return i + 15
        simple_page = SimplePage(title='Simple Page', slug='simple', content='hello')
        self.root.add_child(instance=simple_page)
        with self.assertRaisesRegex(AssertionError, 'Can not create a tests.businesschild under a tests.simplepage'):
            self.assertCanCreate(simple_page, BusinessChild, {})

    def test_assert_can_create_validation_error(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(AssertionError, '\\bslug:\\n[\\s\\S]*\\btitle:\\n'):
            self.assertCanCreate(self.root, SimplePage, {})

    def test_assert_allowed_subpage_types(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertAllowedSubpageTypes(BusinessIndex, {BusinessChild, BusinessSubIndex})
        self.assertAllowedSubpageTypes(BusinessChild, {})
        all_but_business = set(PAGE_MODEL_CLASSES) - {BusinessSubIndex, BusinessChild, BusinessNowherePage, SimpleChildPage}
        self.assertAllowedSubpageTypes(Page, all_but_business)
        with self.assertRaises(AssertionError):
            self.assertAllowedSubpageTypes(BusinessSubIndex, {BusinessSubIndex, BusinessChild})

    def test_assert_allowed_parent_page_types(self):
        if False:
            print('Hello World!')
        self.assertAllowedParentPageTypes(BusinessChild, {BusinessIndex, BusinessSubIndex})
        self.assertAllowedParentPageTypes(BusinessSubIndex, {BusinessIndex})
        all_but_business = set(PAGE_MODEL_CLASSES) - {BusinessSubIndex, BusinessChild, BusinessIndex, SimpleParentPage}
        self.assertAllowedParentPageTypes(BusinessIndex, all_but_business)
        with self.assertRaises(AssertionError):
            self.assertAllowedParentPageTypes(BusinessSubIndex, {BusinessSubIndex, BusinessIndex})

class TestFormDataHelpers(TestCase):

    def test_nested_form_data(self):
        if False:
            i = 10
            return i + 15
        result = nested_form_data({'foo': 'bar', 'parent': {'child': 'field'}})
        self.assertEqual(result, {'foo': 'bar', 'parent-child': 'field'})

    def test_streamfield(self):
        if False:
            while True:
                i = 10
        result = nested_form_data({'content': streamfield([('text', 'Hello, world'), ('text', 'Goodbye, world'), ('coffee', {'type': 'latte', 'milk': 'soya'})])})
        self.assertEqual(result, {'content-count': '3', 'content-0-type': 'text', 'content-0-value': 'Hello, world', 'content-0-order': '0', 'content-0-deleted': '', 'content-1-type': 'text', 'content-1-value': 'Goodbye, world', 'content-1-order': '1', 'content-1-deleted': '', 'content-2-type': 'coffee', 'content-2-value-type': 'latte', 'content-2-value-milk': 'soya', 'content-2-order': '2', 'content-2-deleted': ''})

    def test_inline_formset(self):
        if False:
            while True:
                i = 10
        result = nested_form_data({'lines': inline_formset([{'text': 'Hello'}, {'text': 'World'}])})
        self.assertEqual(result, {'lines-TOTAL_FORMS': '2', 'lines-INITIAL_FORMS': '0', 'lines-MIN_NUM_FORMS': '0', 'lines-MAX_NUM_FORMS': '1000', 'lines-0-text': 'Hello', 'lines-0-ORDER': '0', 'lines-0-DELETE': '', 'lines-1-text': 'World', 'lines-1-ORDER': '1', 'lines-1-DELETE': ''})

    def test_default_rich_text(self):
        if False:
            print('Hello World!')
        result = rich_text('<h2>title</h2><p>para</p>')
        self.assertTrue(content_state_equal(json.loads(result), {'entityMap': {}, 'blocks': [{'inlineStyleRanges': [], 'text': 'title', 'depth': 0, 'type': 'header-two', 'key': '00000', 'entityRanges': []}, {'inlineStyleRanges': [], 'text': 'para', 'depth': 0, 'type': 'unstyled', 'key': '00000', 'entityRanges': []}]}))

    def test_rich_text_with_custom_features(self):
        if False:
            for i in range(10):
                print('nop')
        result = rich_text('<h2>title</h2><p>para</p>', features=['p'])
        self.assertTrue(content_state_equal(json.loads(result), {'entityMap': {}, 'blocks': [{'inlineStyleRanges': [], 'text': 'title', 'depth': 0, 'type': 'unstyled', 'key': '00000', 'entityRanges': []}, {'inlineStyleRanges': [], 'text': 'para', 'depth': 0, 'type': 'unstyled', 'key': '00000', 'entityRanges': []}]}))

    def test_rich_text_with_alternative_editor(self):
        if False:
            for i in range(10):
                print('nop')
        result = rich_text('<h2>title</h2><p>para</p>', editor='custom')
        self.assertEqual(result, '<h2>title</h2><p>para</p>')

class TestDummyExternalStorage(WagtailTestUtils, TestCase):

    def test_save_with_incorrect_file_object_position(self):
        if False:
            return 10
        '\n        Test that DummyExternalStorage correctly warns about attempts\n        to write files that are not rewound to the start\n        '
        png = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc````\x00\x00\x00\x05\x00\x01\xa5\xf6E@\x00\x00\x00\x00IEND\xaeB`\x82'
        simple_png = SimpleUploadedFile(name='test.png', content=png, content_type='image/png')
        simple_png.read()
        with self.assertRaisesMessage(ValueError, 'Content file pointer should be at 0 - got 70 instead'):
            DummyExternalStorage().save('test.png', simple_png)