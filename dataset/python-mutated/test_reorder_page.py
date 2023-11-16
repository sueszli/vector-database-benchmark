from django.test import TestCase
from django.urls import reverse
from wagtail.models import Page
from wagtail.test.testapp.models import BusinessChild, BusinessIndex, SimplePage
from wagtail.test.utils import WagtailTestUtils

class TestPageReorder(WagtailTestUtils, TestCase):
    fixtures = ['test.json']

    def __init__(self, methodName: str=...) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(methodName)

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.root_page = Page.objects.get(id=2)
        self.index_page = SimplePage(title='Simple', slug='simple', content='hello')
        self.root_page.add_child(instance=self.index_page)
        self.child_1 = SimplePage(title='Child 1 of SimplePage', slug='child-1', content='hello')
        self.index_page.add_child(instance=self.child_1)
        self.child_2 = SimplePage(title='Child 2 of SimplePage', slug='child-2', content='hello')
        self.index_page.add_child(instance=self.child_2)
        self.child_3 = SimplePage(title='Child 3 of SimplePage', slug='child-3', content='hello')
        self.index_page.add_child(instance=self.child_3)
        self.user = self.login()

    def test_page_set_page_position_get_request_with_simple_page(self):
        if False:
            while True:
                i = 10
        "\n        Test that GET requests to set_page_position view don't alter the page order.\n        "
        response = self.client.get(reverse('wagtailadmin_pages:set_page_position', args=(self.child_1.id,)))
        self.assertEqual(response.status_code, 200)
        child_slugs = self.index_page.get_children().values_list('slug', flat=True)
        self.assertListEqual(list(child_slugs), ['child-1', 'child-2', 'child-3'])

    def test_page_set_page_position_without_position_argument_moves_it_to_the_end(self):
        if False:
            i = 10
            return i + 15
        response = self.client.post(reverse('wagtailadmin_pages:set_page_position', args=(self.child_1.id,)))
        self.assertEqual(response.status_code, 200)
        child_slugs = self.index_page.get_children().values_list('slug', flat=True)
        self.assertListEqual(list(child_slugs), ['child-2', 'child-3', 'child-1'])

    def test_page_move_page_position_up(self):
        if False:
            for i in range(10):
                print('nop')
        'Moves child 3 to the first position.'
        response = self.client.post(reverse('wagtailadmin_pages:set_page_position', args=(self.child_3.id,)) + '?position=0')
        self.assertEqual(response.status_code, 200)
        child_slugs = self.index_page.get_children().values_list('slug', flat=True)
        self.assertListEqual(list(child_slugs), ['child-3', 'child-1', 'child-2'])

    def test_page_move_page_position_down(self):
        if False:
            while True:
                i = 10
        '\n        Moves child 3 to the first position.'
        response = self.client.post(reverse('wagtailadmin_pages:set_page_position', args=(self.child_1.id,)) + '?position=1')
        self.assertEqual(response.status_code, 200)
        child_slugs = self.index_page.get_children().values_list('slug', flat=True)
        self.assertListEqual(list(child_slugs), ['child-2', 'child-1', 'child-3'])

    def test_page_move_page_position_to_the_same_position(self):
        if False:
            i = 10
            return i + 15
        '\n        Moves child 3 to the first position.'
        response = self.client.post(reverse('wagtailadmin_pages:set_page_position', args=(self.child_1.id,)) + '?position=0')
        self.assertEqual(response.status_code, 200)
        child_slugs = self.index_page.get_children().values_list('slug', flat=True)
        self.assertListEqual(list(child_slugs), ['child-1', 'child-2', 'child-3'])

    def test_page_set_page_position_with_invalid_target_position(self):
        if False:
            print('Hello World!')
        response = self.client.post(reverse('wagtailadmin_pages:set_page_position', args=(self.child_3.id,)) + '?position=99')
        self.assertEqual(response.status_code, 200)
        child_slugs = self.index_page.get_children().values_list('slug', flat=True)
        self.assertListEqual(list(child_slugs), ['child-1', 'child-2', 'child-3'])

class TestPageReorderWithParentPageRestrictions(TestPageReorder):
    """
    This testCase is the same as the TestPageReorder class above, but with a different
    page type: BusinessChild has a parent_page_types restriction.
    This ensures that this restriction doesn't affect the ability to reorder pages.
    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.root_page = Page.objects.get(id=2)
        self.index_page = BusinessIndex(title='Simple', slug='simple')
        self.root_page.add_child(instance=self.index_page)
        self.child_1 = BusinessChild(title='Child 1 of BusinessIndex', slug='child-1')
        self.index_page.add_child(instance=self.child_1)
        self.child_2 = BusinessChild(title='Child 2 of BusinessIndex', slug='child-2')
        self.index_page.add_child(instance=self.child_2)
        self.child_3 = BusinessChild(title='Child 3 of BusinessIndex', slug='child-3')
        self.index_page.add_child(instance=self.child_3)
        self.user = self.login()