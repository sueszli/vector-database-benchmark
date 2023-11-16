from django.test import TestCase
from django.urls import reverse
from wagtail.models import Page
from wagtail.test.testapp.models import FormPageWithRedirect, PageChooserModel, SimplePage
from wagtail.test.utils import WagtailTestUtils

class TestPageUsage(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.user = self.login()
        self.root_page = Page.objects.get(id=2)
        page = SimplePage(title='Hello world!', slug='hello-world', content='hello')
        self.root_page.add_child(instance=page)
        page.save_revision().publish()
        self.page = SimplePage.objects.get(id=page.id)

    def test_no_usage(self):
        if False:
            return 10
        usage_url = reverse('wagtailadmin_pages:usage', args=(self.page.id,))
        response = self.client.get(usage_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/generic/listing.html')
        self.assertContains(response, 'Usage of')
        self.assertContains(response, 'Hello world!')

    def test_has_private_usage(self):
        if False:
            for i in range(10):
                print('nop')
        PageChooserModel.objects.create(page=self.page)
        usage_url = reverse('wagtailadmin_pages:usage', args=(self.page.id,))
        response = self.client.get(usage_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/generic/listing.html')
        self.assertContains(response, 'Usage of')
        self.assertContains(response, 'Hello world!')
        self.assertContains(response, '(Private page chooser model)')
        self.assertContains(response, '<td>Page chooser model</td>', html=True)

    def test_has_editable_usage(self):
        if False:
            return 10
        form_page = FormPageWithRedirect(title='Contact us', slug='contact-us', to_address='to@email.com', from_address='from@email.com', subject='The subject', thank_you_redirect_page=self.page)
        form_page = self.root_page.add_child(instance=form_page)
        usage_url = reverse('wagtailadmin_pages:usage', args=(self.page.id,))
        response = self.client.get(usage_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailadmin/generic/listing.html')
        self.assertContains(response, 'Usage of')
        self.assertContains(response, 'Hello world!')
        self.assertContains(response, 'Contact us')
        self.assertContains(response, reverse('wagtailadmin_pages:edit', args=(form_page.id,)))
        self.assertContains(response, 'Thank you redirect page')
        self.assertContains(response, '<td>Form page with redirect</td>', html=True)