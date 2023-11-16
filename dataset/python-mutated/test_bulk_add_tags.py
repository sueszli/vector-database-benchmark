from django.contrib.auth.models import Permission
from django.test import TestCase
from django.urls import reverse
from wagtail.images import get_image_model
from wagtail.images.tests.utils import get_test_image_file
from wagtail.test.utils import WagtailTestUtils
Image = get_image_model()
test_file = get_test_image_file()

def get_tag_list(image):
    if False:
        print('Hello World!')
    return [tag.name for tag in image.tags.all()]

class TestBulkAddTags(WagtailTestUtils, TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.user = self.login()
        self.new_tags = ['first', 'second']
        self.images = [Image.objects.create(title=f'Test image - {i}', file=test_file) for i in range(1, 6)]
        self.url = reverse('wagtail_bulk_action', args=('wagtailimages', 'image', 'add_tags')) + '?'
        for image in self.images:
            self.url += f'id={image.id}&'
        self.post_data = {'tags': ','.join(self.new_tags)}

    def test_add_tags_with_limited_permissions(self):
        if False:
            for i in range(10):
                print('nop')
        self.user.is_superuser = False
        self.user.user_permissions.add(Permission.objects.get(content_type__app_label='wagtailadmin', codename='access_admin'))
        self.user.save()
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        html = response.content.decode()
        self.assertInHTML("<p>You don't have permission to add tags to these images</p>", html)
        for image in self.images:
            self.assertInHTML(f'<li>{image.title}</li>', html)
        self.client.post(self.url, self.post_data)
        for image in self.images:
            self.assertCountEqual(get_tag_list(Image.objects.get(id=image.id)), [])

    def test_simple(self):
        if False:
            while True:
                i = 10
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'wagtailimages/bulk_actions/confirm_bulk_add_tags.html')

    def test_add_tags(self):
        if False:
            return 10
        response = self.client.post(self.url, self.post_data)
        self.assertEqual(response.status_code, 302)
        for image in self.images:
            self.assertCountEqual(get_tag_list(Image.objects.get(id=image.id)), self.new_tags)