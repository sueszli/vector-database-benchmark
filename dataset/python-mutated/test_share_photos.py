from unittest import skip
from django.test import TestCase
from rest_framework.test import APIClient
from api.models import Photo
from api.tests.utils import create_test_photos, create_test_user, share_test_photos

class SharePhotosTest(TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.client = APIClient()
        self.user1 = create_test_user()
        self.user2 = create_test_user()
        self.client.force_authenticate(user=self.user1)

    def test_share_photos(self):
        if False:
            for i in range(10):
                print('nop')
        photos = create_test_photos(number_of_photos=3, owner=self.user1)
        image_hashes = [p.image_hash for p in photos]
        payload = {'image_hashes': image_hashes, 'shared': True, 'target_user_id': self.user2.id}
        headers = {'Content-Type': 'application/json'}
        response = self.client.post('/api/photosedit/share/', format='json', data=payload, headers=headers)
        data = response.json()
        self.assertTrue(data['status'])
        self.assertEqual(3, data['count'])
        shared_photos = list(Photo.shared_to.through.objects.filter(user_id=self.user2.id, photo_id__in=image_hashes))
        self.assertEqual(3, len(shared_photos))

    def test_unshare_photos(self):
        if False:
            return 10
        photos = create_test_photos(number_of_photos=3, owner=self.user1)
        image_hashes = [p.image_hash for p in photos]
        share_test_photos(image_hashes, self.user2)
        payload = {'image_hashes': image_hashes, 'shared': False, 'target_user_id': self.user2.id}
        headers = {'Content-Type': 'application/json'}
        response = self.client.post('/api/photosedit/share/', format='json', data=payload, headers=headers)
        data = response.json()
        self.assertTrue(data['status'])
        self.assertEqual(3, data['count'])
        shared_photos = list(Photo.shared_to.through.objects.filter(user_id=self.user2.id, photo_id__in=image_hashes))
        self.assertEqual(0, len(shared_photos))

    @skip('BUG!!! scenario not implemented')
    def test_share_other_user_photos(self):
        if False:
            print('Hello World!')
        photos = create_test_photos(number_of_photos=2, owner=self.user2)
        image_hashes = [p.image_hash for p in photos]
        payload = {'image_hashes': image_hashes, 'shared': True, 'target_user_id': self.user1.id}
        headers = {'Content-Type': 'application/json'}
        response = self.client.post('/api/photosedit/share/', format='json', data=payload, headers=headers)
        data = response.json()
        self.assertTrue(data['status'])
        self.assertEqual(0, data['count'])
        shared_photos = list(Photo.shared_to.through.objects.filter(user_id=self.user1.id, photo_id__in=image_hashes))
        self.assertEqual(0, len(shared_photos))