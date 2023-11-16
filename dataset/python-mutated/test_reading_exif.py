import os
from django.test import TestCase
from django.utils import timezone
from faker import Faker
from rest_framework.test import APIClient
from api.models import File, Person, Photo
from api.tests.utils import create_test_user

class ReadFacesFromPhotosTest(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.client = APIClient()
        self.user1 = create_test_user(favorite_min_rating=1)
        self.client.force_authenticate(user=self.user1)

    def test_reading_from_photo(self):
        if False:
            while True:
                i = 10
        file = os.path.dirname(os.path.abspath(__file__)) + '/fixtures/niaz.jpg'
        exif_file = os.path.dirname(os.path.abspath(__file__)) + '/fixtures/niaz.xmp'
        fake = Faker()
        pk = fake.md5()
        os.system('cp ' + file + ' ' + '/tmp/' + str(pk) + '.jpg')
        os.system('cp ' + exif_file + ' ' + '/tmp/' + str(pk) + '.xmp')
        os.system('cp ' + file + ' ' + '/protected_media/thumbnails_big/' + str(pk) + '.jpg')
        photo = Photo(pk=pk, image_hash=pk, aspect_ratio=1, owner=self.user1)
        fileObject = File.create('/tmp/' + str(photo.pk) + '.jpg', self.user1)
        photo.main_file = fileObject
        photo.added_on = timezone.now()
        photo.thumbnail_big = '/protected_media/thumbnails_big/' + str(photo.pk) + '.jpg'
        photo.save()
        photo._extract_faces()
        self.assertEqual(1, len(photo.faces.all()))
        self.assertEqual(3, len(Person.objects.all()))
        self.assertIsNotNone(photo.faces.all()[0].encoding)
        self.assertEqual('Niaz Faridani-Rad', Person.objects.filter(name='Niaz Faridani-Rad').first().name)