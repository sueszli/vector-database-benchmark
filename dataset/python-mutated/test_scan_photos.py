import os
from unittest import skip
from django.test import TestCase
from pyfakefs.fake_filesystem_unittest import Patcher
from rest_framework.test import APIClient
from api.tests.utils import create_test_user
test_images_path = os.path.join(os.path.dirname(__file__), 'test_images')

@skip
class ScanPhotosTestCase(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        with Patcher() as patcher:
            samplephotos_dir = '/data/samplephotos'
            patcher.fs.add_real_directory('/usr/local/lib/python3.11/dist-packages/')
            patcher.fs.add_real_directory(test_images_path, target_path=samplephotos_dir)
            self.client_admin = APIClient()
            self.admin = create_test_user(is_admin=True)
            self.client_admin.force_authenticate(self.admin)
            response = self.client_admin.patch('/api/manage/user/{}/'.format(self.admin.id), {'scan_directory': samplephotos_dir})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['scan_directory'], samplephotos_dir)
            scan_photos_res = self.client_admin.get('/api/scanphotos/')
            self.assertEqual(scan_photos_res.status_code, 200)
            get_photos_res = self.client_admin.get('/api/photos/')
            self.assertEqual(get_photos_res.status_code, 200)
            self.assertTrue(len(get_photos_res.json()['results']) > 0)
            num_photos = len(get_photos_res.json()['results'])
            scan_photos_res = self.client_admin.get('/api/scanphotos/')
            self.assertEqual(scan_photos_res.status_code, 200)
            get_photos_res = self.client_admin.get('/api/photos/')
            self.assertEqual(get_photos_res.status_code, 200)
            self.assertEqual(len(get_photos_res.json()['results']), num_photos)

    def test_setup(self):
        if False:
            while True:
                i = 10
        'make sure setup works'
        pass

    @skip
    def test_auto_albums(self):
        if False:
            i = 10
            return i + 15
        'make sure user can make auto albums, list and retrieve them'
        auto_album_gen_res = self.client_admin.get('/api/autoalbumgen/')
        self.assertEqual(auto_album_gen_res.status_code, 200)
        auto_album_list_res = self.client_admin.get('/api/albums/auto/list/')
        self.assertEqual(auto_album_list_res.status_code, 200)
        for album in auto_album_list_res.json()['results']:
            auto_album_retrieve_res = self.client_admin.get('/api/albums/auto/%d/' % album['id'])
            self.assertEqual(auto_album_retrieve_res.status_code, 200)
            self.assertTrue(len(auto_album_retrieve_res.json()['photos']) > 0)
        num_auto_albums = len(auto_album_list_res.json()['results'])
        auto_album_gen_res = self.client_admin.get('/api/autoalbumgen/')
        self.assertEqual(auto_album_gen_res.status_code, 200)
        auto_album_list_res = self.client_admin.get('/api/albums/auto/list/')
        self.assertEqual(len(auto_album_list_res.json()['results']), num_auto_albums)

    @skip
    def test_place_albums(self):
        if False:
            return 10
        'make sure user can list and retrieve place albums'
        place_album_list_res = self.client_admin.get('/api/albums/place/list/')
        self.assertEqual(place_album_list_res.status_code, 200)
        for album in place_album_list_res.json()['results']:
            place_album_retrieve_res = self.client_admin.get('/api/albums/place/%d/' % album['id'])
            self.assertEqual(place_album_retrieve_res.status_code, 200)

    @skip
    def test_thing_albums(self):
        if False:
            while True:
                i = 10
        'make sure user can list and retrieve thing albums'
        thing_album_list_res = self.client_admin.get('/api/albums/thing/list/')
        self.assertEqual(thing_album_list_res.status_code, 200)
        for album in thing_album_list_res.json()['results']:
            thing_album_retrieve_res = self.client_admin.get('/api/albums/thing/%d/' % album['id'])
            self.assertEqual(thing_album_retrieve_res.status_code, 200)