from django.test import TestCase
from rest_framework.test import APIClient
from api.models import User
from api.tests.utils import create_password

class SetupDirectoryTestCase(TestCase):
    userid = 0

    def setUp(self):
        if False:
            return 10
        self.client = APIClient()
        self.admin = User.objects.create_superuser('test_admin', 'test_admin@test.com', create_password())

    def test_setup_directory(self):
        if False:
            for i in range(10):
                print('nop')
        self.client.force_authenticate(user=self.admin)
        response = self.client.patch(f'/api/manage/user/{self.admin.id}/', {'scan_directory': '/code'})
        self.assertEqual(response.status_code, 200)

    def test_setup_not_existing_directory(self):
        if False:
            print('Hello World!')
        self.client.force_authenticate(user=self.admin)
        response = self.client.patch(f'/api/manage/user/{self.admin.id}/', {'scan_directory': '/non-existent-directory'})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), ['Scan directory does not exist'])