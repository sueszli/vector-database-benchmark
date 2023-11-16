"""Unit tests for takeout_domain.py"""
from __future__ import annotations
from core.domain import takeout_domain
from core.tests import test_utils

class TakeoutDataTests(test_utils.GenericTestBase):

    def test_that_domain_object_is_created_correctly(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        user_data = {'model_name': {'property1': 'value1', 'property2': 'value2'}}
        takeout_data = takeout_domain.TakeoutData(user_data, [])
        self.assertEqual(takeout_data.user_data, user_data)
        self.assertEqual(takeout_data.user_images, [])

class TakeoutImageTests(test_utils.GenericTestBase):

    def test_that_domain_object_is_created_correctly(self) -> None:
        if False:
            print('Hello World!')
        takeout_image_data = takeout_domain.TakeoutImage('b64_fake_image_data', '/test/')
        self.assertEqual(takeout_image_data.b64_image_data, 'b64_fake_image_data')
        self.assertEqual(takeout_image_data.image_export_path, '/test/')