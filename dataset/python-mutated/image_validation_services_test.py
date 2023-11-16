"""Tests for the image validation service."""
from __future__ import annotations
import os
from core import feconf
from core import utils
from core.domain import image_validation_services
from core.tests import test_utils
from typing import Union

class ImageValidationServiceTests(test_utils.GenericTestBase):

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        super().setUp()
        with utils.open_file(os.path.join(feconf.TESTS_DATA_DIR, 'img.png'), 'rb', encoding=None) as f:
            self.raw_image = f.read()

    def _assert_image_validation_error(self, image: Union[str, bytes], filename: str, entity_type: str, expected_error_substring: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Checks that the image passes validation.'
        with self.assertRaisesRegex(utils.ValidationError, expected_error_substring):
            image_validation_services.validate_image_and_filename(image, filename, entity_type)

    def test_image_validation_checks(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._assert_image_validation_error(None, 'image.png', feconf.ENTITY_TYPE_EXPLORATION, 'No image supplied')
        self._assert_image_validation_error(self.raw_image, None, feconf.ENTITY_TYPE_EXPLORATION, 'No filename supplied')
        large_image = '<svg><path d="%s" /></svg>' % ('M150 0 L75 200 L225 200 Z ' * 4000)
        self._assert_image_validation_error(large_image, 'image.svg', feconf.ENTITY_TYPE_EXPLORATION, 'Image exceeds file size limit of 100 KB')
        large_image = '<svg><path d="%s" /></svg>' % ('M150 0 L75 200 L225 200 Z ' * 50000)
        self._assert_image_validation_error(large_image, 'image.svg', feconf.ENTITY_TYPE_BLOG_POST, 'Image exceeds file size limit of 1024 KB')
        invalid_svg = b'<badsvg></badsvg>'
        self._assert_image_validation_error(invalid_svg, 'image.svg', feconf.ENTITY_TYPE_EXPLORATION, 'Unsupported tags/attributes found in the SVG')
        no_xmlns_attribute_svg = invalid_svg = b'<svg></svg>'
        self._assert_image_validation_error(no_xmlns_attribute_svg, 'image.svg', feconf.ENTITY_TYPE_EXPLORATION, "The svg tag does not contains the 'xmlns' attribute.")
        self._assert_image_validation_error(b'not an image', 'image.png', feconf.ENTITY_TYPE_EXPLORATION, 'Image not recognized')
        self._assert_image_validation_error(self.raw_image, '.png', feconf.ENTITY_TYPE_EXPLORATION, 'Invalid filename')
        self._assert_image_validation_error(self.raw_image, 'image/image.png', feconf.ENTITY_TYPE_EXPLORATION, 'Filenames should not include slashes')
        self._assert_image_validation_error(self.raw_image, 'image', feconf.ENTITY_TYPE_EXPLORATION, 'Image filename with no extension')
        self._assert_image_validation_error(self.raw_image, 'image.pdf', feconf.ENTITY_TYPE_EXPLORATION, 'Expected a filename ending in .png')
        base64_encoded_string = 'SGVsbG8gV29ybGQh'
        self._assert_image_validation_error(base64_encoded_string, 'image.svg', feconf.ENTITY_TYPE_EXPLORATION, 'Image not recognized')
        xmlns_attribute_svg = '<svg xmlns="http://www.w3.org/2000/svg" ></svg>'
        base64_encoded_xmlns_attribute_svg = xmlns_attribute_svg.encode('utf-8')
        validated_image = image_validation_services.validate_image_and_filename(base64_encoded_xmlns_attribute_svg, 'image.svg', feconf.ENTITY_TYPE_EXPLORATION)
        self.assertEqual('svg', validated_image)