"""Tests for cloud_translate_services."""
from __future__ import annotations
from core.platform.translate import cloud_translate_services
from core.tests import test_utils

class CloudTranslateServicesUnitTests(test_utils.TestBase):
    """Tests for cloud_translate_services."""

    def test_translate_text_with_invalid_source_language_raises_error(self) -> None:
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(ValueError, 'Invalid source language code: hi'):
            cloud_translate_services.translate_text('hello world', 'hi', 'es')

    def test_translate_text_with_invalid_target_language_raises_error(self) -> None:
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(ValueError, 'Invalid target language code: hi'):
            cloud_translate_services.translate_text('hello world', 'en', 'hi')

    def test_translate_text_with_same_source_target_language_doesnt_call_api(self) -> None:
        if False:
            print('Hello World!')
        with self.swap_to_always_raise(cloud_translate_services.CLIENT, 'translate', error=AssertionError):
            translated_text = cloud_translate_services.translate_text('hello world', 'en', 'en')
            self.assertEqual(translated_text, 'hello world')

    def test_translate_text_with_valid_input_calls_translate_api(self) -> None:
        if False:
            return 10
        with self.swap_to_always_return(cloud_translate_services.CLIENT, 'translate', value={'translatedText': 'hola mundo'}):
            translated_text = cloud_translate_services.translate_text('hello world', 'en', 'es')
            self.assertEqual(translated_text, 'hola mundo')