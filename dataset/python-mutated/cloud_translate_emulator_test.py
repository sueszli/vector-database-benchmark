"""Tests for cloud_translate_emulator."""
from __future__ import annotations
from core.platform.translate import cloud_translate_emulator
from core.tests import test_utils

class CloudTranslateEmulatorUnitTests(test_utils.TestBase):
    """Tests for cloud_translate_emulator."""

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.emulator = cloud_translate_emulator.CloudTranslateEmulator()

    def test_init_prepopulates_responses(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.emulator.expected_responses, self.emulator.PREGENERATED_TRANSLATIONS)

    def test_translate_with_invalid_source_language_raises_error(self) -> None:
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(ValueError, 'Invalid source language code: invalid'):
            self.emulator.translate('hello world', 'invalid', 'es')

    def test_translate_with_invalid_target_language_raises_error(self) -> None:
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(ValueError, 'Invalid target language code: invalid'):
            self.emulator.translate('hello world', 'en', 'invalid')

    def test_translate_with_valid_input_returns_expected_output(self) -> None:
        if False:
            return 10
        translated = self.emulator.translate('hello world', 'en', 'es')
        self.assertEqual('Hola Mundo', translated)

    def test_translate_without_translation_returns_default_string(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        translated = self.emulator.translate('some text', 'en', 'es')
        self.assertEqual(self.emulator.DEFAULT_RESPONSE, translated)

    def test_add_expected_response_adds_retrievable_response(self) -> None:
        if False:
            while True:
                i = 10
        self.emulator.add_expected_response('en', 'es', 'text', 'translation')
        self.assertEqual('translation', self.emulator.translate('text', 'en', 'es'))

    def test_add_expected_response_updates_existing_response(self) -> None:
        if False:
            print('Hello World!')
        self.emulator.add_expected_response('en', 'es', 'text to translate', 'fake translation unchanged')
        self.emulator.add_expected_response('en', 'es', 'text to translate', 'new fake translation')
        self.assertEqual('new fake translation', self.emulator.translate('text to translate', 'en', 'es'))