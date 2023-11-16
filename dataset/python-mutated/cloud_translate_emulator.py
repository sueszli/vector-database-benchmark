"""An emulator that mocks the core.platform.cloud_translate API. This emulator
models the Cloud Translate API.
"""
from __future__ import annotations
from core import utils

class CloudTranslateEmulator:
    """The emulator mocks the translate_text function from the Cloud Translate
    API. This emulator can be used in backend testing, or a local dev
    environment without access to the Cloud Translate API. Expected responses
    must be passed in before using this emulator for testing. See
    PREGENERATED_TRANSLATIONS below for some prepopulated responses.

    This class uses ISO 639-1 compliant language codes to specify languages.
    To learn more about ISO 639-1, see:
        https://www.w3schools.com/tags/ref_language_codes.asp
    """
    PREGENERATED_TRANSLATIONS = {(u'en', u'pt', u'hello world'): u'Olá Mundo', (u'en', u'pt', u'CONTINUE'): u'PROSSEGUIR', (u'en', u'es', u'Please continue.'): u'Por favor continua.', (u'en', u'fr', u'CONTINUE'): u'CONTINUEZ', (u'en', u'fr', u'Please continue.'): u'Continuez s&#39;il vous plaît.', (u'en', u'es', u'CONTINUE'): u'SEGUIR', (u'en', u'zh', u'hello world'): u'你好世界', (u'en', u'es', u'Correct!'): u'¡Correcto!', (u'en', u'zh', u'Correct!'): u'正确的！', (u'en', u'zh', u'CONTINUE'): u'继续', (u'en', u'zh', u'Please continue.'): u'请继续。', (u'en', u'fr', u'Correct!'): u'Correct!', (u'en', u'pt', u'Correct!'): u'Correto!', (u'en', u'es', u'hello world'): u'Hola Mundo', (u'en', u'pt', u'Please continue.'): u'Por favor continue.', (u'en', u'fr', u'hello world'): u'Bonjour le monde'}
    DEFAULT_RESPONSE = 'Default translation for emulator mode. (See core/platform/cloud_translate/cloud_translate_emulator.py for details)'

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Initializes the emulator with pregenerated responses.'
        self.expected_responses = self.PREGENERATED_TRANSLATIONS

    def translate(self, text: str, source_language_code: str, target_language_code: str) -> str:
        if False:
            while True:
                i = 10
        'Returns the saved expected response for a given input. If no\n        response exists for the given input, returns a default response.\n\n        Args:\n            text: str. The text to be translated.\n            source_language_code: str. An allowlisted language code.\n            target_language_code: str. An allowlisted language code.\n\n        Raises:\n            ValueError. Invalid source language code.\n            ValueError. Invalid target language code.\n\n        Returns:\n            str. The translated text.\n        '
        if not utils.is_valid_language_code(source_language_code):
            raise ValueError('Invalid source language code: %s' % source_language_code)
        if not utils.is_valid_language_code(target_language_code):
            raise ValueError('Invalid target language code: %s' % target_language_code)
        key = (source_language_code, target_language_code, text)
        return self.expected_responses.get(key, self.DEFAULT_RESPONSE)

    def add_expected_response(self, source_language_code: str, target_language_code: str, source_text: str, response: str) -> None:
        if False:
            i = 10
            return i + 15
        'Adds an expected response for a given set of inputs.\n\n        Args:\n            source_language_code: str. An allowlisted language code.\n            target_language_code: str. An allowlisted language code.\n            source_text: str. The text to translate.\n            response: str. The expected response for the given inputs.\n        '
        inputs = (source_language_code, target_language_code, source_text)
        self.expected_responses[inputs] = response