"""Provides translate_text functionality from Google Cloud Translate."""
from __future__ import annotations
from core.constants import constants
from google import auth
from google.cloud import translate_v2 as translate
CLIENT = translate.Client(credentials=auth.credentials.AnonymousCredentials() if constants.EMULATOR_MODE else auth.default()[0])
LANGUAGE_CODE_ALLOWLIST = ('en', 'es', 'fr', 'zh', 'pt')

def translate_text(text: str, source_language: str, target_language: str) -> str:
    if False:
        print('Hello World!')
    'Translates text into the target language.\n\n    This method uses ISO 639-1 compliant language codes to specify languages.\n    To learn more about ISO 639-1, see:\n        https://www.w3schools.com/tags/ref_language_codes.asp\n\n    Args:\n        text: str. The text to be translated. If text contains html tags, Cloud\n            Translate only translates content between tags, leaving the tags\n            themselves untouched.\n        source_language: str. An allowlisted language code.\n        target_language: str. An allowlisted language code.\n\n    Raises:\n        ValueError. Invalid source language code.\n        ValueError. Invalid target language code.\n\n    Returns:\n        str. The translated text.\n    '
    if source_language not in LANGUAGE_CODE_ALLOWLIST:
        raise ValueError('Invalid source language code: %s' % source_language)
    if target_language not in LANGUAGE_CODE_ALLOWLIST:
        raise ValueError('Invalid target language code: %s' % target_language)
    if source_language == target_language:
        return text
    result = CLIENT.translate(text, target_language=target_language, source_language=source_language)
    assert isinstance(result, dict)
    translated_text = result['translatedText']
    return translated_text