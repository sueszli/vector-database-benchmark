"""Provides translate_text functionality from the cloud translate emulator.
Responses are prepopulated, to add additional translations, use:
    CLIENT.add_expected_response(
        source_language_code, target_language_code, source_text, response)
See cloud_translate_emulator.py for more details"""
from __future__ import annotations
from core.platform.translate import cloud_translate_emulator
from core.platform.translate import cloud_translate_services
CLIENT = cloud_translate_emulator.CloudTranslateEmulator()

def translate_text(text: str, source_language: str, target_language: str) -> str:
    if False:
        while True:
            i = 10
    'Translates text into the target language.\n\n    For more information on ISO 639-1 see:\n        https://www.w3schools.com/tags/ref_language_codes.asp\n\n    Args:\n        text: str. The text to be translated.\n        source_language: str. An allowlisted ISO 639-1 language code.\n        target_language: str. An allowlisted ISO 639-1 language code.\n\n    Raises:\n        ValueError. Invalid source language code.\n        ValueError. Invalid target language code.\n\n    Returns:\n        str. The translated text.\n    '
    if source_language not in cloud_translate_services.LANGUAGE_CODE_ALLOWLIST:
        raise ValueError('Invalid source language code: %s' % source_language)
    if target_language not in cloud_translate_services.LANGUAGE_CODE_ALLOWLIST:
        raise ValueError('Invalid target language code: %s' % target_language)
    if source_language == target_language:
        return text
    return CLIENT.translate(text, source_language, target_language)