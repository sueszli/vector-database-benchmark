"""
The mycroft.util.lang module provides the main interface for setting up the
lingua-franca (https://github.com/mycroftai/lingua-franca) selected language
"""
from lingua_franca import set_default_lang as _set_default_lf_lang

def set_default_lf_lang(lang_code='en-us'):
    if False:
        return 10
    'Set the default language of Lingua Franca for parsing and formatting.\n\n    Note: this is a temporary method until a global set_default_lang() method\n    can be implemented that updates all Mycroft systems eg STT and TTS.\n    It will be deprecated at the earliest possible point.\n\n    Args:\n        lang (str): BCP-47 language code, e.g. "en-us" or "es-mx"\n    '
    return _set_default_lf_lang(lang_code=lang_code)