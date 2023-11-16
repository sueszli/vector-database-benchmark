from typing import Dict
from TTS.tts.utils.text.japanese.phonemizer import japanese_text_to_phonemes
from TTS.tts.utils.text.phonemizers.base import BasePhonemizer
_DEF_JA_PUNCS = '、.,[]()?!〽~『』「」【】'
_TRANS_TABLE = {'、': ','}

def trans(text):
    if False:
        i = 10
        return i + 15
    for (i, j) in _TRANS_TABLE.items():
        text = text.replace(i, j)
    return text

class JA_JP_Phonemizer(BasePhonemizer):
    """🐸TTS Ja-Jp phonemizer using functions in `TTS.tts.utils.text.japanese.phonemizer`

    TODO: someone with JA knowledge should check this implementation

    Example:

        >>> from TTS.tts.utils.text.phonemizers import JA_JP_Phonemizer
        >>> phonemizer = JA_JP_Phonemizer()
        >>> phonemizer.phonemize("どちらに行きますか？", separator="|")
        'd|o|c|h|i|r|a|n|i|i|k|i|m|a|s|u|k|a|?'

    """
    language = 'ja-jp'

    def __init__(self, punctuations=_DEF_JA_PUNCS, keep_puncs=True, **kwargs):
        if False:
            return 10
        super().__init__(self.language, punctuations=punctuations, keep_puncs=keep_puncs)

    @staticmethod
    def name():
        if False:
            while True:
                i = 10
        return 'ja_jp_phonemizer'

    def _phonemize(self, text: str, separator: str='|') -> str:
        if False:
            print('Hello World!')
        ph = japanese_text_to_phonemes(text)
        if separator is not None or separator != '':
            return separator.join(ph)
        return ph

    def phonemize(self, text: str, separator='|', language=None) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Custom phonemize for JP_JA\n\n        Skip pre-post processing steps used by the other phonemizers.\n        '
        return self._phonemize(text, separator)

    @staticmethod
    def supported_languages() -> Dict:
        if False:
            print('Hello World!')
        return {'ja-jp': 'Japanese (Japan)'}

    def version(self) -> str:
        if False:
            while True:
                i = 10
        return '0.0.1'

    def is_available(self) -> bool:
        if False:
            while True:
                i = 10
        return True