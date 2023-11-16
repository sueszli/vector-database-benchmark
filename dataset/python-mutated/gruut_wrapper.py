import importlib
from typing import List
import gruut
from gruut_ipa import IPA
from TTS.tts.utils.text.phonemizers.base import BasePhonemizer
from TTS.tts.utils.text.punctuation import Punctuation
GRUUT_TRANS_TABLE = str.maketrans('g', 'ɡ')

class Gruut(BasePhonemizer):
    """Gruut wrapper for G2P

    Args:
        language (str):
            Valid language code for the used backend.

        punctuations (str):
            Characters to be treated as punctuation. Defaults to `Punctuation.default_puncs()`.

        keep_puncs (bool):
            If true, keep the punctuations after phonemization. Defaults to True.

        use_espeak_phonemes (bool):
            If true, use espeak lexicons instead of default Gruut lexicons. Defaults to False.

        keep_stress (bool):
            If true, keep the stress characters after phonemization. Defaults to False.

    Example:

        >>> from TTS.tts.utils.text.phonemizers.gruut_wrapper import Gruut
        >>> phonemizer = Gruut('en-us')
        >>> phonemizer.phonemize("Be a voice, not an! echo?", separator="|")
        'b|i| ə| v|ɔ|ɪ|s, n|ɑ|t| ə|n! ɛ|k|o|ʊ?'
    """

    def __init__(self, language: str, punctuations=Punctuation.default_puncs(), keep_puncs=True, use_espeak_phonemes=False, keep_stress=False):
        if False:
            while True:
                i = 10
        super().__init__(language, punctuations=punctuations, keep_puncs=keep_puncs)
        self.use_espeak_phonemes = use_espeak_phonemes
        self.keep_stress = keep_stress

    @staticmethod
    def name():
        if False:
            return 10
        return 'gruut'

    def phonemize_gruut(self, text: str, separator: str='|', tie=False) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Convert input text to phonemes.\n\n        Gruut phonemizes the given `str` by seperating each phoneme character with `separator`, even for characters\n        that constitude a single sound.\n\n        It doesn\'t affect 🐸TTS since it individually converts each character to token IDs.\n\n        Examples::\n            "hello how are you today?" -> `h|ɛ|l|o|ʊ| h|a|ʊ| ɑ|ɹ| j|u| t|ə|d|e|ɪ`\n\n        Args:\n            text (str):\n                Text to be converted to phonemes.\n\n            tie (bool, optional) : When True use a \'͡\' character between\n                consecutive characters of a single phoneme. Else separate phoneme\n                with \'_\'. This option requires espeak>=1.49. Default to False.\n        '
        ph_list = []
        for sentence in gruut.sentences(text, lang=self.language, espeak=self.use_espeak_phonemes):
            for word in sentence:
                if word.is_break:
                    if ph_list:
                        ph_list[-1].append(word.text)
                    else:
                        ph_list.append([word.text])
                elif word.phonemes:
                    word_phonemes = []
                    for word_phoneme in word.phonemes:
                        if not self.keep_stress:
                            word_phoneme = IPA.without_stress(word_phoneme)
                        word_phoneme = word_phoneme.translate(GRUUT_TRANS_TABLE)
                        if word_phoneme:
                            word_phonemes.extend(word_phoneme)
                    if word_phonemes:
                        ph_list.append(word_phonemes)
        ph_words = [separator.join(word_phonemes) for word_phonemes in ph_list]
        ph = f'{separator} '.join(ph_words)
        return ph

    def _phonemize(self, text, separator):
        if False:
            i = 10
            return i + 15
        return self.phonemize_gruut(text, separator, tie=False)

    def is_supported_language(self, language):
        if False:
            i = 10
            return i + 15
        'Returns True if `language` is supported by the backend'
        return gruut.is_language_supported(language)

    @staticmethod
    def supported_languages() -> List:
        if False:
            for i in range(10):
                print('nop')
        'Get a dictionary of supported languages.\n\n        Returns:\n            List: List of language codes.\n        '
        return list(gruut.get_supported_languages())

    def version(self):
        if False:
            i = 10
            return i + 15
        'Get the version of the used backend.\n\n        Returns:\n            str: Version of the used backend.\n        '
        return gruut.__version__

    @classmethod
    def is_available(cls):
        if False:
            while True:
                i = 10
        'Return true if ESpeak is available else false'
        return importlib.util.find_spec('gruut') is not None
if __name__ == '__main__':
    e = Gruut(language='en-us')
    print(e.supported_languages())
    print(e.version())
    print(e.language)
    print(e.name())
    print(e.is_available())
    e = Gruut(language='en-us', keep_puncs=False)
    print('`' + e.phonemize('hello how are you today?') + '`')
    e = Gruut(language='en-us', keep_puncs=True)
    print('`' + e.phonemize('hello how, are you today?') + '`')