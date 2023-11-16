from dataclasses import replace
from typing import Dict
from TTS.tts.configs.shared_configs import CharactersConfig

def parse_symbols():
    if False:
        return 10
    return {'pad': _pad, 'eos': _eos, 'bos': _bos, 'characters': _characters, 'punctuations': _punctuations, 'phonemes': _phonemes}
_pad = '<PAD>'
_eos = '<EOS>'
_bos = '<BOS>'
_blank = '<BLNK>'
_characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_punctuations = "!'(),-.:;? "
_vowels = 'iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ'
_non_pulmonic_consonants = 'ʘɓǀɗǃʄǂɠǁʛ'
_pulmonic_consonants = 'pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ'
_suprasegmentals = 'ˈˌːˑ'
_other_symbols = 'ʍwɥʜʢʡɕʑɺɧʲ'
_diacrilics = 'ɚ˞ɫ'
_phonemes = _vowels + _non_pulmonic_consonants + _pulmonic_consonants + _suprasegmentals + _other_symbols + _diacrilics

class BaseVocabulary:
    """Base Vocabulary class.

    This class only needs a vocabulary dictionary without specifying the characters.

    Args:
        vocab (Dict): A dictionary of characters and their corresponding indices.
    """

    def __init__(self, vocab: Dict, pad: str=None, blank: str=None, bos: str=None, eos: str=None):
        if False:
            print('Hello World!')
        self.vocab = vocab
        self.pad = pad
        self.blank = blank
        self.bos = bos
        self.eos = eos

    @property
    def pad_id(self) -> int:
        if False:
            return 10
        'Return the index of the padding character. If the padding character is not specified, return the length\n        of the vocabulary.'
        return self.char_to_id(self.pad) if self.pad else len(self.vocab)

    @property
    def blank_id(self) -> int:
        if False:
            i = 10
            return i + 15
        'Return the index of the blank character. If the blank character is not specified, return the length of\n        the vocabulary.'
        return self.char_to_id(self.blank) if self.blank else len(self.vocab)

    @property
    def bos_id(self) -> int:
        if False:
            while True:
                i = 10
        'Return the index of the bos character. If the bos character is not specified, return the length of the\n        vocabulary.'
        return self.char_to_id(self.bos) if self.bos else len(self.vocab)

    @property
    def eos_id(self) -> int:
        if False:
            i = 10
            return i + 15
        'Return the index of the eos character. If the eos character is not specified, return the length of the\n        vocabulary.'
        return self.char_to_id(self.eos) if self.eos else len(self.vocab)

    @property
    def vocab(self):
        if False:
            print('Hello World!')
        'Return the vocabulary dictionary.'
        return self._vocab

    @vocab.setter
    def vocab(self, vocab):
        if False:
            for i in range(10):
                print('nop')
        'Set the vocabulary dictionary and character mapping dictionaries.'
        (self._vocab, self._char_to_id, self._id_to_char) = (None, None, None)
        if vocab is not None:
            self._vocab = vocab
            self._char_to_id = {char: idx for (idx, char) in enumerate(self._vocab)}
            self._id_to_char = {idx: char for (idx, char) in enumerate(self._vocab)}

    @staticmethod
    def init_from_config(config, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Initialize from the given config.'
        if config.characters is not None and 'vocab_dict' in config.characters and config.characters.vocab_dict:
            return (BaseVocabulary(config.characters.vocab_dict, config.characters.pad, config.characters.blank, config.characters.bos, config.characters.eos), config)
        return (BaseVocabulary(**kwargs), config)

    def to_config(self) -> 'CharactersConfig':
        if False:
            while True:
                i = 10
        return CharactersConfig(vocab_dict=self._vocab, pad=self.pad, eos=self.eos, bos=self.bos, blank=self.blank, is_unique=False, is_sorted=False)

    @property
    def num_chars(self):
        if False:
            print('Hello World!')
        'Return number of tokens in the vocabulary.'
        return len(self._vocab)

    def char_to_id(self, char: str) -> int:
        if False:
            for i in range(10):
                print('nop')
        'Map a character to an token ID.'
        try:
            return self._char_to_id[char]
        except KeyError as e:
            raise KeyError(f' [!] {repr(char)} is not in the vocabulary.') from e

    def id_to_char(self, idx: int) -> str:
        if False:
            return 10
        'Map an token ID to a character.'
        return self._id_to_char[idx]

class BaseCharacters:
    """🐸BaseCharacters class

        Every new character class should inherit from this.

        Characters are oredered as follows ```[PAD, EOS, BOS, BLANK, CHARACTERS, PUNCTUATIONS]```.

        If you need a custom order, you need to define inherit from this class and override the ```_create_vocab``` method.

        Args:
            characters (str):
                Main set of characters to be used in the vocabulary.

            punctuations (str):
                Characters to be treated as punctuation.

            pad (str):
                Special padding character that would be ignored by the model.

            eos (str):
                End of the sentence character.

            bos (str):
                Beginning of the sentence character.

            blank (str):
                Optional character used between characters by some models for better prosody.

            is_unique (bool):
                Remove duplicates from the provided characters. Defaults to True.
    el
            is_sorted (bool):
                Sort the characters in alphabetical order. Only applies to `self.characters`. Defaults to True.
    """

    def __init__(self, characters: str=None, punctuations: str=None, pad: str=None, eos: str=None, bos: str=None, blank: str=None, is_unique: bool=False, is_sorted: bool=True) -> None:
        if False:
            print('Hello World!')
        self._characters = characters
        self._punctuations = punctuations
        self._pad = pad
        self._eos = eos
        self._bos = bos
        self._blank = blank
        self.is_unique = is_unique
        self.is_sorted = is_sorted
        self._create_vocab()

    @property
    def pad_id(self) -> int:
        if False:
            while True:
                i = 10
        return self.char_to_id(self.pad) if self.pad else len(self.vocab)

    @property
    def blank_id(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self.char_to_id(self.blank) if self.blank else len(self.vocab)

    @property
    def eos_id(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return self.char_to_id(self.eos) if self.eos else len(self.vocab)

    @property
    def bos_id(self) -> int:
        if False:
            print('Hello World!')
        return self.char_to_id(self.bos) if self.bos else len(self.vocab)

    @property
    def characters(self):
        if False:
            while True:
                i = 10
        return self._characters

    @characters.setter
    def characters(self, characters):
        if False:
            for i in range(10):
                print('nop')
        self._characters = characters
        self._create_vocab()

    @property
    def punctuations(self):
        if False:
            while True:
                i = 10
        return self._punctuations

    @punctuations.setter
    def punctuations(self, punctuations):
        if False:
            print('Hello World!')
        self._punctuations = punctuations
        self._create_vocab()

    @property
    def pad(self):
        if False:
            return 10
        return self._pad

    @pad.setter
    def pad(self, pad):
        if False:
            return 10
        self._pad = pad
        self._create_vocab()

    @property
    def eos(self):
        if False:
            return 10
        return self._eos

    @eos.setter
    def eos(self, eos):
        if False:
            for i in range(10):
                print('nop')
        self._eos = eos
        self._create_vocab()

    @property
    def bos(self):
        if False:
            i = 10
            return i + 15
        return self._bos

    @bos.setter
    def bos(self, bos):
        if False:
            print('Hello World!')
        self._bos = bos
        self._create_vocab()

    @property
    def blank(self):
        if False:
            print('Hello World!')
        return self._blank

    @blank.setter
    def blank(self, blank):
        if False:
            for i in range(10):
                print('nop')
        self._blank = blank
        self._create_vocab()

    @property
    def vocab(self):
        if False:
            while True:
                i = 10
        return self._vocab

    @vocab.setter
    def vocab(self, vocab):
        if False:
            for i in range(10):
                print('nop')
        self._vocab = vocab
        self._char_to_id = {char: idx for (idx, char) in enumerate(self.vocab)}
        self._id_to_char = {idx: char for (idx, char) in enumerate(self.vocab)}

    @property
    def num_chars(self):
        if False:
            i = 10
            return i + 15
        return len(self._vocab)

    def _create_vocab(self):
        if False:
            while True:
                i = 10
        _vocab = self._characters
        if self.is_unique:
            _vocab = list(set(_vocab))
        if self.is_sorted:
            _vocab = sorted(_vocab)
        _vocab = list(_vocab)
        _vocab = [self._blank] + _vocab if self._blank is not None and len(self._blank) > 0 else _vocab
        _vocab = [self._bos] + _vocab if self._bos is not None and len(self._bos) > 0 else _vocab
        _vocab = [self._eos] + _vocab if self._eos is not None and len(self._eos) > 0 else _vocab
        _vocab = [self._pad] + _vocab if self._pad is not None and len(self._pad) > 0 else _vocab
        self.vocab = _vocab + list(self._punctuations)
        if self.is_unique:
            duplicates = {x for x in self.vocab if self.vocab.count(x) > 1}
            assert len(self.vocab) == len(self._char_to_id) == len(self._id_to_char), f' [!] There are duplicate characters in the character set. {duplicates}'

    def char_to_id(self, char: str) -> int:
        if False:
            i = 10
            return i + 15
        try:
            return self._char_to_id[char]
        except KeyError as e:
            raise KeyError(f' [!] {repr(char)} is not in the vocabulary.') from e

    def id_to_char(self, idx: int) -> str:
        if False:
            while True:
                i = 10
        return self._id_to_char[idx]

    def print_log(self, level: int=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Prints the vocabulary in a nice format.\n        '
        indent = '\t' * level
        print(f'{indent}| > Characters: {self._characters}')
        print(f'{indent}| > Punctuations: {self._punctuations}')
        print(f'{indent}| > Pad: {self._pad}')
        print(f'{indent}| > EOS: {self._eos}')
        print(f'{indent}| > BOS: {self._bos}')
        print(f'{indent}| > Blank: {self._blank}')
        print(f'{indent}| > Vocab: {self.vocab}')
        print(f'{indent}| > Num chars: {self.num_chars}')

    @staticmethod
    def init_from_config(config: 'Coqpit'):
        if False:
            while True:
                i = 10
        'Init your character class from a config.\n\n        Implement this method for your subclass.\n        '
        if config.characters is not None:
            return (BaseCharacters(**config.characters), config)
        characters = BaseCharacters()
        new_config = replace(config, characters=characters.to_config())
        return (characters, new_config)

    def to_config(self) -> 'CharactersConfig':
        if False:
            for i in range(10):
                print('nop')
        return CharactersConfig(characters=self._characters, punctuations=self._punctuations, pad=self._pad, eos=self._eos, bos=self._bos, blank=self._blank, is_unique=self.is_unique, is_sorted=self.is_sorted)

class IPAPhonemes(BaseCharacters):
    """🐸IPAPhonemes class to manage `TTS.tts` model vocabulary

    Intended to be used with models using IPAPhonemes as input.
    It uses system defaults for the undefined class arguments.

    Args:
        characters (str):
            Main set of case-sensitive characters to be used in the vocabulary. Defaults to `_phonemes`.

        punctuations (str):
            Characters to be treated as punctuation. Defaults to `_punctuations`.

        pad (str):
            Special padding character that would be ignored by the model. Defaults to `_pad`.

        eos (str):
            End of the sentence character. Defaults to `_eos`.

        bos (str):
            Beginning of the sentence character. Defaults to `_bos`.

        blank (str):
            Optional character used between characters by some models for better prosody. Defaults to `_blank`.

        is_unique (bool):
            Remove duplicates from the provided characters. Defaults to True.

        is_sorted (bool):
            Sort the characters in alphabetical order. Defaults to True.
    """

    def __init__(self, characters: str=_phonemes, punctuations: str=_punctuations, pad: str=_pad, eos: str=_eos, bos: str=_bos, blank: str=_blank, is_unique: bool=False, is_sorted: bool=True) -> None:
        if False:
            i = 10
            return i + 15
        super().__init__(characters, punctuations, pad, eos, bos, blank, is_unique, is_sorted)

    @staticmethod
    def init_from_config(config: 'Coqpit'):
        if False:
            print('Hello World!')
        'Init a IPAPhonemes object from a model config\n\n        If characters are not defined in the config, it will be set to the default characters and the config\n        will be updated.\n        '
        if 'characters' in config and config.characters is not None:
            if 'phonemes' in config.characters and config.characters.phonemes is not None:
                config.characters['characters'] = config.characters['phonemes']
            return (IPAPhonemes(characters=config.characters['characters'], punctuations=config.characters['punctuations'], pad=config.characters['pad'], eos=config.characters['eos'], bos=config.characters['bos'], blank=config.characters['blank'], is_unique=config.characters['is_unique'], is_sorted=config.characters['is_sorted']), config)
        if config.characters is not None:
            return (IPAPhonemes(**config.characters), config)
        characters = IPAPhonemes()
        new_config = replace(config, characters=characters.to_config())
        return (characters, new_config)

class Graphemes(BaseCharacters):
    """🐸Graphemes class to manage `TTS.tts` model vocabulary

    Intended to be used with models using graphemes as input.
    It uses system defaults for the undefined class arguments.

    Args:
        characters (str):
            Main set of case-sensitive characters to be used in the vocabulary. Defaults to `_characters`.

        punctuations (str):
            Characters to be treated as punctuation. Defaults to `_punctuations`.

        pad (str):
            Special padding character that would be ignored by the model. Defaults to `_pad`.

        eos (str):
            End of the sentence character. Defaults to `_eos`.

        bos (str):
            Beginning of the sentence character. Defaults to `_bos`.

        is_unique (bool):
            Remove duplicates from the provided characters. Defaults to True.

        is_sorted (bool):
            Sort the characters in alphabetical order. Defaults to True.
    """

    def __init__(self, characters: str=_characters, punctuations: str=_punctuations, pad: str=_pad, eos: str=_eos, bos: str=_bos, blank: str=_blank, is_unique: bool=False, is_sorted: bool=True) -> None:
        if False:
            return 10
        super().__init__(characters, punctuations, pad, eos, bos, blank, is_unique, is_sorted)

    @staticmethod
    def init_from_config(config: 'Coqpit'):
        if False:
            for i in range(10):
                print('nop')
        'Init a Graphemes object from a model config\n\n        If characters are not defined in the config, it will be set to the default characters and the config\n        will be updated.\n        '
        if config.characters is not None:
            if 'phonemes' in config.characters:
                return (Graphemes(characters=config.characters['characters'], punctuations=config.characters['punctuations'], pad=config.characters['pad'], eos=config.characters['eos'], bos=config.characters['bos'], blank=config.characters['blank'], is_unique=config.characters['is_unique'], is_sorted=config.characters['is_sorted']), config)
            return (Graphemes(**config.characters), config)
        characters = Graphemes()
        new_config = replace(config, characters=characters.to_config())
        return (characters, new_config)
if __name__ == '__main__':
    gr = Graphemes()
    ph = IPAPhonemes()
    gr.print_log()
    ph.print_log()