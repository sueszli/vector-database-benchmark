"""Music notation utilities"""
import re
import numpy as np
from numba import jit
from collections import Counter
from .intervals import INTERVALS
from .._cache import cache
from ..util.exceptions import ParameterError
from typing import Dict, List, Iterable, Union, overload
from ..util.decorators import vectorize
from .._typing import _ScalarOrSequence, _FloatLike_co, _SequenceLike, _IterableLike
__all__ = ['key_to_degrees', 'key_to_notes', 'mela_to_degrees', 'mela_to_svara', 'thaat_to_degrees', 'list_mela', 'list_thaat', 'fifths_to_note', 'interval_to_fjs']
THAAT_MAP = dict(bilaval=[0, 2, 4, 5, 7, 9, 11], khamaj=[0, 2, 4, 5, 7, 9, 10], kafi=[0, 2, 3, 5, 7, 9, 10], asavari=[0, 2, 3, 5, 7, 8, 10], bhairavi=[0, 1, 3, 5, 7, 8, 10], kalyan=[0, 2, 4, 6, 7, 9, 11], marva=[0, 1, 4, 6, 7, 9, 11], poorvi=[0, 1, 4, 6, 7, 8, 11], todi=[0, 1, 3, 6, 7, 8, 11], bhairav=[0, 1, 4, 5, 7, 8, 11])
MELAKARTA_MAP = {k: i for (i, k) in enumerate(['kanakangi', 'ratnangi', 'ganamurthi', 'vanaspathi', 'manavathi', 'tanarupi', 'senavathi', 'hanumathodi', 'dhenuka', 'natakapriya', 'kokilapriya', 'rupavathi', 'gayakapriya', 'vakulabharanam', 'mayamalavagaula', 'chakravakom', 'suryakantham', 'hatakambari', 'jhankaradhwani', 'natabhairavi', 'keeravani', 'kharaharapriya', 'gaurimanohari', 'varunapriya', 'mararanjini', 'charukesi', 'sarasangi', 'harikambhoji', 'dheerasankarabharanam', 'naganandini', 'yagapriya', 'ragavardhini', 'gangeyabhushani', 'vagadheeswari', 'sulini', 'chalanatta', 'salagam', 'jalarnavam', 'jhalavarali', 'navaneetham', 'pavani', 'raghupriya', 'gavambodhi', 'bhavapriya', 'subhapanthuvarali', 'shadvidhamargini', 'suvarnangi', 'divyamani', 'dhavalambari', 'namanarayani', 'kamavardhini', 'ramapriya', 'gamanasrama', 'viswambhari', 'syamalangi', 'shanmukhapriya', 'simhendramadhyamam', 'hemavathi', 'dharmavathi', 'neethimathi', 'kanthamani', 'rishabhapriya', 'latangi', 'vachaspathi', 'mechakalyani', 'chitrambari', 'sucharitra', 'jyotisvarupini', 'dhatuvardhini', 'nasikabhushani', 'kosalam', 'rasikapriya'], 1)}
KEY_RE = re.compile('^(?P<tonic>[A-Ga-g])(?P<accidental>[#♯𝄪b!♭𝄫♮n]*):((?P<scale>(maj|min)(or)?)|(?P<mode>(((ion|dor|phryg|lyd|mixolyd|aeol|locr)(ian)?)|phr|mix|aeo|loc)))$')
NOTE_RE = re.compile('^(?P<note>[A-Ga-g])(?P<accidental>[#♯𝄪b!♭𝄫♮n]*)(?P<octave>[+-]?\\d+)?(?P<cents>[+-]\\d+)?$')
MAJOR_DICT = {'ion': {'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G', 'A': 'A', 'B': 'B'}, 'dor': {'C': 'B♭', 'D': 'C', 'E': 'D', 'F': 'E♭', 'G': 'F', 'A': 'G', 'B': 'A'}, 'phr': {'C': 'A♭', 'D': 'B♭', 'E': 'C', 'F': 'D♭', 'G': 'E♭', 'A': 'F', 'B': 'G'}, 'lyd': {'C': 'G', 'D': 'A', 'E': 'B', 'F': 'C', 'G': 'D', 'A': 'E', 'B': 'F♯'}, 'mix': {'C': 'F', 'D': 'G', 'E': 'A', 'F': 'B♭', 'G': 'C', 'A': 'D', 'B': 'E'}, 'aeo': {'C': 'E♭', 'D': 'F', 'E': 'G', 'F': 'A♭', 'G': 'B♭', 'A': 'C', 'B': 'D'}, 'loc': {'C': 'D♭', 'D': 'E♭', 'E': 'F', 'F': 'G♭', 'G': 'A♭', 'A': 'B♭', 'B': 'C'}}
OFFSET_DICT = {'ion': 0, 'dor': 1, 'phr': 2, 'lyd': 3, 'mix': 4, 'aeo': 5, 'loc': 6}
ACC_MAP = {'#': 1, '♮': 0, '': 0, 'n': 0, 'b': -1, '!': -1, '♯': 1, '♭': -1, '𝄪': 2, '𝄫': -2}

def thaat_to_degrees(thaat: str) -> np.ndarray:
    if False:
        print('Hello World!')
    "Construct the svara indices (degrees) for a given thaat\n\n    Parameters\n    ----------\n    thaat : str\n        The name of the thaat\n\n    Returns\n    -------\n    indices : np.ndarray\n        A list of the seven svara indices (starting from 0=Sa)\n        contained in the specified thaat\n\n    See Also\n    --------\n    key_to_degrees\n    mela_to_degrees\n    list_thaat\n\n    Examples\n    --------\n    >>> librosa.thaat_to_degrees('bilaval')\n    array([ 0,  2,  4,  5,  7,  9, 11])\n\n    >>> librosa.thaat_to_degrees('todi')\n    array([ 0,  1,  3,  6,  7,  8, 11])\n    "
    return np.asarray(THAAT_MAP[thaat.lower()])

def mela_to_degrees(mela: Union[str, int]) -> np.ndarray:
    if False:
        print('Hello World!')
    "Construct the svara indices (degrees) for a given melakarta raga\n\n    Parameters\n    ----------\n    mela : str or int\n        Either the name or integer index ([1, 2, ..., 72]) of the melakarta raga\n\n    Returns\n    -------\n    degrees : np.ndarray\n        A list of the seven svara indices (starting from 0=Sa)\n        contained in the specified raga\n\n    See Also\n    --------\n    thaat_to_degrees\n    key_to_degrees\n    list_mela\n\n    Examples\n    --------\n    Melakarta #1 (kanakangi):\n\n    >>> librosa.mela_to_degrees(1)\n    array([0, 1, 2, 5, 7, 8, 9])\n\n    Or using a name directly:\n\n    >>> librosa.mela_to_degrees('kanakangi')\n    array([0, 1, 2, 5, 7, 8, 9])\n    "
    if isinstance(mela, str):
        index = MELAKARTA_MAP[mela.lower()] - 1
    elif 0 < mela <= 72:
        index = mela - 1
    else:
        raise ParameterError(f'mela={mela} must be in range [1, 72]')
    degrees = [0]
    lower = index % 36
    if 0 <= lower < 6:
        degrees.extend([1, 2])
    elif 6 <= lower < 12:
        degrees.extend([1, 3])
    elif 12 <= lower < 18:
        degrees.extend([1, 4])
    elif 18 <= lower < 24:
        degrees.extend([2, 3])
    elif 24 <= lower < 30:
        degrees.extend([2, 4])
    else:
        degrees.extend([3, 4])
    if index < 36:
        degrees.append(5)
    else:
        degrees.append(6)
    degrees.append(7)
    upper = index % 6
    if upper == 0:
        degrees.extend([8, 9])
    elif upper == 1:
        degrees.extend([8, 10])
    elif upper == 2:
        degrees.extend([8, 11])
    elif upper == 3:
        degrees.extend([9, 10])
    elif upper == 4:
        degrees.extend([9, 11])
    else:
        degrees.extend([10, 11])
    return np.array(degrees)

@cache(level=10)
def mela_to_svara(mela: Union[str, int], *, abbr: bool=True, unicode: bool=True) -> List[str]:
    if False:
        return 10
    "Spell the Carnatic svara names for a given melakarta raga\n\n    This function exists to resolve enharmonic equivalences between\n    pitch classes:\n\n        - Ri2 / Ga1\n        - Ri3 / Ga2\n        - Dha2 / Ni1\n        - Dha3 / Ni2\n\n    For svara outside the raga, names are chosen to preserve orderings\n    so that all Ri precede all Ga, and all Dha precede all Ni.\n\n    Parameters\n    ----------\n    mela : str or int\n        the name or numerical index of the melakarta raga\n\n    abbr : bool\n        If `True`, use single-letter svara names: S, R, G, ...\n\n        If `False`, use full names: Sa, Ri, Ga, ...\n\n    unicode : bool\n        If `True`, use unicode symbols for numberings, e.g., Ri₁\n\n        If `False`, use low-order ASCII, e.g., Ri1.\n\n    Returns\n    -------\n    svara : list of strings\n\n        The svara names for each of the 12 pitch classes.\n\n    See Also\n    --------\n    key_to_notes\n    mela_to_degrees\n    list_mela\n\n    Examples\n    --------\n    Melakarta #1 (Kanakangi) uses R1, G1, D1, N1\n\n    >>> librosa.mela_to_svara(1)\n    ['S', 'R₁', 'G₁', 'G₂', 'G₃', 'M₁', 'M₂', 'P', 'D₁', 'N₁', 'N₂', 'N₃']\n\n    #19 (Jhankaradhwani) uses R2 and G2 so the third svara are Ri:\n\n    >>> librosa.mela_to_svara(19)\n    ['S', 'R₁', 'R₂', 'G₂', 'G₃', 'M₁', 'M₂', 'P', 'D₁', 'N₁', 'N₂', 'N₃']\n\n    #31 (Yagapriya) uses R3 and G3, so third and fourth svara are Ri:\n\n    >>> librosa.mela_to_svara(31)\n    ['S', 'R₁', 'R₂', 'R₃', 'G₃', 'M₁', 'M₂', 'P', 'D₁', 'N₁', 'N₂', 'N₃']\n\n    #34 (Vagadheeswari) uses D2 and N2, so Ni1 becomes Dha2:\n\n    >>> librosa.mela_to_svara(34)\n    ['S', 'R₁', 'R₂', 'R₃', 'G₃', 'M₁', 'M₂', 'P', 'D₁', 'D₂', 'N₂', 'N₃']\n\n    #36 (Chalanatta) uses D3 and N3, so Ni2 becomes Dha3:\n\n    >>> librosa.mela_to_svara(36)\n    ['S', 'R₁', 'R₂', 'R₃', 'G₃', 'M₁', 'M₂', 'P', 'D₁', 'D₂', 'D₃', 'N₃']\n\n    # You can also query by raga name instead of index:\n\n    >>> librosa.mela_to_svara('chalanatta')\n    ['S', 'R₁', 'R₂', 'R₃', 'G₃', 'M₁', 'M₂', 'P', 'D₁', 'D₂', 'D₃', 'N₃']\n    "
    svara_map = ['Sa', 'Ri₁', '', '', 'Ga₃', 'Ma₁', 'Ma₂', 'Pa', 'Dha₁', '', '', 'Ni₃']
    if isinstance(mela, str):
        mela_idx = MELAKARTA_MAP[mela.lower()] - 1
    elif 0 < mela <= 72:
        mela_idx = mela - 1
    else:
        raise ParameterError(f'mela={mela} must be in range [1, 72]')
    lower = mela_idx % 36
    if lower < 6:
        svara_map[2] = 'Ga₁'
    else:
        svara_map[2] = 'Ri₂'
    if lower < 30:
        svara_map[3] = 'Ga₂'
    else:
        svara_map[3] = 'Ri₃'
    upper = mela_idx % 6
    if upper == 0:
        svara_map[9] = 'Ni₁'
    else:
        svara_map[9] = 'Dha₂'
    if upper == 5:
        svara_map[10] = 'Dha₃'
    else:
        svara_map[10] = 'Ni₂'
    if abbr:
        t_abbr = str.maketrans({'a': '', 'h': '', 'i': ''})
        svara_map = [s.translate(t_abbr) for s in svara_map]
    if not unicode:
        t_uni = str.maketrans({'₁': '1', '₂': '2', '₃': '3'})
        svara_map = [s.translate(t_uni) for s in svara_map]
    return list(svara_map)

def list_mela() -> Dict[str, int]:
    if False:
        return 10
    "List melakarta ragas by name and index.\n\n    Melakarta raga names are transcribed from [#]_, with the exception of #45\n    (subhapanthuvarali).\n\n    .. [#] Bhagyalekshmy, S. (1990).\n        Ragas in Carnatic music.\n        South Asia Books.\n\n    Returns\n    -------\n    mela_map : dict\n        A dictionary mapping melakarta raga names to indices (1, 2, ..., 72)\n\n    Examples\n    --------\n    >>> librosa.list_mela()\n    {'kanakangi': 1,\n     'ratnangi': 2,\n     'ganamurthi': 3,\n     'vanaspathi': 4,\n     ...}\n\n    See Also\n    --------\n    mela_to_degrees\n    mela_to_svara\n    list_thaat\n    "
    return MELAKARTA_MAP.copy()

def list_thaat() -> List[str]:
    if False:
        return 10
    "List supported thaats by name.\n\n    Returns\n    -------\n    thaats : list\n        A list of supported thaats\n\n    Examples\n    --------\n    >>> librosa.list_thaat()\n    ['bilaval',\n     'khamaj',\n     'kafi',\n     'asavari',\n     'bhairavi',\n     'kalyan',\n     'marva',\n     'poorvi',\n     'todi',\n     'bhairav']\n\n    See Also\n    --------\n    list_mela\n    thaat_to_degrees\n    "
    return list(THAAT_MAP.keys())

@overload
def __note_to_degree(key: str) -> int:
    if False:
        return 10
    ...

@overload
def __note_to_degree(key: _IterableLike[str]) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    ...

@overload
def __note_to_degree(key: Union[str, _IterableLike[str], Iterable[str]]) -> Union[int, np.ndarray]:
    if False:
        i = 10
        return i + 15
    ...

def __note_to_degree(key: Union[str, _IterableLike[str], Iterable[str]]) -> Union[int, np.ndarray]:
    if False:
        return 10
    'Take a note name and return the degree of that note (e.g. \'C#\' -> 1). We allow possibilities like "C#b".\n\n    >>> librosa.__note_to_degree(\'B#\')\n    0\n\n    >>> librosa.__note_to_degree(\'D♮##b\')\n    3\n\n    >>> librosa.__note_to_degree([\'B#\',\'D♮##b\'])\n    array([0,3])\n\n    '
    if not isinstance(key, str):
        return np.array([__note_to_degree(n) for n in key])
    match = NOTE_RE.match(key)
    if not match:
        raise ParameterError(f'Improper key format: {key:s}')
    letter = match.group('note').upper()
    accidental = match.group('accidental')
    pitch_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    counter = Counter(accidental)
    return (pitch_map[letter] + sum([ACC_MAP[acc] * counter[acc] for acc in ACC_MAP])) % 12

@overload
def __simplify_note(key: str, additional_acc: str=..., unicode: bool=...) -> str:
    if False:
        print('Hello World!')
    ...

@overload
def __simplify_note(key: _IterableLike[str], additional_acc: str=..., unicode: bool=...) -> np.ndarray:
    if False:
        i = 10
        return i + 15
    ...

@overload
def __simplify_note(key: Union[str, _IterableLike[str], Iterable[str]], additional_acc: str=..., unicode: bool=...) -> Union[str, np.ndarray]:
    if False:
        while True:
            i = 10
    ...

def __simplify_note(key: Union[str, _IterableLike[str], Iterable[str]], additional_acc: str='', unicode: bool=True) -> Union[str, np.ndarray]:
    if False:
        return 10
    "Take in a note name and simplify by canceling sharp-flat pairs, and doubling accidentals as appropriate.\n\n    >>> librosa.__simplify_note('C♭♯')\n    'C'\n\n    >>> librosa.__simplify_note('C♭♭♭')\n    'C♭𝄫'\n\n    >>> librosa.__simplify_note(['C♭♯', 'C♭♭♭'])\n    array(['C', 'C♭𝄫'], dtype='<U3')\n\n    "
    if not isinstance(key, str):
        return np.array([__simplify_note(n + additional_acc, unicode=unicode) for n in key])
    match = NOTE_RE.match(key + additional_acc)
    if not match:
        raise ParameterError(f'Improper key format: {key:s}')
    letter = match.group('note').upper()
    accidental = match.group('accidental')
    counter = Counter(accidental)
    offset = sum([ACC_MAP[acc] * counter[acc] for acc in ACC_MAP])
    simplified_note = letter
    if offset >= 0:
        simplified_note += '♯' * (offset % 2) + '𝄪' * (offset // 2)
    else:
        simplified_note += '♭' * (offset % 2) + '𝄫' * (abs(offset) // 2)
    if not unicode:
        translations = str.maketrans({'♯': '#', '𝄪': '##', '♭': 'b', '𝄫': 'bb', '♮': 'n'})
        simplified_note = simplified_note.translate(translations)
    return simplified_note

def __mode_to_key(signature: str, unicode: bool=True) -> str:
    if False:
        for i in range(10):
            print('nop')
    "Translate a mode (eg D:dorian) into its equivalent major key. If unicode==True, return the accidentals as unicode symbols, regardless of nature of accidentals in signature. Otherwise, return accidentals as ASCII symbols.\n\n    >>> librosa.__mode_to_key('Db:loc')\n    'E𝄫:maj'\n\n    >>> librosa.__mode_to_key('D♭:loc', unicode = False)\n    'Ebb:maj'\n\n    "
    match = KEY_RE.match(signature)
    if not match:
        raise ParameterError('Improper format: {:s}'.format(signature))
    if match.group('scale') or not match.group('mode'):
        signature = __simplify_note(match.group('tonic').upper() + match.group('accidental'), unicode=unicode) + (':' + match.group('scale') if match.group('scale') else '')
        return signature
    mode = match.group('mode').lower()[:3]
    tonic = MAJOR_DICT[mode][match.group('tonic').upper()]
    return __simplify_note(tonic + match.group('accidental'), unicode=unicode) + ':maj'

@cache(level=10)
def key_to_notes(key: str, *, unicode: bool=True, natural: bool=False) -> List[str]:
    if False:
        return 10
    'List all 12 note names in the chromatic scale, as spelled according to\n    a given key (major or minor) or mode (see below for details and accepted abbreviations).\n\n    This function exists to resolve enharmonic equivalences between different\n    spellings for the same pitch (e.g. C♯ vs D♭), and is primarily useful when producing\n    human-readable outputs (e.g. plotting) for pitch content.\n\n    Note names are decided by the following rules:\n\n    1. If the tonic of the key has an accidental (sharp or flat), that accidental will be\n       used consistently for all notes.\n\n    2. If the tonic does not have an accidental, accidentals will be inferred to minimize\n       the total number used for diatonic scale degrees.\n\n    3. If there is a tie (e.g., in the case of C:maj vs A:min), sharps will be preferred.\n\n    Parameters\n    ----------\n    key : string\n        Must be in the form TONIC:key.  Tonic must be upper case (``CDEFGAB``),\n        key must be lower-case\n        (``major``, ``minor``, ``ionian``, ``dorian``, ``phrygian``, ``lydian``, ``mixolydian``, ``aeolian``, ``locrian``).\n\n        The following abbreviations are supported for the modes: either the first three letters of the mode name\n        (e.g. "mix") or the mode name without "ian" (e.g. "mixolyd").\n\n        Both ``major`` and ``maj`` are supported as mode abbreviations.\n\n        Single and multiple accidentals (``b!♭`` for flat, ``#♯`` for sharp, ``𝄪𝄫`` for double-accidentals, or any combination thereof) are supported.\n\n        Examples: ``C:maj, C:major, Dbb:min, A♭:min, D:aeo, E𝄪:phryg``.\n\n    unicode : bool\n        If ``True`` (default), use Unicode symbols (♯𝄪♭𝄫)for accidentals.\n\n        If ``False``, Unicode symbols will be mapped to low-order ASCII representations::\n\n            ♯ -> #, 𝄪 -> ##, ♭ -> b, 𝄫 -> bb, ♮ -> n\n\n    natural : bool\n        If ``True\'\', mark natural accidentals with a natural symbol (♮).\n\n        If ``False`` (default), do not print natural symbols.\n\n        For example, `note_to_degrees(\'D:maj\')[0]` is `C` if `natural=False` (default) and `C♮` if `natural=True`.\n\n    Returns\n    -------\n    notes : list\n        ``notes[k]`` is the name for semitone ``k`` (starting from C)\n        under the given key.  All chromatic notes (0 through 11) are\n        included.\n\n    See Also\n    --------\n    midi_to_note\n\n    Examples\n    --------\n    `C:maj` will use all sharps\n\n    >>> librosa.key_to_notes(\'C:maj\')\n    [\'C\', \'C♯\', \'D\', \'D♯\', \'E\', \'F\', \'F♯\', \'G\', \'G♯\', \'A\', \'A♯\', \'B\']\n\n    `A:min` has the same notes\n\n    >>> librosa.key_to_notes(\'A:min\')\n    [\'C\', \'C♯\', \'D\', \'D♯\', \'E\', \'F\', \'F♯\', \'G\', \'G♯\', \'A\', \'A♯\', \'B\']\n\n    `A♯:min` will use sharps, but spell note 0 (`C`) as `B♯`\n\n    >>> librosa.key_to_notes(\'A#:min\')\n    [\'B♯\', \'C♯\', \'D\', \'D♯\', \'E\', \'E♯\', \'F♯\', \'G\', \'G♯\', \'A\', \'A♯\', \'B\']\n\n    `G♯:maj` will use a double-sharp to spell note 7 (`G`) as `F𝄪`:\n\n    >>> librosa.key_to_notes(\'G#:maj\')\n    [\'B♯\', \'C♯\', \'D\', \'D♯\', \'E\', \'E♯\', \'F♯\', \'F𝄪\', \'G♯\', \'A\', \'A♯\', \'B\']\n\n    `F♭:min` will use double-flats\n\n    >>> librosa.key_to_notes(\'Fb:min\')\n    [\'D𝄫\', \'D♭\', \'E𝄫\', \'E♭\', \'F♭\', \'F\', \'G♭\', \'A𝄫\', \'A♭\', \'B𝄫\', \'B♭\', \'C♭\']\n\n    `G:loc` uses flats\n\n    >>> librosa.key_to_notes(\'G:loc\')\n    [\'C\', \'D♭\', \'D\', \'E♭\', \'E\', \'F\', \'G♭\', \'G\', \'A♭\', \'A\', \'B♭\', \'B\']\n\n    If `natural=True`, print natural accidentals.\n\n    >>> librosa.key_to_notes(\'G:loc\', natural=True)\n    [\'C\', \'D♭\', \'D♮\', \'E♭\', \'E♮\', \'F\', \'G♭\', \'G\', \'A♭\', \'A♮\', \'B♭\', \'B♮\']\n\n    >>> librosa.key_to_notes(\'D:maj\', natural=True)\n    [\'C♮\', \'C♯\', \'D\', \'D♯\', \'E\', \'F♮\', \'F♯\', \'G\', \'G♯\', \'A\', \'A♯\', \'B\']\n\n    >>> librosa.key_to_notes(\'G#:maj\', unicode = False, natural = True)\n    [\'B#\', \'C#\', \'Dn\', \'D#\', \'En\', \'E#\', \'F#\', \'F##\', \'G#\', \'An\', \'A#\', \'B\']\n\n    '
    match = KEY_RE.match(key)
    if not match:
        raise ParameterError(f'Improper key format: {key:s}')
    pitch_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    tonic = match.group('tonic').upper()
    accidental = match.group('accidental')
    offset = sum([ACC_MAP[acc] for acc in accidental])
    if match.group('mode') or not match.group('scale'):
        equiv = __mode_to_key(key)
        return key_to_notes(equiv, unicode=unicode, natural=natural)
    scale = match.group('scale')[:3].lower()
    multiple = abs(offset) >= 2
    if multiple:
        sign_map = {+1: '♯', -1: '♭'}
        additional_acc = sign_map[np.sign(offset)]
        intermediate_notes = key_to_notes(tonic + additional_acc * (abs(offset) - 1) + ':' + scale, natural=False)
        notes = [__simplify_note(note, additional_acc) for note in intermediate_notes]
        degrees = __note_to_degree(notes)
        notes = np.roll(notes, shift=-np.argwhere(degrees == 0)[0])
        notes = list(notes)
        if not unicode:
            translations = str.maketrans({'♯': '#', '𝄪': '##', '♭': 'b', '𝄫': 'bb', '♮': 'n'})
            notes = list((n.translate(translations) for n in notes))
        return notes
    major = scale == 'maj'
    if major:
        tonic_number = (pitch_map[tonic] + offset) * 7 % 12
    else:
        tonic_number = ((pitch_map[tonic] + offset) * 7 + 9) % 12
    if offset < 0:
        use_sharps = False
    elif offset > 0:
        use_sharps = True
    elif 0 <= tonic_number < 6:
        use_sharps = True
    elif tonic_number > 6:
        use_sharps = False
    notes_sharp = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B']
    notes_flat = ['C', 'D♭', 'D', 'E♭', 'E', 'F', 'G♭', 'G', 'A♭', 'A', 'B♭', 'B']
    sharp_corrections = [(5, 'E♯'), (0, 'B♯'), (7, 'F𝄪'), (2, 'C𝄪'), (9, 'G𝄪'), (4, 'D𝄪'), (11, 'A𝄪')]
    flat_corrections = [(11, 'C♭'), (4, 'F♭'), (9, 'B𝄫'), (2, 'E𝄫'), (7, 'A𝄫'), (0, 'D𝄫')]
    n_sharps = tonic_number
    if tonic_number == 0 and tonic == 'B':
        n_sharps = 12
    if use_sharps:
        for n in range(0, n_sharps - 6 + 1):
            (index, name) = sharp_corrections[n]
            notes_sharp[index] = name
        notes = notes_sharp
    else:
        n_flats = (12 - tonic_number) % 12
        for n in range(0, n_flats - 6 + 1):
            (index, name) = flat_corrections[n]
            notes_flat[index] = name
        notes = notes_flat
    if natural:
        scale_notes = set(key_to_degrees(key))
        for (place, note) in enumerate(notes):
            if __note_to_degree(note) in scale_notes:
                continue
            if len(note) == 1:
                notes[place] = note + '♮'
    if not unicode:
        translations = str.maketrans({'♯': '#', '𝄪': '##', '♭': 'b', '𝄫': 'bb', '♮': 'n'})
        notes = list((n.translate(translations) for n in notes))
    return notes

def key_to_degrees(key: str) -> np.ndarray:
    if False:
        return 10
    'Construct the diatonic scale degrees for a given key.\n\n    Parameters\n    ----------\n    key : str\n        Must be in the form TONIC:key.  Tonic must be upper case (``CDEFGAB``),\n        key must be lower-case\n        (``maj``, ``min``, ``ionian``, ``dorian``, ``phrygian``, ``lydian``, ``mixolydian``, ``aeolian``, ``locrian``).\n\n        The following abbreviations are supported for the modes: either the first three letters of the mode name\n        (e.g. "mix") or the mode name without "ian" (e.g. "mixolyd").\n\n        Both ``major`` and ``maj`` are supported as abbreviations.\n\n        Single and multiple accidentals (``b!♭`` for flat, or ``#♯`` for sharp) are supported.\n\n        Examples: ``C:maj, C:major, Dbb:min, A♭:min, D:aeo, E𝄪:phryg``.\n\n    Returns\n    -------\n    degrees : np.ndarray\n        An array containing the semitone numbers (0=C, 1=C#, ... 11=B)\n        for each of the seven scale degrees in the given key, starting\n        from the tonic.\n\n    See Also\n    --------\n    key_to_notes\n\n    Examples\n    --------\n    >>> librosa.key_to_degrees(\'C:maj\')\n    array([ 0,  2,  4,  5,  7,  9, 11])\n\n    >>> librosa.key_to_degrees(\'C#:maj\')\n    array([ 1,  3,  5,  6,  8, 10,  0])\n\n    >>> librosa.key_to_degrees(\'A:min\')\n    array([ 9, 11,  0,  2,  4,  5,  7])\n\n    >>> librosa.key_to_degrees(\'A:min\')\n    array([ 9, 11,  0,  2,  4,  5,  7])\n\n    '
    notes = dict(maj=np.array([0, 2, 4, 5, 7, 9, 11]), min=np.array([0, 2, 3, 5, 7, 8, 10]))
    match = KEY_RE.match(key)
    if not match:
        raise ParameterError(f'Improper key format: {key:s}')
    if match.group('mode') or not match.group('scale'):
        equiv = __mode_to_key(key)
        offset = OFFSET_DICT[match.group('mode')[:3]]
        return np.roll(key_to_degrees(equiv), -offset)
    pitch_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    tonic = match.group('tonic').upper()
    accidental = match.group('accidental')
    counts = Counter(accidental)
    offset = sum([ACC_MAP[acc] * counts[acc] for acc in ACC_MAP])
    scale = match.group('scale')[:3].lower()
    return (notes[scale] + pitch_map[tonic] + offset) % 12

@cache(level=10)
def fifths_to_note(*, unison: str, fifths: int, unicode: bool=True) -> str:
    if False:
        print('Hello World!')
    'Calculate the note name for a given number of perfect fifths\n    from a specified unison.\n\n    This function is primarily intended as a utility routine for\n    Functional Just System (FJS) notation conversions.\n\n    This function does not assume the "circle of fifths" or equal temperament,\n    so 12 fifths will not generally produce a note of the same pitch class\n    due to the accumulation of accidentals.\n\n    Parameters\n    ----------\n    unison : str\n        The name of the starting (unison) note, e.g., \'C\' or \'Bb\'.\n        Unicode accidentals are supported.\n\n    fifths : integer\n        The number of perfect fifths to deviate from unison.\n\n    unicode : bool\n        If ``True`` (default), use Unicode symbols (♯𝄪♭𝄫)for accidentals.\n\n        If ``False``, accidentals will be encoded as low-order ASCII representations::\n\n            ♯ -> #, 𝄪 -> ##, ♭ -> b, 𝄫 -> bb\n\n    Returns\n    -------\n    note : str\n        The name of the requested note\n\n    Examples\n    --------\n    >>> librosa.fifths_to_note(unison=\'C\', fifths=6)\n    \'F♯\'\n\n    >>> librosa.fifths_to_note(unison=\'G\', fifths=-3)\n    \'B♭\'\n\n    >>> librosa.fifths_to_note(unison=\'Eb\', fifths=11, unicode=False)\n    \'G#\'\n\n    '
    COFMAP = 'FCGDAEB'
    acc_map = {'#': 1, '': 0, 'b': -1, '!': -1, '♯': 1, '𝄪': 2, '♭': -1, '𝄫': -2, '♮': 0, 'n': 0}
    if unicode:
        acc_map_inv = {1: '♯', 2: '𝄪', -1: '♭', -2: '𝄫', 0: ''}
    else:
        acc_map_inv = {1: '#', 2: '##', -1: 'b', -2: 'bb', 0: ''}
    match = NOTE_RE.match(unison)
    if not match:
        raise ParameterError(f'Improper note format: {unison:s}')
    pitch = match.group('note').upper()
    offset = np.sum([acc_map[o] for o in match.group('accidental')])
    circle_idx = COFMAP.index(pitch)
    raw_output = COFMAP[(circle_idx + fifths) % 7]
    acc_index = offset + (circle_idx + fifths) // 7
    acc_str = acc_map_inv[np.sign(acc_index) * 2] * int(abs(acc_index) // 2) + acc_map_inv[np.sign(acc_index)] * int(abs(acc_index) % 2)
    return raw_output + acc_str

@jit(nopython=True, nogil=True, cache=True)
def __o_fold(d):
    if False:
        i = 10
        return i + 15
    'Compute the octave-folded interval.\n\n    This maps intervals to the range [1, 2).\n\n    This is part of the FJS notation converter.\n    It is equivalent to the `red` function described in the FJS\n    documentation.\n    '
    return d * 2.0 ** (-np.floor(np.log2(d)))

@jit(nopython=True, nogil=True, cache=True)
def __bo_fold(d):
    if False:
        print('Hello World!')
    'Compute the balanced, octave-folded interval.\n\n    This maps intervals to the range [sqrt(2)/2, sqrt(2)).\n\n    This is part of the FJS notation converter.\n    It is equivalent to the `reb` function described in the FJS\n    documentation, but with a simpler implementation.\n    '
    return d * 2.0 ** (-np.round(np.log2(d)))

@jit(nopython=True, nogil=True, cache=True)
def __fifth_search(interval, tolerance):
    if False:
        for i in range(10):
            print('nop')
    'Accelerated helper function for finding the number of fifths\n    to get within tolerance of a given interval.\n\n    This implementation will give up after 32 fifths\n    '
    log_tolerance = np.abs(np.log2(tolerance))
    for power in range(32):
        for sign in [1, -1]:
            if np.abs(np.log2(__bo_fold(interval / 3.0 ** (power * sign)))) <= log_tolerance:
                return power * sign
        power += 1
    return power
SUPER_TRANS = str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹')
SUB_TRANS = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')

@overload
def interval_to_fjs(interval: _FloatLike_co, *, unison: str=..., tolerance: float=..., unicode: bool=...) -> str:
    if False:
        return 10
    ...

@overload
def interval_to_fjs(interval: _SequenceLike[_FloatLike_co], *, unison: str=..., tolerance: float=..., unicode: bool=...) -> np.ndarray:
    if False:
        while True:
            i = 10
    ...

@overload
def interval_to_fjs(interval: _ScalarOrSequence[_FloatLike_co], *, unison: str=..., tolerance: float=..., unicode: bool=...) -> Union[str, np.ndarray]:
    if False:
        i = 10
        return i + 15
    ...

@vectorize(otypes='U', excluded=set(['unison', 'tolerance', 'unicode']))
def interval_to_fjs(interval: _ScalarOrSequence[_FloatLike_co], *, unison: str='C', tolerance: float=65.0 / 63, unicode: bool=True) -> Union[str, np.ndarray]:
    if False:
        print('Hello World!')
    "Convert an interval to Functional Just System (FJS) notation.\n\n    See https://misotanni.github.io/fjs/en/index.html for a thorough overview\n    of the FJS notation system, and the examples below.\n\n    FJS conversion works by identifying a Pythagorean interval which is within\n    a specified tolerance of the target interval, which provides the core note\n    name.  If the interval is derived from ratios other than perfect fifths,\n    then the remaining factors are encoded as superscripts for otonal\n    (increasing) intervals and subscripts for utonal (decreasing) intervals.\n\n    Parameters\n    ----------\n    interval : float > 0 or iterable of floats\n        A (just) interval to notate in FJS.\n\n    unison : str\n        The name of the unison note (corresponding to `interval=1`).\n\n    tolerance : float\n        The tolerance threshold for identifying the core note name.\n\n    unicode : bool\n        If ``True`` (default), use Unicode symbols (♯𝄪♭𝄫)for accidentals,\n        and superscripts/subscripts for otonal and utonal accidentals.\n\n        If ``False``, accidentals will be encoded as low-order ASCII representations::\n\n            ♯ -> #, 𝄪 -> ##, ♭ -> b, 𝄫 -> bb\n\n        Otonal and utonal accidentals will be denoted by `^##` and `_##`\n        respectively (see examples below).\n\n    Raises\n    ------\n    ParameterError\n        If the provided interval is not positive\n\n        If the provided interval cannot be identified with a\n        just intonation prime factorization.\n\n    Returns\n    -------\n    note_fjs : str or np.ndarray(dtype=str)\n        The interval(s) relative to the given unison in FJS notation.\n\n    Examples\n    --------\n    Pythagorean intervals appear as expected, with no otonal\n    or utonal extensions:\n\n    >>> librosa.interval_to_fjs(3/2, unison='C')\n    'G'\n    >>> librosa.interval_to_fjs(4/3, unison='F')\n    'B♭'\n\n    A ptolemaic major third will appear with an otonal '5':\n\n    >>> librosa.interval_to_fjs(5/4, unison='A')\n    'C♯⁵'\n\n    And a ptolemaic minor third will appear with utonal '5':\n\n    >>> librosa.interval_to_fjs(6/5, unison='A')\n    'C₅'\n\n    More complex intervals will have compound accidentals.\n    For example:\n\n    >>> librosa.interval_to_fjs(25/14, unison='F#')\n    'E²⁵₇'\n    >>> librosa.interval_to_fjs(25/14, unison='F#', unicode=False)\n    'E^25_7'\n\n    Array inputs are also supported:\n\n    >>> librosa.interval_to_fjs([3/2, 4/3, 5/3])\n    array(['G', 'F', 'A⁵'], dtype='<U2')\n\n    "
    if interval <= 0:
        raise ParameterError(f'Interval={interval} must be strictly positive')
    fifths = __fifth_search(interval, tolerance)
    note_name = fifths_to_note(unison=unison, fifths=fifths, unicode=unicode)
    try:
        interval_b = __o_fold(interval)
        powers = INTERVALS[np.around(interval_b, decimals=6)]
    except KeyError as exc:
        raise ParameterError(f'Unknown interval={interval}') from exc
    powers = {p: powers[p] for p in powers if p > 3}
    otonal = np.prod([p ** powers[p] for p in powers if powers[p] > 0])
    utonal = np.prod([p ** (-powers[p]) for p in powers if powers[p] < 0])
    suffix = ''
    if otonal > 1:
        if unicode:
            suffix += f'{otonal:d}'.translate(SUPER_TRANS)
        else:
            suffix += f'^{otonal}'
    if utonal > 1:
        if unicode:
            suffix += f'{utonal:d}'.translate(SUB_TRANS)
        else:
            suffix += f'_{utonal}'
    return note_name + suffix