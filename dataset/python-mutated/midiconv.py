""" utilities to convert midi note number to and from note names """
import sys
from ._aubio import freqtomidi, miditofreq
__all__ = ['note2midi', 'midi2note', 'freq2note', 'note2freq']
py3 = sys.version_info[0] == 3
if py3:
    str_instances = str
    int_instances = int
else:
    str_instances = (str, unicode)
    int_instances = (int, long)

def note2midi(note):
    if False:
        i = 10
        return i + 15
    "Convert note name to midi note number.\n\n    Input string `note` should be composed of one note root\n    and one octave, with optionally one modifier in between.\n\n    List of valid components:\n\n    - note roots: `C`, `D`, `E`, `F`, `G`, `A`, `B`,\n    - modifiers: `b`, `#`, as well as unicode characters\n      `𝄫`, `♭`, `♮`, `♯` and `𝄪`,\n    - octave numbers: `-1` -> `11`.\n\n    Parameters\n    ----------\n    note : str\n        note name\n\n    Returns\n    -------\n    int\n        corresponding midi note number\n\n    Examples\n    --------\n    >>> aubio.note2midi('C#4')\n    61\n    >>> aubio.note2midi('B♭5')\n    82\n\n    Raises\n    ------\n    TypeError\n        If `note` was not a string.\n    ValueError\n        If an error was found while converting `note`.\n\n    See Also\n    --------\n    midi2note, freqtomidi, miditofreq\n    "
    _valid_notenames = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    _valid_modifiers = {u'𝄫': -2, u'♭': -1, 'b': -1, '♭': -1, u'♮': 0, '♮': 0, None: 0, '#': +1, u'♯': +1, '♯': +1, u'𝄪': +2}
    _valid_octaves = range(-1, 10)
    if not isinstance(note, str_instances):
        msg = 'a string is required, got {:s} ({:s})'
        raise TypeError(msg.format(str(type(note)), repr(note)))
    if len(note) not in range(2, 5):
        msg = 'string of 2 to 4 characters expected, got {:d} ({:s})'
        raise ValueError(msg.format(len(note), note))
    (notename, modifier, octave) = [None] * 3
    if len(note) == 4:
        (notename, modifier, octave_sign, octave) = note
        octave = octave_sign + octave
    elif len(note) == 3:
        (notename, modifier, octave) = note
        if modifier == '-':
            octave = modifier + octave
            modifier = None
    else:
        (notename, octave) = note
    notename = notename.upper()
    octave = int(octave)
    if notename not in _valid_notenames:
        raise ValueError('%s is not a valid note name' % notename)
    if modifier not in _valid_modifiers:
        raise ValueError('%s is not a valid modifier' % modifier)
    if octave not in _valid_octaves:
        raise ValueError('%s is not a valid octave' % octave)
    midi = (octave + 1) * 12 + _valid_notenames[notename] + _valid_modifiers[modifier]
    if midi > 127:
        raise ValueError('%s is outside of the range C-2 to G8' % note)
    return midi

def midi2note(midi):
    if False:
        print('Hello World!')
    "Convert midi note number to note name.\n\n    Parameters\n    ----------\n    midi : int [0, 128]\n        input midi note number\n\n    Returns\n    -------\n    str\n        note name\n\n    Examples\n    --------\n    >>> aubio.midi2note(70)\n    'A#4'\n    >>> aubio.midi2note(59)\n    'B3'\n\n    Raises\n    ------\n    TypeError\n        If `midi` was not an integer.\n    ValueError\n        If `midi` is out of the range `[0, 128]`.\n\n    See Also\n    --------\n    note2midi, miditofreq, freqtomidi\n    "
    if not isinstance(midi, int_instances):
        raise TypeError('an integer is required, got %s' % midi)
    if midi not in range(0, 128):
        msg = 'an integer between 0 and 127 is excepted, got {:d}'
        raise ValueError(msg.format(midi))
    _valid_notenames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return _valid_notenames[midi % 12] + str(int(midi / 12) - 1)

def freq2note(freq):
    if False:
        return 10
    "Convert frequency in Hz to nearest note name.\n\n    Parameters\n    ----------\n    freq : float [0, 23000[\n        input frequency, in Hz\n\n    Returns\n    -------\n    str\n        name of the nearest note\n\n    Example\n    -------\n    >>> aubio.freq2note(440)\n    'A4'\n    >>> aubio.freq2note(220.1)\n    'A3'\n    "
    nearest_note = int(freqtomidi(freq) + 0.5)
    return midi2note(nearest_note)

def note2freq(note):
    if False:
        for i in range(10):
            print('nop')
    "Convert note name to corresponding frequency, in Hz.\n\n    Parameters\n    ----------\n    note : str\n        input note name\n\n    Returns\n    -------\n    freq : float [0, 23000[\n        frequency, in Hz\n\n    Example\n    -------\n    >>> aubio.note2freq('A4')\n    440\n    >>> aubio.note2freq('A3')\n    220.1\n    "
    midi = note2midi(note)
    return miditofreq(midi)