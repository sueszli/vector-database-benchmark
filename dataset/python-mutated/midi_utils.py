from collections import Counter
from typing import List, Tuple, Union

def parse_midi_message_part(byte: Union[int, str, bytes]) -> int:
    if False:
        print('Hello World!')
    if isinstance(byte, int):
        return byte
    elif isinstance(byte, bytes):
        if len(byte) == 1:
            return byte[0]
        else:
            raise NotImplementedError('Not sure how to interpret provided MIDI message.')
    elif isinstance(byte, str):
        try:
            return int(byte)
        except ValueError as e:
            raise NotImplementedError('Not sure how to interpret provided MIDI message.') from e
    raise NotImplementedError('MIDI messages must currently be bytes or lists of byte values.')

def parse_midi_message_string(midi_message_string: str) -> bytes:
    if False:
        while True:
            i = 10
    raise NotImplementedError('MIDI messages must currently be bytes or lists of byte values.')

def normalize_midi_messages(_input) -> List[Tuple[bytes, float]]:
    if False:
        return 10
    '\n    Given a duck-typed Python input, usually an iterable of MIDI messages,\n    normalize the input to a list of tuples of bytes which can be converted\n    into a juce::MidiBuffer on the C++ side.\n    '
    output = []
    for message in _input:
        if hasattr(message, 'bytes') and hasattr(message, 'time'):
            output.append((bytes(message.bytes()), message.time))
        elif (isinstance(message, tuple) or isinstance(message, list)) and len(message) == 2:
            (message, time) = message
            if isinstance(message, str):
                message = parse_midi_message_string(message)
            if isinstance(message, list):
                message = bytes([parse_midi_message_part(x) for x in message])
            elif not isinstance(message, bytes):
                message = bytes(message)
            output.append((message, time))
    all_timestamps = [t for (_, t) in output]
    if len(all_timestamps) > 100 and len(set(all_timestamps)) > 1:
        all_timestamps_histogram = Counter(all_timestamps)
        (most_common_timestamp, num_instances_of_most_common) = all_timestamps_histogram.most_common()[0]
        if num_instances_of_most_common > 100:
            raise ValueError(f'Pedalboard requires MIDI input timestamps to be absolute values, specified as the number of seconds from the start of the returned audio buffer. The provided MIDI data contains {num_instances_of_most_common:,} events at timestamp {most_common_timestamp}, which suggests that the timestamps may be delta values rather than absolute values. Try converting your MIDI message timestamps to absolute values first.')
    return output