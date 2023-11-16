from typing import List, Tuple
import mido
import pytest
from pedalboard.midi_utils import normalize_midi_messages

@pytest.mark.parametrize('_input,expected', [([mido.Message('note_on', note=100, velocity=3, time=0), mido.Message('note_off', note=100, time=5.0)], [(bytes([144, 100, 3]), 0.0), (bytes([128, 100, 64]), 5.0)])])
def test_mido_normalization(_input, expected: List[Tuple[bytes, float]]):
    if False:
        for i in range(10):
            print('nop')
    assert normalize_midi_messages(_input) == expected