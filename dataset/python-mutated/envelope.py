import sys
sys.path.insert(0, f"{__file__.rpartition('/')[0] or '.'}/../../../../frozen/Adafruit_CircuitPython_Wave")
import random
import audiocore
import synthio
from ulab import numpy as np
import adafruit_wave as wave
SAMPLE_SIZE = 1024
VOLUME = 32767
sine = np.array(np.sin(np.linspace(0, 2 * np.pi, SAMPLE_SIZE, endpoint=False)) * VOLUME, dtype=np.int16)
envelope = synthio.Envelope(attack_time=0.05, decay_time=8, release_time=0.25, sustain_level=0)
fast_decay_envelope = synthio.Envelope(attack_time=0.05, decay_time=0.25, release_time=0.25, sustain_level=0)
synth = synthio.Synthesizer(sample_rate=48000)

def synthesize(synth):
    if False:
        i = 10
        return i + 15
    notes = (synthio.Note(frequency=440, waveform=sine, envelope=envelope),)
    synth.press(notes)
    yield 360
    notes[0].envelope = fast_decay_envelope
    yield 180
    synth.release_all()

def synthesize2(synth):
    if False:
        return 10
    notes = (synthio.Note(frequency=440, waveform=sine, envelope=envelope),)
    synth.press(notes)
    yield 360
    synth.release_all()
    yield 180

def chain(*args):
    if False:
        i = 10
        return i + 15
    for a in args:
        yield from a
with wave.open('envelope.wav', 'w') as f:
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(48000)
    for n in chain(synthesize(synth), synthesize2(synth)):
        for i in range(n):
            (result, data) = audiocore.get_buffer(synth)
            f.writeframes(data)