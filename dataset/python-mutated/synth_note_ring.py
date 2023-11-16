from synthnotehelper import *

@synth_test
def gen(synth):
    if False:
        i = 10
        return i + 15
    l = LFO(sweep, rate=4, once=True)
    yield [l]
    n = Note(128, ring_waveform=sine, ring_frequency=6, ring_bend=l)
    synth.press(n)
    yield (1 / 4)