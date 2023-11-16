""" A simple demo using aubio and pyaudio to play beats in real time

Note you will need to have pyaudio installed: `pip install pyaudio`.

Examples:
    ./demo_tapthebeat.py ~/Music/track1.ogg

When compiled with ffmpeg/libav, you should be able to open remote streams. For
instance using youtube-dl (`pip install youtube-dl`):

    ./demo_tapthebeat.py `youtube-dl -xg https://youtu.be/zZbM9n9j3_g`

"""
import sys
import time
import pyaudio
import aubio
import numpy as np
win_s = 1024
hop_s = win_s // 2
if len(sys.argv) < 2:
    print('Usage: %s <filename> [samplerate]' % sys.argv[0])
    sys.exit(1)
filename = sys.argv[1]
samplerate = 0
if len(sys.argv) > 2:
    samplerate = int(sys.argv[2])
a_source = aubio.source(filename, samplerate, hop_s)
samplerate = a_source.samplerate
a_tempo = aubio.tempo('default', win_s, hop_s, samplerate)
click = 0.7 * np.sin(2.0 * np.pi * np.arange(hop_s) / hop_s * samplerate / 3000.0)

def pyaudio_callback(_in_data, _frame_count, _time_info, _status):
    if False:
        while True:
            i = 10
    (samples, read) = a_source()
    is_beat = a_tempo(samples)
    if is_beat:
        samples += click
    audiobuf = samples.tobytes()
    if read < hop_s:
        return (audiobuf, pyaudio.paComplete)
    return (audiobuf, pyaudio.paContinue)
p = pyaudio.PyAudio()
pyaudio_format = pyaudio.paFloat32
frames_per_buffer = hop_s
n_channels = 1
stream = p.open(format=pyaudio_format, channels=n_channels, rate=samplerate, output=True, frames_per_buffer=frames_per_buffer, stream_callback=pyaudio_callback)
stream.start_stream()
while stream.is_active():
    time.sleep(0.1)
stream.stop_stream()
stream.close()
p.terminate()