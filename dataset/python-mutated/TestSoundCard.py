import numpy as np

def test_sounddevice_lib():
    if False:
        print('Hello World!')
    import time
    import numpy as np
    from sounddevice import InputStream, OutputStream, sleep as sd_sleep
    ' \n    if no portaudio installed:\n    Traceback (most recent call last):\n  File "TestSoundCard.py", line 42, in <module>\n    test_sounddevice_lib()\n  File "TestSoundCard.py", line 5, in test_sounddevice_lib\n    import sounddevice as sd\n  File "/usr/lib/python3.6/site-packages/sounddevice.py", line 64, in <module>\n    raise OSError(\'PortAudio library not found\')\n  OSError: PortAudio library not found\n\n    '
    duration = 2.5
    rx_buffer = np.ones((10 ** 6, 2), dtype=np.float32)
    global current_rx, current_tx
    current_rx = 0
    current_tx = 0

    def rx_callback(indata: np.ndarray, frames: int, time, status):
        if False:
            print('Hello World!')
        global current_rx
        if status:
            print(status)
        rx_buffer[current_rx:current_rx + frames] = indata
        current_rx += frames

    def tx_callback(outdata: np.ndarray, frames: int, time, status):
        if False:
            print('Hello World!')
        global current_tx
        if status:
            print(status)
        outdata[:] = rx_buffer[current_tx:current_tx + frames]
        current_tx += frames
    with InputStream(channels=2, callback=rx_callback):
        sd_sleep(int(duration * 1000))
    print('Current rx', current_rx)
    with OutputStream(channels=2, callback=tx_callback):
        sd_sleep(int(duration * 1000))
    print('Current tx', current_tx)

def test_pyaudio():
    if False:
        while True:
            i = 10
    import pyaudio
    CHUNK = 1024
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=2, rate=48000, input=True, frames_per_buffer=CHUNK)
    print('* recording')
    frames = []
    for i in range(0, 100):
        data = stream.read(CHUNK)
        frames.append(data)
    print('* done recording')
    stream.stop_stream()
    stream.close()
    p.terminate()
    data = b''.join(frames)
    print('* playing')
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=2, rate=48000, output=True)
    for i in range(0, len(data), CHUNK):
        stream.write(data[i:i + CHUNK])
    stream.stop_stream()
    stream.close()
    p.terminate()
    print('* done playing')
if __name__ == '__main__':
    test_pyaudio()