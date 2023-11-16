from __future__ import division, absolute_import, with_statement, print_function, unicode_literals
from renpy.compat import PY2, basestring, bchr, bord, chr, open, pystr, range, round, str, tobytes, unicode
import renpy
import emscripten
import pygame
from json import dumps
import renpy.audio.renpysound as renpysound
video_only = False

def call(function, *args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calls a method on `function`.\n    '
    emscripten.run_script('renpyAudio.{}.apply(null, {});'.format(function, dumps(args)))

def call_int(function, *args):
    if False:
        print('Hello World!')
    '\n    Calls a method on `function`.\n    '
    return emscripten.run_script_int('renpyAudio.{}.apply(null, {});'.format(function, dumps(args)))

def call_str(function, *args):
    if False:
        return 10
    '\n    Calls a method on `function`.\n    '
    rv = emscripten.run_script_string('renpyAudio.{}.apply(null, {});'.format(function, dumps(args)))
    return rv
audio_channels = set()

def set_movie_channel(channel, movie):
    if False:
        while True:
            i = 10
    if video_only and (not movie):
        audio_channels.add(channel)
renpysound.set_movie_channel = set_movie_channel
renpysound_funcs = {}

def proxy_with_channel(func):
    if False:
        return 10
    '\n    Call the webaudio function instead of the renpysound function for audio channels if browser\n    supports audio decoding.\n    Always call the webaudio function instead of the renpysound function for video channels.\n    '

    def hook(channel, *args, **kwargs):
        if False:
            print('Hello World!')
        if video_only and channel in audio_channels:
            return renpysound_funcs[func.__name__](channel, *args, **kwargs)
        return func(channel, *args, **kwargs)
    if func.__name__ not in renpysound_funcs:
        renpysound_funcs[func.__name__] = getattr(renpysound, func.__name__)
    setattr(renpysound, func.__name__, hook)
    return func

def proxy_call_both(func):
    if False:
        print('Hello World!')
    '\n    Call renpysound function followed by webaudio function if browser does not support\n    audio decoding.\n    Only call the webaudio function if browser supports audio decoding.\n    '

    def hook(*args, **kwargs):
        if False:
            while True:
                i = 10
        if video_only:
            ret1 = renpysound_funcs[func.__name__](*args, **kwargs)
            ret2 = func(*args, **kwargs)
            return ret1 and ret2
        return func(*args, **kwargs)
    if func.__name__ not in renpysound_funcs:
        renpysound_funcs[func.__name__] = getattr(renpysound, func.__name__)
    setattr(renpysound, func.__name__, hook)
    return func

@proxy_with_channel
def play(channel, file, name, paused=False, fadein=0, tight=False, start=0, end=0, relative_volume=1.0):
    if False:
        i = 10
        return i + 15
    '\n    Plays `file` on `channel`. This clears the playing and queued samples and\n    replaces them with this file.\n\n    `name`\n        A python object giving a readable name for the file.\n\n    `paused`\n        If True, playback is paused rather than started.\n\n    `fadein`\n        The time it should take the fade the music in, in seconds.\n\n    `tight`\n        If true, the file is played in tight mode. This means that fadeouts\n        can span between this file and the file queued after it.\n\n    `start`\n        A time in the file to start playing.\n\n    `end`\n        A time in the file to end playing.    `\n\n    `relative_volume`\n        A number between 0 and 1 that controls the relative volume of this file\n    '
    try:
        if not isinstance(file, basestring):
            file = file.name
    except Exception:
        return
    call('stop', channel)
    call('queue', channel, file, name, paused, fadein, tight, start, end, relative_volume)

@proxy_with_channel
def queue(channel, file, name, fadein=0, tight=False, start=0, end=0, relative_volume=1.0):
    if False:
        for i in range(10):
            print('nop')
    '\n    Queues `file` on `channel` to play when the current file ends. If no file is\n    playing, plays it.\n\n    The other arguments are as for play.\n    '
    try:
        if not isinstance(file, basestring):
            file = file.name
    except Exception:
        return
    call('queue', channel, file, name, False, fadein, tight, start, end, relative_volume)

@proxy_with_channel
def stop(channel):
    if False:
        while True:
            i = 10
    '\n    Immediately stops `channel`, and unqueues any queued audio file.\n    '
    call('stop', channel)

@proxy_with_channel
def dequeue(channel, even_tight=False):
    if False:
        i = 10
        return i + 15
    '\n    Dequeues the queued sound file.\n\n    `even_tight`\n        If true, a queued sound file that is tight is not dequeued. If false,\n        a file marked as tight is dequeued.\n    '
    call('dequeue', channel, even_tight)

@proxy_with_channel
def queue_depth(channel):
    if False:
        return 10
    '\n    Returns the queue depth of the channel. 0 if no file is playing, 1 if\n    a files is playing but there is no queued file, and 2 if a file is playing\n    and one is queued.\n    '
    return emscripten.run_script_int('renpyAudio.queue_depth({})'.format(channel))

@proxy_with_channel
def playing_name(channel):
    if False:
        return 10
    '\n    Returns the `name`  argument of the playing sound. This was passed into\n    `play` or `queue`.\n    '
    rv = call_str('playing_name', channel)
    if rv:
        return rv
    return None

@proxy_with_channel
def pause(channel):
    if False:
        return 10
    '\n    Pauses `channel`.\n    '
    call('pause', channel)

@proxy_with_channel
def unpause(channel):
    if False:
        return 10
    '\n    Unpauses `channel`.\n    '
    call('unpause', channel)

@proxy_call_both
def unpause_all_at_start():
    if False:
        for i in range(10):
            print('nop')
    '\n    Unpauses all channels that are paused.\n    '
    call('unpauseAllAtStart')

@proxy_with_channel
def fadeout(channel, delay):
    if False:
        i = 10
        return i + 15
    '\n    Fades out `channel` over `delay` seconds.\n    '
    call('fadeout', channel, delay)

@proxy_with_channel
def busy(channel):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns true if `channel` is currently playing something, and false\n    otherwise\n    '
    return queue_depth(channel) > 0

@proxy_with_channel
def get_pos(channel):
    if False:
        i = 10
        return i + 15
    '\n    Returns the position of the audio file playing in `channel`. Returns None\n    if not file is is playing or it is not known.\n    '
    rv = call_int('get_pos', channel)
    if rv >= 0:
        return rv / 1000.0
    else:
        return None

@proxy_with_channel
def get_duration(channel):
    if False:
        return 10
    '\n    Returns the duration of the audio file playing in `channel`, or None if no\n    file is playing or it is not known.\n    '
    rv = call_int('get_duration', channel)
    if rv >= 0:
        return rv / 1000.0
    else:
        return None

@proxy_with_channel
def set_volume(channel, volume):
    if False:
        i = 10
        return i + 15
    '\n    Sets the primary volume for `channel` to `volume`, a number between\n    0 and 1. This volume control is perceptual, taking into account the\n    logarithmic nature of human hearing.\n    '
    call('set_volume', channel, volume)

@proxy_with_channel
def set_pan(channel, pan, delay):
    if False:
        for i in range(10):
            print('nop')
    "\n    Sets the pan for channel.\n\n    `pan`\n        A number between -1 and 1 that control the placement of the audio.\n        If this is -1, then all audio is sent to the left channel.\n        If it's 0, then the two channels are equally balanced. If it's 1,\n        then all audio is sent to the right ear.\n\n    `delay`\n        The amount of time it takes for the panning to occur.\n    "
    call('set_pan', channel, pan, delay)

@proxy_with_channel
def set_secondary_volume(channel, volume, delay):
    if False:
        i = 10
        return i + 15
    '\n    Sets the secondary volume for channel. This is linear, and is multiplied\n    with the primary volume and scale factors derived from pan to find the\n    actual multiplier used on the samples.\n\n    `delay`\n        The time it takes for the change in volume to happen.\n    '
    call('set_secondary_volume', channel, volume, delay)

@proxy_with_channel
def get_volume(channel):
    if False:
        print('Hello World!')
    '\n    Gets the primary volume associated with `channel`.\n    '
    return call_int('get_volume', channel)

@proxy_with_channel
def video_ready(channel):
    if False:
        while True:
            i = 10
    '\n    Returns true if the video playing on `channel` has a frame ready for\n    presentation.\n    '
    if not video_supported():
        return False
    return call_int('video_ready', channel)
channel_size = {}

@proxy_with_channel
def read_video(channel):
    if False:
        i = 10
        return i + 15
    '\n    Returns the frame of video playing on `channel`. This is returned as a GLTexture.\n    '
    if not video_supported():
        return None
    video_size = channel_size.get(channel)
    if video_size is None:
        info = call_str('get_video_size', channel)
        if len(info) == 0:
            return None
        video_size = [int(s) for s in info.split('x')]
        channel_size[channel] = video_size
    tex = renpy.gl2.gl2texture.Texture(video_size, renpy.display.draw.texture_loader, generate=True)
    res = call_int('read_video', channel, tex.get_number(), *video_size)
    if res == 0:
        return tex
    if res > 0:
        return None
    if res == -1:
        del channel_size[channel]
        return read_video(channel)
    return None
NO_VIDEO = 0
NODROP_VIDEO = 1
DROP_VIDEO = 2

@proxy_with_channel
def set_video(channel, video, loop=False):
    if False:
        return 10
    '\n    Sets a flag that determines if this channel will attempt to decode video.\n    '
    if video != renpysound.NO_VIDEO and (not video_supported()):
        import sys
        print('Warning: video playback is not supported on this browser', file=sys.stderr)
    call('set_video', channel, video, loop)

def video_supported():
    if False:
        return 10
    return renpy.session['renderer'] in ('gl2', 'gles2')
loaded = False

def load_script():
    if False:
        for i in range(10):
            print('nop')
    '\n    Loads the javascript required for webaudio to work.\n    '
    global loaded
    if not loaded:
        js = renpy.loader.load('_audio.js').read().decode('utf-8')
        emscripten.run_script(js)
    loaded = True

@proxy_call_both
def init(freq, stereo, samples, status=False, equal_mono=False, linear_fades=False):
    if False:
        print('Hello World!')
    '\n    Initializes the audio system with the given parameters. The parameters are\n    just informational - the audio system should be able to play all supported\n    files.\n    '
    load_script()
    return True

@proxy_call_both
def quit():
    if False:
        return 10
    '\n    De-initializes the audio system.\n    '

@proxy_call_both
def periodic():
    if False:
        for i in range(10):
            print('nop')
    '\n    Called periodically (at 20 Hz).\n    '

@proxy_call_both
def advance_time():
    if False:
        print('Hello World!')
    '\n    Called to advance time at the start of a frame.\n    '

@proxy_call_both
def sample_surfaces(rgb, rgba):
    if False:
        return 10
    '\n    Called to provide sample surfaces to the display system. The surfaces\n    returned by read_video should be in the same format as these.\n    '
    return

def can_play_types(types):
    if False:
        print('Hello World!')
    '\n    Webaudio-specific. Returns 1 if the audio system can play all the mime\n    types in the list, 0 if it cannot.\n    '
    load_script()
    return call_int('can_play_types', types)