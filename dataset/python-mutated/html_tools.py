"""Implements ``ipython_display``, a function to embed images/videos/audio in the
IPython Notebook.
"""
import inspect
import os
from base64 import b64encode
from moviepy.audio.AudioClip import AudioClip
from moviepy.tools import extensions_dict
from moviepy.video.io.ffmpeg_reader import ffmpeg_parse_infos
from moviepy.video.VideoClip import ImageClip, VideoClip
try:
    from IPython.display import HTML
    ipython_available = True

    class HTML2(HTML):

        def __add__(self, other):
            if False:
                return 10
            return HTML2(self.data + other.data)
except ImportError:

    def HTML2(content):
        if False:
            while True:
                i = 10
        return content
    ipython_available = False
sorry = "Sorry, seems like your browser doesn't support HTML5 audio/video"
templates = {'audio': "<audio controls><source %(options)s  src='data:audio/%(ext)s;base64,%(data)s'>" + sorry + '</audio>', 'image': "<img %(options)s src='data:image/%(ext)s;base64,%(data)s'>", 'video': "<video %(options)ssrc='data:video/%(ext)s;base64,%(data)s' controls>" + sorry + '</video>'}

def html_embed(clip, filetype=None, maxduration=60, rd_kwargs=None, center=True, **html_kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Returns HTML5 code embedding the clip.\n\n    Parameters\n    ----------\n\n    clip : moviepy.Clip.Clip\n      Either a file name, or a clip to preview.\n      Either an image, a sound or a video. Clips will actually be\n      written to a file and embedded as if a filename was provided.\n\n    filetype : str, optional\n      One of \'video\',\'image\',\'audio\'. If None is given, it is determined\n      based on the extension of ``filename``, but this can bug.\n\n    maxduration : float, optional\n      An error will be raised if the clip\'s duration is more than the indicated\n      value (in seconds), to avoid spoiling the browser\'s cache and the RAM.\n\n    rd_kwargs : dict, optional\n      Keyword arguments for the rendering, like ``dict(fps=15, bitrate="50k")``.\n      Allow you to give some options to the render process. You can, for\n      example, disable the logger bar passing ``dict(logger=None)``.\n\n    center : bool, optional\n      If true (default), the content will be wrapped in a\n      ``<div align=middle>`` HTML container, so the content will be displayed\n      at the center.\n\n    html_kwargs\n      Allow you to give some options, like ``width=260``, ``autoplay=True``,\n      ``loop=1`` etc.\n\n    Examples\n    --------\n\n    >>> from moviepy.editor import *\n    >>> # later ...\n    >>> html_embed(clip, width=360)\n    >>> html_embed(clip.audio)\n\n    >>> clip.write_gif("test.gif")\n    >>> html_embed(\'test.gif\')\n\n    >>> clip.save_frame("first_frame.jpeg")\n    >>> html_embed("first_frame.jpeg")\n    '
    if rd_kwargs is None:
        rd_kwargs = {}
    if 'Clip' in str(clip.__class__):
        TEMP_PREFIX = '__temp__'
        if isinstance(clip, ImageClip):
            filename = TEMP_PREFIX + '.png'
            kwargs = {'filename': filename, 'with_mask': True}
            argnames = inspect.getfullargspec(clip.save_frame).args
            kwargs.update({key: value for (key, value) in rd_kwargs.items() if key in argnames})
            clip.save_frame(**kwargs)
        elif isinstance(clip, VideoClip):
            filename = TEMP_PREFIX + '.mp4'
            kwargs = {'filename': filename, 'preset': 'ultrafast'}
            kwargs.update(rd_kwargs)
            clip.write_videofile(**kwargs)
        elif isinstance(clip, AudioClip):
            filename = TEMP_PREFIX + '.mp3'
            kwargs = {'filename': filename}
            kwargs.update(rd_kwargs)
            clip.write_audiofile(**kwargs)
        else:
            raise ValueError('Unknown class for the clip. Cannot embed and preview.')
        return html_embed(filename, maxduration=maxduration, rd_kwargs=rd_kwargs, center=center, **html_kwargs)
    filename = clip
    options = ' '.join(["%s='%s'" % (str(k), str(v)) for (k, v) in html_kwargs.items()])
    (name, ext) = os.path.splitext(filename)
    ext = ext[1:]
    if filetype is None:
        ext = filename.split('.')[-1].lower()
        if ext == 'gif':
            filetype = 'image'
        elif ext in extensions_dict:
            filetype = extensions_dict[ext]['type']
        else:
            raise ValueError("No file type is known for the provided file. Please provide argument `filetype` (one of 'image', 'video', 'sound') to the ipython display function.")
    if filetype == 'video':
        exts_htmltype = {'mp4': 'mp4', 'webm': 'webm', 'ogv': 'ogg'}
        allowed_exts = ' '.join(exts_htmltype.keys())
        try:
            ext = exts_htmltype[ext]
        except Exception:
            raise ValueError('This video extension cannot be displayed in the IPython Notebook. Allowed extensions: ' + allowed_exts)
    if filetype in ['audio', 'video']:
        duration = ffmpeg_parse_infos(filename, decode_file=True)['duration']
        if duration > maxduration:
            raise ValueError("The duration of video %s (%.1f) exceeds the 'maxduration' attribute. You can increase 'maxduration', by passing 'maxduration' parameter to ipython_display function. But note that embedding large videos may take all the memory away!" % (filename, duration))
    with open(filename, 'rb') as file:
        data = b64encode(file.read()).decode('utf-8')
    template = templates[filetype]
    result = template % {'data': data, 'options': options, 'ext': ext}
    if center:
        result = '<div align=middle>%s</div>' % result
    return result

def ipython_display(clip, filetype=None, maxduration=60, t=None, fps=None, rd_kwargs=None, center=True, **html_kwargs):
    if False:
        print('Hello World!')
    'Displays clip content in an IPython Notebook.\n\n    Remarks: If your browser doesn\'t support HTML5, this should warn you.\n    If nothing is displayed, maybe your file or filename is wrong.\n    Important: The media will be physically embedded in the notebook.\n\n    Parameters\n    ----------\n\n    clip : moviepy.Clip.Clip\n      Either the name of a file, or a clip to preview. The clip will actually\n      be written to a file and embedded as if a filename was provided.\n\n    filetype : str, optional\n      One of ``"video"``, ``"image"`` or ``"audio"``. If None is given, it is\n      determined based on the extension of ``filename``, but this can bug.\n\n    maxduration : float, optional\n      An error will be raised if the clip\'s duration is more than the indicated\n      value (in seconds), to avoid spoiling the browser\'s cache and the RAM.\n\n    t : float, optional\n      If not None, only the frame at time t will be displayed in the notebook,\n      instead of a video of the clip.\n\n    fps : int, optional\n      Enables to specify an fps, as required for clips whose fps is unknown.\n\n    rd_kwargs : dict, optional\n      Keyword arguments for the rendering, like ``dict(fps=15, bitrate="50k")``.\n      Allow you to give some options to the render process. You can, for\n      example, disable the logger bar passing ``dict(logger=None)``.\n\n    center : bool, optional\n      If true (default), the content will be wrapped in a\n      ``<div align=middle>`` HTML container, so the content will be displayed\n      at the center.\n\n    kwargs\n      Allow you to give some options, like ``width=260``, etc. When editing\n      looping gifs, a good choice is ``loop=1, autoplay=1``.\n\n    Examples\n    --------\n\n    >>> from moviepy.editor import *\n    >>> # later ...\n    >>> clip.ipython_display(width=360)\n    >>> clip.audio.ipython_display()\n\n    >>> clip.write_gif("test.gif")\n    >>> ipython_display(\'test.gif\')\n\n    >>> clip.save_frame("first_frame.jpeg")\n    >>> ipython_display("first_frame.jpeg")\n    '
    if not ipython_available:
        raise ImportError('Only works inside an IPython Notebook')
    if rd_kwargs is None:
        rd_kwargs = {}
    if fps is not None:
        rd_kwargs['fps'] = fps
    if t is not None:
        clip = clip.to_ImageClip(t)
    return HTML2(html_embed(clip, filetype=filetype, maxduration=maxduration, center=center, rd_kwargs=rd_kwargs, **html_kwargs))