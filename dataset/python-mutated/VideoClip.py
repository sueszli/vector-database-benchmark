"""Implements VideoClip (base class for video clips) and its main subclasses:

- Animated clips:     VideoFileClip, ImageSequenceClip, BitmapClip
- Static image clips: ImageClip, ColorClip, TextClip,
"""
import copy as _copy
import os
import subprocess as sp
import tempfile
from numbers import Real
import numpy as np
import proglog
from imageio import imread, imsave
from PIL import Image
from moviepy.Clip import Clip
from moviepy.config import IMAGEMAGICK_BINARY
from moviepy.decorators import add_mask_if_none, apply_to_mask, convert_masks_to_RGB, convert_parameter_to_seconds, convert_path_to_string, outplace, requires_duration, requires_fps, use_clip_fps_by_default
from moviepy.tools import cross_platform_popen_params, extensions_dict, find_extension, subprocess_call
from moviepy.video.io.ffmpeg_writer import ffmpeg_write_video
from moviepy.video.io.gif_writers import write_gif, write_gif_with_image_io, write_gif_with_tempfiles
from moviepy.video.tools.drawing import blit

class VideoClip(Clip):
    """Base class for video clips.

    See ``VideoFileClip``, ``ImageClip`` etc. for more user-friendly classes.


    Parameters
    ----------

    is_mask
      `True` if the clip is going to be used as a mask.


    Attributes
    ----------

    size
      The size of the clip, (width,height), in pixels.

    w, h
      The width and height of the clip, in pixels.

    is_mask
      Boolean set to `True` if the clip is a mask.

    make_frame
      A function ``t-> frame at time t`` where ``frame`` is a
      w*h*3 RGB array.

    mask (default None)
      VideoClip mask attached to this clip. If mask is ``None``,
                The video clip is fully opaque.

    audio (default None)
      An AudioClip instance containing the audio of the video clip.

    pos
      A function ``t->(x,y)`` where ``x,y`` is the position
      of the clip when it is composed with other clips.
      See ``VideoClip.set_pos`` for more details

    relative_pos
      See variable ``pos``.

    layer
      Indicates which clip is rendered on top when two clips overlap in
      a CompositeVideoClip. The highest number is rendered on top.
      Default is 0.

    """

    def __init__(self, make_frame=None, is_mask=False, duration=None, has_constant_size=True):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.mask = None
        self.audio = None
        self.pos = lambda t: (0, 0)
        self.relative_pos = False
        self.layer = 0
        if make_frame:
            self.make_frame = make_frame
            self.size = self.get_frame(0).shape[:2][::-1]
        self.is_mask = is_mask
        self.has_constant_size = has_constant_size
        if duration is not None:
            self.duration = duration
            self.end = duration

    @property
    def w(self):
        if False:
            return 10
        'Returns the width of the video.'
        return self.size[0]

    @property
    def h(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the height of the video.'
        return self.size[1]

    @property
    def aspect_ratio(self):
        if False:
            return 10
        'Returns the aspect ratio of the video.'
        return self.w / float(self.h)

    @property
    @requires_duration
    @requires_fps
    def n_frames(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the number of frames of the video.'
        return int(self.duration * self.fps)

    def __copy__(self):
        if False:
            while True:
                i = 10
        "Mixed copy of the clip.\n\n        Returns a shallow copy of the clip whose mask and audio will\n        be shallow copies of the clip's mask and audio if they exist.\n\n        This method is intensively used to produce new clips every time\n        there is an outplace transformation of the clip (clip.resize,\n        clip.subclip, etc.)\n\n        Acts like a deepcopy except for the fact that readers and other\n        possible unpickleables objects are not copied.\n        "
        cls = self.__class__
        new_clip = cls.__new__(cls)
        for attr in self.__dict__:
            value = getattr(self, attr)
            if attr in ('mask', 'audio'):
                value = _copy.copy(value)
            setattr(new_clip, attr, value)
        return new_clip
    copy = __copy__

    @convert_parameter_to_seconds(['t'])
    @convert_masks_to_RGB
    def save_frame(self, filename, t=0, with_mask=True):
        if False:
            i = 10
            return i + 15
        "Save a clip's frame to an image file.\n\n        Saves the frame of clip corresponding to time ``t`` in ``filename``.\n        ``t`` can be expressed in seconds (15.35), in (min, sec),\n        in (hour, min, sec), or as a string: '01:03:05.35'.\n\n        Parameters\n        ----------\n\n        filename : str\n          Name of the file in which the frame will be stored.\n\n        t : float or tuple or str, optional\n          Moment of the frame to be saved. As default, the first frame will be\n          saved.\n\n        with_mask : bool, optional\n          If is ``True`` the mask is saved in the alpha layer of the picture\n          (only works with PNGs).\n        "
        im = self.get_frame(t)
        if with_mask and self.mask is not None:
            mask = 255 * self.mask.get_frame(t)
            im = np.dstack([im, mask]).astype('uint8')
        else:
            im = im.astype('uint8')
        imsave(filename, im)

    @requires_duration
    @use_clip_fps_by_default
    @convert_masks_to_RGB
    @convert_path_to_string(['filename', 'temp_audiofile', 'temp_audiofile_path'])
    def write_videofile(self, filename, fps=None, codec=None, bitrate=None, audio=True, audio_fps=44100, preset='medium', audio_nbytes=4, audio_codec=None, audio_bitrate=None, audio_bufsize=2000, temp_audiofile=None, temp_audiofile_path='', remove_temp=True, write_logfile=False, threads=None, ffmpeg_params=None, logger='bar', pixel_format=None):
        if False:
            return 10
        'Write the clip to a videofile.\n\n        Parameters\n        ----------\n\n        filename\n          Name of the video file to write in, as a string or a path-like object.\n          The extension must correspond to the "codec" used (see below),\n          or simply be \'.avi\' (which will work with any codec).\n\n        fps\n          Number of frames per second in the resulting video file. If None is\n          provided, and the clip has an fps attribute, this fps will be used.\n\n        codec\n          Codec to use for image encoding. Can be any codec supported\n          by ffmpeg. If the filename is has extension \'.mp4\', \'.ogv\', \'.webm\',\n          the codec will be set accordingly, but you can still set it if you\n          don\'t like the default. For other extensions, the output filename\n          must be set accordingly.\n\n          Some examples of codecs are:\n\n          - ``\'libx264\'`` (default codec for file extension ``.mp4``)\n            makes well-compressed videos (quality tunable using \'bitrate\').\n          - ``\'mpeg4\'`` (other codec for extension ``.mp4``) can be an alternative\n            to ``\'libx264\'``, and produces higher quality videos by default.\n          - ``\'rawvideo\'`` (use file extension ``.avi``) will produce\n            a video of perfect quality, of possibly very huge size.\n          - ``png`` (use file extension ``.avi``) will produce a video\n            of perfect quality, of smaller size than with ``rawvideo``.\n          - ``\'libvorbis\'`` (use file extension ``.ogv``) is a nice video\n            format, which is completely free/ open source. However not\n            everyone has the codecs installed by default on their machine.\n          - ``\'libvpx\'`` (use file extension ``.webm``) is tiny a video\n            format well indicated for web videos (with HTML5). Open source.\n\n        audio\n          Either ``True``, ``False``, or a file name.\n          If ``True`` and the clip has an audio clip attached, this\n          audio clip will be incorporated as a soundtrack in the movie.\n          If ``audio`` is the name of an audio file, this audio file\n          will be incorporated as a soundtrack in the movie.\n\n        audio_fps\n          frame rate to use when generating the sound.\n\n        temp_audiofile\n          the name of the temporary audiofile, as a string or path-like object,\n          to be created and then used to write the complete video, if any.\n\n        temp_audiofile_path\n          the location that the temporary audiofile is placed, as a\n          string or path-like object. Defaults to the current working directory.\n\n        audio_codec\n          Which audio codec should be used. Examples are \'libmp3lame\'\n          for \'.mp3\', \'libvorbis\' for \'ogg\', \'libfdk_aac\':\'m4a\',\n          \'pcm_s16le\' for 16-bit wav and \'pcm_s32le\' for 32-bit wav.\n          Default is \'libmp3lame\', unless the video extension is \'ogv\'\n          or \'webm\', at which case the default is \'libvorbis\'.\n\n        audio_bitrate\n          Audio bitrate, given as a string like \'50k\', \'500k\', \'3000k\'.\n          Will determine the size/quality of audio in the output file.\n          Note that it mainly an indicative goal, the bitrate won\'t\n          necessarily be the this in the final file.\n\n        preset\n          Sets the time that FFMPEG will spend optimizing the compression.\n          Choices are: ultrafast, superfast, veryfast, faster, fast, medium,\n          slow, slower, veryslow, placebo. Note that this does not impact\n          the quality of the video, only the size of the video file. So\n          choose ultrafast when you are in a hurry and file size does not\n          matter.\n\n        threads\n          Number of threads to use for ffmpeg. Can speed up the writing of\n          the video on multicore computers.\n\n        ffmpeg_params\n          Any additional ffmpeg parameters you would like to pass, as a list\n          of terms, like [\'-option1\', \'value1\', \'-option2\', \'value2\'].\n\n        write_logfile\n          If true, will write log files for the audio and the video.\n          These will be files ending with \'.log\' with the name of the\n          output file in them.\n\n        logger\n          Either ``"bar"`` for progress bar or ``None`` or any Proglog logger.\n\n        pixel_format\n          Pixel format for the output video file.\n\n        Examples\n        --------\n\n        >>> from moviepy import VideoFileClip\n        >>> clip = VideoFileClip("myvideo.mp4").subclip(100,120)\n        >>> clip.write_videofile("my_new_video.mp4")\n        >>> clip.close()\n\n        '
        (name, ext) = os.path.splitext(os.path.basename(filename))
        ext = ext[1:].lower()
        logger = proglog.default_bar_logger(logger)
        if codec is None:
            try:
                codec = extensions_dict[ext]['codec'][0]
            except KeyError:
                raise ValueError("MoviePy couldn't find the codec associated with the filename. Provide the 'codec' parameter in write_videofile.")
        if audio_codec is None:
            if ext in ['ogv', 'webm']:
                audio_codec = 'libvorbis'
            else:
                audio_codec = 'libmp3lame'
        elif audio_codec == 'raw16':
            audio_codec = 'pcm_s16le'
        elif audio_codec == 'raw32':
            audio_codec = 'pcm_s32le'
        audiofile = audio if isinstance(audio, str) else None
        make_audio = audiofile is None and audio is True and (self.audio is not None)
        if make_audio and temp_audiofile:
            audiofile = temp_audiofile
        elif make_audio:
            audio_ext = find_extension(audio_codec)
            audiofile = os.path.join(temp_audiofile_path, name + Clip._TEMP_FILES_PREFIX + 'wvf_snd.%s' % audio_ext)
        logger(message='MoviePy - Building video %s.' % filename)
        if make_audio:
            self.audio.write_audiofile(audiofile, audio_fps, audio_nbytes, audio_bufsize, audio_codec, bitrate=audio_bitrate, write_logfile=write_logfile, logger=logger)
        ffmpeg_write_video(self, filename, fps, codec, bitrate=bitrate, preset=preset, write_logfile=write_logfile, audiofile=audiofile, threads=threads, ffmpeg_params=ffmpeg_params, logger=logger, pixel_format=pixel_format)
        if remove_temp and make_audio:
            if os.path.exists(audiofile):
                os.remove(audiofile)
        logger(message='MoviePy - video ready %s' % filename)

    @requires_duration
    @use_clip_fps_by_default
    @convert_masks_to_RGB
    def write_images_sequence(self, name_format, fps=None, with_mask=True, logger='bar'):
        if False:
            i = 10
            return i + 15
        'Writes the videoclip to a sequence of image files.\n\n        Parameters\n        ----------\n\n        name_format\n          A filename specifying the numerotation format and extension\n          of the pictures. For instance "frame%03d.png" for filenames\n          indexed with 3 digits and PNG format. Also possible:\n          "some_folder/frame%04d.jpeg", etc.\n\n        fps\n          Number of frames per second to consider when writing the\n          clip. If not specified, the clip\'s ``fps`` attribute will\n          be used if it has one.\n\n        with_mask\n          will save the clip\'s mask (if any) as an alpha canal (PNGs only).\n\n        logger\n          Either ``"bar"`` for progress bar or ``None`` or any Proglog logger.\n\n\n        Returns\n        -------\n\n        names_list\n          A list of all the files generated.\n\n        Notes\n        -----\n\n        The resulting image sequence can be read using e.g. the class\n        ``ImageSequenceClip``.\n\n        '
        logger = proglog.default_bar_logger(logger)
        timings = np.arange(0, self.duration, 1.0 / fps)
        filenames = []
        for (i, t) in logger.iter_bar(t=list(enumerate(timings))):
            name = name_format % i
            filenames.append(name)
            self.save_frame(name, t, with_mask=with_mask)
        return filenames

    @requires_duration
    @convert_masks_to_RGB
    @convert_path_to_string('filename')
    def write_gif(self, filename, fps=None, program='imageio', opt='nq', fuzz=1, loop=0, dispose=False, colors=None, tempfiles=False, logger='bar', pixel_format=None):
        if False:
            i = 10
            return i + 15
        "Write the VideoClip to a GIF file.\n\n        Converts a VideoClip into an animated GIF using ImageMagick\n        or ffmpeg.\n\n        Parameters\n        ----------\n\n        filename\n          Name of the resulting gif file, as a string or a path-like object.\n\n        fps\n          Number of frames per second (see note below). If it\n          isn't provided, then the function will look for the clip's\n          ``fps`` attribute (VideoFileClip, for instance, have one).\n\n        program\n          Software to use for the conversion, either 'imageio' (this will use\n          the library FreeImage through ImageIO), or 'ImageMagick', or 'ffmpeg'.\n\n        opt\n          Optimalization to apply. If program='imageio', opt must be either 'wu'\n          (Wu) or 'nq' (Neuquant). If program='ImageMagick',\n          either 'optimizeplus' or 'OptimizeTransparency'.\n\n        fuzz\n          (ImageMagick only) Compresses the GIF by considering that\n          the colors that are less than fuzz% different are in fact\n          the same.\n\n        tempfiles\n          Writes every frame to a file instead of passing them in the RAM.\n          Useful on computers with little RAM. Can only be used with\n          ImageMagick' or 'ffmpeg'.\n\n        progress_bar\n          If True, displays a progress bar\n\n        pixel_format\n          Pixel format for the output gif file. If is not specified\n          'rgb24' will be used as the default format unless ``clip.mask``\n          exist, then 'rgba' will be used. This option is only going to\n          be accepted if ``program=ffmpeg`` or when ``tempfiles=True``\n\n\n        Notes\n        -----\n\n        The gif will be playing the clip in real time (you can\n        only change the frame rate). If you want the gif to be played\n        slower than the clip you will use ::\n\n            >>> # slow down clip 50% and make it a gif\n            >>> myClip.multiply_speed(0.5).to_gif('myClip.gif')\n\n        "
        if program == 'imageio':
            write_gif_with_image_io(self, filename, fps=fps, opt=opt, loop=loop, colors=colors, logger=logger)
        elif tempfiles:
            opt = 'optimizeplus' if opt == 'nq' else 'OptimizeTransparency'
            write_gif_with_tempfiles(self, filename, fps=fps, program=program, opt=opt, fuzz=fuzz, loop=loop, dispose=dispose, colors=colors, logger=logger, pixel_format=pixel_format)
        else:
            opt = 'optimizeplus' if opt == 'nq' else 'OptimizeTransparency'
            write_gif(self, filename, fps=fps, program=program, opt=opt, fuzz=fuzz, loop=loop, dispose=dispose, colors=colors, logger=logger, pixel_format=pixel_format)

    def subfx(self, fx, start_time=0, end_time=None, **kwargs):
        if False:
            print('Hello World!')
        'Apply a transformation to a part of the clip.\n\n        Returns a new clip in which the function ``fun`` (clip->clip)\n        has been applied to the subclip between times `start_time` and `end_time`\n        (in seconds).\n\n        Examples\n        --------\n\n        >>> # The scene between times t=3s and t=6s in ``clip`` will be\n        >>> # be played twice slower in ``new_clip``\n        >>> new_clip = clip.subapply(lambda c:c.multiply_speed(0.5) , 3,6)\n\n        '
        left = None if start_time == 0 else self.subclip(0, start_time)
        center = self.subclip(start_time, end_time).fx(fx, **kwargs)
        right = None if end_time is None else self.subclip(start_time=end_time)
        clips = [clip for clip in [left, center, right] if clip is not None]
        from moviepy.video.compositing.concatenate import concatenate_videoclips
        return concatenate_videoclips(clips).with_start(self.start)

    def image_transform(self, image_func, apply_to=None):
        if False:
            i = 10
            return i + 15
        'Modifies the images of a clip by replacing the frame `get_frame(t)` by\n        another frame,  `image_func(get_frame(t))`.\n        '
        apply_to = apply_to or []
        return self.transform(lambda get_frame, t: image_func(get_frame(t)), apply_to)

    def fill_array(self, pre_array, shape=(0, 0)):
        if False:
            print('Hello World!')
        'TODO: needs documentation.'
        pre_shape = pre_array.shape
        dx = shape[0] - pre_shape[0]
        dy = shape[1] - pre_shape[1]
        post_array = pre_array
        if dx < 0:
            post_array = pre_array[:shape[0]]
        elif dx > 0:
            x_1 = [[[1, 1, 1]] * pre_shape[1]] * dx
            post_array = np.vstack((pre_array, x_1))
        if dy < 0:
            post_array = post_array[:, :shape[1]]
        elif dy > 0:
            x_1 = [[[1, 1, 1]] * dy] * post_array.shape[0]
            post_array = np.hstack((post_array, x_1))
        return post_array

    def blit_on(self, picture, t):
        if False:
            print('Hello World!')
        "Returns the result of the blit of the clip's frame at time `t`\n        on the given `picture`, the position of the clip being given\n        by the clip's ``pos`` attribute. Meant for compositing.\n        "
        (wf, hf) = picture.size
        ct = t - self.start
        img = self.get_frame(ct).astype('uint8')
        im_img = Image.fromarray(img)
        if self.mask is not None:
            mask = (self.mask.get_frame(ct) * 255).astype('uint8')
            im_mask = Image.fromarray(mask).convert('L')
            if im_img.size != im_mask.size:
                bg_size = (max(im_img.size[0], im_mask.size[0]), max(im_img.size[1], im_mask.size[1]))
                im_img_bg = Image.new('RGB', bg_size, 'black')
                im_img_bg.paste(im_img, (0, 0))
                im_mask_bg = Image.new('L', bg_size, 0)
                im_mask_bg.paste(im_mask, (0, 0))
                (im_img, im_mask) = (im_img_bg, im_mask_bg)
        else:
            im_mask = None
        (wi, hi) = im_img.size
        pos = self.pos(ct)
        if isinstance(pos, str):
            pos = {'center': ['center', 'center'], 'left': ['left', 'center'], 'right': ['right', 'center'], 'top': ['center', 'top'], 'bottom': ['center', 'bottom']}[pos]
        else:
            pos = list(pos)
        if self.relative_pos:
            for (i, dim) in enumerate([wf, hf]):
                if not isinstance(pos[i], str):
                    pos[i] = dim * pos[i]
        if isinstance(pos[0], str):
            D = {'left': 0, 'center': (wf - wi) / 2, 'right': wf - wi}
            pos[0] = D[pos[0]]
        if isinstance(pos[1], str):
            D = {'top': 0, 'center': (hf - hi) / 2, 'bottom': hf - hi}
            pos[1] = D[pos[1]]
        pos = map(int, pos)
        return blit(im_img, picture, pos, mask=im_mask)

    def add_mask(self):
        if False:
            print('Hello World!')
        'Add a mask VideoClip to the VideoClip.\n\n        Returns a copy of the clip with a completely opaque mask\n        (made of ones). This makes computations slower compared to\n        having a None mask but can be useful in many cases. Choose\n\n        Set ``constant_size`` to  `False` for clips with moving\n        image size.\n        '
        if self.has_constant_size:
            mask = ColorClip(self.size, 1.0, is_mask=True)
            return self.with_mask(mask.with_duration(self.duration))
        else:

            def make_frame(t):
                if False:
                    print('Hello World!')
                return np.ones(self.get_frame(t).shape[:2], dtype=float)
            mask = VideoClip(is_mask=True, make_frame=make_frame)
            return self.with_mask(mask.with_duration(self.duration))

    def on_color(self, size=None, color=(0, 0, 0), pos=None, col_opacity=None):
        if False:
            for i in range(10):
                print('nop')
        "Place the clip on a colored background.\n\n        Returns a clip made of the current clip overlaid on a color\n        clip of a possibly bigger size. Can serve to flatten transparent\n        clips.\n\n        Parameters\n        ----------\n\n        size\n          Size (width, height) in pixels of the final clip.\n          By default it will be the size of the current clip.\n\n        color\n          Background color of the final clip ([R,G,B]).\n\n        pos\n          Position of the clip in the final clip. 'center' is the default\n\n        col_opacity\n          Parameter in 0..1 indicating the opacity of the colored\n          background.\n        "
        from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
        if size is None:
            size = self.size
        if pos is None:
            pos = 'center'
        if col_opacity is not None:
            colorclip = ColorClip(size, color=color, duration=self.duration).with_opacity(col_opacity)
            result = CompositeVideoClip([colorclip, self.with_position(pos)])
        else:
            result = CompositeVideoClip([self.with_position(pos)], size=size, bg_color=color)
        if isinstance(self, ImageClip) and (not hasattr(pos, '__call__')) and (self.mask is None or isinstance(self.mask, ImageClip)):
            new_result = result.to_ImageClip()
            if result.mask is not None:
                new_result.mask = result.mask.to_ImageClip()
            return new_result.with_duration(result.duration)
        return result

    @outplace
    def with_make_frame(self, mf):
        if False:
            return 10
        "Change the clip's ``get_frame``.\n\n        Returns a copy of the VideoClip instance, with the make_frame\n        attribute set to `mf`.\n        "
        self.make_frame = mf
        self.size = self.get_frame(0).shape[:2][::-1]

    @outplace
    def with_audio(self, audioclip):
        if False:
            i = 10
            return i + 15
        'Attach an AudioClip to the VideoClip.\n\n        Returns a copy of the VideoClip instance, with the `audio`\n        attribute set to ``audio``, which must be an AudioClip instance.\n        '
        self.audio = audioclip

    @outplace
    def with_mask(self, mask):
        if False:
            for i in range(10):
                print('nop')
        "Set the clip's mask.\n\n        Returns a copy of the VideoClip with the mask attribute set to\n        ``mask``, which must be a greyscale (values in 0-1) VideoClip.\n        "
        assert mask is None or mask.is_mask
        self.mask = mask

    @add_mask_if_none
    @outplace
    def with_opacity(self, opacity):
        if False:
            while True:
                i = 10
        'Set the opacity/transparency level of the clip.\n\n        Returns a semi-transparent copy of the clip where the mask is\n        multiplied by ``op`` (any float, normally between 0 and 1).\n        '
        self.mask = self.mask.image_transform(lambda pic: opacity * pic)

    @apply_to_mask
    @outplace
    def with_position(self, pos, relative=False):
        if False:
            i = 10
            return i + 15
        'Set the clip\'s position in compositions.\n\n        Sets the position that the clip will have when included\n        in compositions. The argument ``pos`` can be either a couple\n        ``(x,y)`` or a function ``t-> (x,y)``. `x` and `y` mark the\n        location of the top left corner of the clip, and can be\n        of several types.\n\n        Examples\n        --------\n\n        >>> clip.with_position((45,150)) # x=45, y=150\n        >>>\n        >>> # clip horizontally centered, at the top of the picture\n        >>> clip.with_position(("center","top"))\n        >>>\n        >>> # clip is at 40% of the width, 70% of the height:\n        >>> clip.with_position((0.4,0.7), relative=True)\n        >>>\n        >>> # clip\'s position is horizontally centered, and moving up !\n        >>> clip.with_position(lambda t: (\'center\', 50+t) )\n\n        '
        self.relative_pos = relative
        if hasattr(pos, '__call__'):
            self.pos = pos
        else:
            self.pos = lambda t: pos

    @apply_to_mask
    @outplace
    def with_layer(self, layer):
        if False:
            while True:
                i = 10
        "Set the clip's layer in compositions. Clips with a greater ``layer``\n        attribute will be displayed on top of others.\n\n        Note: Only has effect when the clip is used in a CompositeVideoClip.\n        "
        self.layer = layer

    @convert_parameter_to_seconds(['t'])
    def to_ImageClip(self, t=0, with_mask=True, duration=None):
        if False:
            print('Hello World!')
        "\n        Returns an ImageClip made out of the clip's frame at time ``t``,\n        which can be expressed in seconds (15.35), in (min, sec),\n        in (hour, min, sec), or as a string: '01:03:05.35'.\n        "
        new_clip = ImageClip(self.get_frame(t), is_mask=self.is_mask, duration=duration)
        if with_mask and self.mask is not None:
            new_clip.mask = self.mask.to_ImageClip(t)
        return new_clip

    def to_mask(self, canal=0):
        if False:
            return 10
        'Return a mask a video clip made from the clip.'
        if self.is_mask:
            return self
        else:
            new_clip = self.image_transform(lambda pic: 1.0 * pic[:, :, canal] / 255)
            new_clip.is_mask = True
            return new_clip

    def to_RGB(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a non-mask video clip made from the mask video clip.'
        if self.is_mask:
            new_clip = self.image_transform(lambda pic: np.dstack(3 * [255 * pic]).astype('uint8'))
            new_clip.is_mask = False
            return new_clip
        else:
            return self

    @outplace
    def without_audio(self):
        if False:
            i = 10
            return i + 15
        "Remove the clip's audio.\n\n        Return a copy of the clip with audio set to None.\n        "
        self.audio = None

    @outplace
    def afx(self, fun, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        "Transform the clip's audio.\n\n        Return a new clip whose audio has been transformed by ``fun``.\n        "
        self.audio = self.audio.fx(fun, *args, **kwargs)

    def __add__(self, other):
        if False:
            return 10
        if isinstance(other, VideoClip):
            from moviepy.video.compositing.concatenate import concatenate_videoclips
            method = 'chain' if self.size == other.size else 'compose'
            return concatenate_videoclips([self, other], method=method)
        return super(VideoClip, self).__add__(other)

    def __or__(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        Implement the or (self | other) to produce a video with self and other\n        placed side by side horizontally.\n        '
        if isinstance(other, VideoClip):
            from moviepy.video.compositing.CompositeVideoClip import clips_array
            return clips_array([[self, other]])
        return super(VideoClip, self).__or__(other)

    def __truediv__(self, other):
        if False:
            print('Hello World!')
        '\n        Implement division (self / other) to produce a video with self\n        placed on top of other.\n        '
        if isinstance(other, VideoClip):
            from moviepy.video.compositing.CompositeVideoClip import clips_array
            return clips_array([[self], [other]])
        return super(VideoClip, self).__or__(other)

    def __matmul__(self, n):
        if False:
            print('Hello World!')
        if not isinstance(n, Real):
            return NotImplemented
        from moviepy.video.fx.rotate import rotate
        return rotate(self, n)

    def __and__(self, mask):
        if False:
            for i in range(10):
                print('nop')
        return self.with_mask(mask)

class DataVideoClip(VideoClip):
    """
    Class of video clips whose successive frames are functions
    of successive datasets

    Parameters
    ----------
    data
      A list of datasets, each dataset being used for one frame of the clip

    data_to_frame
      A function d -> video frame, where d is one element of the list `data`

    fps
      Number of frames per second in the animation
    """

    def __init__(self, data, data_to_frame, fps, is_mask=False, has_constant_size=True):
        if False:
            while True:
                i = 10
        self.data = data
        self.data_to_frame = data_to_frame
        self.fps = fps

        def make_frame(t):
            if False:
                return 10
            return self.data_to_frame(self.data[int(self.fps * t)])
        VideoClip.__init__(self, make_frame, is_mask=is_mask, duration=1.0 * len(data) / fps, has_constant_size=has_constant_size)

class UpdatedVideoClip(VideoClip):
    """
    Class of clips whose make_frame requires some objects to
    be updated. Particularly practical in science where some
    algorithm needs to make some steps before a new frame can
    be generated.

    UpdatedVideoClips have the following make_frame:

    >>> def make_frame(t):
    >>>     while self.world.clip_t < t:
    >>>         world.update() # updates, and increases world.clip_t
    >>>     return world.to_frame()

    Parameters
    ----------

    world
      An object with the following attributes:
      - world.clip_t: the clip's time corresponding to the world's state.
      - world.update() : update the world's state, (including increasing
      world.clip_t of one time step).
      - world.to_frame() : renders a frame depending on the world's state.

    is_mask
      True if the clip is a WxH mask with values in 0-1

    duration
      Duration of the clip, in seconds

    """

    def __init__(self, world, is_mask=False, duration=None):
        if False:
            i = 10
            return i + 15
        self.world = world

        def make_frame(t):
            if False:
                while True:
                    i = 10
            while self.world.clip_t < t:
                world.update()
            return world.to_frame()
        VideoClip.__init__(self, make_frame=make_frame, is_mask=is_mask, duration=duration)
"---------------------------------------------------------------------\n\n    ImageClip (base class for all 'static clips') and its subclasses\n    ColorClip and TextClip.\n    I would have liked to put these in a separate file but Python is bad\n    at cyclic imports.\n\n---------------------------------------------------------------------"

class ImageClip(VideoClip):
    """Class for non-moving VideoClips.

    A video clip originating from a picture. This clip will simply
    display the given picture at all times.

    Examples
    --------

    >>> clip = ImageClip("myHouse.jpeg")
    >>> clip = ImageClip( someArray ) # a Numpy array represent

    Parameters
    ----------

    img
      Any picture file (png, tiff, jpeg, etc.) as a string or a path-like object,
      or any array representing an RGB image (for instance a frame from a VideoClip).

    is_mask
      Set this parameter to `True` if the clip is a mask.

    transparent
      Set this parameter to `True` (default) if you want the alpha layer
      of the picture (if it exists) to be used as a mask.

    Attributes
    ----------

    img
      Array representing the image of the clip.

    """

    def __init__(self, img, is_mask=False, transparent=True, fromalpha=False, duration=None):
        if False:
            for i in range(10):
                print('nop')
        VideoClip.__init__(self, is_mask=is_mask, duration=duration)
        if not isinstance(img, np.ndarray):
            img = imread(img)
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                if fromalpha:
                    img = 1.0 * img[:, :, 3] / 255
                elif is_mask:
                    img = 1.0 * img[:, :, 0] / 255
                elif transparent:
                    self.mask = ImageClip(1.0 * img[:, :, 3] / 255, is_mask=True)
                    img = img[:, :, :3]
            elif is_mask:
                img = 1.0 * img[:, :, 0] / 255
        self.make_frame = lambda t: img
        self.size = img.shape[:2][::-1]
        self.img = img

    def transform(self, func, apply_to=None, keep_duration=True):
        if False:
            return 10
        'General transformation filter.\n\n        Equivalent to VideoClip.transform. The result is no more an\n        ImageClip, it has the class VideoClip (since it may be animated)\n        '
        if apply_to is None:
            apply_to = []
        new_clip = VideoClip.transform(self, func, apply_to=apply_to, keep_duration=keep_duration)
        new_clip.__class__ = VideoClip
        return new_clip

    @outplace
    def image_transform(self, image_func, apply_to=None):
        if False:
            return 10
        "Image-transformation filter.\n\n        Does the same as VideoClip.image_transform, but for ImageClip the\n        transformed clip is computed once and for all at the beginning,\n        and not for each 'frame'.\n        "
        if apply_to is None:
            apply_to = []
        arr = image_func(self.get_frame(0))
        self.size = arr.shape[:2][::-1]
        self.make_frame = lambda t: arr
        self.img = arr
        for attr in apply_to:
            a = getattr(self, attr, None)
            if a is not None:
                new_a = a.image_transform(image_func)
                setattr(self, attr, new_a)

    @outplace
    def time_transform(self, time_func, apply_to=None, keep_duration=False):
        if False:
            i = 10
            return i + 15
        "Time-transformation filter.\n\n        Applies a transformation to the clip's timeline\n        (see Clip.time_transform).\n\n        This method does nothing for ImageClips (but it may affect their\n        masks or their audios). The result is still an ImageClip.\n        "
        if apply_to is None:
            apply_to = ['mask', 'audio']
        for attr in apply_to:
            a = getattr(self, attr, None)
            if a is not None:
                new_a = a.time_transform(time_func)
                setattr(self, attr, new_a)

class ColorClip(ImageClip):
    """An ImageClip showing just one color.

    Parameters
    ----------

    size
      Size (width, height) in pixels of the clip.

    color
      If argument ``is_mask`` is False, ``color`` indicates
      the color in RGB of the clip (default is black). If `is_mask``
      is True, ``color`` must be  a float between 0 and 1 (default is 1)

    is_mask
      Set to true if the clip will be used as a mask.

    """

    def __init__(self, size, color=None, is_mask=False, duration=None):
        if False:
            i = 10
            return i + 15
        (w, h) = size
        if is_mask:
            shape = (h, w)
            if color is None:
                color = 0
            elif not np.isscalar(color):
                raise Exception('Color has to be a scalar when mask is true')
        else:
            if color is None:
                color = (0, 0, 0)
            elif not hasattr(color, '__getitem__'):
                raise Exception('Color has to contain RGB of the clip')
            elif isinstance(color, str):
                raise Exception('Color cannot be string. Color has to contain RGB of the clip')
            shape = (h, w, len(color))
        super().__init__(np.tile(color, w * h).reshape(shape), is_mask=is_mask, duration=duration)

class TextClip(ImageClip):
    """Class for autogenerated text clips.

    Creates an ImageClip originating from a script-generated text image.
    Requires ImageMagick.

    Parameters
    ----------

    text
      A string of the text to write. Can be replaced by argument
      ``filename``.

    filename
      The name of a file in which there is the text to write,
      as a string or a path-like object.
      Can be provided instead of argument ``txt``

    size
      Size of the picture in pixels. Can be auto-set if
      method='label', but mandatory if method='caption'.
      the height can be None, it will then be auto-determined.

    bg_color
      Color of the background. See ``TextClip.list('color')``
      for a list of acceptable names.

    color
      Color of the text. See ``TextClip.list('color')`` for a
      list of acceptable names.

    font
      Name of the font to use. See ``TextClip.list('font')`` for
      the list of fonts you can use on your computer.

    stroke_color
      Color of the stroke (=contour line) of the text. If ``None``,
      there will be no stroke.

    stroke_width
      Width of the stroke, in pixels. Can be a float, like 1.5.

    method
      Either 'label' (default, the picture will be autosized so as to fit
      exactly the size) or 'caption' (the text will be drawn in a picture
      with fixed size provided with the ``size`` argument). If `caption`,
      the text will be wrapped automagically (sometimes it is buggy, not
      my fault, complain to the ImageMagick crew) and can be aligned or
      centered (see parameter ``align``).

    kerning
      Changes the default spacing between letters. For
      instance ``kerning=-1`` will make the letters 1 pixel nearer from
      ach other compared to the default spacing.

    align
      center | East | West | South | North . Will only work if ``method``
      is set to ``caption``

    transparent
      ``True`` (default) if you want to take into account the
      transparency in the image.
    """

    @convert_path_to_string('filename')
    def __init__(self, text=None, filename=None, size=None, color='black', bg_color='transparent', font_size=None, font='Courier', stroke_color=None, stroke_width=1, method='label', kerning=None, align='center', interline=None, tempfilename=None, temptxt=None, transparent=True, remove_temp=True, print_cmd=False):
        if False:
            return 10
        if text is not None:
            if temptxt is None:
                (temptxt_fd, temptxt) = tempfile.mkstemp(suffix='.txt')
                try:
                    os.write(temptxt_fd, bytes(text, 'UTF8'))
                except TypeError:
                    os.write(temptxt_fd, text)
                os.close(temptxt_fd)
            text = '@' + temptxt
        elif filename is not None:
            text = '@' + filename
        else:
            raise ValueError("You must provide either 'text' or 'filename' arguments to TextClip")
        if size is not None:
            size = ('' if size[0] is None else str(size[0]), '' if size[1] is None else str(size[1]))
        cmd = [IMAGEMAGICK_BINARY, '-background', bg_color, '-fill', color, '-font', font]
        if font_size is not None:
            cmd += ['-pointsize', '%d' % font_size]
        if kerning is not None:
            cmd += ['-kerning', '%0.1f' % kerning]
        if stroke_color is not None:
            cmd += ['-stroke', stroke_color, '-strokewidth', '%.01f' % stroke_width]
        if size is not None:
            cmd += ['-size', '%sx%s' % (size[0], size[1])]
        if align is not None:
            cmd += ['-gravity', align]
        if interline is not None:
            cmd += ['-interline-spacing', '%d' % interline]
        if tempfilename is None:
            (tempfile_fd, tempfilename) = tempfile.mkstemp(suffix='.png')
            os.close(tempfile_fd)
        cmd += ['%s:%s' % (method, text), '-type', 'truecolormatte', 'PNG32:%s' % tempfilename]
        if print_cmd:
            print(' '.join(cmd))
        try:
            subprocess_call(cmd, logger=None)
        except (IOError, OSError) as err:
            error = f"MoviePy Error: creation of {filename} failed because of the following error:\n\n{err}.\n\n.This error can be due to the fact that ImageMagick is not installed on your computer, or (for Windows users) that you didn't specify the path to the ImageMagick binary. Check the documentation."
            raise IOError(error)
        ImageClip.__init__(self, tempfilename, transparent=transparent)
        self.text = text
        self.color = color
        self.stroke_color = stroke_color
        if remove_temp:
            if tempfilename is not None and os.path.exists(tempfilename):
                os.remove(tempfilename)
            if temptxt is not None and os.path.exists(temptxt):
                os.remove(temptxt)

    @staticmethod
    def list(arg):
        if False:
            for i in range(10):
                print('nop')
        'Returns a list of all valid entries for the ``font`` or ``color`` argument of\n        ``TextClip``.\n        '
        popen_params = cross_platform_popen_params({'stdout': sp.PIPE, 'stderr': sp.DEVNULL, 'stdin': sp.DEVNULL})
        process = sp.Popen([IMAGEMAGICK_BINARY, '-list', arg], encoding='utf-8', **popen_params)
        result = process.communicate()[0]
        lines = result.splitlines()
        if arg == 'font':
            return [line[8:] for line in lines if line.startswith('  Font:')]
        elif arg == 'color':
            return [line.split(' ')[0] for line in lines[5:]]
        else:
            raise Exception("MoviePy Error: Argument must equal 'font' or 'color'")

    @staticmethod
    def search(string, arg):
        if False:
            for i in range(10):
                print('nop')
        'Returns the of all valid entries which contain ``string`` for the\n        argument ``arg`` of ``TextClip``, for instance\n\n        >>> # Find all the available fonts which contain "Courier"\n        >>> print(TextClip.search(\'Courier\', \'font\'))\n        '
        string = string.lower()
        names_list = TextClip.list(arg)
        return [name for name in names_list if string in name.lower()]

class BitmapClip(VideoClip):
    """Clip made of color bitmaps. Mainly designed for testing purposes."""
    DEFAULT_COLOR_DICT = {'R': (255, 0, 0), 'G': (0, 255, 0), 'B': (0, 0, 255), 'O': (0, 0, 0), 'W': (255, 255, 255), 'A': (89, 225, 62), 'C': (113, 157, 108), 'D': (215, 182, 143), 'E': (57, 26, 252), 'F': (225, 135, 33)}

    @convert_parameter_to_seconds(['duration'])
    def __init__(self, bitmap_frames, *, fps=None, duration=None, color_dict=None, is_mask=False):
        if False:
            return 10
        'Creates a VideoClip object from a bitmap representation. Primarily used\n        in the test suite.\n\n        Parameters\n        ----------\n\n        bitmap_frames\n          A list of frames. Each frame is a list of strings. Each string\n          represents a row of colors. Each color represents an (r, g, b) tuple.\n          Example input (2 frames, 5x3 pixel size)::\n\n              [["RRRRR",\n                "RRBRR",\n                "RRBRR"],\n               ["RGGGR",\n                "RGGGR",\n                "RGGGR"]]\n\n        fps\n          The number of frames per second to display the clip at. `duration` will\n          calculated from the total number of frames. If both `fps` and `duration`\n          are set, `duration` will be ignored.\n\n        duration\n          The total duration of the clip. `fps` will be calculated from the total\n          number of frames. If both `fps` and `duration` are set, `duration` will\n          be ignored.\n\n        color_dict\n          A dictionary that can be used to set specific (r, g, b) values that\n          correspond to the letters used in ``bitmap_frames``.\n          eg ``{"A": (50, 150, 150)}``.\n\n          Defaults to::\n\n              {\n                "R": (255, 0, 0),\n                "G": (0, 255, 0),\n                "B": (0, 0, 255),\n                "O": (0, 0, 0),  # "O" represents black\n                "W": (255, 255, 255),\n                # "A", "C", "D", "E", "F" represent arbitrary colors\n                "A": (89, 225, 62),\n                "C": (113, 157, 108),\n                "D": (215, 182, 143),\n                "E": (57, 26, 252),\n              }\n\n        is_mask\n          Set to ``True`` if the clip is going to be used as a mask.\n        '
        assert fps is not None or duration is not None
        self.color_dict = color_dict if color_dict else self.DEFAULT_COLOR_DICT
        frame_list = []
        for input_frame in bitmap_frames:
            output_frame = []
            for row in input_frame:
                output_frame.append([self.color_dict[color] for color in row])
            frame_list.append(np.array(output_frame))
        frame_array = np.array(frame_list)
        self.total_frames = len(frame_array)
        if fps is None:
            fps = self.total_frames / duration
        else:
            duration = self.total_frames / fps
        VideoClip.__init__(self, make_frame=lambda t: frame_array[int(t * fps)], is_mask=is_mask, duration=duration)
        self.fps = fps

    def to_bitmap(self, color_dict=None):
        if False:
            while True:
                i = 10
        'Returns a valid bitmap list that represents each frame of the clip.\n        If `color_dict` is not specified, then it will use the same `color_dict`\n        that was used to create the clip.\n        '
        color_dict = color_dict or self.color_dict
        bitmap = []
        for frame in self.iter_frames():
            bitmap.append([])
            for line in frame:
                bitmap[-1].append('')
                for pixel in line:
                    letter = list(color_dict.keys())[list(color_dict.values()).index(tuple(pixel))]
                    bitmap[-1][-1] += letter
        return bitmap