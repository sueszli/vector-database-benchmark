"""Implements the central object of MoviePy, the Clip, and all the methods that
are common to the two subclasses of Clip, VideoClip and AudioClip.
"""
import copy as _copy
from functools import reduce
from numbers import Real
from operator import add
import numpy as np
import proglog
from moviepy.decorators import apply_to_audio, apply_to_mask, convert_parameter_to_seconds, outplace, requires_duration, use_clip_fps_by_default

class Clip:
    """Base class of all clips (VideoClips and AudioClips).

    Attributes
    ----------

    start : float
      When the clip is included in a composition, time of the
      composition at which the clip starts playing (in seconds).

    end : float
      When the clip is included in a composition, time of the
      composition at which the clip stops playing (in seconds).

    duration : float
      Duration of the clip (in seconds). Some clips are infinite, in
      this case their duration will be ``None``.
    """
    _TEMP_FILES_PREFIX = 'TEMP_MPY_'

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.start = 0
        self.end = None
        self.duration = None
        self.memoize = False
        self.memoized_t = None
        self.memoized_frame = None

    def copy(self):
        if False:
            print('Hello World!')
        'Allows the usage of ``.copy()`` in clips as chained methods invocation.'
        return _copy.copy(self)

    @convert_parameter_to_seconds(['t'])
    def get_frame(self, t):
        if False:
            i = 10
            return i + 15
        'Gets a numpy array representing the RGB picture of the clip,\n        or (mono or stereo) value for a sound clip, at time ``t``.\n\n        Parameters\n        ----------\n\n        t : float or tuple or str\n          Moment of the clip whose frame will be returned.\n        '
        if self.memoize:
            if t == self.memoized_t:
                return self.memoized_frame
            else:
                frame = self.make_frame(t)
                self.memoized_t = t
                self.memoized_frame = frame
                return frame
        else:
            return self.make_frame(t)

    def transform(self, func, apply_to=None, keep_duration=True):
        if False:
            for i in range(10):
                print('nop')
        'General processing of a clip.\n\n        Returns a new Clip whose frames are a transformation\n        (through function ``func``) of the frames of the current clip.\n\n        Parameters\n        ----------\n\n        func : function\n          A function with signature (gf,t -> frame) where ``gf`` will\n          represent the current clip\'s ``get_frame`` method,\n          i.e. ``gf`` is a function (t->image). Parameter `t` is a time\n          in seconds, `frame` is a picture (=Numpy array) which will be\n          returned by the transformed clip (see examples below).\n\n        apply_to : {"mask", "audio", ["mask", "audio"]}, optional\n          Can be either ``\'mask\'``, or ``\'audio\'``, or\n          ``[\'mask\',\'audio\']``.\n          Specifies if the filter should also be applied to the\n          audio or the mask of the clip, if any.\n\n        keep_duration : bool, optional\n          Set to True if the transformation does not change the\n          ``duration`` of the clip.\n\n        Examples\n        --------\n\n        In the following ``new_clip`` a 100 pixels-high clip whose video\n        content scrolls from the top to the bottom of the frames of\n        ``clip`` at 50 pixels per second.\n\n        >>> filter = lambda get_frame,t : get_frame(t)[int(t):int(t)+50, :]\n        >>> new_clip = clip.transform(filter, apply_to=\'mask\')\n\n        '
        if apply_to is None:
            apply_to = []
        new_clip = self.with_make_frame(lambda t: func(self.get_frame, t))
        if not keep_duration:
            new_clip.duration = None
            new_clip.end = None
        if isinstance(apply_to, str):
            apply_to = [apply_to]
        for attribute in apply_to:
            attribute_value = getattr(new_clip, attribute, None)
            if attribute_value is not None:
                new_attribute_value = attribute_value.transform(func, keep_duration=keep_duration)
                setattr(new_clip, attribute, new_attribute_value)
        return new_clip

    def time_transform(self, time_func, apply_to=None, keep_duration=False):
        if False:
            while True:
                i = 10
        '\n        Returns a Clip instance playing the content of the current clip\n        but with a modified timeline, time ``t`` being replaced by another\n        time `time_func(t)`.\n\n        Parameters\n        ----------\n\n        time_func : function\n          A function ``t -> new_t``.\n\n        apply_to : {"mask", "audio", ["mask", "audio"]}, optional\n          Can be either \'mask\', or \'audio\', or [\'mask\',\'audio\'].\n          Specifies if the filter ``transform`` should also be applied to the\n          audio or the mask of the clip, if any.\n\n        keep_duration : bool, optional\n          ``False`` (default) if the transformation modifies the\n          ``duration`` of the clip.\n\n        Examples\n        --------\n\n        >>> # plays the clip (and its mask and sound) twice faster\n        >>> new_clip = clip.time_transform(lambda t: 2*t, apply_to=[\'mask\', \'audio\'])\n        >>>\n        >>> # plays the clip starting at t=3, and backwards:\n        >>> new_clip = clip.time_transform(lambda t: 3-t)\n\n        '
        if apply_to is None:
            apply_to = []
        return self.transform(lambda get_frame, t: get_frame(time_func(t)), apply_to, keep_duration=keep_duration)

    def fx(self, func, *args, **kwargs):
        if False:
            return 10
        'Returns the result of ``func(self, *args, **kwargs)``, for instance\n\n        >>> new_clip = clip.fx(resize, 0.2, method="bilinear")\n\n        is equivalent to\n\n        >>> new_clip = resize(clip, 0.2, method="bilinear")\n\n        The motivation of fx is to keep the name of the effect near its\n        parameters when the effects are chained:\n\n        >>> from moviepy.video.fx import multiply_volume, resize, mirrorx\n        >>> clip.fx(multiply_volume, 0.5).fx(resize, 0.3).fx(mirrorx)\n        >>> # Is equivalent, but clearer than\n        >>> mirrorx(resize(multiply_volume(clip, 0.5), 0.3))\n        '
        return func(self, *args, **kwargs)

    @apply_to_mask
    @apply_to_audio
    @convert_parameter_to_seconds(['t'])
    @outplace
    def with_start(self, t, change_end=True):
        if False:
            for i in range(10):
                print('nop')
        "Returns a copy of the clip, with the ``start`` attribute set\n        to ``t``, which can be expressed in seconds (15.35), in (min, sec),\n        in (hour, min, sec), or as a string: '01:03:05.35'.\n\n        These changes are also applied to the ``audio`` and ``mask``\n        clips of the current clip, if they exist.\n\n        Parameters\n        ----------\n\n        t : float or tuple or str\n          New ``start`` attribute value for the clip.\n\n        change_end : bool optional\n          Indicates if the ``end`` attribute value must be changed accordingly,\n          if possible. If ``change_end=True`` and the clip has a ``duration``\n          attribute, the ``end`` attribute of the clip will be updated to\n          ``start + duration``. If ``change_end=False`` and the clip has a\n          ``end`` attribute, the ``duration`` attribute of the clip will be\n          updated to ``end - start``.\n        "
        self.start = t
        if self.duration is not None and change_end:
            self.end = t + self.duration
        elif self.end is not None:
            self.duration = self.end - self.start

    @apply_to_mask
    @apply_to_audio
    @convert_parameter_to_seconds(['t'])
    @outplace
    def with_end(self, t):
        if False:
            i = 10
            return i + 15
        "Returns a copy of the clip, with the ``end`` attribute set to ``t``,\n        which can be expressed in seconds (15.35), in (min, sec), in\n        (hour, min, sec), or as a string: '01:03:05.35'. Also sets the duration\n        of the mask and audio, if any, of the returned clip.\n\n        Parameters\n        ----------\n\n        t : float or tuple or str\n          New ``end`` attribute value for the clip.\n        "
        self.end = t
        if self.end is None:
            return
        if self.start is None:
            if self.duration is not None:
                self.start = max(0, t - self.duration)
        else:
            self.duration = self.end - self.start

    @apply_to_mask
    @apply_to_audio
    @convert_parameter_to_seconds(['duration'])
    @outplace
    def with_duration(self, duration, change_end=True):
        if False:
            while True:
                i = 10
        "Returns a copy of the clip, with the  ``duration`` attribute set to\n        ``t``, which can be expressed in seconds (15.35), in (min, sec), in\n        (hour, min, sec), or as a string: '01:03:05.35'. Also sets the duration\n        of the mask and audio, if any, of the returned clip.\n\n        If ``change_end is False``, the start attribute of the clip will be\n        modified in function of the duration and the preset end of the clip.\n\n        Parameters\n        ----------\n\n        duration : float\n          New duration attribute value for the clip.\n\n        change_end : bool, optional\n          If ``True``, the ``end`` attribute value of the clip will be adjusted\n          accordingly to the new duration using ``clip.start + duration``.\n        "
        self.duration = duration
        if change_end:
            self.end = None if duration is None else self.start + duration
        else:
            if self.duration is None:
                raise ValueError('Cannot change clip start when new duration is None')
            self.start = self.end - duration

    @outplace
    def with_make_frame(self, make_frame):
        if False:
            for i in range(10):
                print('nop')
        'Sets a ``make_frame`` attribute for the clip. Useful for setting\n        arbitrary/complicated videoclips.\n\n        Parameters\n        ----------\n\n        make_frame : function\n          New frame creator function for the clip.\n        '
        self.make_frame = make_frame

    def with_fps(self, fps, change_duration=False):
        if False:
            print('Hello World!')
        'Returns a copy of the clip with a new default fps for functions like\n        write_videofile, iterframe, etc.\n\n        Parameters\n        ----------\n\n        fps : int\n          New ``fps`` attribute value for the clip.\n\n        change_duration : bool, optional\n          If ``change_duration=True``, then the video speed will change to\n          match the new fps (conserving all frames 1:1). For example, if the\n          fps is halved in this mode, the duration will be doubled.\n        '
        if change_duration:
            from moviepy.video.fx.multiply_speed import multiply_speed
            newclip = multiply_speed(self, fps / self.fps)
        else:
            newclip = self.copy()
        newclip.fps = fps
        return newclip

    @outplace
    def with_is_mask(self, is_mask):
        if False:
            i = 10
            return i + 15
        'Says whether the clip is a mask or not.\n\n        Parameters\n        ----------\n\n        is_mask : bool\n          New ``is_mask`` attribute value for the clip.\n        '
        self.is_mask = is_mask

    @outplace
    def with_memoize(self, memoize):
        if False:
            i = 10
            return i + 15
        'Sets whether the clip should keep the last frame read in memory.\n\n        Parameters\n        ----------\n\n        memoize : bool\n          Indicates if the clip should keep the last frame read in memory.\n        '
        self.memoize = memoize

    @convert_parameter_to_seconds(['t'])
    def is_playing(self, t):
        if False:
            print('Hello World!')
        "If ``t`` is a time, returns true if t is between the start and the end\n        of the clip. ``t`` can be expressed in seconds (15.35), in (min, sec), in\n        (hour, min, sec), or as a string: '01:03:05.35'. If ``t`` is a numpy\n        array, returns False if none of the ``t`` is in the clip, else returns a\n        vector [b_1, b_2, b_3...] where b_i is true if tti is in the clip.\n        "
        if isinstance(t, np.ndarray):
            (tmin, tmax) = (t.min(), t.max())
            if self.end is not None and tmin >= self.end:
                return False
            if tmax < self.start:
                return False
            result = 1 * (t >= self.start)
            if self.end is not None:
                result *= t <= self.end
            return result
        else:
            return t >= self.start and (self.end is None or t < self.end)

    @convert_parameter_to_seconds(['start_time', 'end_time'])
    @apply_to_mask
    @apply_to_audio
    def subclip(self, start_time=0, end_time=None):
        if False:
            print('Hello World!')
        "Returns a clip playing the content of the current clip between times\n        ``start_time`` and ``end_time``, which can be expressed in seconds\n        (15.35), in (min, sec), in (hour, min, sec), or as a string:\n        '01:03:05.35'.\n\n        The ``mask`` and ``audio`` of the resulting subclip will be subclips of\n        ``mask`` and ``audio`` the original clip, if they exist.\n\n        It's equivalent to slice the clip as a sequence, like\n        ``clip[t_start:t_end]``.\n\n        Parameters\n        ----------\n\n        start_time : float or tuple or str, optional\n          Moment that will be chosen as the beginning of the produced clip. If\n          is negative, it is reset to ``clip.duration + start_time``.\n\n        end_time : float or tuple or str, optional\n          Moment that will be chosen as the end of the produced clip. If not\n          provided, it is assumed to be the duration of the clip (potentially\n          infinite). If is negative, it is reset to ``clip.duration + end_time``.\n          For instance:\n\n          >>> # cut the last two seconds of the clip:\n          >>> new_clip = clip.subclip(0, -2)\n\n          If ``end_time`` is provided or if the clip has a duration attribute,\n          the duration of the returned clip is set automatically.\n        "
        if start_time < 0:
            start_time = self.duration + start_time
        if self.duration is not None and start_time >= self.duration:
            raise ValueError('start_time (%.02f) ' % start_time + "should be smaller than the clip's " + 'duration (%.02f).' % self.duration)
        new_clip = self.time_transform(lambda t: t + start_time, apply_to=[])
        if end_time is None and self.duration is not None:
            end_time = self.duration
        elif end_time is not None and end_time < 0:
            if self.duration is None:
                raise ValueError('Subclip with negative times (here %s) can only be extracted from clips with a ``duration``' % str((start_time, end_time)))
            else:
                end_time = self.duration + end_time
        if end_time is not None:
            new_clip.duration = end_time - start_time
            new_clip.end = new_clip.start + new_clip.duration
        return new_clip

    @convert_parameter_to_seconds(['start_time', 'end_time'])
    def cutout(self, start_time, end_time):
        if False:
            return 10
        "\n        Returns a clip playing the content of the current clip but\n        skips the extract between ``start_time`` and ``end_time``, which can be\n        expressed in seconds (15.35), in (min, sec), in (hour, min, sec),\n        or as a string: '01:03:05.35'.\n\n        If the original clip has a ``duration`` attribute set,\n        the duration of the returned clip  is automatically computed as\n        `` duration - (end_time - start_time)``.\n\n        The resulting clip's ``audio`` and ``mask`` will also be cutout\n        if they exist.\n\n        Parameters\n        ----------\n\n        start_time : float or tuple or str\n          Moment from which frames will be ignored in the resulting output.\n\n        end_time : float or tuple or str\n          Moment until which frames will be ignored in the resulting output.\n        "
        new_clip = self.time_transform(lambda t: t + (t >= start_time) * (end_time - start_time), apply_to=['audio', 'mask'])
        if self.duration is not None:
            return new_clip.with_duration(self.duration - (end_time - start_time))
        else:
            return new_clip

    @requires_duration
    @use_clip_fps_by_default
    def iter_frames(self, fps=None, with_times=False, logger=None, dtype=None):
        if False:
            print('Hello World!')
        'Iterates over all the frames of the clip.\n\n        Returns each frame of the clip as a HxWxN Numpy array,\n        where N=1 for mask clips and N=3 for RGB clips.\n\n        This function is not really meant for video editing. It provides an\n        easy way to do frame-by-frame treatment of a video, for fields like\n        science, computer vision...\n\n        Parameters\n        ----------\n\n        fps : int, optional\n          Frames per second for clip iteration. Is optional if the clip already\n          has a ``fps`` attribute.\n\n        with_times : bool, optional\n          Ff ``True`` yield tuples of ``(t, frame)`` where ``t`` is the current\n          time for the frame, otherwise only a ``frame`` object.\n\n        logger : str, optional\n          Either ``"bar"`` for progress bar or ``None`` or any Proglog logger.\n\n        dtype : type, optional\n          Type to cast Numpy array frames. Use ``dtype="uint8"`` when using the\n          pictures to write video, images...\n\n        Examples\n        --------\n\n        >>> # prints the maximum of red that is contained\n        >>> # on the first line of each frame of the clip.\n        >>> from moviepy import VideoFileClip\n        >>> myclip = VideoFileClip(\'myvideo.mp4\')\n        >>> print ( [frame[0,:,0].max()\n                     for frame in myclip.iter_frames()])\n        '
        logger = proglog.default_bar_logger(logger)
        for frame_index in logger.iter_bar(frame_index=np.arange(0, int(self.duration * fps))):
            t = frame_index / fps
            frame = self.get_frame(t)
            if dtype is not None and frame.dtype != dtype:
                frame = frame.astype(dtype)
            if with_times:
                yield (t, frame)
            else:
                yield frame

    def close(self):
        if False:
            while True:
                i = 10
        'Release any resources that are in use.'
        pass

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        if not isinstance(other, Clip):
            return NotImplemented
        self_length = self.duration * self.fps
        other_length = other.duration * other.fps
        if self_length != other_length:
            return False
        for (frame1, frame2) in zip(self.iter_frames(), other.iter_frames()):
            if not np.array_equal(frame1, frame2):
                return False
        return True

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        '\n        Support the Context Manager protocol,\n        to ensure that resources are cleaned up.\n        '
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            return 10
        self.close()

    def __getitem__(self, key):
        if False:
            return 10
        "\n        Support extended slice and index operations over\n        a clip object.\n\n        Simple slicing is implemented via `subclip`.\n        So, ``clip[t_start:t_end]`` is equivalent to\n        ``clip.subclip(t_start, t_end)``. If ``t_start`` is not\n        given, default to ``0``, if ``t_end`` is not given,\n        default to ``self.duration``.\n\n        The slice object optionally support a third argument as\n        a ``speed`` coefficient (that could be negative),\n        ``clip[t_start:t_end:speed]``.\n\n        For example ``clip[::-1]`` returns a reversed (a time_mirror fx)\n        the video and ``clip[:5:2]`` returns the segment from 0 to 5s\n        accelerated to 2x (ie. resulted duration would be 2.5s)\n\n        In addition, a tuple of slices is supported, resulting in the concatenation\n        of each segment. For example ``clip[(:1, 2:)]`` return a clip\n        with the segment from 1 to 2s removed.\n\n        If ``key`` is not a slice or tuple, we assume it's a time\n        value (expressed in any format supported by `cvsec`)\n        and return the frame at that time, passing the key\n        to ``get_frame``.\n        "
        apply_to = ['mask', 'audio']
        if isinstance(key, slice):
            clip = self.subclip(key.start or 0, key.stop or self.duration)
            if key.step:
                factor = abs(key.step)
                if factor != 1:
                    clip = clip.time_transform(lambda t: factor * t, apply_to=apply_to, keep_duration=True)
                    clip = clip.with_duration(1.0 * clip.duration / factor)
                if key.step < 0:
                    clip = clip.time_transform(lambda t: clip.duration - t - 1, keep_duration=True, apply_to=apply_to)
            return clip
        elif isinstance(key, tuple):
            return reduce(add, (self[k] for k in key))
        else:
            return self.get_frame(key)

    def __del__(self):
        if False:
            while True:
                i = 10
        pass

    def __add__(self, other):
        if False:
            return 10
        return NotImplemented

    def __mul__(self, n):
        if False:
            while True:
                i = 10
        if not isinstance(n, Real):
            return NotImplemented
        from moviepy.video.fx.loop import loop
        return loop(self, n)