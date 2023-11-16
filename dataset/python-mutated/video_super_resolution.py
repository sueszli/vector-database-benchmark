import os.path as osp
from collections import OrderedDict
import cv2
from cv2 import CAP_PROP_FOURCC, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, CAP_PROP_POS_FRAMES, VideoWriter_fourcc
from .util import check_file_exist, mkdir_or_exist, track_progress

class Cache:

    def __init__(self, capacity):
        if False:
            i = 10
            return i + 15
        self._cache = OrderedDict()
        self._capacity = int(capacity)
        if capacity <= 0:
            raise ValueError('capacity must be a positive integer')

    @property
    def capacity(self):
        if False:
            return 10
        return self._capacity

    @property
    def size(self):
        if False:
            while True:
                i = 10
        return len(self._cache)

    def put(self, key, val):
        if False:
            while True:
                i = 10
        if key in self._cache:
            return
        if len(self._cache) >= self.capacity:
            self._cache.popitem(last=False)
        self._cache[key] = val

    def get(self, key, default=None):
        if False:
            print('Hello World!')
        val = self._cache[key] if key in self._cache else default
        return val

class VideoReader:
    """Video class with similar usage to a list object.
    This video warpper class provides convenient apis to access frames.
    There exists an issue of OpenCV's VideoCapture class that jumping to a
    certain frame may be inaccurate. It is fixed in this class by checking
    the position after jumping each time.
    Cache is used when decoding videos. So if the same frame is visited for
    the second time, there is no need to decode again if it is stored in the
    cache.
    Example:

    >>> import mmcv
    >>> v = mmcv.VideoReader('sample.mp4')
    >>> len(v)  # get the total frame number with `len()`
    120
    >>> for img in v:  # v is iterable
    >>>     mmcv.imshow(img)
    >>> v[5]  # get the 6th frame
    """

    def __init__(self, filename, cache_capacity=10):
        if False:
            print('Hello World!')
        if not filename.startswith(('https://', 'http://')):
            check_file_exist(filename, 'Video file not found: ' + filename)
        self._vcap = cv2.VideoCapture(filename)
        assert cache_capacity > 0
        self._cache = Cache(cache_capacity)
        self._position = 0
        self._width = int(self._vcap.get(CAP_PROP_FRAME_WIDTH))
        self._height = int(self._vcap.get(CAP_PROP_FRAME_HEIGHT))
        self._fps = self._vcap.get(CAP_PROP_FPS)
        self._frame_cnt = int(self._vcap.get(CAP_PROP_FRAME_COUNT))
        self._fourcc = self._vcap.get(CAP_PROP_FOURCC)

    @property
    def vcap(self):
        if False:
            print('Hello World!')
        ':obj:`cv2.VideoCapture`: The raw VideoCapture object.'
        return self._vcap

    @property
    def opened(self):
        if False:
            for i in range(10):
                print('nop')
        'bool: Indicate whether the video is opened.'
        return self._vcap.isOpened()

    @property
    def width(self):
        if False:
            for i in range(10):
                print('nop')
        'int: Width of video frames.'
        return self._width

    @property
    def height(self):
        if False:
            while True:
                i = 10
        'int: Height of video frames.'
        return self._height

    @property
    def resolution(self):
        if False:
            for i in range(10):
                print('nop')
        'tuple: Video resolution (width, height).'
        return (self._width, self._height)

    @property
    def fps(self):
        if False:
            i = 10
            return i + 15
        'float: FPS of the video.'
        return self._fps

    @property
    def frame_cnt(self):
        if False:
            i = 10
            return i + 15
        'int: Total frames of the video.'
        return self._frame_cnt

    @property
    def fourcc(self):
        if False:
            return 10
        'str: "Four character code" of the video.'
        return self._fourcc

    @property
    def position(self):
        if False:
            for i in range(10):
                print('nop')
        'int: Current cursor position, indicating frame decoded.'
        return self._position

    def _get_real_position(self):
        if False:
            i = 10
            return i + 15
        return int(round(self._vcap.get(CAP_PROP_POS_FRAMES)))

    def _set_real_position(self, frame_id):
        if False:
            for i in range(10):
                print('nop')
        self._vcap.set(CAP_PROP_POS_FRAMES, frame_id)
        pos = self._get_real_position()
        for _ in range(frame_id - pos):
            self._vcap.read()
        self._position = frame_id

    def read(self):
        if False:
            while True:
                i = 10
        'Read the next frame.\n        If the next frame have been decoded before and in the cache, then\n        return it directly, otherwise decode, cache and return it.\n        Returns:\n            ndarray or None: Return the frame if successful, otherwise None.\n        '
        if self._cache:
            img = self._cache.get(self._position)
            if img is not None:
                ret = True
            else:
                if self._position != self._get_real_position():
                    self._set_real_position(self._position)
                (ret, img) = self._vcap.read()
                if ret:
                    self._cache.put(self._position, img)
        else:
            (ret, img) = self._vcap.read()
        if ret:
            self._position += 1
        return img

    def get_frame(self, frame_id):
        if False:
            while True:
                i = 10
        'Get frame by index.\n        Args:\n            frame_id (int): Index of the expected frame, 0-based.\n        Returns:\n            ndarray or None: Return the frame if successful, otherwise None.\n        '
        if frame_id < 0 or frame_id >= self._frame_cnt:
            raise IndexError(f'"frame_id" must be between 0 and {self._frame_cnt - 1}')
        if frame_id == self._position:
            return self.read()
        if self._cache:
            img = self._cache.get(frame_id)
            if img is not None:
                self._position = frame_id + 1
                return img
        self._set_real_position(frame_id)
        (ret, img) = self._vcap.read()
        if ret:
            if self._cache:
                self._cache.put(self._position, img)
            self._position += 1
        return img

    def current_frame(self):
        if False:
            print('Hello World!')
        'Get the current frame (frame that is just visited).\n        Returns:\n            ndarray or None: If the video is fresh, return None, otherwise\n                return the frame.\n        '
        if self._position == 0:
            return None
        return self._cache.get(self._position - 1)

    def cvt2frames(self, frame_dir, file_start=0, filename_tmpl='{:06d}.jpg', start=0, max_num=0, show_progress=True):
        if False:
            return 10
        'Convert a video to frame images.\n        Args:\n            frame_dir (str): Output directory to store all the frame images.\n            file_start (int): Filenames will start from the specified number.\n            filename_tmpl (str): Filename template with the index as the\n                placeholder.\n            start (int): The starting frame index.\n            max_num (int): Maximum number of frames to be written.\n            show_progress (bool): Whether to show a progress bar.\n        '
        mkdir_or_exist(frame_dir)
        if max_num == 0:
            task_num = self.frame_cnt - start
        else:
            task_num = min(self.frame_cnt - start, max_num)
        if task_num <= 0:
            raise ValueError('start must be less than total frame number')
        if start > 0:
            self._set_real_position(start)

        def write_frame(file_idx):
            if False:
                return 10
            img = self.read()
            if img is None:
                return
            filename = osp.join(frame_dir, filename_tmpl.format(file_idx))
            cv2.imwrite(filename, img)
        if show_progress:
            track_progress(write_frame, range(file_start, file_start + task_num))
        else:
            for i in range(task_num):
                write_frame(file_start + i)

    def __len__(self):
        if False:
            return 10
        return self.frame_cnt

    def __getitem__(self, index):
        if False:
            i = 10
            return i + 15
        if isinstance(index, slice):
            return [self.get_frame(i) for i in range(*index.indices(self.frame_cnt))]
        if index < 0:
            index += self.frame_cnt
            if index < 0:
                raise IndexError('index out of range')
        return self.get_frame(index)

    def __iter__(self):
        if False:
            print('Hello World!')
        self._set_real_position(0)
        return self

    def __next__(self):
        if False:
            print('Hello World!')
        img = self.read()
        if img is not None:
            return img
        else:
            raise StopIteration
    next = __next__

    def __enter__(self):
        if False:
            return 10
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            print('Hello World!')
        self._vcap.release()