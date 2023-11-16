"""Abstraction layer to resize images using PIL, ImageMagick, or a
public resizing proxy if neither is available.
"""
import os
import os.path
import platform
import re
import subprocess
from itertools import chain
from tempfile import NamedTemporaryFile
from urllib.parse import urlencode
from beets import logging, util
from beets.util import bytestring_path, displayable_path, py3_path, syspath
PROXY_URL = 'https://images.weserv.nl/'
log = logging.getLogger('beets')

def resize_url(url, maxwidth, quality=0):
    if False:
        while True:
            i = 10
    'Return a proxied image URL that resizes the original image to\n    maxwidth (preserving aspect ratio).\n    '
    params = {'url': url.replace('http://', ''), 'w': maxwidth}
    if quality > 0:
        params['q'] = quality
    return '{}?{}'.format(PROXY_URL, urlencode(params))

def temp_file_for(path):
    if False:
        return 10
    'Return an unused filename with the same extension as the\n    specified path.\n    '
    ext = os.path.splitext(path)[1]
    with NamedTemporaryFile(suffix=py3_path(ext), delete=False) as f:
        return bytestring_path(f.name)

class LocalBackendNotAvailableError(Exception):
    pass
_NOT_AVAILABLE = object()

class LocalBackend:

    @classmethod
    def available(cls):
        if False:
            print('Hello World!')
        try:
            cls.version()
            return True
        except LocalBackendNotAvailableError:
            return False

class IMBackend(LocalBackend):
    NAME = 'ImageMagick'
    _version = None
    _legacy = None

    @classmethod
    def version(cls):
        if False:
            for i in range(10):
                print('nop')
        'Obtain and cache ImageMagick version.\n\n        Raises `LocalBackendNotAvailableError` if not available.\n        '
        if cls._version is None:
            for (cmd_name, legacy) in (('magick', False), ('convert', True)):
                try:
                    out = util.command_output([cmd_name, '--version']).stdout
                except (subprocess.CalledProcessError, OSError) as exc:
                    log.debug('ImageMagick version check failed: {}', exc)
                    cls._version = _NOT_AVAILABLE
                else:
                    if b'imagemagick' in out.lower():
                        pattern = b'.+ (\\d+)\\.(\\d+)\\.(\\d+).*'
                        match = re.search(pattern, out)
                        if match:
                            cls._version = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
                            cls._legacy = legacy
        if cls._version is _NOT_AVAILABLE:
            raise LocalBackendNotAvailableError()
        else:
            return cls._version

    def __init__(self):
        if False:
            return 10
        'Initialize a wrapper around ImageMagick for local image operations.\n\n        Stores the ImageMagick version and legacy flag. If ImageMagick is not\n        available, raise an Exception.\n        '
        self.version()
        if self._legacy:
            self.convert_cmd = ['convert']
            self.identify_cmd = ['identify']
            self.compare_cmd = ['compare']
        else:
            self.convert_cmd = ['magick']
            self.identify_cmd = ['magick', 'identify']
            self.compare_cmd = ['magick', 'compare']

    def resize(self, maxwidth, path_in, path_out=None, quality=0, max_filesize=0):
        if False:
            for i in range(10):
                print('nop')
        'Resize using ImageMagick.\n\n        Use the ``magick`` program or ``convert`` on older versions. Return\n        the output path of resized image.\n        '
        path_out = path_out or temp_file_for(path_in)
        log.debug('artresizer: ImageMagick resizing {0} to {1}', displayable_path(path_in), displayable_path(path_out))
        cmd = self.convert_cmd + [syspath(path_in, prefix=False), '-resize', f'{maxwidth}x>', '-interlace', 'none']
        if quality > 0:
            cmd += ['-quality', f'{quality}']
        if max_filesize > 0:
            cmd += ['-define', f'jpeg:extent={max_filesize}b']
        cmd.append(syspath(path_out, prefix=False))
        try:
            util.command_output(cmd)
        except subprocess.CalledProcessError:
            log.warning('artresizer: IM convert failed for {0}', displayable_path(path_in))
            return path_in
        return path_out

    def get_size(self, path_in):
        if False:
            i = 10
            return i + 15
        cmd = self.identify_cmd + ['-format', '%w %h', syspath(path_in, prefix=False)]
        try:
            out = util.command_output(cmd).stdout
        except subprocess.CalledProcessError as exc:
            log.warning('ImageMagick size query failed')
            log.debug('`convert` exited with (status {}) when getting size with command {}:\n{}', exc.returncode, cmd, exc.output.strip())
            return None
        try:
            return tuple(map(int, out.split(b' ')))
        except IndexError:
            log.warning('Could not understand IM output: {0!r}', out)
            return None

    def deinterlace(self, path_in, path_out=None):
        if False:
            print('Hello World!')
        path_out = path_out or temp_file_for(path_in)
        cmd = self.convert_cmd + [syspath(path_in, prefix=False), '-interlace', 'none', syspath(path_out, prefix=False)]
        try:
            util.command_output(cmd)
            return path_out
        except subprocess.CalledProcessError:
            return path_in

    def get_format(self, filepath):
        if False:
            print('Hello World!')
        cmd = self.identify_cmd + ['-format', '%[magick]', syspath(filepath)]
        try:
            return util.command_output(cmd).stdout
        except subprocess.CalledProcessError:
            return None

    def convert_format(self, source, target, deinterlaced):
        if False:
            while True:
                i = 10
        cmd = self.convert_cmd + [syspath(source), *(['-interlace', 'none'] if deinterlaced else []), syspath(target)]
        try:
            subprocess.check_call(cmd, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            return target
        except subprocess.CalledProcessError:
            return source

    @property
    def can_compare(self):
        if False:
            for i in range(10):
                print('nop')
        return self.version() > (6, 8, 7)

    def compare(self, im1, im2, compare_threshold):
        if False:
            while True:
                i = 10
        is_windows = platform.system() == 'Windows'
        convert_cmd = self.convert_cmd + [syspath(im2, prefix=False), syspath(im1, prefix=False), '-colorspace', 'gray', 'MIFF:-']
        compare_cmd = self.compare_cmd + ['-define', 'phash:colorspaces=sRGB,HCLp', '-metric', 'PHASH', '-', 'null:']
        log.debug('comparing images with pipeline {} | {}', convert_cmd, compare_cmd)
        convert_proc = subprocess.Popen(convert_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=not is_windows)
        compare_proc = subprocess.Popen(compare_cmd, stdin=convert_proc.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=not is_windows)
        convert_proc.stdout.close()
        convert_stderr = convert_proc.stderr.read()
        convert_proc.stderr.close()
        convert_proc.wait()
        if convert_proc.returncode:
            log.debug('ImageMagick convert failed with status {}: {!r}', convert_proc.returncode, convert_stderr)
            return None
        (stdout, stderr) = compare_proc.communicate()
        if compare_proc.returncode:
            if compare_proc.returncode != 1:
                log.debug('ImageMagick compare failed: {0}, {1}', displayable_path(im2), displayable_path(im1))
                return None
            out_str = stderr
        else:
            out_str = stdout
        try:
            phash_diff = float(out_str)
        except ValueError:
            log.debug('IM output is not a number: {0!r}', out_str)
            return None
        log.debug('ImageMagick compare score: {0}', phash_diff)
        return phash_diff <= compare_threshold

    @property
    def can_write_metadata(self):
        if False:
            i = 10
            return i + 15
        return True

    def write_metadata(self, file, metadata):
        if False:
            i = 10
            return i + 15
        assignments = list(chain.from_iterable((('-set', k, v) for (k, v) in metadata.items())))
        command = self.convert_cmd + [file, *assignments, file]
        util.command_output(command)

class PILBackend(LocalBackend):
    NAME = 'PIL'

    @classmethod
    def version(cls):
        if False:
            print('Hello World!')
        try:
            __import__('PIL', fromlist=['Image'])
        except ImportError:
            raise LocalBackendNotAvailableError()

    def __init__(self):
        if False:
            print('Hello World!')
        'Initialize a wrapper around PIL for local image operations.\n\n        If PIL is not available, raise an Exception.\n        '
        self.version()

    def resize(self, maxwidth, path_in, path_out=None, quality=0, max_filesize=0):
        if False:
            for i in range(10):
                print('nop')
        'Resize using Python Imaging Library (PIL).  Return the output path\n        of resized image.\n        '
        path_out = path_out or temp_file_for(path_in)
        from PIL import Image
        log.debug('artresizer: PIL resizing {0} to {1}', displayable_path(path_in), displayable_path(path_out))
        try:
            im = Image.open(syspath(path_in))
            size = (maxwidth, maxwidth)
            im.thumbnail(size, Image.Resampling.LANCZOS)
            if quality == 0:
                quality = -1
            im.save(py3_path(path_out), quality=quality, progressive=False)
            if max_filesize > 0:
                if quality > 0:
                    lower_qual = quality
                else:
                    lower_qual = 95
                for i in range(5):
                    filesize = os.stat(syspath(path_out)).st_size
                    log.debug('PIL Pass {0} : Output size: {1}B', i, filesize)
                    if filesize <= max_filesize:
                        return path_out
                    lower_qual -= 10
                    if lower_qual < 10:
                        lower_qual = 10
                    im.save(py3_path(path_out), quality=lower_qual, optimize=True, progressive=False)
                log.warning('PIL Failed to resize file to below {0}B', max_filesize)
                return path_out
            else:
                return path_out
        except OSError:
            log.error("PIL cannot create thumbnail for '{0}'", displayable_path(path_in))
            return path_in

    def get_size(self, path_in):
        if False:
            while True:
                i = 10
        from PIL import Image
        try:
            im = Image.open(syspath(path_in))
            return im.size
        except OSError as exc:
            log.error('PIL could not read file {}: {}', displayable_path(path_in), exc)
            return None

    def deinterlace(self, path_in, path_out=None):
        if False:
            return 10
        path_out = path_out or temp_file_for(path_in)
        from PIL import Image
        try:
            im = Image.open(syspath(path_in))
            im.save(py3_path(path_out), progressive=False)
            return path_out
        except IOError:
            return path_in

    def get_format(self, filepath):
        if False:
            print('Hello World!')
        from PIL import Image, UnidentifiedImageError
        try:
            with Image.open(syspath(filepath)) as im:
                return im.format
        except (ValueError, TypeError, UnidentifiedImageError, FileNotFoundError):
            log.exception('failed to detect image format for {}', filepath)
            return None

    def convert_format(self, source, target, deinterlaced):
        if False:
            print('Hello World!')
        from PIL import Image, UnidentifiedImageError
        try:
            with Image.open(syspath(source)) as im:
                im.save(py3_path(target), progressive=not deinterlaced)
                return target
        except (ValueError, TypeError, UnidentifiedImageError, FileNotFoundError, OSError):
            log.exception('failed to convert image {} -> {}', source, target)
            return source

    @property
    def can_compare(self):
        if False:
            while True:
                i = 10
        return False

    def compare(self, im1, im2, compare_threshold):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    @property
    def can_write_metadata(self):
        if False:
            while True:
                i = 10
        return True

    def write_metadata(self, file, metadata):
        if False:
            return 10
        from PIL import Image, PngImagePlugin
        im = Image.open(syspath(file))
        meta = PngImagePlugin.PngInfo()
        for (k, v) in metadata.items():
            meta.add_text(k, v, 0)
        im.save(py3_path(file), 'PNG', pnginfo=meta)

class Shareable(type):
    """A pseudo-singleton metaclass that allows both shared and
    non-shared instances. The ``MyClass.shared`` property holds a
    lazily-created shared instance of ``MyClass`` while calling
    ``MyClass()`` to construct a new object works as usual.
    """

    def __init__(cls, name, bases, dict):
        if False:
            while True:
                i = 10
        super().__init__(name, bases, dict)
        cls._instance = None

    @property
    def shared(cls):
        if False:
            i = 10
            return i + 15
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
BACKEND_CLASSES = [IMBackend, PILBackend]

class ArtResizer(metaclass=Shareable):
    """A singleton class that performs image resizes."""

    def __init__(self):
        if False:
            print('Hello World!')
        'Create a resizer object with an inferred method.'
        for backend_cls in BACKEND_CLASSES:
            try:
                self.local_method = backend_cls()
                log.debug(f'artresizer: method is {self.local_method.NAME}')
                break
            except LocalBackendNotAvailableError:
                continue
        else:
            log.debug('artresizer: method is WEBPROXY')
            self.local_method = None

    @property
    def method(self):
        if False:
            print('Hello World!')
        if self.local:
            return self.local_method.NAME
        else:
            return 'WEBPROXY'

    def resize(self, maxwidth, path_in, path_out=None, quality=0, max_filesize=0):
        if False:
            return 10
        'Manipulate an image file according to the method, returning a\n        new path. For PIL or IMAGEMAGIC methods, resizes the image to a\n        temporary file and encodes with the specified quality level.\n        For WEBPROXY, returns `path_in` unmodified.\n        '
        if self.local:
            return self.local_method.resize(maxwidth, path_in, path_out, quality=quality, max_filesize=max_filesize)
        else:
            return path_in

    def deinterlace(self, path_in, path_out=None):
        if False:
            while True:
                i = 10
        'Deinterlace an image.\n\n        Only available locally.\n        '
        if self.local:
            return self.local_method.deinterlace(path_in, path_out)
        else:
            return path_in

    def proxy_url(self, maxwidth, url, quality=0):
        if False:
            while True:
                i = 10
        'Modifies an image URL according the method, returning a new\n        URL. For WEBPROXY, a URL on the proxy server is returned.\n        Otherwise, the URL is returned unmodified.\n        '
        if self.local:
            return url
        else:
            return resize_url(url, maxwidth, quality)

    @property
    def local(self):
        if False:
            for i in range(10):
                print('nop')
        'A boolean indicating whether the resizing method is performed\n        locally (i.e., PIL or ImageMagick).\n        '
        return self.local_method is not None

    def get_size(self, path_in):
        if False:
            while True:
                i = 10
        'Return the size of an image file as an int couple (width, height)\n        in pixels.\n\n        Only available locally.\n        '
        if self.local:
            return self.local_method.get_size(path_in)
        else:
            return path_in

    def get_format(self, path_in):
        if False:
            i = 10
            return i + 15
        'Returns the format of the image as a string.\n\n        Only available locally.\n        '
        if self.local:
            return self.local_method.get_format(path_in)
        else:
            return None

    def reformat(self, path_in, new_format, deinterlaced=True):
        if False:
            print('Hello World!')
        'Converts image to desired format, updating its extension, but\n        keeping the same filename.\n\n        Only available locally.\n        '
        if not self.local:
            return path_in
        new_format = new_format.lower()
        new_format = {'jpeg': 'jpg'}.get(new_format, new_format)
        (fname, ext) = os.path.splitext(path_in)
        path_new = fname + b'.' + new_format.encode('utf8')
        result_path = path_in
        try:
            result_path = self.local_method.convert_format(path_in, path_new, deinterlaced)
        finally:
            if result_path != path_in:
                os.unlink(path_in)
        return result_path

    @property
    def can_compare(self):
        if False:
            i = 10
            return i + 15
        'A boolean indicating whether image comparison is available'
        if self.local:
            return self.local_method.can_compare
        else:
            return False

    def compare(self, im1, im2, compare_threshold):
        if False:
            print('Hello World!')
        'Return a boolean indicating whether two images are similar.\n\n        Only available locally.\n        '
        if self.local:
            return self.local_method.compare(im1, im2, compare_threshold)
        else:
            return None

    @property
    def can_write_metadata(self):
        if False:
            while True:
                i = 10
        'A boolean indicating whether writing image metadata is supported.'
        if self.local:
            return self.local_method.can_write_metadata
        else:
            return False

    def write_metadata(self, file, metadata):
        if False:
            i = 10
            return i + 15
        'Write key-value metadata to the image file.\n\n        Only available locally. Currently, expects the image to be a PNG file.\n        '
        if self.local:
            self.local_method.write_metadata(file, metadata)
        else:
            pass