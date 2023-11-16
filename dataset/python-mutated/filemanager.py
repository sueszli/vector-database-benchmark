import logging
import os
import os.path as op
import re
import uuid
from flask.globals import _request_ctx_stack
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
from wtforms import ValidationError
try:
    from flask import _app_ctx_stack
except ImportError:
    _app_ctx_stack = None
app_stack = _app_ctx_stack or _request_ctx_stack
log = logging.getLogger(__name__)
try:
    from PIL import Image, ImageOps
except ImportError:
    Image = None
    ImageOps = None

class FileManager(object):

    def __init__(self, base_path=None, relative_path='', namegen=None, allowed_extensions=None, permission=493, **kwargs):
        if False:
            print('Hello World!')
        ctx = app_stack.top
        if 'UPLOAD_FOLDER' in ctx.app.config and (not base_path):
            base_path = ctx.app.config['UPLOAD_FOLDER']
        if not base_path:
            raise Exception('Config key UPLOAD_FOLDER is mandatory')
        self.base_path = base_path
        self.relative_path = relative_path
        self.namegen = namegen or uuid_namegen
        if not allowed_extensions and 'FILE_ALLOWED_EXTENSIONS' in ctx.app.config:
            self.allowed_extensions = ctx.app.config['FILE_ALLOWED_EXTENSIONS']
        else:
            self.allowed_extensions = allowed_extensions
        self.permission = permission
        self._should_delete = False

    def is_file_allowed(self, filename):
        if False:
            return 10
        if not self.allowed_extensions:
            return True
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in self.allowed_extensions

    def generate_name(self, obj, file_data):
        if False:
            while True:
                i = 10
        return self.namegen(file_data)

    def get_path(self, filename):
        if False:
            print('Hello World!')
        if not self.base_path:
            raise ValueError('FileUploadField field requires base_path to be set.')
        return op.join(self.base_path, filename)

    def delete_file(self, filename):
        if False:
            return 10
        path = self.get_path(filename)
        if op.exists(path):
            os.remove(path)

    def save_file(self, data, filename):
        if False:
            print('Hello World!')
        filename_ = secure_filename(filename)
        path = self.get_path(filename_)
        if not op.exists(op.dirname(path)):
            os.makedirs(os.path.dirname(path), self.permission)
        data.save(path)
        return filename_

class ImageManager(FileManager):
    """
    Image Manager will manage your image files referenced on SQLAlchemy Model
    will save files on IMG_UPLOAD_FOLDER as <uuid>_sep_<filename>
    """
    keep_image_formats = ('PNG',)

    def __init__(self, base_path=None, relative_path=None, max_size=None, namegen=None, allowed_extensions=None, thumbgen=None, thumbnail_size=None, permission=493, **kwargs):
        if False:
            print('Hello World!')
        if Image is None:
            raise Exception('PIL library was not found')
        ctx = app_stack.top
        if 'IMG_SIZE' in ctx.app.config and (not max_size):
            self.max_size = ctx.app.config['IMG_SIZE']
        if 'IMG_UPLOAD_URL' in ctx.app.config and (not relative_path):
            relative_path = ctx.app.config['IMG_UPLOAD_URL']
        if not relative_path:
            raise Exception('Config key IMG_UPLOAD_URL is mandatory')
        if 'IMG_UPLOAD_FOLDER' in ctx.app.config and (not base_path):
            base_path = ctx.app.config['IMG_UPLOAD_FOLDER']
        if not base_path:
            raise Exception('Config key IMG_UPLOAD_FOLDER is mandatory')
        self.thumbnail_fn = thumbgen or thumbgen_filename
        self.thumbnail_size = thumbnail_size
        self.image = None
        if not allowed_extensions:
            allowed_extensions = ('gif', 'jpg', 'jpeg', 'png', 'tiff')
        super(ImageManager, self).__init__(base_path=base_path, relative_path=relative_path, namegen=namegen, allowed_extensions=allowed_extensions, permission=permission, **kwargs)

    def get_url(self, filename):
        if False:
            while True:
                i = 10
        if isinstance(filename, FileStorage):
            return filename.filename
        return self.relative_path + filename

    def get_url_thumbnail(self, filename):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(filename, FileStorage):
            return filename.filename
        return self.relative_path + thumbgen_filename(filename)

    def delete_file(self, filename):
        if False:
            while True:
                i = 10
        super(ImageManager, self).delete_file(filename)
        self.delete_thumbnail(filename)

    def delete_thumbnail(self, filename):
        if False:
            i = 10
            return i + 15
        path = self.get_path(self.thumbnail_fn(filename))
        if op.exists(path):
            os.remove(path)

    def save_file(self, data, filename, size=None, thumbnail_size=None):
        if False:
            while True:
                i = 10
        '\n        Saves an image File\n\n        :param data: FileStorage from Flask form upload field\n        :param filename: Filename with full path\n\n        '
        max_size = size or self.max_size
        thumbnail_size = thumbnail_size or self.thumbnail_size
        if data and isinstance(data, FileStorage):
            try:
                self.image = Image.open(data)
            except Exception as e:
                raise ValidationError('Invalid image: %s' % e)
        path = self.get_path(filename)
        if not op.exists(op.dirname(path)):
            os.makedirs(os.path.dirname(path), self.permission)
        (filename, format) = self.get_save_format(filename, self.image)
        if self.image and (self.image.format != format or max_size):
            if max_size:
                image = self.resize(self.image, max_size)
            else:
                image = self.image
            self.save_image(image, self.get_path(filename), format)
        else:
            data.seek(0)
            data.save(path)
        self.save_thumbnail(data, filename, format, thumbnail_size)
        return filename

    def save_thumbnail(self, data, filename, format, thumbnail_size=None):
        if False:
            i = 10
            return i + 15
        thumbnail_size = thumbnail_size or self.thumbnail_size
        if self.image and thumbnail_size:
            path = self.get_path(self.thumbnail_fn(filename))
            self.save_image(self.resize(self.image, thumbnail_size), path, format)

    def resize(self, image, size):
        if False:
            for i in range(10):
                print('nop')
        '\n        Resizes the image\n\n        :param image: The image object\n        :param size: size is PIL tuple (width, heigth, force) ex: (200,100,True)\n        '
        (width, height, force) = size
        if image.size[0] > width or image.size[1] > height:
            if force:
                return ImageOps.fit(self.image, (width, height), Image.ANTIALIAS)
            else:
                thumb = self.image.copy()
                thumb.thumbnail((width, height), Image.ANTIALIAS)
                return thumb
        return image

    def save_image(self, image, path, format='JPEG'):
        if False:
            while True:
                i = 10
        if image.mode not in ('RGB', 'RGBA'):
            image = image.convert('RGBA')
        with open(path, 'wb') as fp:
            image.save(fp, format)

    def get_save_format(self, filename, image):
        if False:
            print('Hello World!')
        if image.format not in self.keep_image_formats:
            (name, ext) = op.splitext(filename)
            filename = '%s.jpg' % name
            return (filename, 'JPEG')
        return (filename, image.format)

def uuid_namegen(file_data):
    if False:
        return 10
    return str(uuid.uuid1()) + '_sep_' + file_data.filename

def get_file_original_name(name):
    if False:
        while True:
            i = 10
    "\n    Use this function to get the user's original filename.\n    Filename is concatenated with <UUID>_sep_<FILE NAME>, to avoid collisions.\n    Use this function on your models on an additional function\n\n    ::\n\n        class ProjectFiles(Base):\n            id = Column(Integer, primary_key=True)\n            file = Column(FileColumn, nullable=False)\n\n            def file_name(self):\n                return get_file_original_name(str(self.file))\n\n    :param name:\n        The file name from model\n    :return:\n        Returns the user's original filename removes <UUID>_sep_\n    "
    re_match = re.findall('.*_sep_(.*)', name)
    if re_match:
        return re_match[0]
    else:
        return 'Not valid'

def uuid_originalname(uuid_filename):
    if False:
        i = 10
        return i + 15
    return uuid_filename.split('_sep_')[1]

def thumbgen_filename(filename):
    if False:
        i = 10
        return i + 15
    (name, ext) = op.splitext(filename)
    return '%s_thumb%s' % (name, ext)