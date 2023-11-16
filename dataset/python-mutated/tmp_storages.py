"""
Copyright (c) Bojan Mihelac and individual contributors.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

    1. Redistributions of source code must retain the above copyright notice,
       this list of conditions and the following disclaimer.

    2. Redistributions in binary form must reproduce the above copyright
       notice, this list of conditions and the following disclaimer in the
       documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import os
import tempfile
from uuid import uuid4
from django.core.cache import cache
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage

class BaseStorage:

    def __init__(self, name=None):
        if False:
            for i in range(10):
                print('nop')
        self.name = name

    def save(self, data, mode='w'):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def read(self, read_mode='r'):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def remove(self):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

class TempFolderStorage(BaseStorage):

    def open(self, mode='r'):
        if False:
            print('Hello World!')
        if self.name:
            return open(self.get_full_path(), mode)
        else:
            tmp_file = tempfile.NamedTemporaryFile(delete=False)
            self.name = tmp_file.name
            return tmp_file

    def save(self, data, mode='w'):
        if False:
            for i in range(10):
                print('nop')
        with self.open(mode=mode) as file:
            file.write(data)

    def read(self, mode='r'):
        if False:
            for i in range(10):
                print('nop')
        with self.open(mode=mode) as file:
            return file.read()

    def remove(self):
        if False:
            i = 10
            return i + 15
        os.remove(self.get_full_path())

    def get_full_path(self):
        if False:
            return 10
        return os.path.join(tempfile.gettempdir(), self.name)

class CacheStorage(BaseStorage):
    """
    By default memcache maximum size per key is 1MB, be careful with large files.
    """
    CACHE_LIFETIME = 86400
    CACHE_PREFIX = 'django-import-export-'

    def save(self, data, mode=None):
        if False:
            while True:
                i = 10
        if not self.name:
            self.name = uuid4().hex
        cache.set(self.CACHE_PREFIX + self.name, data, self.CACHE_LIFETIME)

    def read(self, read_mode='r'):
        if False:
            print('Hello World!')
        return cache.get(self.CACHE_PREFIX + self.name)

    def remove(self):
        if False:
            i = 10
            return i + 15
        cache.delete(self.name)

class MediaStorage(BaseStorage):
    MEDIA_FOLDER = 'django-import-export'

    def save(self, data, mode=None):
        if False:
            for i in range(10):
                print('nop')
        if not self.name:
            self.name = uuid4().hex
        default_storage.save(self.get_full_path(), ContentFile(data))

    def read(self, read_mode='rb'):
        if False:
            for i in range(10):
                print('nop')
        with default_storage.open(self.get_full_path(), mode=read_mode) as f:
            return f.read()

    def remove(self):
        if False:
            for i in range(10):
                print('nop')
        default_storage.delete(self.get_full_path())

    def get_full_path(self):
        if False:
            return 10
        return os.path.join(self.MEDIA_FOLDER, self.name)