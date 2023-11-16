import os
import shutil
import tarfile
import time
from io import BytesIO
from setup import Command, download_securely, is_ci

class ReVendor(Command):
    CAN_USE_SYSTEM_VERSION = True

    def add_options(self, parser):
        if False:
            for i in range(10):
                print('nop')
        parser.add_option('--path-to-%s' % self.NAME, help='Path to the extracted %s source' % self.TAR_NAME)
        parser.add_option('--%s-url' % self.NAME, default=self.DOWNLOAD_URL, help='URL to %s source archive in tar.gz format' % self.TAR_NAME)
        if self.CAN_USE_SYSTEM_VERSION:
            parser.add_option('--system-%s' % self.NAME, default=False, action='store_true', help='Treat %s as system copy and symlink instead of copy' % self.TAR_NAME)

    def download_vendor_release(self, tdir, url):
        if False:
            while True:
                i = 10
        self.info('Downloading %s:' % self.TAR_NAME, url)
        num = 5 if is_ci else 1
        for i in range(num):
            try:
                raw = download_securely(url)
            except Exception as err:
                if i == num - 1:
                    raise
                self.info(f'Download failed with error "{err}" sleeping and retrying...')
                time.sleep(2)
        with tarfile.open(fileobj=BytesIO(raw)) as tf:
            tf.extractall(tdir)
            if len(os.listdir(tdir)) == 1:
                return self.j(tdir, os.listdir(tdir)[0])
            else:
                return tdir

    def add_file_pre(self, name, raw):
        if False:
            for i in range(10):
                print('nop')
        pass

    def add_file(self, path, name):
        if False:
            print('Hello World!')
        with open(path, 'rb') as f:
            raw = f.read()
        self.add_file_pre(name, raw)
        dest = self.j(self.vendored_dir, *name.split('/'))
        base = os.path.dirname(dest)
        if not os.path.exists(base):
            os.makedirs(base)
        if self.use_symlinks:
            os.symlink(path, dest)
        else:
            with open(dest, 'wb') as f:
                f.write(raw)

    def add_tree(self, base, prefix, ignore=lambda n: False):
        if False:
            i = 10
            return i + 15
        for (dirpath, dirnames, filenames) in os.walk(base):
            for fname in filenames:
                f = os.path.join(dirpath, fname)
                name = prefix + '/' + os.path.relpath(f, base).replace(os.sep, '/')
                if not ignore(name):
                    self.add_file(f, name)

    @property
    def vendored_dir(self):
        if False:
            i = 10
            return i + 15
        return self.j(self.RESOURCES, self.NAME)

    def clean(self):
        if False:
            i = 10
            return i + 15
        if os.path.exists(self.vendored_dir):
            shutil.rmtree(self.vendored_dir)