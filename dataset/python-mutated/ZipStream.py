import io
import os
import zipfile

class ZipStream(object):

    def __init__(self, dir_path):
        if False:
            print('Hello World!')
        self.dir_path = dir_path
        self.pos = 0
        self.buff_pos = 0
        self.zf = zipfile.ZipFile(self, 'w', zipfile.ZIP_DEFLATED, allowZip64=True)
        self.buff = io.BytesIO()
        self.file_list = self.getFileList()

    def getFileList(self):
        if False:
            for i in range(10):
                print('nop')
        for (root, dirs, files) in os.walk(self.dir_path):
            for file in files:
                file_path = root + '/' + file
                relative_path = os.path.join(os.path.relpath(root, self.dir_path), file)
                yield (file_path, relative_path)
        self.zf.close()

    def read(self, size=60 * 1024):
        if False:
            return 10
        for (file_path, relative_path) in self.file_list:
            self.zf.write(file_path, relative_path)
            if self.buff.tell() >= size:
                break
        self.buff.seek(0)
        back = self.buff.read()
        self.buff.truncate(0)
        self.buff.seek(0)
        self.buff_pos += len(back)
        return back

    def write(self, data):
        if False:
            print('Hello World!')
        self.pos += len(data)
        self.buff.write(data)

    def tell(self):
        if False:
            print('Hello World!')
        return self.pos

    def seek(self, pos, whence=0):
        if False:
            print('Hello World!')
        if pos >= self.buff_pos:
            self.buff.seek(pos - self.buff_pos, whence)
            self.pos = pos

    def flush(self):
        if False:
            print('Hello World!')
        pass
if __name__ == '__main__':
    zs = ZipStream('.')
    out = open('out.zip', 'wb')
    while 1:
        data = zs.read()
        print('Write %s' % len(data))
        if not data:
            break
        out.write(data)
    out.close()