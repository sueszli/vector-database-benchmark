class FakeFileSystem:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.files = {}
        self.dirs = {}

    def contents_of(self, path):
        if False:
            for i in range(10):
                print('nop')
        return self.files[path]

    def exists(self, path):
        if False:
            i = 10
            return i + 15
        return path in self.files

    def entries_if_dir_exists(self, path):
        if False:
            print('Hello World!')
        return self.dirs.get(path, [])

    def create_fake_file(self, path, contents=''):
        if False:
            for i in range(10):
                print('nop')
        import os
        self.files[path] = contents
        self.create_fake_dir(os.path.dirname(path), os.path.basename(path))

    def create_fake_dir(self, dir_path, *dir_entries):
        if False:
            for i in range(10):
                print('nop')
        self.dirs[dir_path] = dir_entries