import vaex.file

class ArrowOpener:

    @classmethod
    def quick_test(cls, path, *args, **kwargs):
        if False:
            while True:
                i = 10
        return vaex.file.ext(path) == '.arrow'

    @classmethod
    def can_open(cls, path, *args, **kwargs):
        if False:
            while True:
                i = 10
        return cls.quick_test(path, *args, **kwargs)

    @staticmethod
    def open(path, *args, **kwargs):
        if False:
            print('Hello World!')
        from .dataset import open
        return open(path, *args, **kwargs)

class FeatherOpener(ArrowOpener):

    @classmethod
    def quick_test(cls, path, *args, **kwargs):
        if False:
            while True:
                i = 10
        return vaex.file.ext(path) == '.feather'

    @classmethod
    def can_open(cls, path, *args, **kwargs):
        if False:
            return 10
        return cls.quick_test(path, *args, **kwargs)

    @staticmethod
    def open(path, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        from .dataset import open
        return open(path, *args, **kwargs)

class ParquetOpener:

    @classmethod
    def quick_test(cls, path, *args, **kwargs):
        if False:
            print('Hello World!')
        ext = vaex.file.ext(path)
        return ext == '.parquet' or ext == ''

    @classmethod
    def can_open(cls, path, *args, **kwargs):
        if False:
            print('Hello World!')
        return cls.quick_test(path, *args, **kwargs)

    @staticmethod
    def open(path, *args, **kwargs):
        if False:
            return 10
        from .dataset import open_parquet
        return open_parquet(path, *args, **kwargs)