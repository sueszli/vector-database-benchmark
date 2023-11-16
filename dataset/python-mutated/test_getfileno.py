from rich._fileno import get_fileno

def test_get_fileno():
    if False:
        print('Hello World!')

    class FileLike:

        def fileno(self) -> int:
            if False:
                while True:
                    i = 10
            return 123
    assert get_fileno(FileLike()) == 123

def test_get_fileno_missing():
    if False:
        while True:
            i = 10

    class FileLike:
        pass
    assert get_fileno(FileLike()) is None

def test_get_fileno_broken():
    if False:
        for i in range(10):
            print('nop')

    class FileLike:

        def fileno(self) -> int:
            if False:
                for i in range(10):
                    print('nop')
            1 / 0
            return 123
    assert get_fileno(FileLike()) is None