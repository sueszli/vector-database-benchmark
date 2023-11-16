from utils_cv.common.gpu import db_num_workers, linux_with_gpu, is_binder, is_linux, is_windows, which_processor, system_info

def test_which_processor():
    if False:
        i = 10
        return i + 15
    which_processor()

def test_is_linux():
    if False:
        print('Hello World!')
    assert type(is_linux()) == bool

def test_is_windows():
    if False:
        while True:
            i = 10
    assert type(is_windows()) == bool

def test_linux_with_gpu():
    if False:
        while True:
            i = 10
    assert type(linux_with_gpu()) == bool

def test_is_binder():
    if False:
        i = 10
        return i + 15
    assert is_binder() == False

def test_db_num_workers():
    if False:
        for i in range(10):
            print('nop')
    if is_windows():
        assert db_num_workers() == 0
        assert db_num_workers(non_windows_num_workers=7) == 0
    else:
        assert db_num_workers() == 16
        assert db_num_workers(non_windows_num_workers=7) == 7

def test_system_info():
    if False:
        print('Hello World!')
    system_info()