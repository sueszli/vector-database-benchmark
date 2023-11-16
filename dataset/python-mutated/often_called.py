import cython

@cython.profile(False)
def my_often_called_function():
    if False:
        return 10
    pass