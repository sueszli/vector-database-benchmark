class TimeWrapper(object):
    """
    Overview:
        Abstract class method that defines ``TimeWrapper`` class

    Interface:
        ``wrapper``, ``start_time``, ``end_time``
    """

    @classmethod
    def wrapper(cls, fn):
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Classmethod wrapper, wrap a function and automatically return its running time\n\n        - fn (:obj:`function`): The function to be wrap and timed\n        '

        def time_func(*args, **kwargs):
            if False:
                return 10
            cls.start_time()
            ret = fn(*args, **kwargs)
            t = cls.end_time()
            return (ret, t)
        return time_func

    @classmethod
    def start_time(cls):
        if False:
            print('Hello World!')
        '\n        Overview:\n            Abstract classmethod, start timing\n        '
        raise NotImplementedError

    @classmethod
    def end_time(cls):
        if False:
            while True:
                i = 10
        '\n        Overview:\n            Abstract classmethod, stop timing\n        '
        raise NotImplementedError