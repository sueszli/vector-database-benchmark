from deeplake.core.compute.provider import ComputeProvider, get_progress_bar

class SerialProvider(ComputeProvider):

    def __init__(self):
        if False:
            print('Hello World!')
        pass

    def map(self, func, iterable):
        if False:
            print('Hello World!')
        return list(map(func, iterable))

    def map_with_progress_bar(self, func, iterable, total_length: int, desc=None, pbar=None, pqueue=None):
        if False:
            print('Hello World!')
        progress_bar = pbar or get_progress_bar(total_length, desc)

        def sub_func(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')

            def pg_callback(value: int):
                if False:
                    return 10
                progress_bar.update(value)
            return func(pg_callback, *args, **kwargs)
        result = self.map(sub_func, iterable)
        return result

    def create_queue(self):
        if False:
            return 10
        return None

    def close(self):
        if False:
            print('Hello World!')
        return