from DynamicLibraryWithoutArgspec import DynamicLibraryWithoutArgspec

class DynamicLibraryWithKwargsSupportWithoutArgspec(DynamicLibraryWithoutArgspec):

    def run_keyword(self, name, args, kwargs):
        if False:
            return 10
        return getattr(self, name)(*args, **kwargs)

    def do_something_with_kwargs(self, a, b=2, c=3, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        print(a, b, c, ' '.join(('%s:%s' % (k, v) for (k, v) in kwargs.items())))