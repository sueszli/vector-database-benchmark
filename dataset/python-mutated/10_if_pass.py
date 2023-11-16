from weakref import ref

class _localimpl:

    def create_dict(self, thread):
        if False:
            for i in range(10):
                print('nop')
        'Create a new dict for the current thread, and return it.'
        localdict = {}
        idt = id(thread)

        def thread_deleted(_, idt=idt):
            if False:
                while True:
                    i = 10
            local = wrlocal()
            if local is not None:
                pass
        wrlocal = ref(self, local_deleted)
        return localdict