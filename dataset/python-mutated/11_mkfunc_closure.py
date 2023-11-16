class _OverlappedFuture(futures.Future):

    def __init__(self, ov, *, loop=None):
        if False:
            print('Hello World!')
        super().__init__(loop=loop)