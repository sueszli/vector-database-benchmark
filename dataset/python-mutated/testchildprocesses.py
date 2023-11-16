import psutil

class KillLeftoverChildrenTestMixin:
    _children_on_start = None

    @staticmethod
    def _get_process_children():
        if False:
            for i in range(10):
                print('nop')
        p = psutil.Process()
        return set([c.pid for c in p.children(recursive=True)])

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self._children_on_start = self._get_process_children()

    def tearDown(self):
        if False:
            print('Hello World!')
        super().tearDown()
        nkotb = self._get_process_children() - self._children_on_start
        for k in nkotb:
            try:
                p = psutil.Process(k)
                p.kill()
            except psutil.Error:
                pass