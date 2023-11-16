from typing import TYPE_CHECKING
from tensorboardX import SummaryWriter
if TYPE_CHECKING:
    from ding.framework import Parallel

class DistributedWriter(SummaryWriter):
    """
    Overview:
        A simple subclass of SummaryWriter that supports writing to one process in multi-process mode.
        The best way is to use it in conjunction with the ``router`` to take advantage of the message             and event components of the router (see ``writer.plugin``).
    """
    root = None

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self._default_writer_to_disk = kwargs.get('write_to_disk') if 'write_to_disk' in kwargs else True
        kwargs['write_to_disk'] = False
        super().__init__(*args, **kwargs)
        self._in_parallel = False
        self._router = None
        self._is_writer = False
        self._lazy_initialized = False

    @classmethod
    def get_instance(cls, *args, **kwargs) -> 'DistributedWriter':
        if False:
            print('Hello World!')
        '\n        Overview:\n            Get instance and set the root level instance on the first called. If args and kwargs is none,\n            this method will return root instance.\n        '
        if args or kwargs:
            ins = cls(*args, **kwargs)
            if cls.root is None:
                cls.root = ins
            return ins
        else:
            return cls.root

    def plugin(self, router: 'Parallel', is_writer: bool=False) -> 'DistributedWriter':
        if False:
            for i in range(10):
                print('nop')
        '\n        Overview:\n            Plugin ``router``, so when using this writer with active router, it will automatically send requests                to the main writer instead of writing it to the disk. So we can collect data from multiple processes                and write them into one file.\n        Examples:\n            >>> DistributedWriter().plugin(router, is_writer=True)\n        '
        if router.is_active:
            self._in_parallel = True
            self._router = router
            self._is_writer = is_writer
            if is_writer:
                self.initialize()
            self._lazy_initialized = True
            router.on('distributed_writer', self._on_distributed_writer)
        return self

    def _on_distributed_writer(self, fn_name: str, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        if self._is_writer:
            getattr(self, fn_name)(*args, **kwargs)

    def initialize(self):
        if False:
            return 10
        self.close()
        self._write_to_disk = self._default_writer_to_disk
        self._get_file_writer()
        self._lazy_initialized = True

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        self.close()

def enable_parallel(fn_name, fn):
    if False:
        for i in range(10):
            print('nop')

    def _parallel_fn(self: DistributedWriter, *args, **kwargs):
        if False:
            while True:
                i = 10
        if not self._lazy_initialized:
            self.initialize()
        if self._in_parallel and (not self._is_writer):
            self._router.emit('distributed_writer', fn_name, *args, **kwargs)
        else:
            fn(self, *args, **kwargs)
    return _parallel_fn
ready_to_parallel_fns = ['add_audio', 'add_custom_scalars', 'add_custom_scalars_marginchart', 'add_custom_scalars_multilinechart', 'add_embedding', 'add_figure', 'add_graph', 'add_graph_deprecated', 'add_histogram', 'add_histogram_raw', 'add_hparams', 'add_image', 'add_image_with_boxes', 'add_images', 'add_mesh', 'add_onnx_graph', 'add_openvino_graph', 'add_pr_curve', 'add_pr_curve_raw', 'add_scalar', 'add_scalars', 'add_text', 'add_video']
for fn_name in ready_to_parallel_fns:
    if hasattr(DistributedWriter, fn_name):
        setattr(DistributedWriter, fn_name, enable_parallel(fn_name, getattr(DistributedWriter, fn_name)))
distributed_writer = DistributedWriter()