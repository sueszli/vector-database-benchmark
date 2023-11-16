import copy
import six
from chainer.training import extension
from chainer import backend
from chainer.dataset import convert
from chainer import function
from chainer.utils import argument
import chainerx as chx

class GenericMultiNodeEvaluator(extension.Extension):
    """Generic multi-node evaluator for non-allreducable evaluation.

    This is to evaluate a Dataset that cannot evenly divided across
    all processes in the communicator, for evaluation calculation that
    is not applicable to a simple add-and-devide style averaging among
    processes.

    Users are recommeneded to implement its own local calculation
    ``calc_local()`` (e.g.  at each distributed GPU) and aggregation
    ``aggregate()`` of its results. Although it has built-in
    implementaiton of those two methods.

    It has several drawbacks; 1) Additional implementation of
    aggregation required to users, and 2) no compatibility with
    :class:`~chainer.training.extensions.Evaluator`.

    .. note:: No automatic support of Reporter is provided; Set it up
       at ``initialize()`` method

    Args:
        comm:
            ChainerMN communicator object
        iterator:
            An iterator for test dataset. Must be non-repeated.
        target (callable):
            A model to evaluate with test dataset
        device (int or chainer.backend.Device):
            A device indicator to send data with converter. Not used
            when the converter is not using any devices.
        converter (callable):
            A converter. Default value is
            :func:`chainer.dataset.concat_examples` .
        root (int):
            Rank number of root process to run bcast and gather with.
        progress_hook (callable):
            A callable that receives single argument for indicators. The
            callable is only callled at root process.

    """
    trigger = (1, 'epoch')
    default_name = 'validation'
    priority = extension.PRIORITY_WRITER
    name = None

    def __init__(self, comm, iterator, target, device=None, converter=convert.concat_examples, root=0, **kwargs):
        if False:
            i = 10
            return i + 15
        (progress_hook,) = argument.parse_kwargs(kwargs, ('progress_hook', None))
        self.comm = comm
        self.iterator = iterator
        self._targets = {'main': target}
        self.converter = converter
        if device is not None:
            device = backend.get_device(device)
        self.device = device
        self._progress_hook = progress_hook
        assert 0 <= root and root < self.comm.size
        self.root = root

    def __call__(self, trainer):
        if False:
            print('Hello World!')
        if hasattr(self.iterator, 'reset'):
            self.iterator.reset()
            it = self.iterator
        else:
            it = copy.copy(self.iterator)
        if self.comm is not None:
            gen = self._evaluate_local(it)
            if self.comm.rank == self.root:
                total_result = self.aggregate([result for result in gen])
            else:
                for _ in gen:
                    pass
                total_result = None
        else:
            gen = self._evaluate_local_single(self, it)
            total_result = self.aggregate([result for result in gen])
        return total_result

    def calc_local(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'A generic method for local calculation.\n\n        Override this method to run its local calculation.  Otherwise,\n        results are calculated with original target and test dataset.\n\n        Args:\n            args:\n                Result of converter when it is tuple.\n            kwargs:\n                Result of converter when it is dict.\n\n        Returns:\n            Arbrary value may be returned, but must not be ``None``.\n\n        '
        target = self._targets['main']
        return target(*args, **kwargs)

    def aggregate(self, results):
        if False:
            for i in range(10):
                print('nop')
        'A generic aggregation method.\n\n        Override this method for original aggregation calculation. By\n        default, it just does nothing but returns the input. This\n        method is called once and only once across the cluster, at\n        root process. Reporting can be run here.\n\n        Args:\n            results (list):\n                List of return value of ``calc_local()`` obtained from\n                all nodes..\n\n        '
        return results

    def _evaluate_local_single(self, iterator):
        if False:
            i = 10
            return i + 15
        for batch in iterator:
            in_arrays = convert._call_converter(self.converter, batch, self.device)
            with function.no_backprop_mode():
                if isinstance(in_arrays, tuple):
                    results = self.calc_local(*in_arrays)
                elif isinstance(in_arrays, dict):
                    results = self.calc_local(**in_arrays)
                else:
                    results = self.calc_local(in_arrays)
            if self._progress_hook:
                self._progress_hook(batch)
            yield results

    def _evaluate_local(self, iterator):
        if False:
            print('Hello World!')
        gather_interval = 8
        all_done = None
        while not all_done:
            all_done = None
            results = None
            for _ in range(gather_interval):
                try:
                    batch = iterator.next()
                    in_arrays = convert._call_converter(self.converter, batch, self.device)
                    with function.no_backprop_mode():
                        if isinstance(in_arrays, tuple):
                            results = self.calc_local(*in_arrays)
                        elif isinstance(in_arrays, dict):
                            results = self.calc_local(**in_arrays)
                        else:
                            results = self.calc_local(in_arrays)
                    if self.comm.rank == self.root and self._progress_hook:
                        self._progress_hook(batch)
                except StopIteration:
                    batch = None
                    results = None
                results = self.comm.gather_obj(results, root=self.root)
                if self.comm.rank == self.root:
                    valid_results = [r for r in results if r is not None]
                    for result in valid_results:
                        yield result
                    all_done = len(valid_results) == 0
            all_done = self.comm.bcast_obj(all_done, root=self.root)
        return

def create_multi_node_evaluator(actual_evaluator, communicator):
    if False:
        for i in range(10):
            print('nop')
    'Create a multi node evaluator from a normal evaluator.\n\n    Actually this method patches the evaluator to work in multi node\n    environment. This method adds several hidden attributes starting\n    with `_mn_` prefix.\n\n    Args:\n        actual_evaluator: evaluator to be patched\n            (e.g., ``chainer.training.extensions.Evaluator``)\n        communicator: ChainerMN communicator\n\n    Returns:\n        The multi-node patched ``actual_evaluator``.\n\n    .. note:: After patched, original evaluator does not work\n              correctly in non-MPI environment.\n\n    '
    actual_evaluator._mn_original_evaluate = actual_evaluator.evaluate
    actual_evaluator._mn_communicator = communicator

    def new_evaluate(self):
        if False:
            while True:
                i = 10
        local_mean_dict = self._mn_original_evaluate()
        arrays = list(local_mean_dict.values())
        if len(arrays) > 0:
            array0 = list(local_mean_dict.values())[0]
            xp = backend.get_array_module(array0)
            if xp == chx and array0.device.backend.name == 'cuda':
                local_mean_dict = {name: chx.to_numpy(value) for (name, value) in local_mean_dict.items()}
        global_mean_dict = {name: self._mn_communicator.allreduce_obj(value) / self._mn_communicator.size for (name, value) in sorted(local_mean_dict.items())}
        return global_mean_dict
    actual_evaluator.evaluate = six.create_bound_method(new_evaluate, actual_evaluator)
    return actual_evaluator