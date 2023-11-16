import io
from chainer.serializers import load_npz
from chainer.serializers import save_npz
from chainer.training.extension import Extension
from chainer.training.extensions._snapshot import _find_latest_snapshot

def multi_node_snapshot(comm, snapshot, replica_sets):
    if False:
        for i in range(10):
            print('nop')
    "Create trainer extension for multi-node snapshots\n\n    Provides generis multi-node snapshot saving and auto-load feature\n    at multi-node environment, leveraging power of single-node\n    snapshot.\n\n    In many cases snapshot target may differ, e.g. only trainer of\n    rank 0 process often has extensions such as ``LogReport`` and so\n    on, to not confuse terminal output. Just loading at one process\n    and broadcasting it to other processes does not work in that case.\n\n    This wrapper addresses that issue by defining sets of replicas\n    where within the set the target object is replicated and supposed\n    to be same among processes. For example, a trainer example, only\n    the trainer at rank ``0`` has special extensions and others\n    doesn't::\n\n        trainer = Trainer(updater)\n        if comm.rank == 0:\n            trainer.extend(extensions.DumpGraph('main/loss'))\n            trainer.extend(extensions.LogReport())\n            trainer.extend(extensions.PrintReport(\n                ['epoch', 'main/loss', 'validation/main/loss',\n                 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))\n            trainer.extend(extensions.ProgressBar())\n\n    This case can be described with two replica sets, where each set\n    can be represented as single integer that indicates rank number,\n    or iterable set/list/generator of integers like this::\n\n        replica_sets = [[0], range(1, comm.size)]\n\n    Here the first replica set is described as ``[0]``, or simply in\n    short just ``0``, and the second replica set is ``range(1,\n    comm.size)``, representing rest of processes other than ``0``. The\n    remaining list can be omitted. Thus in that case, it can be\n    simplified more::\n\n        replica_sets = [0,]\n\n    In this case, the snapshot will be saved at rank ``0`` process and\n    at rank ``1`` process. The latter represents the replica set of\n    ``range(1, comm.size)`` . In this case autoloading at\n    initialization of snapshot extension works after the restart\n    cleanly, even though the size of the communicator differs.\n\n    Once the replica sets are defined, it can be easily extended::\n\n        replica_sets = [0,]\n        snapshot = multi_node_snapshot(comm, extensions.snapshot(),\n                                       replica_sets)\n        trainer.extend(snapshot, trigger=(1, 'epoch'))\n\n\n    More example tuples of replica set representation follows:\n\n    ===================== ===== ==============================================\n    code                  nproc actual sets\n    ===================== ===== ==============================================\n    ``[0]``               ``4`` ``[{0}, {1, 2, 3}]``\n    ``[0, 1]``            ``4`` ``[{0}, {1}, {2, 3}]``\n    ``[0, 1], [2, 3]]``   ``4`` ``[{0, 1}, {2, 3}]``\n    ``[]``                ``4`` ``[{0, 1, 2, 3}]``\n    ``[range(0, 8, 2)]``  ``8`` ``[set(range(0, 8, 2)), set(range(1, 8, 2))]``\n    ===================== ===== ==============================================\n\n    Args:\n        comm (ChainerMN communicator): communicater object\n        snapshot: Snapshot extension object obtained via\n              :meth:`~chainer.training.extensions.snapshot` .\n        replica_sets: list of replica set definition, where\n              a replica set can be defined by single integer\n              as rank number, or iterable integers.\n\n    Returns:\n        Trainer extension that wraps ``snapshot`` and properly\n        controles number of snapshots.\n\n    "
    return _MultiNodeSnapshot(comm, snapshot, replica_sets)

def _parse_replica_sets(replica_sets, size):
    if False:
        print('Hello World!')
    sets = []
    for replica_set in replica_sets:
        if isinstance(replica_set, int):
            assert replica_set >= 0
            assert replica_set < size
            sets.append({replica_set})
        else:
            for i in replica_set:
                assert i >= 0
                assert i < size
            sets.append(set(replica_set))
    if size > sum((len(s) for s in sets)):
        all_ranks = set(range(size))
        all_exp = set()
        for s in sets:
            all_exp |= s
        rest = all_ranks - all_exp
        if rest:
            sets.append(rest)
    assert size == sum((len(s) for s in sets))
    all_sum = set()
    for s in sets:
        all_sum |= s
    assert size == len(all_sum)
    return sets

class _MultiNodeSnapshot(Extension):

    def __init__(self, comm, snapshot, replica_sets):
        if False:
            for i in range(10):
                print('nop')
        assert comm is not None
        assert snapshot is not None
        self.comm = comm
        self.snapshot = snapshot
        if callable(snapshot.filename):
            filename_fun = snapshot.filename

            def append_rank(trainer):
                if False:
                    while True:
                        i = 10
                filename = filename_fun(trainer)
                return '{}.{}'.format(filename, comm.rank)
            snapshot.filename = append_rank
        else:
            filename = '{}.{}'.format(snapshot.filename, comm.rank)
            snapshot.filename = filename
        sets = _parse_replica_sets(replica_sets, comm.size)
        self.master = None
        self.replica_set = []
        for s in sets:
            if self.comm.rank in s:
                self.master = min(s)
                self.replica_set = s
                break
        assert self.master is not None
        assert self.comm.rank in self.replica_set

    @property
    def is_master(self):
        if False:
            return 10
        return self.master == self.comm.rank

    def initialize(self, trainer):
        if False:
            print('Hello World!')
        if self.is_master:
            self.snapshot.initialize(trainer)
        if not self.snapshot.autoload:
            return
        if self.snapshot._target is None:
            target = trainer
        else:
            target = self.snapshot._target
        if self.is_master:
            filename = _find_latest_snapshot(self.snapshot.filename, trainer.out)
            if filename is None:
                data = None
            else:
                buf = io.BytesIO()
                save_npz(buf, target)
                data = buf.getvalue()
            for rank in self.replica_set:
                if rank == self.comm.rank:
                    continue
                self.comm.send_obj(data, rank)
        else:
            data = self.comm.recv_obj(self.master)
            if data is None:
                return
            load_npz(io.BytesIO(data), target)

    def on_error(self, trainer, e, t):
        if False:
            while True:
                i = 10
        if self.is_master:
            self.snapshot.on_error(trainer, e, t)

    def __call__(self, trainer):
        if False:
            print('Hello World!')
        if self.is_master:
            self.snapshot(trainer)

    def finalize(self):
        if False:
            i = 10
            return i + 15
        if self.is_master:
            self.snapshot.finalize()