from collections import OrderedDict
from threading import Lock
from typing import Callable, Optional
from uuid import UUID

class ModelCache:
    """Cache for model prediction functions on executors.

    This requires the `spark.python.worker.reuse` configuration to be set to `true`, otherwise a
    new python worker (with an empty cache) will be started for every task.

    If a python worker is idle for more than one minute (per the IDLE_WORKER_TIMEOUT_NS setting in
    PythonWorkerFactory.scala), it will be killed, effectively clearing the cache until a new python
    worker is started.

    Caching large models can lead to out-of-memory conditions, which may require adjusting spark
    memory configurations, e.g. `spark.executor.memoryOverhead`.
    """
    _models: OrderedDict = OrderedDict()
    _capacity: int = 3
    _lock: Lock = Lock()

    @staticmethod
    def add(uuid: UUID, predict_fn: Callable) -> None:
        if False:
            i = 10
            return i + 15
        with ModelCache._lock:
            ModelCache._models[uuid] = predict_fn
            ModelCache._models.move_to_end(uuid)
            if len(ModelCache._models) > ModelCache._capacity:
                ModelCache._models.popitem(last=False)

    @staticmethod
    def get(uuid: UUID) -> Optional[Callable]:
        if False:
            for i in range(10):
                print('nop')
        with ModelCache._lock:
            predict_fn = ModelCache._models.get(uuid)
            if predict_fn:
                ModelCache._models.move_to_end(uuid)
            return predict_fn