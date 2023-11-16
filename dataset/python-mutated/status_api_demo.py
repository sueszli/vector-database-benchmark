import time
import threading
import queue as Queue
from typing import Any, Callable, List, Tuple
from pyspark import SparkConf, SparkContext

def delayed(seconds: int) -> Callable[[Any], Any]:
    if False:
        i = 10
        return i + 15

    def f(x: int) -> int:
        if False:
            for i in range(10):
                print('nop')
        time.sleep(seconds)
        return x
    return f

def call_in_background(f: Callable[..., Any], *args: Any) -> Queue.Queue:
    if False:
        for i in range(10):
            print('nop')
    result: Queue.Queue = Queue.Queue(1)
    t = threading.Thread(target=lambda : result.put(f(*args)))
    t.daemon = True
    t.start()
    return result

def main() -> None:
    if False:
        while True:
            i = 10
    conf = SparkConf().set('spark.ui.showConsoleProgress', 'false')
    sc = SparkContext(appName='PythonStatusAPIDemo', conf=conf)

    def run() -> List[Tuple[int, int]]:
        if False:
            while True:
                i = 10
        rdd = sc.parallelize(range(10), 10).map(delayed(2))
        reduced = rdd.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)
        return reduced.map(delayed(2)).collect()
    result = call_in_background(run)
    status = sc.statusTracker()
    while result.empty():
        ids = status.getJobIdsForGroup()
        for id in ids:
            job = status.getJobInfo(id)
            assert job is not None
            print('Job', id, 'status: ', job.status)
            for sid in job.stageIds:
                info = status.getStageInfo(sid)
                if info:
                    print('Stage %d: %d tasks total (%d active, %d complete)' % (sid, info.numTasks, info.numActiveTasks, info.numCompletedTasks))
        time.sleep(1)
    print('Job results are:', result.get())
    sc.stop()
if __name__ == '__main__':
    main()