import time
import ray
from ray.util.metrics import Counter, Gauge, Histogram
ray.init(_metrics_export_port=8080)

@ray.remote
class MyActor:

    def __init__(self, name):
        if False:
            i = 10
            return i + 15
        self._curr_count = 0
        self.counter = Counter('num_requests', description='Number of requests processed by the actor.', tag_keys=('actor_name',))
        self.counter.set_default_tags({'actor_name': name})
        self.gauge = Gauge('curr_count', description='Current count held by the actor. Goes up and down.', tag_keys=('actor_name',))
        self.gauge.set_default_tags({'actor_name': name})
        self.histogram = Histogram('request_latency', description='Latencies of requests in ms.', boundaries=[0.1, 1], tag_keys=('actor_name',))
        self.histogram.set_default_tags({'actor_name': name})

    def process_request(self, num):
        if False:
            for i in range(10):
                print('nop')
        start = time.time()
        self._curr_count += num
        self.counter.inc()
        self.gauge.set(self._curr_count)
        self.histogram.observe(1000 * (time.time() - start))
        return self._curr_count
print('Starting actor.')
my_actor = MyActor.remote('my_actor')
print('Calling actor.')
my_actor.process_request.remote(-10)
print('Calling actor.')
my_actor.process_request.remote(5)
print('Metrics should be exported.')
print('See http://localhost:8080 (this may take a few seconds to load).')
time.sleep(30)
print('Exiting!')