import redis
import json
import time
import logging
from django.conf import settings
from django.apps import apps
from awx.main.consumers import emit_channel_notification
from awx.main.utils import is_testing
root_key = settings.SUBSYSTEM_METRICS_REDIS_KEY_PREFIX
logger = logging.getLogger('awx.main.analytics')

class BaseM:

    def __init__(self, field, help_text):
        if False:
            i = 10
            return i + 15
        self.field = field
        self.help_text = help_text
        self.current_value = 0
        self.metric_has_changed = False

    def reset_value(self, conn):
        if False:
            return 10
        conn.hset(root_key, self.field, 0)
        self.current_value = 0

    def inc(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.current_value += value
        self.metric_has_changed = True

    def set(self, value):
        if False:
            i = 10
            return i + 15
        self.current_value = value
        self.metric_has_changed = True

    def get(self):
        if False:
            while True:
                i = 10
        return self.current_value

    def decode(self, conn):
        if False:
            i = 10
            return i + 15
        value = conn.hget(root_key, self.field)
        return self.decode_value(value)

    def to_prometheus(self, instance_data):
        if False:
            for i in range(10):
                print('nop')
        output_text = f'# HELP {self.field} {self.help_text}\n# TYPE {self.field} gauge\n'
        for instance in instance_data:
            if self.field in instance_data[instance]:
                output_text += f'{self.field}{{node="{instance}"}} {instance_data[instance][self.field]}\n'
        return output_text

class FloatM(BaseM):

    def decode_value(self, value):
        if False:
            i = 10
            return i + 15
        if value is not None:
            return float(value)
        else:
            return 0.0

    def store_value(self, conn):
        if False:
            i = 10
            return i + 15
        if self.metric_has_changed:
            conn.hincrbyfloat(root_key, self.field, self.current_value)
            self.current_value = 0
            self.metric_has_changed = False

class IntM(BaseM):

    def decode_value(self, value):
        if False:
            i = 10
            return i + 15
        if value is not None:
            return int(value)
        else:
            return 0

    def store_value(self, conn):
        if False:
            return 10
        if self.metric_has_changed:
            conn.hincrby(root_key, self.field, self.current_value)
            self.current_value = 0
            self.metric_has_changed = False

class SetIntM(BaseM):

    def decode_value(self, value):
        if False:
            i = 10
            return i + 15
        if value is not None:
            return int(value)
        else:
            return 0

    def store_value(self, conn):
        if False:
            print('Hello World!')
        if self.metric_has_changed:
            conn.hset(root_key, self.field, self.current_value)
            self.metric_has_changed = False

class SetFloatM(SetIntM):

    def decode_value(self, value):
        if False:
            return 10
        if value is not None:
            return float(value)
        else:
            return 0

class HistogramM(BaseM):

    def __init__(self, field, help_text, buckets):
        if False:
            while True:
                i = 10
        self.buckets = buckets
        self.buckets_to_keys = {}
        for b in buckets:
            self.buckets_to_keys[b] = IntM(field + '_' + str(b), '')
        self.inf = IntM(field + '_inf', '')
        self.sum = IntM(field + '_sum', '')
        super(HistogramM, self).__init__(field, help_text)

    def reset_value(self, conn):
        if False:
            for i in range(10):
                print('nop')
        conn.hset(root_key, self.field, 0)
        self.inf.reset_value(conn)
        self.sum.reset_value(conn)
        for b in self.buckets_to_keys.values():
            b.reset_value(conn)
        super(HistogramM, self).reset_value(conn)

    def observe(self, value):
        if False:
            while True:
                i = 10
        for b in self.buckets:
            if value <= b:
                self.buckets_to_keys[b].inc(1)
                break
        self.sum.inc(value)
        self.inf.inc(1)

    def decode(self, conn):
        if False:
            while True:
                i = 10
        values = {'counts': []}
        for b in self.buckets_to_keys:
            values['counts'].append(self.buckets_to_keys[b].decode(conn))
        values['sum'] = self.sum.decode(conn)
        values['inf'] = self.inf.decode(conn)
        return values

    def store_value(self, conn):
        if False:
            while True:
                i = 10
        for b in self.buckets:
            self.buckets_to_keys[b].store_value(conn)
        self.sum.store_value(conn)
        self.inf.store_value(conn)

    def to_prometheus(self, instance_data):
        if False:
            while True:
                i = 10
        output_text = f'# HELP {self.field} {self.help_text}\n# TYPE {self.field} histogram\n'
        for instance in instance_data:
            for (i, b) in enumerate(self.buckets):
                output_text += f'''{self.field}_bucket{{le="{b}",node="{instance}"}} {sum(instance_data[instance][self.field]['counts'][0:i + 1])}\n'''
            output_text += f'''{self.field}_bucket{{le="+Inf",node="{instance}"}} {instance_data[instance][self.field]['inf']}\n'''
            output_text += f'''{self.field}_count{{node="{instance}"}} {instance_data[instance][self.field]['inf']}\n'''
            output_text += f'''{self.field}_sum{{node="{instance}"}} {instance_data[instance][self.field]['sum']}\n'''
        return output_text

class Metrics:

    def __init__(self, auto_pipe_execute=False, instance_name=None):
        if False:
            return 10
        self.pipe = redis.Redis.from_url(settings.BROKER_URL).pipeline()
        self.conn = redis.Redis.from_url(settings.BROKER_URL)
        self.last_pipe_execute = time.time()
        self.metrics_have_changed = True
        self.pipe_execute_interval = settings.SUBSYSTEM_METRICS_INTERVAL_SAVE_TO_REDIS
        self.send_metrics_interval = settings.SUBSYSTEM_METRICS_INTERVAL_SEND_METRICS
        self.auto_pipe_execute = auto_pipe_execute
        Instance = apps.get_model('main', 'Instance')
        if instance_name:
            self.instance_name = instance_name
        elif is_testing():
            self.instance_name = 'awx_testing'
        else:
            self.instance_name = Instance.objects.my_hostname()
        METRICSLIST = [SetIntM('callback_receiver_events_queue_size_redis', 'Current number of events in redis queue'), IntM('callback_receiver_events_popped_redis', 'Number of events popped from redis'), IntM('callback_receiver_events_in_memory', 'Current number of events in memory (in transfer from redis to db)'), IntM('callback_receiver_batch_events_errors', 'Number of times batch insertion failed'), FloatM('callback_receiver_events_insert_db_seconds', 'Total time spent saving events to database'), IntM('callback_receiver_events_insert_db', 'Number of events batch inserted into database'), IntM('callback_receiver_events_broadcast', 'Number of events broadcast to other control plane nodes'), HistogramM('callback_receiver_batch_events_insert_db', 'Number of events batch inserted into database', settings.SUBSYSTEM_METRICS_BATCH_INSERT_BUCKETS), SetFloatM('callback_receiver_event_processing_avg_seconds', 'Average processing time per event per callback receiver batch'), FloatM('subsystem_metrics_pipe_execute_seconds', 'Time spent saving metrics to redis'), IntM('subsystem_metrics_pipe_execute_calls', 'Number of calls to pipe_execute'), FloatM('subsystem_metrics_send_metrics_seconds', 'Time spent sending metrics to other nodes'), SetFloatM('task_manager_get_tasks_seconds', 'Time spent in loading tasks from db'), SetFloatM('task_manager_start_task_seconds', 'Time spent starting task'), SetFloatM('task_manager_process_running_tasks_seconds', 'Time spent processing running tasks'), SetFloatM('task_manager_process_pending_tasks_seconds', 'Time spent processing pending tasks'), SetFloatM('task_manager__schedule_seconds', 'Time spent in running the entire _schedule'), IntM('task_manager__schedule_calls', 'Number of calls to _schedule, after lock is acquired'), SetFloatM('task_manager_recorded_timestamp', 'Unix timestamp when metrics were last recorded'), SetIntM('task_manager_tasks_started', 'Number of tasks started'), SetIntM('task_manager_running_processed', 'Number of running tasks processed'), SetIntM('task_manager_pending_processed', 'Number of pending tasks processed'), SetIntM('task_manager_tasks_blocked', 'Number of tasks blocked from running'), SetFloatM('task_manager_commit_seconds', 'Time spent in db transaction, including on_commit calls'), SetFloatM('dependency_manager_get_tasks_seconds', 'Time spent loading pending tasks from db'), SetFloatM('dependency_manager_generate_dependencies_seconds', 'Time spent generating dependencies for pending tasks'), SetFloatM('dependency_manager__schedule_seconds', 'Time spent in running the entire _schedule'), IntM('dependency_manager__schedule_calls', 'Number of calls to _schedule, after lock is acquired'), SetFloatM('dependency_manager_recorded_timestamp', 'Unix timestamp when metrics were last recorded'), SetIntM('dependency_manager_pending_processed', 'Number of pending tasks processed'), SetFloatM('workflow_manager__schedule_seconds', 'Time spent in running the entire _schedule'), IntM('workflow_manager__schedule_calls', 'Number of calls to _schedule, after lock is acquired'), SetFloatM('workflow_manager_recorded_timestamp', 'Unix timestamp when metrics were last recorded'), SetFloatM('workflow_manager_spawn_workflow_graph_jobs_seconds', 'Time spent spawning workflow tasks'), SetFloatM('workflow_manager_get_tasks_seconds', 'Time spent loading workflow tasks from db'), SetIntM('dispatcher_pool_scale_up_events', 'Number of times local dispatcher scaled up a worker since startup'), SetIntM('dispatcher_pool_active_task_count', 'Number of active tasks in the worker pool when last task was submitted'), SetIntM('dispatcher_pool_max_worker_count', 'Highest number of workers in worker pool in last collection interval, about 20s'), SetFloatM('dispatcher_availability', 'Fraction of time (in last collection interval) dispatcher was able to receive messages')]
        self.METRICS = {}
        for m in METRICSLIST:
            self.METRICS[m.field] = m
        self.previous_send_metrics = SetFloatM('send_metrics_time', 'Timestamp of previous send_metrics call')

    def reset_values(self):
        if False:
            i = 10
            return i + 15
        for m in self.METRICS.values():
            m.reset_value(self.conn)
        self.metrics_have_changed = True
        self.conn.delete(root_key + '_lock')
        for m in self.conn.scan_iter(root_key + '_instance_*'):
            self.conn.delete(m)

    def inc(self, field, value):
        if False:
            for i in range(10):
                print('nop')
        if value != 0:
            self.METRICS[field].inc(value)
            self.metrics_have_changed = True
            if self.auto_pipe_execute is True:
                self.pipe_execute()

    def set(self, field, value):
        if False:
            print('Hello World!')
        self.METRICS[field].set(value)
        self.metrics_have_changed = True
        if self.auto_pipe_execute is True:
            self.pipe_execute()

    def get(self, field):
        if False:
            i = 10
            return i + 15
        return self.METRICS[field].get()

    def decode(self, field):
        if False:
            for i in range(10):
                print('nop')
        return self.METRICS[field].decode(self.conn)

    def observe(self, field, value):
        if False:
            while True:
                i = 10
        self.METRICS[field].observe(value)
        self.metrics_have_changed = True
        if self.auto_pipe_execute is True:
            self.pipe_execute()

    def serialize_local_metrics(self):
        if False:
            print('Hello World!')
        data = self.load_local_metrics()
        return json.dumps(data)

    def load_local_metrics(self):
        if False:
            i = 10
            return i + 15
        data = {}
        for field in self.METRICS:
            data[field] = self.METRICS[field].decode(self.conn)
        return data

    def should_pipe_execute(self):
        if False:
            return 10
        if self.metrics_have_changed is False:
            return False
        if time.time() - self.last_pipe_execute > self.pipe_execute_interval:
            return True
        else:
            return False

    def pipe_execute(self):
        if False:
            return 10
        if self.metrics_have_changed is True:
            duration_to_save = time.perf_counter()
            for m in self.METRICS:
                self.METRICS[m].store_value(self.pipe)
            self.pipe.execute()
            self.last_pipe_execute = time.time()
            self.metrics_have_changed = False
            duration_to_save = time.perf_counter() - duration_to_save
            self.METRICS['subsystem_metrics_pipe_execute_seconds'].inc(duration_to_save)
            self.METRICS['subsystem_metrics_pipe_execute_calls'].inc(1)
            duration_to_save = time.perf_counter()
            self.send_metrics()
            duration_to_save = time.perf_counter() - duration_to_save
            self.METRICS['subsystem_metrics_send_metrics_seconds'].inc(duration_to_save)

    def send_metrics(self):
        if False:
            for i in range(10):
                print('nop')
        lock = self.conn.lock(root_key + '_lock')
        if not lock.acquire(blocking=False):
            return
        try:
            current_time = time.time()
            if current_time - self.previous_send_metrics.decode(self.conn) > self.send_metrics_interval:
                serialized_metrics = self.serialize_local_metrics()
                payload = {'instance': self.instance_name, 'metrics': serialized_metrics}
                self.conn.set(root_key + '_instance_' + self.instance_name, serialized_metrics)
                emit_channel_notification('metrics', payload)
                self.previous_send_metrics.set(current_time)
                self.previous_send_metrics.store_value(self.conn)
        finally:
            try:
                lock.release()
            except Exception as exc:
                logger.warning(f'Error releasing subsystem metrics redis lock, error: {str(exc)}')

    def load_other_metrics(self, request):
        if False:
            return 10
        instances_filter = request.query_params.getlist('node')
        instance_names = [self.instance_name]
        for m in self.conn.scan_iter(root_key + '_instance_*'):
            instance_names.append(m.decode('UTF-8').split('_instance_')[1])
        instance_names.sort()
        instance_data = {}
        for instance in instance_names:
            if len(instances_filter) == 0 or instance in instances_filter:
                instance_data_from_redis = self.conn.get(root_key + '_instance_' + instance)
                if instance_data_from_redis:
                    instance_data[instance] = json.loads(instance_data_from_redis.decode('UTF-8'))
        return instance_data

    def generate_metrics(self, request):
        if False:
            return 10
        instance_data = self.load_other_metrics(request)
        metrics_filter = request.query_params.getlist('metric')
        output_text = ''
        if instance_data:
            for field in self.METRICS:
                if len(metrics_filter) == 0 or field in metrics_filter:
                    output_text += self.METRICS[field].to_prometheus(instance_data)
        return output_text

def metrics(request):
    if False:
        for i in range(10):
            print('nop')
    m = Metrics()
    return m.generate_metrics(request)