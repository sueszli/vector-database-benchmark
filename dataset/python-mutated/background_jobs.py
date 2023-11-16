import gc
import os
import socket
import time
from collections import defaultdict
from collections.abc import Callable
from functools import lru_cache
from typing import Any, NoReturn
from uuid import uuid4
import redis
from redis.exceptions import BusyLoadingError, ConnectionError
from rq import Callback, Queue, Worker
from rq.exceptions import NoSuchJobError
from rq.job import Job, JobStatus
from rq.logutils import setup_loghandlers
from rq.worker import DequeueStrategy
from rq.worker_pool import WorkerPool
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed
import frappe
import frappe.monitor
from frappe import _
from frappe.utils import CallbackManager, cint, cstr, get_bench_id
from frappe.utils.commands import log
from frappe.utils.deprecations import deprecation_warning
from frappe.utils.redis_queue import RedisQueue
RQ_JOB_FAILURE_TTL = 7 * 24 * 60 * 60
RQ_FAILED_JOBS_LIMIT = 1000
RQ_RESULTS_TTL = 10 * 60
_redis_queue_conn = None

@lru_cache
def get_queues_timeout():
    if False:
        for i in range(10):
            print('nop')
    common_site_config = frappe.get_conf()
    custom_workers_config = common_site_config.get('workers', {})
    default_timeout = 300
    return {'short': default_timeout, 'default': default_timeout, 'long': 1500, **{worker: config.get('timeout', default_timeout) for (worker, config) in custom_workers_config.items()}}

def enqueue(method: str | Callable, queue: str='default', timeout: int | None=None, event=None, is_async: bool=True, job_name: str | None=None, now: bool=False, enqueue_after_commit: bool=False, *, on_success: Callable=None, on_failure: Callable=None, at_front: bool=False, job_id: str=None, deduplicate=False, **kwargs) -> Job | Any:
    if False:
        return 10
    "\n\tEnqueue method to be executed using a background worker\n\n\t:param method: method string or method object\n\t:param queue: should be either long, default or short\n\t:param timeout: should be set according to the functions\n\t:param event: this is passed to enable clearing of jobs from queues\n\t:param is_async: if is_async=False, the method is executed immediately, else via a worker\n\t:param job_name: [DEPRECATED] can be used to name an enqueue call, which can be used to prevent duplicate calls\n\t:param now: if now=True, the method is executed via frappe.call\n\t:param kwargs: keyword arguments to be passed to the method\n\t:param deduplicate: do not re-queue job if it's already queued, requires job_id.\n\t:param job_id: Assigning unique job id, which can be checked using `is_job_enqueued`\n\t"
    is_async = kwargs.pop('async', is_async)
    if deduplicate:
        if not job_id:
            frappe.throw(_('`job_id` paramater is required for deduplication.'))
        job = get_job(job_id)
        if job and job.get_status() in (JobStatus.QUEUED, JobStatus.STARTED):
            frappe.logger().debug(f'Not queueing job {job.id} because it is in queue already')
            return
        elif job:
            job.delete()
    job_id = create_job_id(job_id)
    if job_name:
        deprecation_warning('Using enqueue with `job_name` is deprecated, use `job_id` instead.')
    if not is_async and (not frappe.flags.in_test):
        deprecation_warning('Using enqueue with is_async=False outside of tests is not recommended, use now=True instead.')
    call_directly = now or (not is_async and (not frappe.flags.in_test))
    if call_directly:
        return frappe.call(method, **kwargs)
    try:
        q = get_queue(queue, is_async=is_async)
    except ConnectionError:
        if frappe.local.flags.in_migrate:
            print(f'Redis queue is unreachable: Executing {method} synchronously')
            return frappe.call(method, **kwargs)
        raise
    if not timeout:
        timeout = get_queues_timeout().get(queue) or 300
    queue_args = {'site': frappe.local.site, 'user': frappe.session.user, 'method': method, 'event': event, 'job_name': job_name or cstr(method), 'is_async': is_async, 'kwargs': kwargs}
    on_failure = on_failure or truncate_failed_registry

    def enqueue_call():
        if False:
            for i in range(10):
                print('nop')
        return q.enqueue_call(execute_job, on_success=Callback(func=on_success) if on_success else None, on_failure=Callback(func=on_failure) if on_failure else None, timeout=timeout, kwargs=queue_args, at_front=at_front, failure_ttl=frappe.conf.get('rq_job_failure_ttl') or RQ_JOB_FAILURE_TTL, result_ttl=frappe.conf.get('rq_results_ttl') or RQ_RESULTS_TTL, job_id=job_id)
    if enqueue_after_commit:
        frappe.db.after_commit.add(enqueue_call)
        return
    return enqueue_call()

def enqueue_doc(doctype, name=None, method=None, queue='default', timeout=300, now=False, **kwargs):
    if False:
        print('Hello World!')
    'Enqueue a method to be run on a document'
    return enqueue('frappe.utils.background_jobs.run_doc_method', doctype=doctype, name=name, doc_method=method, queue=queue, timeout=timeout, now=now, **kwargs)

def run_doc_method(doctype, name, doc_method, **kwargs):
    if False:
        i = 10
        return i + 15
    getattr(frappe.get_doc(doctype, name), doc_method)(**kwargs)

def execute_job(site, method, event, job_name, kwargs, user=None, is_async=True, retry=0):
    if False:
        for i in range(10):
            print('nop')
    'Executes job in a worker, performs commit/rollback and logs if there is any error'
    retval = None
    if is_async:
        frappe.connect(site)
        if os.environ.get('CI'):
            frappe.flags.in_test = True
        if user:
            frappe.set_user(user)
    if isinstance(method, str):
        method_name = method
        method = frappe.get_attr(method)
    else:
        method_name = cstr(method.__name__)
    frappe.local.job = frappe._dict(site=site, method=method_name, job_name=job_name, kwargs=kwargs, user=user, after_job=CallbackManager())
    for before_job_task in frappe.get_hooks('before_job'):
        frappe.call(before_job_task, method=method_name, kwargs=kwargs, transaction_type='job')
    try:
        retval = method(**kwargs)
    except (frappe.db.InternalError, frappe.RetryBackgroundJobError) as e:
        frappe.db.rollback()
        if retry < 5 and (isinstance(e, frappe.RetryBackgroundJobError) or (frappe.db.is_deadlocked(e) or frappe.db.is_timedout(e))):
            frappe.destroy()
            time.sleep(retry + 1)
            return execute_job(site, method, event, job_name, kwargs, is_async=is_async, retry=retry + 1)
        else:
            frappe.log_error(title=method_name)
            raise
    except Exception:
        frappe.db.rollback()
        frappe.log_error(title=method_name)
        frappe.db.commit()
        print(frappe.get_traceback())
        raise
    else:
        frappe.db.commit()
        return retval
    finally:
        for after_job_task in frappe.get_hooks('after_job'):
            frappe.call(after_job_task, method=method_name, kwargs=kwargs, result=retval)
        frappe.local.job.after_job.run()
        if is_async:
            frappe.destroy()

def start_worker(queue: str | None=None, quiet: bool=False, rq_username: str | None=None, rq_password: str | None=None, burst: bool=False, strategy: DequeueStrategy | None=DequeueStrategy.DEFAULT) -> NoReturn | None:
    if False:
        for i in range(10):
            print('nop')
    'Wrapper to start rq worker. Connects to redis and monitors these queues.'
    if not strategy:
        strategy = DequeueStrategy.DEFAULT
    _freeze_gc()
    with frappe.init_site():
        redis_connection = get_redis_conn(username=rq_username, password=rq_password)
        if queue:
            queue = [q.strip() for q in queue.split(',')]
        queues = get_queue_list(queue, build_queue_name=True)
        queue_name = queue and generate_qname(queue)
    if os.environ.get('CI'):
        setup_loghandlers('ERROR')
    set_niceness()
    logging_level = 'INFO'
    if quiet:
        logging_level = 'WARNING'
    worker = Worker(queues, name=get_worker_name(queue_name), connection=redis_connection)
    worker.work(logging_level=logging_level, burst=burst, date_format='%Y-%m-%d %H:%M:%S', log_format='%(asctime)s,%(msecs)03d %(message)s', dequeue_strategy=strategy)

def start_worker_pool(queue: str | None=None, num_workers: int=1, quiet: bool=False, burst: bool=False) -> NoReturn:
    if False:
        for i in range(10):
            print('nop')
    'Start worker pool with specified number of workers.\n\n\tWARNING: This feature is considered "EXPERIMENTAL".\n\t'
    _freeze_gc()
    with frappe.init_site():
        redis_connection = get_redis_conn()
        if queue:
            queue = [q.strip() for q in queue.split(',')]
        queues = get_queue_list(queue, build_queue_name=True)
    if os.environ.get('CI'):
        setup_loghandlers('ERROR')
    set_niceness()
    logging_level = 'INFO'
    if quiet:
        logging_level = 'WARNING'
    pool = WorkerPool(queues=queues, connection=redis_connection, num_workers=num_workers)
    pool.start(logging_level=logging_level, burst=burst)

def _freeze_gc():
    if False:
        return 10
    if frappe._tune_gc:
        gc.collect()
        gc.freeze()

def get_worker_name(queue):
    if False:
        return 10
    'When limiting worker to a specific queue, also append queue name to default worker name'
    name = None
    if queue:
        name = '{uuid}.{hostname}.{pid}.{queue}'.format(uuid=uuid4().hex, hostname=socket.gethostname(), pid=os.getpid(), queue=queue)
    return name

def get_jobs(site=None, queue=None, key='method'):
    if False:
        while True:
            i = 10
    'Gets jobs per queue or per site or both'
    jobs_per_site = defaultdict(list)

    def add_to_dict(job):
        if False:
            return 10
        if key in job.kwargs:
            jobs_per_site[job.kwargs['site']].append(job.kwargs[key])
        elif key in job.kwargs.get('kwargs', {}):
            jobs_per_site[job.kwargs['site']].append(job.kwargs['kwargs'][key])
    for _queue in get_queue_list(queue):
        q = get_queue(_queue)
        jobs = q.jobs + get_running_jobs_in_queue(q)
        for job in jobs:
            if job.kwargs.get('site'):
                if job.kwargs['site'] == site or site is None:
                    add_to_dict(job)
            else:
                print('No site found in job', job.__dict__)
    return jobs_per_site

def get_queue_list(queue_list=None, build_queue_name=False):
    if False:
        while True:
            i = 10
    'Defines possible queues. Also wraps a given queue in a list after validating.'
    default_queue_list = list(get_queues_timeout())
    if queue_list:
        if isinstance(queue_list, str):
            queue_list = [queue_list]
        for queue in queue_list:
            validate_queue(queue, default_queue_list)
    else:
        queue_list = default_queue_list
    return [generate_qname(qtype) for qtype in queue_list] if build_queue_name else queue_list

def get_workers(queue=None):
    if False:
        print('Hello World!')
    'Returns a list of Worker objects tied to a queue object if queue is passed, else returns a list of all workers'
    if queue:
        return Worker.all(queue=queue)
    else:
        return Worker.all(get_redis_conn())

def get_running_jobs_in_queue(queue):
    if False:
        return 10
    'Returns a list of Jobs objects that are tied to a queue object and are currently running'
    jobs = []
    workers = get_workers(queue)
    for worker in workers:
        current_job = worker.get_current_job()
        if current_job:
            jobs.append(current_job)
    return jobs

def get_queue(qtype, is_async=True):
    if False:
        i = 10
        return i + 15
    'Returns a Queue object tied to a redis connection'
    validate_queue(qtype)
    return Queue(generate_qname(qtype), connection=get_redis_conn(), is_async=is_async)

def validate_queue(queue, default_queue_list=None):
    if False:
        i = 10
        return i + 15
    if not default_queue_list:
        default_queue_list = list(get_queues_timeout())
    if queue not in default_queue_list:
        frappe.throw(_('Queue should be one of {0}').format(', '.join(default_queue_list)))

@retry(retry=retry_if_exception_type((BusyLoadingError, ConnectionError)), stop=stop_after_attempt(5), wait=wait_fixed(1), reraise=True)
def get_redis_conn(username=None, password=None):
    if False:
        while True:
            i = 10
    if not hasattr(frappe.local, 'conf'):
        raise Exception('You need to call frappe.init')
    elif not frappe.local.conf.redis_queue:
        raise Exception('redis_queue missing in common_site_config.json')
    global _redis_queue_conn
    cred = frappe._dict()
    if frappe.conf.get('use_rq_auth'):
        if username:
            cred['username'] = username
            cred['password'] = password
        else:
            cred['username'] = frappe.get_site_config().rq_username or get_bench_id()
            cred['password'] = frappe.get_site_config().rq_password
    elif os.environ.get('RQ_ADMIN_PASWORD'):
        cred['username'] = 'default'
        cred['password'] = os.environ.get('RQ_ADMIN_PASWORD')
    try:
        if not cred:
            return get_redis_connection_without_auth()
        else:
            return RedisQueue.get_connection(**cred)
    except (redis.exceptions.AuthenticationError, redis.exceptions.ResponseError):
        log(f"Wrong credentials used for {cred.username or 'default user'}. You can reset credentials using `bench create-rq-users` CLI and restart the server", colour='red')
        raise
    except Exception:
        log(f'Please make sure that Redis Queue runs @ {frappe.get_conf().redis_queue}', colour='red')
        raise

def get_redis_connection_without_auth():
    if False:
        i = 10
        return i + 15
    global _redis_queue_conn
    if not _redis_queue_conn:
        _redis_queue_conn = RedisQueue.get_connection()
    return _redis_queue_conn

def get_queues(connection=None) -> list[Queue]:
    if False:
        for i in range(10):
            print('nop')
    'Get all the queues linked to the current bench.'
    queues = Queue.all(connection=connection or get_redis_conn())
    return [q for q in queues if is_queue_accessible(q)]

def generate_qname(qtype: str) -> str:
    if False:
        i = 10
        return i + 15
    'Generate qname by combining bench ID and queue type.\n\n\tqnames are useful to define namespaces of customers.\n\t'
    if isinstance(qtype, list):
        qtype = ','.join(qtype)
    return f'{get_bench_id()}:{qtype}'

def is_queue_accessible(qobj: Queue) -> bool:
    if False:
        i = 10
        return i + 15
    'Checks whether queue is relate to current bench or not.'
    accessible_queues = [generate_qname(q) for q in list(get_queues_timeout())]
    return qobj.name in accessible_queues

def enqueue_test_job():
    if False:
        i = 10
        return i + 15
    enqueue('frappe.utils.background_jobs.test_job', s=100)

def test_job(s):
    if False:
        i = 10
        return i + 15
    import time
    print('sleeping...')
    time.sleep(s)

def create_job_id(job_id: str) -> str:
    if False:
        print('Hello World!')
    'Generate unique job id for deduplication'
    if not job_id:
        job_id = str(uuid4())
    return f'{frappe.local.site}::{job_id}'

def is_job_enqueued(job_id: str) -> bool:
    if False:
        print('Hello World!')
    return get_job_status(job_id) in (JobStatus.QUEUED, JobStatus.STARTED)

def get_job_status(job_id: str) -> JobStatus | None:
    if False:
        return 10
    'Get RQ job status, returns None if job is not found.'
    job = get_job(job_id)
    if job:
        return job.get_status()

def get_job(job_id: str) -> Job:
    if False:
        print('Hello World!')
    try:
        return Job.fetch(create_job_id(job_id), connection=get_redis_conn())
    except NoSuchJobError:
        return None
BACKGROUND_PROCESS_NICENESS = 10

def set_niceness():
    if False:
        while True:
            i = 10
    "Background processes should have slightly lower priority than web processes.\n\n\tCalling this function increments the niceness of process by configured value or default.\n\tNote: This function should be called only once in process' lifetime.\n\t"
    conf = frappe.get_conf()
    nice_increment = BACKGROUND_PROCESS_NICENESS
    configured_niceness = conf.get('background_process_niceness')
    if configured_niceness is not None:
        nice_increment = cint(configured_niceness)
    os.nice(nice_increment)

def truncate_failed_registry(job, connection, type, value, traceback):
    if False:
        return 10
    "Ensures that number of failed jobs don't exceed specified limits."
    from frappe.utils import create_batch
    conf = frappe.get_conf(site=job.kwargs.get('site'))
    limit = (conf.get('rq_failed_jobs_limit') or RQ_FAILED_JOBS_LIMIT) - 1
    for queue in get_queues(connection=connection):
        fail_registry = queue.failed_job_registry
        failed_jobs = fail_registry.get_job_ids()[limit:]
        for job_ids in create_batch(failed_jobs, 100):
            for job_obj in Job.fetch_many(job_ids=job_ids, connection=connection):
                job_obj and fail_registry.remove(job_obj, delete_job=True)