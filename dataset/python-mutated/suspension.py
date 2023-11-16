from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from redis import Redis
    from rq.worker import BaseWorker
WORKERS_SUSPENDED = 'rq:suspended'

def is_suspended(connection: 'Redis', worker: Optional['BaseWorker']=None):
    if False:
        return 10
    'Checks whether a Worker is suspendeed on a given connection\n    PS: pipeline returns a list of responses\n    Ref: https://github.com/andymccurdy/redis-py#pipelines\n\n    Args:\n        connection (Redis): The Redis Connection\n        worker (Optional[Worker], optional): The Worker. Defaults to None.\n    '
    with connection.pipeline() as pipeline:
        if worker is not None:
            worker.heartbeat(pipeline=pipeline)
        pipeline.exists(WORKERS_SUSPENDED)
        return pipeline.execute()[-1]

def suspend(connection: 'Redis', ttl: Optional[int]=None):
    if False:
        print('Hello World!')
    '\n    Suspends.\n    TTL of 0 will invalidate right away.\n\n    Args:\n        connection (Redis): The Redis connection to use..\n        ttl (Optional[int], optional): time to live in seconds. Defaults to `None`\n    '
    connection.set(WORKERS_SUSPENDED, 1)
    if ttl is not None:
        connection.expire(WORKERS_SUSPENDED, ttl)

def resume(connection: 'Redis'):
    if False:
        print('Hello World!')
    '\n    Resumes.\n\n    Args:\n        connection (Redis): The Redis connection to use..\n    '
    return connection.delete(WORKERS_SUSPENDED)