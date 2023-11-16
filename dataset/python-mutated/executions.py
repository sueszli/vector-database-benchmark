from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from uuid import uuid4
from redis import Redis
if TYPE_CHECKING:
    from redis.client import Pipeline
from .job import Job
from .registry import BaseRegistry, StartedJobRegistry
from .utils import as_text, current_timestamp, utcnow

class Execution:
    """Class to represent an execution of a job."""

    def __init__(self, id: str, job_id: str, connection: Redis):
        if False:
            return 10
        self.id = id
        self.job_id = job_id
        self.connection = connection
        now = utcnow()
        self.created_at = now
        self.last_heartbeat = now

    def __eq__(self, other: object) -> bool:
        if False:
            return 10
        if not isinstance(other, Execution):
            return False
        return self.id == other.id

    @property
    def key(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'rq:execution:{self.composite_key}'

    @property
    def job(self) -> Job:
        if False:
            return 10
        return Job(id=self.job_id, connection=self.connection)

    @property
    def composite_key(self):
        if False:
            print('Hello World!')
        return f'{self.job_id}:{self.id}'

    @classmethod
    def fetch(cls, id: str, job_id: str, connection: Redis) -> 'Execution':
        if False:
            while True:
                i = 10
        'Fetch an execution from Redis.'
        execution = cls(id=id, job_id=job_id, connection=connection)
        execution.refresh()
        return execution

    def refresh(self):
        if False:
            print('Hello World!')
        'Refresh execution data from Redis.'
        data = self.connection.hgetall(self.key)
        if not data:
            raise ValueError(f'Execution {self.id} not found in Redis')
        self.created_at = datetime.fromtimestamp(float(data[b'created_at']))
        self.last_heartbeat = datetime.fromtimestamp(float(data[b'last_heartbeat']))

    @classmethod
    def from_composite_key(cls, composite_key: str, connection: Redis) -> 'Execution':
        if False:
            for i in range(10):
                print('nop')
        'A combination of job_id and execution_id separated by a colon.'
        (job_id, id) = composite_key.split(':')
        return cls(id=id, job_id=job_id, connection=connection)

    @classmethod
    def create(cls, job: Job, ttl: int, pipeline: 'Pipeline') -> 'Execution':
        if False:
            while True:
                i = 10
        'Save execution data to Redis.'
        id = uuid4().hex
        execution = cls(id=id, job_id=job.id, connection=job.connection)
        execution.save(ttl=ttl, pipeline=pipeline)
        ExecutionRegistry(job_id=job.id, connection=pipeline).add(execution=execution, ttl=ttl, pipeline=pipeline)
        job.started_job_registry.add_execution(execution, pipeline=pipeline, ttl=ttl, xx=False)
        return execution

    def save(self, ttl: int, pipeline: Optional['Pipeline']=None):
        if False:
            while True:
                i = 10
        'Save execution data to Redis and JobExecutionRegistry.'
        connection = pipeline if pipeline is not None else self.connection
        connection.hset(self.key, mapping=self.serialize())
        connection.expire(self.key, ttl)

    def delete(self, job: Job, pipeline: 'Pipeline'):
        if False:
            i = 10
            return i + 15
        'Delete an execution from Redis.'
        pipeline.delete(self.key)
        job.started_job_registry.remove_execution(execution=self, job=job, pipeline=pipeline)
        ExecutionRegistry(job_id=self.job_id, connection=self.connection).remove(execution=self, pipeline=pipeline)

    def serialize(self) -> Dict:
        if False:
            return 10
        return {'id': self.id, 'created_at': self.created_at.timestamp(), 'last_heartbeat': self.last_heartbeat.timestamp()}

    def heartbeat(self, started_job_registry: StartedJobRegistry, ttl: int, pipeline: 'Pipeline'):
        if False:
            print('Hello World!')
        'Update execution heartbeat.'
        self.last_heartbeat = utcnow()
        pipeline.hset(self.key, 'last_heartbeat', self.last_heartbeat.timestamp())
        pipeline.expire(self.key, ttl)
        started_job_registry.add(self.job, ttl, pipeline=pipeline, xx=True)
        ExecutionRegistry(job_id=self.job_id, connection=pipeline).add(execution=self, ttl=ttl, pipeline=pipeline)

class ExecutionRegistry(BaseRegistry):
    """Class to represent a registry of job executions.
    Each job has its own execution registry.
    """
    key_template = 'rq:executions:{0}'

    def __init__(self, job_id: str, connection: Redis):
        if False:
            return 10
        self.connection = connection
        self.job_id = job_id
        self.key = self.key_template.format(job_id)

    def cleanup(self, timestamp: Optional[float]=None):
        if False:
            for i in range(10):
                print('nop')
        'Remove expired jobs from registry.\n\n        Removes jobs with an expiry time earlier than timestamp, specified as\n        seconds since the Unix epoch. timestamp defaults to call time if\n        unspecified.\n        '
        score = timestamp if timestamp is not None else current_timestamp()
        self.connection.zremrangebyscore(self.key, 0, score)

    def add(self, execution: Execution, ttl: int, pipeline: 'Pipeline') -> Any:
        if False:
            while True:
                i = 10
        "Register an execution to registry with expiry time of now + ttl, unless it's -1 which is set to +inf\n\n        Args:\n            execution (Execution): The Execution to add\n            ttl (int, optional): The time to live. Defaults to 0.\n            pipeline (Optional[Pipeline], optional): The Redis Pipeline. Defaults to None.\n            xx (bool, optional): .... Defaults to False.\n\n        Returns:\n            result (int): The ZADD command result\n        "
        score = current_timestamp() + ttl
        pipeline.zadd(self.key, {execution.id: score + 60})
        pipeline.expire(self.key, ttl + 60)
        return

    def remove(self, execution: Execution, pipeline: 'Pipeline') -> Any:
        if False:
            i = 10
            return i + 15
        'Remove an execution from registry.'
        return pipeline.zrem(self.key, execution.id)

    def get_execution_ids(self, start: int=0, end: int=-1) -> List[str]:
        if False:
            while True:
                i = 10
        'Returns all executions IDs in registry'
        self.cleanup()
        return [as_text(job_id) for job_id in self.connection.zrange(self.key, start, end)]

    def get_executions(self, start: int=0, end: int=-1) -> List[Execution]:
        if False:
            return 10
        'Returns all executions IDs in registry'
        execution_ids = self.get_execution_ids(start, end)
        executions = []
        for execution_id in execution_ids:
            executions.append(Execution.fetch(id=execution_id, job_id=self.job_id, connection=self.connection))
        return executions

    def delete(self, job: Job, pipeline: 'Pipeline'):
        if False:
            i = 10
            return i + 15
        'Delete the registry.'
        executions = self.get_executions()
        for execution in executions:
            execution.delete(pipeline=pipeline, job=job)
        pipeline.delete(self.key)