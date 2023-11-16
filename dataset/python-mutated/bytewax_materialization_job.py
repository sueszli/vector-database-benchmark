from typing import Optional
from kubernetes import client
from feast.infra.materialization.batch_materialization_engine import MaterializationJob, MaterializationJobStatus

class BytewaxMaterializationJob(MaterializationJob):

    def __init__(self, job_id, namespace, error: Optional[BaseException]=None):
        if False:
            print('Hello World!')
        super().__init__()
        self._job_id = job_id
        self.namespace = namespace
        self._error: Optional[BaseException] = error
        self.batch_v1 = client.BatchV1Api()

    def error(self):
        if False:
            i = 10
            return i + 15
        return self._error

    def status(self):
        if False:
            i = 10
            return i + 15
        if self._error is not None:
            return MaterializationJobStatus.ERROR
        else:
            job_status = self.batch_v1.read_namespaced_job_status(self.job_id(), self.namespace).status
            if job_status.active is not None:
                if job_status.completion_time is None:
                    return MaterializationJobStatus.RUNNING
            else:
                if job_status.completion_time is not None and job_status.conditions[0].type == 'Complete':
                    return MaterializationJobStatus.SUCCEEDED
                if job_status.conditions is not None and job_status.conditions[0].type == 'Failed':
                    self._error = Exception(f'Job {self.job_id()} failed with reason: {job_status.conditions[0].message}')
                    return MaterializationJobStatus.ERROR
                return MaterializationJobStatus.WAITING

    def should_be_retried(self):
        if False:
            i = 10
            return i + 15
        return False

    def job_id(self):
        if False:
            print('Hello World!')
        return f'dataflow-{self._job_id}'

    def url(self):
        if False:
            while True:
                i = 10
        return None