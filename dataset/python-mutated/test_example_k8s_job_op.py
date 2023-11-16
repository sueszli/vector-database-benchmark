from dagster_k8s import k8s_job_op
from dagster import job
first_op = k8s_job_op.configured({'image': 'busybox', 'command': ['/bin/sh', '-c'], 'args': ['echo HELLO']}, name='first_op')
second_op = k8s_job_op.configured({'image': 'busybox', 'command': ['/bin/sh', '-c'], 'args': ['echo GOODBYE']}, name='second_op')

@job
def full_job():
    if False:
        print('Hello World!')
    second_op(first_op())

def test_k8s_job_op():
    if False:
        while True:
            i = 10
    assert full_job