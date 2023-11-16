from dagster_docker import docker_container_op
from dagster import job
first_op = docker_container_op.configured({'image': 'busybox', 'command': ['echo HELLO']}, name='first_op')
second_op = docker_container_op.configured({'image': 'busybox', 'command': ['echo GOODBYE']}, name='second_op')

@job
def full_job():
    if False:
        i = 10
        return i + 15
    second_op(first_op())

def test_docker_container_op():
    if False:
        print('Hello World!')
    assert full_job