from __future__ import annotations
from invoke import Context, task

@task
def docker_build(context: Context) -> None:
    if False:
        while True:
            i = 10
    pass

@task(docker_build)
def docker_push(context: Context) -> None:
    if False:
        while True:
            i = 10
    pass