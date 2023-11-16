from typing import Tuple, Optional
import ray
from ray import workflow

@ray.remote
def intentional_fail() -> str:
    if False:
        print('Hello World!')
    raise RuntimeError('oops')

@ray.remote
def cry(error: Exception) -> None:
    if False:
        print('Hello World!')
    print('Sadly', error)

@ray.remote
def celebrate(result: str) -> None:
    if False:
        return 10
    print('Success!', result)

@ray.remote
def send_email(result: str) -> None:
    if False:
        while True:
            i = 10
    print('Sending email', result)

@ray.remote
def exit_handler(res: Tuple[Optional[str], Optional[Exception]]) -> None:
    if False:
        while True:
            i = 10
    (result, error) = res
    email = send_email.bind(f'Raw result: {result}, {error}')
    if error:
        handler = cry.bind(error)
    else:
        handler = celebrate.bind(result)
    return workflow.continuation(wait_all.bind(handler, email))

@ray.remote
def wait_all(*deps):
    if False:
        print('Hello World!')
    return 'done'
if __name__ == '__main__':
    res = intentional_fail.options(**workflow.options(catch_exceptions=True)).bind()
    print(workflow.run(exit_handler.bind(res)))