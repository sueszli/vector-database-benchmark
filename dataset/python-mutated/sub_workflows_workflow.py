import ray
from ray import workflow

@ray.remote
def hello(name: str) -> str:
    if False:
        i = 10
        return i + 15
    return workflow.continuation(format_name.bind(name))

@ray.remote
def format_name(name: str) -> str:
    if False:
        while True:
            i = 10
    return f'hello, {name}'

@ray.remote
def report(msg: str) -> None:
    if False:
        print('Hello World!')
    print(msg)
if __name__ == '__main__':
    r1 = hello.bind('Kristof')
    r2 = report.bind(r1)
    workflow.run(r2)