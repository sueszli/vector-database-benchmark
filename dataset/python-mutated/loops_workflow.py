import ray
from ray import workflow

@ray.remote
def hello(msg: str) -> None:
    if False:
        i = 10
        return i + 15
    print(msg)

@ray.remote
def wait_all(*args) -> None:
    if False:
        i = 10
        return i + 15
    pass
if __name__ == '__main__':
    children = []
    for msg in ['hello world', 'goodbye world']:
        children.append(hello.bind(msg))
    workflow.run(wait_all.bind(*children))