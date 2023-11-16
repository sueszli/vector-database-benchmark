import ray
from ray import workflow

@ray.remote
def hello(msg: str) -> None:
    if False:
        print('Hello World!')
    print(msg)
if __name__ == '__main__':
    workflow.run(hello.bind('hello world'))