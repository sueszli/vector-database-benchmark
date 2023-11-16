import ray
from ray import workflow

@ray.remote
def compose_greeting(greeting: str, name: str) -> str:
    if False:
        print('Hello World!')
    return greeting + ': ' + name

@ray.remote
def main_workflow(name: str) -> str:
    if False:
        i = 10
        return i + 15
    return workflow.continuation(compose_greeting.bind('Hello', name))
if __name__ == '__main__':
    print(workflow.run(main_workflow.bind('Alice')))