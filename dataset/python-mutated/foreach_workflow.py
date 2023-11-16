from typing import List
import ray
from ray import workflow

@ray.remote
def start():
    if False:
        return 10
    titles = ['Stranger Things', 'House of Cards', 'Narcos']
    children = [a.bind(t) for t in titles]
    return workflow.continuation(end.bind(children))

@ray.remote
def a(title: str) -> str:
    if False:
        i = 10
        return i + 15
    return f'{title} processed'

@ray.remote
def end(results: 'List[ray.ObjectRef[str]]') -> str:
    if False:
        i = 10
        return i + 15
    return '\n'.join(ray.get(results))
if __name__ == '__main__':
    workflow.run(start.bind())