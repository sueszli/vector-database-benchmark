import ray
from ray import workflow

@ray.remote
def handle_heads() -> str:
    if False:
        return 10
    return 'It was heads'

@ray.remote
def handle_tails() -> str:
    if False:
        while True:
            i = 10
    print('It was tails, retrying')
    return workflow.continuation(flip_coin.bind())

@ray.remote
def flip_coin() -> str:
    if False:
        while True:
            i = 10
    import random

    @ray.remote
    def decide(heads: bool) -> str:
        if False:
            i = 10
            return i + 15
        if heads:
            return workflow.continuation(handle_heads.bind())
        else:
            return workflow.continuation(handle_tails.bind())
    return workflow.continuation(decide.bind(random.random() > 0.5))
if __name__ == '__main__':
    print(workflow.run(flip_coin.bind()))