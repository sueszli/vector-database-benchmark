import ray
from ray import workflow

@ray.remote
def handle_heads() -> str:
    if False:
        i = 10
        return i + 15
    return 'It was heads'

@ray.remote
def handle_tails() -> str:
    if False:
        print('Hello World!')
    return 'It was tails'

@ray.remote
def flip_coin() -> str:
    if False:
        return 10
    import random

    @ray.remote
    def decide(heads: bool) -> str:
        if False:
            i = 10
            return i + 15
        return workflow.continuation(handle_heads.bind() if heads else handle_tails.bind())
    return workflow.continuation(decide.bind(random.random() > 0.5))
if __name__ == '__main__':
    print(workflow.run(flip_coin.bind()))