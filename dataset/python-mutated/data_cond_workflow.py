import ray
from ray import workflow

def make_request(url: str) -> str:
    if False:
        return 10
    return '42'

@ray.remote
def get_size() -> int:
    if False:
        return 10
    return int(make_request('https://www.example.com/callA'))

@ray.remote
def small(result: int) -> str:
    if False:
        for i in range(10):
            print('nop')
    return make_request('https://www.example.com/SmallFunc')

@ray.remote
def medium(result: int) -> str:
    if False:
        i = 10
        return i + 15
    return make_request('https://www.example.com/MediumFunc')

@ray.remote
def large(result: int) -> str:
    if False:
        while True:
            i = 10
    return make_request('https://www.example.com/LargeFunc')

@ray.remote
def decide(result: int) -> str:
    if False:
        i = 10
        return i + 15
    if result < 10:
        return workflow.continuation(small.bind(result))
    elif result < 100:
        return workflow.continuation(medium.bind(result))
    else:
        return workflow.continuation(large.bind(result))
if __name__ == '__main__':
    print(workflow.run(decide.bind(get_size.bind())))