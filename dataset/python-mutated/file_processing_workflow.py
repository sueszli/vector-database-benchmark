from typing import List
import ray
from ray import workflow
FILES_TO_PROCESS = ['file-{}'.format(i) for i in range(100)]

def download(url: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    return 'contents' * 10000

def process(contents: str) -> str:
    if False:
        return 10
    return 'processed: ' + contents

def upload(contents: str) -> None:
    if False:
        while True:
            i = 10
    pass

@ray.remote
def upload_all(file_contents: List[ray.ObjectRef]) -> None:
    if False:
        for i in range(10):
            print('nop')

    @ray.remote
    def upload_one(contents: str) -> None:
        if False:
            while True:
                i = 10
        upload(contents)
    children = [upload_one.bind(f) for f in file_contents]

    @ray.remote
    def wait_all(*deps) -> None:
        if False:
            print('Hello World!')
        pass
    return wait_all.bind(*children)

@ray.remote
def process_all(file_contents: List[ray.ObjectRef]) -> None:
    if False:
        while True:
            i = 10

    @ray.remote
    def process_one(contents: str) -> str:
        if False:
            i = 10
            return i + 15
        return process(contents)
    children = [process_one.bind(f) for f in file_contents]
    return upload_all.bind(children)

@ray.remote
def download_all(urls: List[str]) -> None:
    if False:
        for i in range(10):
            print('nop')

    @ray.remote
    def download_one(url: str) -> str:
        if False:
            return 10
        return download(url)
    children = [download_one.bind(u) for u in urls]
    return process_all.bind(children)
if __name__ == '__main__':
    res = download_all.bind(FILES_TO_PROCESS)
    workflow.run(res)