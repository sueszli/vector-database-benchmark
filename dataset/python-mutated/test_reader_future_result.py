from typing import Final, Sequence, cast
import anyio
import httpx
from typing_extensions import TypedDict
from returns.context import RequiresContextFutureResultE
from returns.functions import tap
from returns.future import FutureResultE, future_safe
from returns.iterables import Fold
from returns.pipeline import managed
from returns.result import ResultE, safe
_URL: Final = 'https://jsonplaceholder.typicode.com/posts/{0}'

class _Post(TypedDict):
    id: int
    user_id: int
    title: str
    body: str

def _close(client: httpx.AsyncClient, raw_value: ResultE[Sequence[str]]) -> FutureResultE[None]:
    if False:
        return 10
    return future_safe(client.aclose)()

def _fetch_post(post_id: int) -> RequiresContextFutureResultE[_Post, httpx.AsyncClient]:
    if False:
        return 10
    context: RequiresContextFutureResultE[httpx.AsyncClient, httpx.AsyncClient] = RequiresContextFutureResultE.ask()
    return context.bind_future_result(lambda client: future_safe(client.get)(_URL.format(post_id))).bind_result(safe(tap(httpx.Response.raise_for_status))).map(lambda response: cast(_Post, response.json()))

def _show_titles(number_of_posts: int) -> RequiresContextFutureResultE[Sequence[str], httpx.AsyncClient]:
    if False:
        while True:
            i = 10

    def factory(post: _Post) -> str:
        if False:
            return 10
        return post['title']
    titles = [_fetch_post(post_id).map(factory) for post_id in range(1, number_of_posts + 1)]
    return Fold.collect(titles, RequiresContextFutureResultE.from_value(()))
if __name__ == '__main__':
    managed_httpx = managed(_show_titles(3), _close)
    future_result = managed_httpx(FutureResultE.from_value(httpx.AsyncClient(timeout=5)))
    print(anyio.run(future_result.awaitable))