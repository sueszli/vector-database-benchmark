import asyncio
from typing import Final, Sequence, cast
import httpx
from typing_extensions import TypedDict
from returns.future import FutureResultE, future_safe
from returns.io import IOResultE
from returns.iterables import Fold
_URL: Final = 'https://jsonplaceholder.typicode.com/posts/{0}'

class _Post(TypedDict):
    id: int
    user_id: int
    title: str
    body: str

@future_safe
async def _fetch_post(post_id: int) -> _Post:
    async with httpx.AsyncClient(timeout=5) as client:
        response = await client.get(_URL.format(post_id))
        response.raise_for_status()
        return cast(_Post, response.json())

def _show_titles(number_of_posts: int) -> Sequence[FutureResultE[str]]:
    if False:
        while True:
            i = 10

    def factory(post: _Post) -> str:
        if False:
            return 10
        return post['title']
    return [_fetch_post(post_id).map(factory) for post_id in range(1, number_of_posts + 1)]

async def main() -> IOResultE[Sequence[str]]:
    """
    Main entrypoint for the async world.

    Let's fetch 3 titles of posts asynchronously.
    We use `gather` to run requests in "parallel".
    """
    futures: Sequence[IOResultE[str]] = await asyncio.gather(*_show_titles(3))
    return Fold.collect(futures, IOResultE.from_value(()))
if __name__ == '__main__':
    print(asyncio.run(main()))