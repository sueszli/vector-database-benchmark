import asyncio
from openai import OpenAI, AsyncOpenAI

def sync_main() -> None:
    if False:
        for i in range(10):
            print('nop')
    client = OpenAI()
    response = client.completions.create(model='text-davinci-002', prompt='1,2,3,', max_tokens=5, temperature=0, stream=True)
    first = next(response)
    print(f'got response data: {first.model_dump_json(indent=2)}')
    for data in response:
        print(data.model_dump_json())

async def async_main() -> None:
    client = AsyncOpenAI()
    response = await client.completions.create(model='text-davinci-002', prompt='1,2,3,', max_tokens=5, temperature=0, stream=True)
    first = await response.__anext__()
    print(f'got response data: {first.model_dump_json(indent=2)}')
    async for data in response:
        print(data.model_dump_json())
sync_main()
asyncio.run(async_main())