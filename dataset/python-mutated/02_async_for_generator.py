def make_arange(n):
    if False:
        i = 10
        return i + 15
    return (i * 2 async for i in n)

async def run(m):
    return [i async for i in m]

async def run_list(pair, f):
    return [i for pair in p async for i in f]