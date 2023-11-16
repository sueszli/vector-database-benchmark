def __getattr__(attr):
    if False:
        i = 10
        return i + 15
    import asyncio
    return getattr(asyncio, attr)