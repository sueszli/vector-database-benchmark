def not_bug():
    if False:
        while True:
            i = 10
    cache_token = 5

    def register():
        if False:
            print('Hello World!')
        nonlocal cache_token
        return cache_token == 5
    return register()
assert not_bug()