def some_func(unformatted, args):
    if False:
        print('Hello World!')
    print('I am some_func')
    return 0

async def some_async_func(unformatted, args):
    print('I am some_async_func')
    await asyncio.sleep(1)

class SomeClass(Unformatted, SuperClasses):

    def some_method(self, unformatted, args):
        if False:
            for i in range(10):
                print('nop')
        print('I am some_method')
        return 0

    async def some_async_method(self, unformatted, args):
        print('I am some_async_method')
        await asyncio.sleep(1)
if unformatted_call(args):
    print('First branch')
elif another_unformatted_call(args):
    print('Second branch')
else:
    print('Last branch')
while some_condition(unformatted, args):
    print('Do something')
for i in some_iter(unformatted, args):
    print('Do something')

async def test_async_for():
    async for i in some_async_iter(unformatted, args):
        print('Do something')
try:
    some_call()
except UnformattedError as ex:
    handle_exception()
finally:
    finally_call()
with give_me_context(unformatted, args):
    print('Do something')

async def test_async_with():
    async with give_me_async_context(unformatted, args):
        print('Do something')