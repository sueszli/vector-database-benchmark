"""File docstring. Should not be printed."""
import asyncio
import contextlib
async_loop = asyncio.new_event_loop()
'no block'
a = 'printed'
('This should be', a)
if True:
    'IF'
if False:
    pass
elif True:
    'ELIF'
if False:
    pass
else:
    'ELSE'
for ii in range(1):
    'FOR'
while True:
    'WHILE'
    break

@contextlib.contextmanager
def context_mgr():
    if False:
        while True:
            i = 10
    try:
        yield
    finally:
        pass
with context_mgr():
    'WITH'
try:
    'TRY'
except:
    raise
try:
    raise RuntimeError('shenanigans!')
except RuntimeError:
    'EXCEPT'
try:
    pass
finally:
    'FINALLY'

def func(value):
    if False:
        i = 10
        return i + 15
    value
func('FUNCTION')

async def async_func(value):
    value
async_loop.run_until_complete(async_func('ASYNC FUNCTION'))

async def async_for():

    async def async_iter():
        yield
    async for _ in async_iter():
        'ASYNC FOR'
async_loop.run_until_complete(async_for())

async def async_with():

    @contextlib.asynccontextmanager
    async def async_context_mgr():
        try:
            yield
        finally:
            pass
    async with async_context_mgr():
        'ASYNC WITH'
async_loop.run_until_complete(async_with())

def docstrings():
    if False:
        i = 10
        return i + 15
    'Docstring. Should not be printed.'

    def nested():
        if False:
            for i in range(10):
                print('nop')
        'Multiline docstring.\n        Should not be printed.'
        pass

    class Foo(object):
        """Class docstring. Should not be printed."""
        pass
    nested()
docstrings()

def my_func():
    if False:
        i = 10
        return i + 15
    'my_func: this help block should be printed.'
    pass
my_func

class MyClass:
    """MyClass: this help block should be printed."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        'This should not be printed.'
MyClass
my_instance = MyClass()
my_instance