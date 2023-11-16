import functools
import pytest

@pytest.fixture()
def my_fixture():
    if False:
        for i in range(10):
            print('nop')
    return 0

@pytest.fixture()
def my_fixture():
    if False:
        i = 10
        return i + 15
    resource = acquire_resource()
    yield resource
    resource.release()

@pytest.fixture()
def my_fixture():
    if False:
        while True:
            i = 10
    request = get_request()
    request.addfinalizer(finalizer)
    return request

def create_resource(arg, request):
    if False:
        print('Hello World!')
    resource = Resource(arg)
    request.addfinalizer(resource.release)
    return resource

@pytest.fixture()
def resource_factory(request):
    if False:
        return 10
    return functools.partial(create_resource, request=request)

@pytest.fixture()
def resource_factory(request):
    if False:
        return 10

    def create_resource(arg) -> Resource:
        if False:
            return 10
        resource = Resource(arg)
        request.addfinalizer(resource.release)
        return resource
    return create_resource

@pytest.fixture()
def my_fixture(request):
    if False:
        i = 10
        return i + 15
    resource = acquire_resource()
    request.addfinalizer(resource.release)
    return resource

@pytest.fixture()
def my_fixture(request):
    if False:
        return 10
    resource = acquire_resource()
    request.addfinalizer(resource.release)
    yield resource
    resource