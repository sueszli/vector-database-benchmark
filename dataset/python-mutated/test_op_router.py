import pytest
from werkzeug.exceptions import NotFound
from werkzeug.routing import Map, Rule
from localstack.aws.protocol.op_router import GreedyPathConverter, RestServiceOperationRouter
from localstack.aws.spec import list_services, load_service
from localstack.http import Request

def _collect_services():
    if False:
        return 10
    for service in list_services():
        if service.protocol.startswith('rest'):
            yield service.service_name

@pytest.mark.parametrize('service', _collect_services())
@pytest.mark.param
def test_create_op_router_works_for_every_service(service):
    if False:
        while True:
            i = 10
    router = RestServiceOperationRouter(load_service(service))
    try:
        router.match(Request('GET', '/'))
    except NotFound:
        pass

def test_greedy_path_converter():
    if False:
        i = 10
        return i + 15
    router = Map(converters={'path': GreedyPathConverter}, merge_slashes=False)
    router.add(Rule('/test-bucket/<path:p>'))
    router.add(Rule('/some-route/<path:p>/bar'))
    matcher = router.bind('')
    assert matcher.match('/test-bucket//foo/bar') == (None, {'p': '/foo/bar'})
    assert matcher.match('/test-bucket//foo//bar') == (None, {'p': '/foo//bar'})
    assert matcher.match('/test-bucket//foo/bar/') == (None, {'p': '/foo/bar/'})
    assert matcher.match('/some-route//foo/bar') == (None, {'p': '/foo'})
    assert matcher.match('/some-route//foo//bar') == (None, {'p': '/foo/'})
    assert matcher.match('/some-route//foo/bar/bar') == (None, {'p': '/foo/bar'})
    with pytest.raises(NotFound):
        matcher.match('/some-route//foo/baz')

def test_s3_head_request():
    if False:
        print('Hello World!')
    router = RestServiceOperationRouter(load_service('s3'))
    (op, _) = router.match(Request('GET', '/my-bucket/my-key/'))
    assert op.name == 'GetObject'
    (op, _) = router.match(Request('HEAD', '/my-bucket/my-key/'))
    assert op.name == 'HeadObject'

def test_basic_param_extraction():
    if False:
        i = 10
        return i + 15
    router = RestServiceOperationRouter(load_service('apigateway'))
    (op, params) = router.match(Request('POST', '/restapis/myrestapi/deployments'))
    assert op.name == 'CreateDeployment'
    assert params == {'restapi_id': 'myrestapi'}
    with pytest.raises(NotFound):
        router.match(Request('POST', '/restapis/myrestapi//deployments'))

def test_trailing_slashes_are_not_strict():
    if False:
        print('Hello World!')
    router = RestServiceOperationRouter(load_service('lambda'))
    (op, _) = router.match(Request('GET', '/2015-03-31/functions'))
    assert op.name == 'ListFunctions'
    (op, _) = router.match(Request('GET', '/2015-03-31/functions/'))
    assert op.name == 'ListFunctions'
    (op, _) = router.match(Request('POST', '/2015-03-31/functions'))
    assert op.name == 'CreateFunction'
    (op, _) = router.match(Request('POST', '/2015-03-31/functions/'))
    assert op.name == 'CreateFunction'

def test_s3_query_args_routing():
    if False:
        while True:
            i = 10
    router = RestServiceOperationRouter(load_service('s3'))
    (op, params) = router.match(Request('DELETE', '/mybucket?delete'))
    assert op.name == 'DeleteBucket'
    assert params == {'Bucket': 'mybucket'}
    (op, params) = router.match(Request('DELETE', '/mybucket/?delete'))
    assert op.name == 'DeleteBucket'
    assert params == {'Bucket': 'mybucket'}
    (op, params) = router.match(Request('DELETE', '/mybucket/mykey?delete'))
    assert op.name == 'DeleteObject'
    assert params == {'Bucket': 'mybucket', 'Key': 'mykey'}
    (op, params) = router.match(Request('DELETE', '/mybucket/mykey/?delete'))
    assert op.name == 'DeleteObject'
    assert params == {'Bucket': 'mybucket', 'Key': 'mykey'}

def test_s3_bucket_operation_with_trailing_slashes():
    if False:
        return 10
    router = RestServiceOperationRouter(load_service('s3'))
    (op, params) = router.match(Request('GET', '/mybucket'))
    assert op.name == 'ListObjects'
    assert params == {'Bucket': 'mybucket'}
    (op, params) = router.match(Request('Get', '/mybucket/'))
    assert op.name == 'ListObjects'
    assert params == {'Bucket': 'mybucket'}

def test_s3_object_operation_with_trailing_slashes():
    if False:
        i = 10
        return i + 15
    router = RestServiceOperationRouter(load_service('s3'))
    (op, params) = router.match(Request('GET', '/mybucket/mykey'))
    assert op.name == 'GetObject'
    assert params == {'Bucket': 'mybucket', 'Key': 'mykey'}
    (op, params) = router.match(Request('GET', '/mybucket/mykey/'))
    assert op.name == 'GetObject'
    assert params == {'Bucket': 'mybucket', 'Key': 'mykey'}

def test_s3_bucket_operation_with_double_slashes():
    if False:
        print('Hello World!')
    router = RestServiceOperationRouter(load_service('s3'))
    (op, params) = router.match(Request('GET', '/mybucket//mykey'))
    assert op.name == 'GetObject'
    assert params == {'Bucket': 'mybucket', 'Key': '/mykey'}