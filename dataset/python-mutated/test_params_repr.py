from typing import Any, List
from dirty_equals import IsOneOf
from fastapi.params import Body, Cookie, Depends, Header, Param, Path, Query
test_data: List[Any] = ['teststr', None, ..., 1, []]

def get_user():
    if False:
        for i in range(10):
            print('nop')
    return {}

def test_param_repr_str():
    if False:
        for i in range(10):
            print('nop')
    assert repr(Param('teststr')) == 'Param(teststr)'

def test_param_repr_none():
    if False:
        while True:
            i = 10
    assert repr(Param(None)) == 'Param(None)'

def test_param_repr_ellipsis():
    if False:
        return 10
    assert repr(Param(...)) == IsOneOf('Param(PydanticUndefined)', 'Param(Ellipsis)')

def test_param_repr_number():
    if False:
        for i in range(10):
            print('nop')
    assert repr(Param(1)) == 'Param(1)'

def test_param_repr_list():
    if False:
        while True:
            i = 10
    assert repr(Param([])) == 'Param([])'

def test_path_repr():
    if False:
        while True:
            i = 10
    assert repr(Path()) == IsOneOf('Path(PydanticUndefined)', 'Path(Ellipsis)')
    assert repr(Path(...)) == IsOneOf('Path(PydanticUndefined)', 'Path(Ellipsis)')

def test_query_repr_str():
    if False:
        for i in range(10):
            print('nop')
    assert repr(Query('teststr')) == 'Query(teststr)'

def test_query_repr_none():
    if False:
        print('Hello World!')
    assert repr(Query(None)) == 'Query(None)'

def test_query_repr_ellipsis():
    if False:
        while True:
            i = 10
    assert repr(Query(...)) == IsOneOf('Query(PydanticUndefined)', 'Query(Ellipsis)')

def test_query_repr_number():
    if False:
        print('Hello World!')
    assert repr(Query(1)) == 'Query(1)'

def test_query_repr_list():
    if False:
        i = 10
        return i + 15
    assert repr(Query([])) == 'Query([])'

def test_header_repr_str():
    if False:
        i = 10
        return i + 15
    assert repr(Header('teststr')) == 'Header(teststr)'

def test_header_repr_none():
    if False:
        print('Hello World!')
    assert repr(Header(None)) == 'Header(None)'

def test_header_repr_ellipsis():
    if False:
        while True:
            i = 10
    assert repr(Header(...)) == IsOneOf('Header(PydanticUndefined)', 'Header(Ellipsis)')

def test_header_repr_number():
    if False:
        for i in range(10):
            print('nop')
    assert repr(Header(1)) == 'Header(1)'

def test_header_repr_list():
    if False:
        i = 10
        return i + 15
    assert repr(Header([])) == 'Header([])'

def test_cookie_repr_str():
    if False:
        while True:
            i = 10
    assert repr(Cookie('teststr')) == 'Cookie(teststr)'

def test_cookie_repr_none():
    if False:
        return 10
    assert repr(Cookie(None)) == 'Cookie(None)'

def test_cookie_repr_ellipsis():
    if False:
        return 10
    assert repr(Cookie(...)) == IsOneOf('Cookie(PydanticUndefined)', 'Cookie(Ellipsis)')

def test_cookie_repr_number():
    if False:
        for i in range(10):
            print('nop')
    assert repr(Cookie(1)) == 'Cookie(1)'

def test_cookie_repr_list():
    if False:
        return 10
    assert repr(Cookie([])) == 'Cookie([])'

def test_body_repr_str():
    if False:
        while True:
            i = 10
    assert repr(Body('teststr')) == 'Body(teststr)'

def test_body_repr_none():
    if False:
        print('Hello World!')
    assert repr(Body(None)) == 'Body(None)'

def test_body_repr_ellipsis():
    if False:
        print('Hello World!')
    assert repr(Body(...)) == IsOneOf('Body(PydanticUndefined)', 'Body(Ellipsis)')

def test_body_repr_number():
    if False:
        for i in range(10):
            print('nop')
    assert repr(Body(1)) == 'Body(1)'

def test_body_repr_list():
    if False:
        i = 10
        return i + 15
    assert repr(Body([])) == 'Body([])'

def test_depends_repr():
    if False:
        for i in range(10):
            print('nop')
    assert repr(Depends()) == 'Depends(NoneType)'
    assert repr(Depends(get_user)) == 'Depends(get_user)'
    assert repr(Depends(use_cache=False)) == 'Depends(NoneType, use_cache=False)'
    assert repr(Depends(get_user, use_cache=False)) == 'Depends(get_user, use_cache=False)'