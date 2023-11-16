"""
Default scope function.

`Paddle` manages Scope as programming language's scope.  It just a
thread-local stack of Scope. Top of that stack is current scope, the bottom
of that stack is all scopes' parent.

Invoking `var/find_var`  can `new/find` variable in current scope.
Invoking `enter_local_scope/leave_local_scope` can create or destroy local
scope.

A `scoped_function` will take a `function` as input. That function will be
invoked in a new local scope.
"""
import threading
import paddle.base.core
__tl_scope__ = threading.local()
__all__ = []

def get_cur_scope():
    if False:
        while True:
            i = 10
    '\n    Get current scope.\n    :rtype: paddle.base.core.Scope\n    '
    cur_scope_stack = getattr(__tl_scope__, 'cur_scope', None)
    if cur_scope_stack is None:
        __tl_scope__.cur_scope = []
    if len(__tl_scope__.cur_scope) == 0:
        __tl_scope__.cur_scope.append(paddle.base.core.Scope())
    return __tl_scope__.cur_scope[-1]

def enter_local_scope():
    if False:
        return 10
    '\n    Enter a new local scope\n    '
    cur_scope = get_cur_scope()
    new_scope = cur_scope.new_scope()
    __tl_scope__.cur_scope.append(new_scope)

def leave_local_scope():
    if False:
        print('Hello World!')
    '\n    Leave local scope\n    '
    __tl_scope__.cur_scope.pop()
    get_cur_scope().drop_kids()

def var(name):
    if False:
        return 10
    '\n    create variable in current scope.\n    '
    return get_cur_scope().var(name)

def find_var(name):
    if False:
        for i in range(10):
            print('nop')
    '\n    get variable in current scope.\n    '
    return get_cur_scope().find_var(name)

def scoped_function(func):
    if False:
        print('Hello World!')
    '\n    invoke `func` in new scope.\n\n    :param func: a callable function that will be run in new scope.\n    :type func: callable\n    '
    enter_local_scope()
    try:
        func()
    except:
        raise
    finally:
        leave_local_scope()