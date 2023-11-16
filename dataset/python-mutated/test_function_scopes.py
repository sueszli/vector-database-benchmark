"""Scopes and Namespaces.

@see: https://docs.python.org/3/tutorial/classes.html#scopes-and-namespaces-example

A NAMESPACE is a mapping from names to objects. Most namespaces are currently implemented as Python
dictionaries, but that’s normally not noticeable in any way (except for performance), and it may
change in the future. Examples of namespaces are: the set of built-in names (containing functions
such as abs(), and built-in exception names); the global names in a module; and the local names
in a function invocation. In a sense the set of attributes of an object also form a namespace.
The important thing to know about namespaces is that there is absolutely no relation between names
in different namespaces; for instance, two different modules may both define a function maximize
without confusion — users of the modules must prefix it with the module name.

By the way, we use the word attribute for any name following a dot — for example, in the expression
z.real, real is an attribute of the object z. Strictly speaking, references to names in modules are
attribute references: in the expression modname.func_name, modname is a module object and func_name
is an attribute of it. In this case there happens to be a straightforward mapping between the
module’s attributes and the global names defined in the module: they share the same namespace!

A SCOPE is a textual region of a Python program where a namespace is directly accessible.
“Directly accessible” here means that an unqualified reference to a name attempts to find the name
in the namespace.

Although scopes are determined statically, they are used dynamically. At any time during execution,
there are at least three nested scopes whose namespaces are directly accessible:
- the innermost scope, which is searched first, contains the local names.
- the scopes of any enclosing functions, which are searched starting with the nearest enclosing
scope, contains non-local, but also non-global names.
- the next-to-last scope contains the current module’s global names.
- the outermost scope (searched last) is the namespace containing built-in names.

BE CAREFUL!!!
-------------
Changing global or nonlocal variables from within an inner function might be a BAD
practice and might lead to harder debugging and to more fragile code! Do this only if you know
what you're doing.
"""
test_variable = 'initial global value'

def test_function_scopes():
    if False:
        while True:
            i = 10
    'Scopes and Namespaces Example'
    test_variable = 'initial value inside test function'

    def do_local():
        if False:
            i = 10
            return i + 15
        test_variable = 'local value'
        return test_variable

    def do_nonlocal():
        if False:
            return 10
        nonlocal test_variable
        test_variable = 'nonlocal value'
        return test_variable

    def do_global():
        if False:
            while True:
                i = 10
        global test_variable
        test_variable = 'global value'
        return test_variable
    assert test_variable == 'initial value inside test function'
    do_local()
    assert test_variable == 'initial value inside test function'
    do_nonlocal()
    assert test_variable == 'nonlocal value'
    do_global()
    assert test_variable == 'nonlocal value'

def test_global_variable_access():
    if False:
        return 10
    'Testing global variable access from within a function'
    global test_variable
    assert test_variable == 'global value'