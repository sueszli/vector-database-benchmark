from typing import Union

class Node:

    def __init__(self, id) -> None:
        if False:
            print('Hello World!')
        self.id = id

    def send(self, vc) -> None:
        if False:
            i = 10
            return i + 15
        ...

    @classmethod
    def get(cls, id) -> 'Node':
        if False:
            i = 10
            return i + 15
        return cls(id)

def user_controlled_input():
    if False:
        for i in range(10):
            print('nop')
    return 'evil'

def permissive_context():
    if False:
        return 10
    return 0

def combine_tainted_user_and_dangerous_vc():
    if False:
        for i in range(10):
            print('nop')
    id = user_controlled_input()
    vc = permissive_context()
    Node.get(id).send(vc)

def demonstrate_triggered_context(vc):
    if False:
        print('Hello World!')
    id = user_controlled_input()
    Node.get(id).send(vc)

def demonstrate_triggered_input(id):
    if False:
        while True:
            i = 10
    vc = permissive_context()
    Node.get(id).send(vc)

def issue_with_triggered_input():
    if False:
        while True:
            i = 10
    id = user_controlled_input()
    demonstrate_triggered_input(id)

def issue_with_triggered_context():
    if False:
        for i in range(10):
            print('nop')
    vc = permissive_context()
    demonstrate_triggered_context(vc)

def no_issue_with_wrong_label():
    if False:
        return 10
    vc = permissive_context()
    demonstrate_triggered_input(vc)

def wrapper(id, vc):
    if False:
        while True:
            i = 10
    Node.get(id).send(vc)

def no_issue_with_wrapper_call():
    if False:
        while True:
            i = 10
    id = user_controlled_input()
    vc = permissive_context()
    wrapper(id, vc)

def test_other_input():
    if False:
        for i in range(10):
            print('nop')
    return 'other'

def combines_tests_and_context(test, vc):
    if False:
        i = 10
        return i + 15
    return None

def a_source():
    if False:
        print('Hello World!')
    return None

def b_source():
    if False:
        while True:
            i = 10
    return None

def issue_with_test_a_and_b():
    if False:
        i = 10
        return i + 15
    combines_tests_and_context(a_source(), permissive_context())
    combines_tests_and_context(b_source(), permissive_context())

def a_sink(arg):
    if False:
        i = 10
        return i + 15
    return

def b_sink(arg):
    if False:
        for i in range(10):
            print('nop')
    return

def transform_t(arg):
    if False:
        while True:
            i = 10
    return

def sanitize_source_a_tito(arg):
    if False:
        while True:
            i = 10
    return arg

def sanitize_source_b_tito(arg):
    if False:
        return 10
    return arg

def sanitize_sink_a_tito(arg):
    if False:
        for i in range(10):
            print('nop')
    return arg

def no_issue_with_transform():
    if False:
        i = 10
        return i + 15
    x = a_source()
    y = transform_t(x)
    combines_tests_and_context(y, permissive_context())

def no_sink_with_transform(x):
    if False:
        for i in range(10):
            print('nop')
    y = transform_t(x)
    combines_tests_and_context(a_source(), y)

def issue_with_sanitizer():
    if False:
        i = 10
        return i + 15
    x = a_source()
    y = sanitize_sink_a_tito(x)
    combines_tests_and_context(y, permissive_context())

def no_sink_with_sanitizer(x):
    if False:
        print('Hello World!')
    y = sanitize_source_b_tito(sanitize_source_a_tito(x))
    combines_tests_and_context(y, permissive_context())

def user_controlled_input_wrapper():
    if False:
        while True:
            i = 10
    return user_controlled_input()

def demonstrate_triggered_context_more_hops(vc):
    if False:
        print('Hello World!')
    id = user_controlled_input_wrapper()
    Node.get(id).send(vc)

def issue_with_triggered_context_more_hops():
    if False:
        return 10
    vc = permissive_context()
    demonstrate_triggered_context_more_hops(vc)

class A:

    def multi_sink(self, user_controlled, permissive_context):
        if False:
            i = 10
            return i + 15
        pass

class B:

    def multi_sink(self, user_controlled, permissive_context):
        if False:
            print('Hello World!')
        pass

def muliple_main_issues_1(a_or_b: Union[A, B]):
    if False:
        i = 10
        return i + 15
    a_or_b.multi_sink(user_controlled_input(), permissive_context())

def muliple_main_issues_2():
    if False:
        while True:
            i = 10
    vc = permissive_context()
    multiple_triggered_context(vc)

def multiple_triggered_context(vc):
    if False:
        print('Hello World!')
    id1 = user_controlled_input()
    Node.get(id1).send(vc)
    id2 = user_controlled_input()
    Node.get(id2).send(vc)

def false_negative_triggered_context(vc):
    if False:
        i = 10
        return i + 15
    id = user_controlled_input()
    wrapper(id, vc)

def no_issue_with_wrapper_call_2():
    if False:
        return 10
    vc = permissive_context()
    false_negative_triggered_context(vc)