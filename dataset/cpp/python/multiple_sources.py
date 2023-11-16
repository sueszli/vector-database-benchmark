# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union


class Node:
    def __init__(self, id) -> None:
        self.id = id

    def send(self, vc) -> None:
        ...

    @classmethod
    def get(cls, id) -> "Node":
        return cls(id)


def user_controlled_input():
    return "evil"


def permissive_context():
    return 0


def combine_tainted_user_and_dangerous_vc():
    id = user_controlled_input()
    vc = permissive_context()
    Node.get(id).send(vc)


def demonstrate_triggered_context(vc):
    id = user_controlled_input()
    Node.get(id).send(vc)


def demonstrate_triggered_input(id):
    vc = permissive_context()
    Node.get(id).send(vc)


def issue_with_triggered_input():
    id = user_controlled_input()
    demonstrate_triggered_input(id)


def issue_with_triggered_context():
    vc = permissive_context()
    demonstrate_triggered_context(vc)


def no_issue_with_wrong_label():
    vc = permissive_context()
    demonstrate_triggered_input(vc)


def wrapper(id, vc):
    Node.get(id).send(vc)


def no_issue_with_wrapper_call():
    id = user_controlled_input()
    vc = permissive_context()
    # Since we do not propagate the untriggered partial sinks,
    # we miss the issue here
    wrapper(id, vc)


def test_other_input():
    return "other"


def combines_tests_and_context(test, vc):
    return None


def a_source():
    return None


def b_source():
    return None


def issue_with_test_a_and_b():
    combines_tests_and_context(a_source(), permissive_context())
    combines_tests_and_context(b_source(), permissive_context())


def a_sink(arg):
    return


def b_sink(arg):
    return


def transform_t(arg):
    return


def sanitize_source_a_tito(arg):
    return arg


def sanitize_source_b_tito(arg):
    return arg


def sanitize_sink_a_tito(arg):
    return arg


def no_issue_with_transform():
    x = a_source()
    y = transform_t(x)
    combines_tests_and_context(y, permissive_context())


def no_sink_with_transform(x):
    y = transform_t(x)
    combines_tests_and_context(a_source(), y)


def issue_with_sanitizer():
    x = a_source()
    y = sanitize_sink_a_tito(x)
    combines_tests_and_context(y, permissive_context())


def no_sink_with_sanitizer(x):
    y = sanitize_source_b_tito(sanitize_source_a_tito(x))
    combines_tests_and_context(y, permissive_context())


def user_controlled_input_wrapper():
    return user_controlled_input()


def demonstrate_triggered_context_more_hops(vc):
    # More hops enable showing the source trace as a subtrace
    id = user_controlled_input_wrapper()
    Node.get(id).send(vc)


def issue_with_triggered_context_more_hops():
    vc = permissive_context()
    demonstrate_triggered_context_more_hops(vc)


class A:
    def multi_sink(self, user_controlled, permissive_context):
        pass


class B:
    def multi_sink(self, user_controlled, permissive_context):
        pass


def muliple_main_issues_1(a_or_b: Union[A, B]):
    # Due to multiple potential callables at the same call site
    a_or_b.multi_sink(user_controlled_input(), permissive_context())


def muliple_main_issues_2():
    # Due to joining the issue handles from multiple call sites
    vc = permissive_context()
    multiple_triggered_context(vc)


def multiple_triggered_context(vc):
    id1 = user_controlled_input()
    Node.get(id1).send(vc)
    id2 = user_controlled_input()
    Node.get(id2).send(vc)


def false_negative_triggered_context(vc):
    id = user_controlled_input()
    # Since we do not propagate the untriggered partial sinks,
    # we do not trigger a partial sink here
    wrapper(id, vc)


def no_issue_with_wrapper_call_2():
    vc = permissive_context()
    false_negative_triggered_context(vc)
