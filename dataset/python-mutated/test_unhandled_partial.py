from typing import List

def test_unhandled_partial_gh1467():
    if False:
        while True:
            i = 10
    'Test that previously failed due to incorrect partial unification.\n\n    Submitted by BAH.\n\n    Fixed in https://github.com/osohq/oso/pull/1467.'
    POLICY = 'actor User {}\n\nallow(actor, action, resource) if has_permission(actor, action, resource);\n\nresource A {\n    permissions = ["Read"];\n    roles = ["User"];\n\n    "Read" if "User";\n}\n\nhas_role(user: User, "User", a: A) if\n    a_role in a.groups and\n    "Read" = a_role.p and\n    a_role.group_id in user.group_ids;\n\nresource Aprime {\n    relations = { a: A };\n    permissions = ["Read"];\n\n    "Read" if "User" on "a";\n}\n\nhas_relation(subject: A, "a", object: Aprime) if\n    subject = object;\n    '
    from dataclasses import dataclass
    from oso import Oso, Variable
    from polar import Expression, Pattern

    @dataclass
    class Group:
        permission: str
        group_id: int

    @dataclass
    class A:
        groups: List[Group]

    @dataclass
    class Aprime(A):
        pass

    @dataclass
    class User:
        group_ids: List[int]
    oso = Oso()
    oso.register_class(Aprime)
    oso.register_class(A)
    oso.register_class(User)
    oso.load_str(POLICY)
    constraint = Expression('And', [Expression('Isa', [Variable('resource'), Pattern('Aprime', {})])])
    results = list(oso.query_rule('allow', User(group_ids=[0]), 'Read', Variable('resource'), accept_expression=True, bindings={'resource': constraint}))
    print(results)