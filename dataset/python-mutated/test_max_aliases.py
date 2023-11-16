from typing import Optional
import strawberry
from strawberry.extensions.max_aliases import MaxAliasesLimiter

@strawberry.type
class Human:
    name: str
    email: str

@strawberry.type
class Query:

    @strawberry.field
    def user(self, name: Optional[str]=None, email: Optional[str]=None) -> Human:
        if False:
            i = 10
            return i + 15
        return Human(name='Jane Doe', email='jane@example.com')
    version: str
    user1: Human
    user2: Human
    user3: Human

def test_2_aliases_same_content():
    if False:
        i = 10
        return i + 15
    query = '\n    {\n      matt: user(name: "matt") {\n        email\n      }\n      matt_alias: user(name: "matt") {\n        email\n      }\n    }\n    '
    result = _execute_with_max_aliases(query, 1)
    assert len(result.errors) == 1
    assert result.errors[0].message == '2 aliases found. Allowed: 1'

def test_2_aliases_different_content():
    if False:
        while True:
            i = 10
    query = '\n    query read {\n      matt: user(name: "matt") {\n        email\n      }\n      matt_alias: user(name: "matt42") {\n        email\n      }\n    }\n    '
    result = _execute_with_max_aliases(query, 1)
    assert len(result.errors) == 1
    assert result.errors[0].message == '2 aliases found. Allowed: 1'

def test_multiple_aliases_some_overlap_in_content():
    if False:
        return 10
    query = '\n    query read {\n      matt: user(name: "matt") {\n        email\n      }\n      jane: user(name: "jane") {\n        email\n      }\n      matt_alias: user(name: "matt") {\n        email\n      }\n    }\n    '
    result = _execute_with_max_aliases(query, 1)
    assert len(result.errors) == 1
    assert result.errors[0].message == '3 aliases found. Allowed: 1'

def test_multiple_arguments():
    if False:
        i = 10
        return i + 15
    query = '\n    query read {\n      matt: user(name: "matt", email: "matt@example.com") {\n        email\n      }\n      jane: user(name: "jane") {\n        email\n      }\n      matt_alias: user(name: "matt", email: "matt@example.com") {\n        email\n      }\n    }\n    '
    result = _execute_with_max_aliases(query, 1)
    assert len(result.errors) == 1
    assert result.errors[0].message == '3 aliases found. Allowed: 1'

def test_alias_in_nested_field():
    if False:
        return 10
    query = '\n    query read {\n      matt: user(name: "matt") {\n        email_address: email\n      }\n    }\n    '
    result = _execute_with_max_aliases(query, 1)
    assert len(result.errors) == 1
    assert result.errors[0].message == '2 aliases found. Allowed: 1'

def test_alias_in_fragment():
    if False:
        return 10
    query = '\n    fragment humanInfo on Human {\n      email_address: email\n    }\n    query read {\n      matt: user(name: "matt") {\n        ...humanInfo\n      }\n    }\n    '
    result = _execute_with_max_aliases(query, 1)
    assert len(result.errors) == 1
    assert result.errors[0].message == '2 aliases found. Allowed: 1'

def test_2_top_level_1_nested():
    if False:
        return 10
    query = '{\n      matt: user(name: "matt") {\n        email_address: email\n      }\n      matt_alias: user(name: "matt") {\n        email\n      }\n    }\n    '
    result = _execute_with_max_aliases(query, 2)
    assert len(result.errors) == 1
    assert result.errors[0].message == '3 aliases found. Allowed: 2'

def test_no_error_one_aliased_one_without():
    if False:
        print('Hello World!')
    query = '\n    {\n      user(name: "matt") {\n        email\n      }\n      matt_alias: user(name: "matt") {\n        email\n      }\n    }\n    '
    result = _execute_with_max_aliases(query, 1)
    assert not result.errors

def test_no_error_for_multiple_but_not_too_many_aliases():
    if False:
        print('Hello World!')
    query = '{\n      matt: user(name: "matt") {\n        email\n      }\n      matt_alias: user(name: "matt") {\n        email\n      }\n    }\n    '
    result = _execute_with_max_aliases(query, 2)
    assert not result.errors

def _execute_with_max_aliases(query: str, max_alias_count: int):
    if False:
        i = 10
        return i + 15
    schema = strawberry.Schema(Query, extensions=[MaxAliasesLimiter(max_alias_count=max_alias_count)])
    return schema.execute_sync(query)