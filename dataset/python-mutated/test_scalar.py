import textwrap
from strawberry.schema_codegen import codegen

def test_scalar():
    if False:
        print('Hello World!')
    schema = '\n    scalar LocalDate @specifiedBy(url: "https://scalars.graphql.org/andimarek/local-date.html")\n    '
    expected = textwrap.dedent('\n        import strawberry\n        from typing import NewType\n\n        LocalDate = strawberry.scalar(NewType("LocalDate", object), specified_by_url="https://scalars.graphql.org/andimarek/local-date.html", serialize=lambda v: v, parse_value=lambda v: v)\n        ').strip()
    assert codegen(schema).strip() == expected

def test_scalar_with_description():
    if False:
        print('Hello World!')
    schema = '\n    "A date without a time-zone in the ISO-8601 calendar system, such as 2007-12-03."\n    scalar LocalDate\n    '
    expected = textwrap.dedent('\n        import strawberry\n        from typing import NewType\n\n        LocalDate = strawberry.scalar(NewType("LocalDate", object), description="A date without a time-zone in the ISO-8601 calendar system, such as 2007-12-03.", serialize=lambda v: v, parse_value=lambda v: v)\n        ').strip()
    assert codegen(schema).strip() == expected

def test_builtin_scalars():
    if False:
        for i in range(10):
            print('nop')
    schema = '\n    scalar JSON\n    scalar Date\n    scalar Time\n    scalar DateTime\n    scalar UUID\n    scalar Decimal\n\n    type Example {\n        a: JSON!\n        b: Date!\n        c: Time!\n        d: DateTime!\n        e: UUID!\n        f: Decimal!\n    }\n    '
    expected = textwrap.dedent('\n        import strawberry\n        from datetime import date\n        from datetime import datetime\n        from datetime import time\n        from decimal import Decimal\n        from uuid import UUID\n\n        @strawberry.type\n        class Example:\n            a: strawberry.JSON\n            b: date\n            c: time\n            d: datetime\n            e: UUID\n            f: Decimal\n        ').strip()
    assert codegen(schema).strip() == expected