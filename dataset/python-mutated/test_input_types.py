import textwrap
from strawberry.schema_codegen import codegen

def test_codegen_input_type():
    if False:
        return 10
    schema = '\n    input Example {\n        a: Int!\n        b: Float!\n        c: Boolean!\n        d: String!\n        e: ID!\n        f: [Int!]!\n        g: [Float!]!\n        h: [Boolean!]!\n        i: [String!]!\n        j: [ID!]!\n        k: [Int]\n        l: [Float]\n        m: [Boolean]\n        n: [String]\n        o: [ID]\n    }\n    '
    expected = textwrap.dedent('\n        import strawberry\n\n        @strawberry.input\n        class Example:\n            a: int\n            b: float\n            c: bool\n            d: str\n            e: strawberry.ID\n            f: list[int]\n            g: list[float]\n            h: list[bool]\n            i: list[str]\n            j: list[strawberry.ID]\n            k: list[int | None] | None\n            l: list[float | None] | None\n            m: list[bool | None] | None\n            n: list[str | None] | None\n            o: list[strawberry.ID | None] | None\n        ').strip()
    assert codegen(schema).strip() == expected