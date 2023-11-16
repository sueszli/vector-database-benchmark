import ast
import textwrap
from pathlib import Path
from pytest_snapshot.plugin import Snapshot
from strawberry.schema_codegen import codegen
HERE = Path(__file__).parent

def test_long_descriptions(snapshot: Snapshot):
    if False:
        while True:
            i = 10
    snapshot.snapshot_dir = HERE / 'snapshots'
    schema = '\n    """A connection to a list of items."""\n    type FilmCharactersConnection {\n    """Information to aid in pagination."""\n    pageInfo: PageInfo!\n\n    """A list of edges."""\n    edges: [FilmCharactersEdge]\n\n    """\n    A count of the total number of objects in this connection, ignoring pagination.\n    This allows a client to fetch the first five objects by passing "5" as the\n    argument to "first", then fetch the total count so it could display "5 of 83",\n    for example.\n    """\n    totalCount: Int\n\n    """\n    A list of all of the objects returned in the connection. This is a convenience\n    field provided for quickly exploring the API; rather than querying for\n    "{ edges { node } }" when no edge data is needed, this field can be be used\n    instead. Note that when clients like Relay need to fetch the "cursor" field on\n    the edge to enable efficient pagination, this shortcut cannot be used, and the\n    full "{ edges { node } }" version should be used instead.\n    """\n    characters: [Person]\n    }\n    '
    output = codegen(schema)
    ast.parse(output)
    snapshot.assert_match(output, 'long_descriptions.py')

def test_can_convert_descriptions_with_quotes():
    if False:
        return 10
    schema = '\n    """A type of person or character within the "Star Wars" Universe."""\n    type Species {\n        """The classification of this species, such as "mammal" or "reptile"."""\n        classification: String!\n    }\n    '
    output = codegen(schema)
    expected_output = textwrap.dedent('\n        import strawberry\n\n        @strawberry.type(description=\'A type of person or character within the "Star Wars" Universe.\')\n        class Species:\n            classification: str = strawberry.field(description=\'The classification of this species, such as "mammal" or "reptile".\')\n        ').lstrip()
    assert output == expected_output