import tempfile
import time
from pathlib import Path
import strawberry
from strawberry.extensions import pyinstrument

def a():
    if False:
        print('Hello World!')
    time.sleep(0.1)
    return b()

def b():
    if False:
        while True:
            i = 10
    time.sleep(0.1)
    return c()

def c():
    if False:
        print('Hello World!')
    time.sleep(0.1)
    return 4

def test_basic_pyinstrument():
    if False:
        while True:
            i = 10
    REPORT_PATH = tempfile.NamedTemporaryFile().name

    @strawberry.type
    class Query:

        @strawberry.field
        def the_field(self) -> int:
            if False:
                for i in range(10):
                    print('nop')
            return a()
    schema = strawberry.Schema(query=Query, extensions=[pyinstrument.PyInstrument(report_path=REPORT_PATH)])
    result = schema.execute_sync('{ theField }')
    assert not result.errors
    assert result.data['theField'] == 4
    content = Path(REPORT_PATH, encoding='utf-8').read_text()
    assert '"function": "a"' in content
    assert '"function": "b"' in content
    assert '"function": "c"' in content
    assert content.count('"function": "sleep"') == 3