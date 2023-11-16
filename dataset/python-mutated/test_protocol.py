import io
from rich.abc import RichRenderable
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

class Foo:

    def __rich__(self) -> Text:
        if False:
            while True:
                i = 10
        return Text('Foo')

def test_rich_cast():
    if False:
        i = 10
        return i + 15
    foo = Foo()
    console = Console(file=io.StringIO())
    console.print(foo)
    assert console.file.getvalue() == 'Foo\n'

class Fake:

    def __getattr__(self, name):
        if False:
            return 10
        return 12

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return 'Fake()'

def test_rich_cast_fake():
    if False:
        while True:
            i = 10
    fake = Fake()
    console = Console(file=io.StringIO())
    console.print(fake)
    assert console.file.getvalue() == 'Fake()\n'

def test_rich_cast_container():
    if False:
        i = 10
        return i + 15
    foo = Foo()
    console = Console(file=io.StringIO(), legacy_windows=False)
    console.print(Panel.fit(foo, padding=0))
    assert console.file.getvalue() == '╭───╮\n│Foo│\n╰───╯\n'

def test_abc():
    if False:
        print('Hello World!')
    foo = Foo()
    assert isinstance(foo, RichRenderable)
    assert isinstance(Text('hello'), RichRenderable)
    assert isinstance(Panel('hello'), RichRenderable)
    assert not isinstance(foo, str)
    assert not isinstance('foo', RichRenderable)
    assert not isinstance([], RichRenderable)

def test_cast_deep():
    if False:
        print('Hello World!')

    class B:

        def __rich__(self) -> Foo:
            if False:
                return 10
            return Foo()

    class A:

        def __rich__(self) -> B:
            if False:
                print('Hello World!')
            return B()
    console = Console(file=io.StringIO())
    console.print(A())
    assert console.file.getvalue() == 'Foo\n'

def test_cast_recursive():
    if False:
        for i in range(10):
            print('nop')

    class B:

        def __rich__(self) -> 'A':
            if False:
                i = 10
                return i + 15
            return A()

        def __repr__(self) -> str:
            if False:
                i = 10
                return i + 15
            return '<B>'

    class A:

        def __rich__(self) -> B:
            if False:
                i = 10
                return i + 15
            return B()

        def __repr__(self) -> str:
            if False:
                while True:
                    i = 10
            return '<A>'
    console = Console(file=io.StringIO())
    console.print(A())
    assert console.file.getvalue() == '<B>\n'