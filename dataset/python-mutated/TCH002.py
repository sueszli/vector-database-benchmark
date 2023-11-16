"""Tests to determine accurate detection of typing-only imports."""

def f():
    if False:
        for i in range(10):
            print('nop')
    import pandas as pd
    x: pd.DataFrame

def f():
    if False:
        i = 10
        return i + 15
    from pandas import DataFrame
    x: DataFrame

def f():
    if False:
        for i in range(10):
            print('nop')
    from pandas import DataFrame as df
    x: df

def f():
    if False:
        print('Hello World!')
    import pandas as pd
    x: pd.DataFrame = 1

def f():
    if False:
        return 10
    from pandas import DataFrame
    x: DataFrame = 2

def f():
    if False:
        return 10
    from pandas import DataFrame as df
    x: df = 3

def f():
    if False:
        while True:
            i = 10
    import pandas as pd
    x: 'pd.DataFrame' = 1

def f():
    if False:
        return 10
    import pandas as pd
    x = dict['pd.DataFrame', 'pd.DataFrame']

def f():
    if False:
        i = 10
        return i + 15
    import pandas as pd
    print(pd)

def f():
    if False:
        for i in range(10):
            print('nop')
    from pandas import DataFrame
    print(DataFrame)

def f():
    if False:
        for i in range(10):
            print('nop')
    from pandas import DataFrame

    def f():
        if False:
            while True:
                i = 10
        print(DataFrame)

def f():
    if False:
        i = 10
        return i + 15
    from typing import Dict, Any

    def example() -> Any:
        if False:
            return 10
        return 1
    x: Dict[int] = 20

def f():
    if False:
        while True:
            i = 10
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from typing import Dict
    x: Dict[int] = 20

def f():
    if False:
        i = 10
        return i + 15
    from pathlib import Path

    class ImportVisitor(ast.NodeTransformer):

        def __init__(self, cwd: Path) -> None:
            if False:
                while True:
                    i = 10
            self.cwd = cwd
            origin = Path(spec.origin)

    class ExampleClass:

        def __init__(self):
            if False:
                for i in range(10):
                    print('nop')
            self.cwd = Path(pandas.getcwd())

def f():
    if False:
        for i in range(10):
            print('nop')
    import pandas

    class Migration:
        enum = pandas

def f():
    if False:
        for i in range(10):
            print('nop')
    import pandas

    class Migration:
        enum = pandas.EnumClass

def f():
    if False:
        return 10
    from typing import TYPE_CHECKING
    from pandas import y
    if TYPE_CHECKING:
        _type = x
    else:
        _type = y

def f():
    if False:
        i = 10
        return i + 15
    from typing import TYPE_CHECKING
    from pandas import y
    if TYPE_CHECKING:
        _type = x
    elif True:
        _type = y

def f():
    if False:
        print('Hello World!')
    from typing import cast
    import pandas as pd
    x = cast(pd.DataFrame, 2)

def f():
    if False:
        for i in range(10):
            print('nop')
    import pandas as pd
    x = dict[pd.DataFrame, pd.DataFrame]

def f():
    if False:
        for i in range(10):
            print('nop')
    import pandas as pd

def f():
    if False:
        while True:
            i = 10
    from pandas import DataFrame
    x: DataFrame = 2

def f():
    if False:
        for i in range(10):
            print('nop')
    from pandas import DataFrame
    x: DataFrame = 2

def f():
    if False:
        print('Hello World!')
    global Member
    from module import Member
    x: Member = 1

def f():
    if False:
        while True:
            i = 10
    from typing_extensions import TYPE_CHECKING
    from pandas import y
    if TYPE_CHECKING:
        _type = x
    elif True:
        _type = y