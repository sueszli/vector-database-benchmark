from typing import List, overload

class Series:

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        pass

class DataFrame:

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        pass

    @overload
    def __getitem__(self, key: str) -> Series:
        if False:
            while True:
                i = 10
        pass

    @overload
    def __getitem__(self, key: List[str]) -> 'DataFrame':
        if False:
            while True:
                i = 10
        pass

    def __getitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        return None

    @overload
    def __setitem__(self, key: str, newvalue: Series) -> None:
        if False:
            for i in range(10):
                print('nop')
        pass

    @overload
    def __setitem__(self, key: List[str], newvalue: 'DataFrame') -> None:
        if False:
            while True:
                i = 10
        pass

    def __setitem__(self, key, newvalue):
        if False:
            print('Hello World!')
        return None

    def apply(self, func, axis: int=0) -> Series:
        if False:
            for i in range(10):
                print('nop')
        pass

def sink(arg: DataFrame):
    if False:
        print('Hello World!')
    pass

def source() -> DataFrame:
    if False:
        while True:
            i = 10
    pass

def clear_df() -> DataFrame:
    if False:
        while True:
            i = 10
    pass

def map(arg):
    if False:
        for i in range(10):
            print('nop')
    pass

def map2(arg1, arg2):
    if False:
        while True:
            i = 10
    pass

def issue1():
    if False:
        for i in range(10):
            print('nop')
    df = source()
    df['a'] = df['b']
    sink(df)

def issue2():
    if False:
        while True:
            i = 10
    df = source()
    df2 = df[['a', 'b']]
    sink(df2)

def issue3():
    if False:
        return 10
    df = source()
    df2 = clear_df()
    df2['a'] = df['b']
    sink(df2)

def issue4():
    if False:
        print('Hello World!')
    df = source()
    df2 = clear_df()
    df2[['b', 'a']] = df[['a', 'b']]
    sink(df2)

def issue5():
    if False:
        return 10
    df = source()
    var = 'a'
    df['b'] = df[var]
    sink(df)

def issue6():
    if False:
        for i in range(10):
            print('nop')
    df = source()
    df2 = clear_df()
    df2['a'] = df.apply(lambda x: map(x['a']), axis=1)
    sink(df2)

def issue7():
    if False:
        return 10
    df = source()
    df2 = clear_df()
    df2['a'] = df.apply(lambda x: map(x), axis=1)
    sink(df2)

def issue8():
    if False:
        print('Hello World!')
    df = source()
    df2 = clear_df()
    df2['a'] = df.apply(lambda x: map2(x['a'], x['b']), axis=1)
    sink(df2)

def issue9():
    if False:
        return 10
    df = source()
    df2 = clear_df()
    df2['a'] = df2.apply(lambda x: map(df), axis=1)
    sink(df2)