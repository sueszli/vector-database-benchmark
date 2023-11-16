from builtins import _test_sink, _test_source
from dataclasses import dataclass
from typing import final

@dataclass
class DataClass:
    bad: int
    benign: str

def bad_is_tainted():
    if False:
        print('Hello World!')
    context = DataClass(bad=_test_source(), benign=1)
    _test_sink(context)
    return context

def benign_is_untainted():
    if False:
        return 10
    context = DataClass(bad=_test_source(), benign=1)
    _test_sink(context.benign)
    return context

@dataclass
class DataClassWIthInit:
    bad: int

    def __init__(self, bad: int) -> None:
        if False:
            print('Hello World!')
        self.bad = bad
        _test_sink(bad)

def issue_in_dataclass_constructor() -> None:
    if False:
        while True:
            i = 10
    DataClassWIthInit(bad=_test_source())

@dataclass
class WeirdDataClass:
    bad: int
    bad_sink: int

    def __init__(self, bad: int, another: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        object.__setattr__(self, 'bad', bad)
        object.__setattr__(self, 'bad_sink', another)

def test_weird_dataclass_taint() -> WeirdDataClass:
    if False:
        i = 10
        return i + 15
    return WeirdDataClass(bad=1, another=2)

def issue_with_weird_dataclass():
    if False:
        while True:
            i = 10
    wdc = WeirdDataClass(bad=1, another=2)
    _test_sink(wdc.bad)

@final
@dataclass(frozen=True)
class DataClassWithSource:
    tainted: int
    not_tainted: str

def test_dataclass_with_source(context: DataClassWithSource) -> None:
    if False:
        while True:
            i = 10
    _test_sink(context.tainted)
    _test_sink(context.not_tainted)

@final
@dataclass(frozen=True)
class DataClassWithOtherSource:
    tainted: int
    not_tainted: str

def test_dataclass_with_other_source(context: DataClassWithOtherSource) -> None:
    if False:
        while True:
            i = 10
    _test_sink(context.tainted)
    _test_sink(context.not_tainted)

@dataclass
class DataClassWithClassAttributeTaintedDirectly:
    bad: int
    benign: str

def test_class_attr_model_tainted_directly() -> None:
    if False:
        print('Hello World!')
    DataClassWithClassAttributeTaintedDirectly(bad=1, benign=_test_source())
    DataClassWithClassAttributeTaintedDirectly(bad=_test_source(), benign='1')
    data_object_no_issue = DataClassWithClassAttributeTaintedDirectly(bad=1, benign='1')
    data_object_no_issue.benign = _test_source()
    data_object_issue = DataClassWithClassAttributeTaintedDirectly(bad=1, benign='1')
    data_object_issue.bad = _test_source()

@dataclass
class DataClassWithClassAttributeTaintedInConstructor:
    bad: int
    benign: str

def test_class_attr_model_tainted_in_constructor() -> None:
    if False:
        return 10
    DataClassWithClassAttributeTaintedInConstructor(bad=1, benign=_test_source())
    DataClassWithClassAttributeTaintedInConstructor(bad=_test_source(), benign='1')
    data_object_no_issue = DataClassWithClassAttributeTaintedInConstructor(bad=1, benign='1')
    data_object_no_issue.benign = _test_source()
    data_object_issue = DataClassWithClassAttributeTaintedInConstructor(bad=1, benign='1')
    data_object_issue.bad = _test_source()

def test_constructor_tito(x: int, y: str) -> DataClass:
    if False:
        return 10
    return DataClass(bad=x, benign=y)

@dataclass
class DataClassSwapArguments:
    foo: str
    bar: str

    def __init__(self, foo: str, bar: str) -> None:
        if False:
            i = 10
            return i + 15
        self.foo = bar
        self.bar = foo

def test_dataclass_parameter_path(dc: DataClass):
    if False:
        while True:
            i = 10
    _test_sink(dc.bad)

def test_dataclass_positional_parameter(x: int, y: str) -> None:
    if False:
        i = 10
        return i + 15
    _test_sink(DataClass(x, y))