from typing import Any
import pytest
from litestar._openapi.typescript_converter.types import TypeScriptAnonymousInterface, TypeScriptArray, TypeScriptConst, TypeScriptEnum, TypeScriptInterface, TypeScriptIntersection, TypeScriptLiteral, TypeScriptNamespace, TypeScriptPrimitive, TypeScriptProperty, TypeScriptType, TypeScriptUnion

@pytest.mark.parametrize('value', ('string', 'number', 'boolean', 'any', 'null', 'undefined', 'symbol'))
def test_typescript_primitive(value: Any) -> None:
    if False:
        while True:
            i = 10
    assert TypeScriptPrimitive(value).write() == value

def test_typescript_intersection() -> None:
    if False:
        for i in range(10):
            print('nop')
    intersection = TypeScriptIntersection(types=(TypeScriptPrimitive('string'), TypeScriptPrimitive('number')))
    assert intersection.write() == 'string & number'

def test_typescript_union() -> None:
    if False:
        print('Hello World!')
    union = TypeScriptUnion(types=(TypeScriptPrimitive('string'), TypeScriptPrimitive('number')))
    assert union.write() == 'number | string'

@pytest.mark.parametrize('value, expected', (('abc', '"abc"'), (123, '123'), (100.123, '100.123'), (True, 'true'), (False, 'false')))
def test_typescript_literal(value: Any, expected: str) -> None:
    if False:
        while True:
            i = 10
    assert TypeScriptLiteral(value).write() == expected

@pytest.mark.parametrize('value, expected', ((TypeScriptPrimitive('string'), 'string[]'), (TypeScriptUnion(types=(TypeScriptPrimitive('string'), TypeScriptPrimitive('number'))), '(number | string)[]')))
def test_typescript_array(value: Any, expected: str) -> None:
    if False:
        return 10
    assert TypeScriptArray(item_type=value).write() == expected

def test_typescript_property() -> None:
    if False:
        while True:
            i = 10
    prop = TypeScriptProperty(required=True, key='myKey', value=TypeScriptPrimitive('string'))
    assert prop.write() == 'myKey: string;'
    prop.required = False
    assert prop.write() == 'myKey?: string;'

def test_typescript_anonymous_interface() -> None:
    if False:
        for i in range(10):
            print('nop')
    first_prop = TypeScriptProperty(required=True, key='aProp', value=TypeScriptPrimitive('string'))
    second_prop = TypeScriptProperty(required=True, key='bProp', value=TypeScriptPrimitive('number'))
    interface = TypeScriptAnonymousInterface(properties=(first_prop, second_prop))
    assert interface.write() == '{\n\taProp: string;\n\tbProp: number;\n}'

def test_typescript_named_interface() -> None:
    if False:
        print('Hello World!')
    first_prop = TypeScriptProperty(required=True, key='aProp', value=TypeScriptPrimitive('string'))
    second_prop = TypeScriptProperty(required=True, key='bProp', value=TypeScriptPrimitive('number'))
    interface = TypeScriptInterface(name='MyInterface', properties=(first_prop, second_prop))
    assert interface.write() == 'export interface MyInterface {\n\taProp: string;\n\tbProp: number;\n};'

def test_typescript_enum() -> None:
    if False:
        i = 10
        return i + 15
    enum = TypeScriptEnum(name='MyEnum', values=(('FIRST', 'a'), ('SECOND', 'b')))
    assert enum.write() == 'export enum MyEnum {\n\tFIRST = "a",\n\tSECOND = "b",\n};'

def test_typescript_type() -> None:
    if False:
        print('Hello World!')
    ts_type = TypeScriptType(name='MyUnion', value=TypeScriptUnion(types=(TypeScriptPrimitive('string'), TypeScriptPrimitive('number'))))
    assert ts_type.write() == 'export type MyUnion = number | string;'

def test_typescript_const() -> None:
    if False:
        print('Hello World!')
    const = TypeScriptConst(name='MyConstant', value=TypeScriptPrimitive('number'))
    assert const.write() == 'export const MyConstant: number;'

def test_typescript_namespace() -> None:
    if False:
        while True:
            i = 10
    first_prop = TypeScriptProperty(required=True, key='aProp', value=TypeScriptPrimitive('string'))
    second_prop = TypeScriptProperty(required=True, key='bProp', value=TypeScriptPrimitive('number'))
    interface = TypeScriptInterface(name='MyInterface', properties=(first_prop, second_prop))
    enum = TypeScriptEnum(name='MyEnum', values=(('FIRST', 'a'), ('SECOND', 'b')))
    namespace = TypeScriptNamespace('MyNamespace', values=(interface, enum))
    assert namespace.write() == 'export namespace MyNamespace {\n\texport enum MyEnum {\n\tFIRST = "a",\n\tSECOND = "b",\n};\n\n\texport interface MyInterface {\n\taProp: string;\n\tbProp: number;\n};\n};'