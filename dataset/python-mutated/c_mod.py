from typing import List
import a_mod
import b_mod
import x_mod
from a_mod import AObject as C_AObject
from b_mod import BObject as C_BObject
import strawberry

def c_inheritance_resolver() -> List['CInheritance']:
    if False:
        i = 10
        return i + 15
    pass

def c_composition_resolver() -> List['CComposition']:
    if False:
        print('Hello World!')
    pass

def c_composition_by_name_resolver() -> List['CCompositionByName']:
    if False:
        return 10
    pass

@strawberry.type
class CInheritance(a_mod.AObject, b_mod.BObject):
    pass

@strawberry.type
class CComposition:
    a_list: List[a_mod.AObject]
    b_list: List[b_mod.BObject]

@strawberry.type
class CCompositionByName:
    a_list: List['C_AObject']
    b_list: List['C_BObject']

    @strawberry.field
    def a_method(self) -> List['C_AObject']:
        if False:
            for i in range(10):
                print('nop')
        return self.a_list

    @strawberry.field
    def b_method(self) -> List['C_BObject']:
        if False:
            i = 10
            return i + 15
        return self.b_list

@strawberry.type
class CCompositionByNameWithResolvers:
    a_list: List['C_AObject'] = strawberry.field(resolver=a_mod.a_resolver)
    b_list: List['C_BObject'] = strawberry.field(resolver=b_mod.b_resolver)

@strawberry.type
class CCompositionByNameWithTypelessResolvers:
    a_list: List['C_AObject'] = strawberry.field(resolver=x_mod.typeless_resolver)
    b_list: List['C_BObject'] = strawberry.field(resolver=x_mod.typeless_resolver)

@strawberry.type
class CCompositionOnlyResolvers:
    a_list = strawberry.field(resolver=a_mod.a_resolver)
    b_list = strawberry.field(resolver=b_mod.b_resolver)