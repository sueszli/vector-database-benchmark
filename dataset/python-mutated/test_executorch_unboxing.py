import unittest
from types import ModuleType
from torchgen import local
from torchgen.api import cpp as aten_cpp, types as aten_types
from torchgen.api.types import ArgName, BaseCType, ConstRefCType, MutRefCType, NamedCType
from torchgen.executorch.api import et_cpp as et_cpp, types as et_types
from torchgen.executorch.api.unboxing import Unboxing
from torchgen.model import BaseTy, BaseType, ListType, OptionalType, Type

def aten_argumenttype_type_wrapper(t: Type, *, mutable: bool, binds: ArgName, remove_non_owning_ref_types: bool=False) -> NamedCType:
    if False:
        return 10
    return aten_cpp.argumenttype_type(t, mutable=mutable, binds=binds, remove_non_owning_ref_types=remove_non_owning_ref_types)
ATEN_UNBOXING = Unboxing(argument_type_gen=aten_argumenttype_type_wrapper)
ET_UNBOXING = Unboxing(argument_type_gen=et_cpp.argumenttype_type)

class TestUnboxing(unittest.TestCase):
    """
    Could use torch.testing._internal.common_utils to reduce boilerplate.
    GH CI job doesn't build torch before running tools unit tests, hence
    manually adding these parametrized tests.
    """

    @local.parametrize(use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False)
    def test_symint_argument_translate_ctype_aten(self) -> None:
        if False:
            i = 10
            return i + 15
        symint_list_type = ListType(elem=BaseType(BaseTy.SymInt), size=None)
        (out_name, ctype, _, _) = ATEN_UNBOXING.argumenttype_evalue_convert(t=symint_list_type, arg_name='size', mutable=False)
        self.assertEqual(out_name, 'size_list_out')
        self.assertIsInstance(ctype, BaseCType)
        self.assertEqual(ctype, aten_types.BaseCType(aten_types.intArrayRefT))

    @local.parametrize(use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False)
    def test_symint_argument_translate_ctype_executorch(self) -> None:
        if False:
            i = 10
            return i + 15
        symint_list_type = ListType(elem=BaseType(BaseTy.SymInt), size=None)
        (out_name, ctype, _, _) = ET_UNBOXING.argumenttype_evalue_convert(t=symint_list_type, arg_name='size', mutable=False)
        self.assertEqual(out_name, 'size_list_out')
        self.assertIsInstance(ctype, et_types.ArrayRefCType)
        self.assertEqual(ctype, et_types.ArrayRefCType(elem=BaseCType(aten_types.longT)))

    @local.parametrize(use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False)
    def _test_const_tensor_argument_translate_ctype(self, unboxing: Unboxing, types: ModuleType) -> None:
        if False:
            return 10
        tensor_type = BaseType(BaseTy.Tensor)
        (out_name, ctype, _, _) = unboxing.argumenttype_evalue_convert(t=tensor_type, arg_name='self', mutable=False)
        self.assertEqual(out_name, 'self_base')
        self.assertEqual(ctype, ConstRefCType(BaseCType(types.tensorT)))

    def test_const_tensor_argument_translate_ctype_aten(self) -> None:
        if False:
            i = 10
            return i + 15
        self._test_const_tensor_argument_translate_ctype(ATEN_UNBOXING, aten_types)

    def test_const_tensor_argument_translate_ctype_executorch(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._test_const_tensor_argument_translate_ctype(ET_UNBOXING, et_types)

    @local.parametrize(use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False)
    def _test_mutable_tensor_argument_translate_ctype(self, unboxing: Unboxing, types: ModuleType) -> None:
        if False:
            while True:
                i = 10
        tensor_type = BaseType(BaseTy.Tensor)
        (out_name, ctype, _, _) = unboxing.argumenttype_evalue_convert(t=tensor_type, arg_name='out', mutable=True)
        self.assertEqual(out_name, 'out_base')
        self.assertEqual(ctype, MutRefCType(BaseCType(types.tensorT)))

    def test_mutable_tensor_argument_translate_ctype_aten(self) -> None:
        if False:
            return 10
        self._test_mutable_tensor_argument_translate_ctype(ATEN_UNBOXING, aten_types)

    def test_mutable_tensor_argument_translate_ctype_executorch(self) -> None:
        if False:
            i = 10
            return i + 15
        self._test_mutable_tensor_argument_translate_ctype(ET_UNBOXING, et_types)

    @local.parametrize(use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False)
    def _test_tensor_list_argument_translate_ctype(self, unboxing: Unboxing, types: ModuleType) -> None:
        if False:
            while True:
                i = 10
        tensor_list_type = ListType(elem=BaseType(BaseTy.Tensor), size=None)
        (out_name, ctype, _, _) = unboxing.argumenttype_evalue_convert(t=tensor_list_type, arg_name='out', mutable=True)
        self.assertEqual(out_name, 'out_list_out')
        self.assertEqual(ctype, BaseCType(types.tensorListT))

    def test_tensor_list_argument_translate_ctype_aten(self) -> None:
        if False:
            print('Hello World!')
        self._test_tensor_list_argument_translate_ctype(ATEN_UNBOXING, aten_types)

    def test_tensor_list_argument_translate_ctype_executorch(self) -> None:
        if False:
            return 10
        self._test_tensor_list_argument_translate_ctype(ET_UNBOXING, et_types)

    @local.parametrize(use_const_ref_for_mutable_tensors=False, use_ilistref_for_tensor_lists=False)
    def _test_optional_int_argument_translate_ctype(self, unboxing: Unboxing, types: ModuleType) -> None:
        if False:
            return 10
        optional_int_type = OptionalType(elem=BaseType(BaseTy.int))
        (out_name, ctype, _, _) = unboxing.argumenttype_evalue_convert(t=optional_int_type, arg_name='something', mutable=True)
        self.assertEqual(out_name, 'something_opt_out')
        self.assertEqual(ctype, types.OptionalCType(BaseCType(types.longT)))

    def test_optional_int_argument_translate_ctype_aten(self) -> None:
        if False:
            while True:
                i = 10
        self._test_optional_int_argument_translate_ctype(ATEN_UNBOXING, aten_types)

    def test_optional_int_argument_translate_ctype_executorch(self) -> None:
        if False:
            i = 10
            return i + 15
        self._test_optional_int_argument_translate_ctype(ET_UNBOXING, et_types)