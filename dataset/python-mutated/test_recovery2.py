"""Tests for recovering after errors."""
from pytype.tests import test_base

class RecoveryTests(test_base.BaseTest):
    """Tests for recovering after errors."""

    def test_function_with_unknown_decorator(self):
        if False:
            print('Hello World!')
        self.InferWithErrors('\n      from nowhere import decorator  # import-error\n      @decorator\n      def f():\n        name_error  # name-error\n      @decorator\n      def g(x: int) -> None:\n        x.upper()  # attribute-error\n    ', deep=True)

    def test_complex_init(self):
        if False:
            while True:
                i = 10
        'Test that we recover when __init__ triggers a utils.TooComplexError.'
        (_, errors) = self.InferWithErrors('\n      from typing import AnyStr, Optional\n      class X:\n        def __init__(self,\n                     literal: Optional[int] = None,\n                     target_index: Optional[int] = None,\n                     register_range_first: Optional[int] = None,\n                     register_range_last: Optional[int] = None,\n                     method_ref: Optional[AnyStr] = None,\n                     field_ref: Optional[AnyStr] = None,\n                     string_ref: Optional[AnyStr] = None,\n                     type_ref: Optional[AnyStr] = None) -> None:\n          pass\n        def foo(self, x: other_module.X) -> None:  # name-error[e]\n          pass\n    ', deep=True)
        self.assertErrorRegexes(errors, {'e': 'other_module'})

class RecoveryTestsPython3(test_base.BaseTest):
    """Tests for recovering after errors(python3 only)."""

    def test_bad_call_parameter(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer('\n          def f():\n            return "%s" % chr("foo")\n        ', report_errors=False)
        self.assertTypesMatchPytd(ty, '\n          def f() -> str: ...\n        ')

    def test_bad_function(self):
        if False:
            i = 10
            return i + 15
        ty = self.Infer("\n        import time\n        def f():\n          return time.unknown_function(3)\n        def g():\n          return '%s' % f()\n      ", report_errors=False)
        self.assertTypesMatchPytd(ty, '\n        import time\n        from typing import Any\n        def f() -> Any: ...\n        def g() -> str: ...\n      ')
if __name__ == '__main__':
    test_base.main()