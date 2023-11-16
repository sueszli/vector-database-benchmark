from sympy.testing.pytest import warns_deprecated_sympy

def test_compatibility_submodule():
    if False:
        for i in range(10):
            print('nop')
    with warns_deprecated_sympy():
        import sympy.core.compatibility