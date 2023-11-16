from numba import njit
from numba.tests.gdb_support import GdbMIDriver
from numba.tests.support import TestCase, needs_subprocess
import unittest

def foo_factory(n):
    if False:
        while True:
            i = 10

    @njit(debug=True)
    def foo(x):
        if False:
            print('Hello World!')
        z = 7 + n
        return (x, z)
    return foo
(foo1, foo2, foo3) = [foo_factory(x) for x in range(3)]

@njit(debug=True)
def call_foo():
    if False:
        i = 10
        return i + 15
    a = foo1(10)
    b = foo2(20)
    c = foo3(30)
    return (a, b, c)

@needs_subprocess
class Test(TestCase):

    def test(self):
        if False:
            i = 10
            return i + 15
        call_foo()
        driver = GdbMIDriver(__file__)
        vsym = '__main__::foo_factory::_3clocals_3e::foo[abi:v2]'
        driver.set_breakpoint(symbol=vsym)
        driver.run()
        driver.check_hit_breakpoint(number=1)
        driver.assert_regex_output('^.*foo\\[abi:v2\\].*line="11"')
        driver.stack_list_arguments(2)
        expect = '[frame={level="0",args=[{name="x",type="Literal[int](10)",value="10"}]}]'
        driver.assert_output(expect)
        driver.set_breakpoint(symbol='foo')
        driver.cont()
        driver.check_hit_breakpoint(number=2)
        driver.assert_regex_output('^.*foo\\[abi:v3\\].*line="11"')
        driver.stack_list_arguments(2)
        expect = '[frame={level="0",args=[{name="x",type="Literal[int](20)",value="20"}]}]'
        driver.assert_output(expect)
        driver.cont()
        driver.check_hit_breakpoint(number=2)
        driver.assert_regex_output('^.*foo\\[abi:v4\\].*line="11"')
        driver.stack_list_arguments(2)
        expect = '[frame={level="0",args=[{name="x",type="Literal[int](30)",value="30"}]}]'
        driver.assert_output(expect)
        driver.quit()
if __name__ == '__main__':
    unittest.main()