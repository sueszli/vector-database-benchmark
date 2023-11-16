import os
import pytest
from util.Flag import Flag

class TestFlag:

    def testFlagging(self):
        if False:
            i = 10
            return i + 15
        flag = Flag()

        @flag.admin
        @flag.no_multiuser
        def testFn(anything):
            if False:
                i = 10
                return i + 15
            return anything
        assert 'admin' in flag.db['testFn']
        assert 'no_multiuser' in flag.db['testFn']

    def testSubclassedFlagging(self):
        if False:
            while True:
                i = 10
        flag = Flag()

        class Test:

            @flag.admin
            @flag.no_multiuser
            def testFn(anything):
                if False:
                    for i in range(10):
                        print('nop')
                return anything

        class SubTest(Test):
            pass
        assert 'admin' in flag.db['testFn']
        assert 'no_multiuser' in flag.db['testFn']

    def testInvalidFlag(self):
        if False:
            for i in range(10):
                print('nop')
        flag = Flag()
        with pytest.raises(Exception) as err:

            @flag.no_multiuser
            @flag.unknown_flag
            def testFn(anything):
                if False:
                    for i in range(10):
                        print('nop')
                return anything
        assert 'Invalid flag' in str(err.value)