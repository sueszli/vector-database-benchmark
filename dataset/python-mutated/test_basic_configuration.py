from unittest import TestCase as VanillaTestCase
import pytest
from django.db import IntegrityError
from django.test import TestCase as DjangoTestCase
from hypothesis import HealthCheck, Verbosity, given, settings
from hypothesis.errors import InvalidArgument
from hypothesis.extra.django import TestCase, TransactionTestCase
from hypothesis.internal.compat import GRAALPY, PYPY
from hypothesis.strategies import integers
from tests.django.toystore.models import Company

class SomeStuff:

    @settings(suppress_health_check=[HealthCheck.too_slow, HealthCheck.differing_executors])
    @given(integers())
    def test_is_blank_slate(self, unused):
        if False:
            i = 10
            return i + 15
        Company.objects.create(name='MickeyCo')

    def test_normal_test_1(self):
        if False:
            while True:
                i = 10
        Company.objects.create(name='MickeyCo')

    def test_normal_test_2(self):
        if False:
            while True:
                i = 10
        Company.objects.create(name='MickeyCo')

class TestConstraintsWithTransactions(SomeStuff, TestCase):
    pass
if not (PYPY or GRAALPY):

    class TestConstraintsWithoutTransactions(SomeStuff, TransactionTestCase):
        pass

class TestWorkflow(VanillaTestCase):

    def test_does_not_break_later_tests(self):
        if False:
            for i in range(10):
                print('nop')

        def break_the_db(i):
            if False:
                return 10
            Company.objects.create(name='MickeyCo')
            Company.objects.create(name='MickeyCo')

        class LocalTest(TestCase):

            @given(integers().map(break_the_db))
            @settings(suppress_health_check=list(HealthCheck), verbosity=Verbosity.quiet)
            def test_does_not_break_other_things(self, unused):
                if False:
                    while True:
                        i = 10
                pass

            def test_normal_test_1(self):
                if False:
                    print('Hello World!')
                Company.objects.create(name='MickeyCo')
        t = LocalTest('test_normal_test_1')
        try:
            t.test_does_not_break_other_things()
        except IntegrityError:
            pass
        t.test_normal_test_1()

    def test_given_needs_hypothesis_test_case(self):
        if False:
            while True:
                i = 10

        class LocalTest(DjangoTestCase):

            @given(integers())
            def tst(self, i):
                if False:
                    for i in range(10):
                        print('nop')
                raise AssertionError('InvalidArgument should be raised in @given')
        with pytest.raises(InvalidArgument):
            LocalTest('tst').tst()