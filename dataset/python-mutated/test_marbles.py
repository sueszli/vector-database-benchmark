import unittest
from reactivex.testing.marbles import marbles_testing
from reactivex.testing.reactivetest import ReactiveTest

class TestTestContext(unittest.TestCase):

    def test_start_with_cold_never(self):
        if False:
            return 10
        with marbles_testing() as (start, cold, hot, exp):
            obs = cold('----')
            '           012345678901234567890'

            def create():
                if False:
                    for i in range(10):
                        print('nop')
                return obs
            results = start(create)
            expected = []
            assert results == expected

    def test_start_with_cold_empty(self):
        if False:
            while True:
                i = 10
        with marbles_testing() as (start, cold, hot, exp):
            obs = cold('------|')
            '           012345678901234567890'

            def create():
                if False:
                    for i in range(10):
                        print('nop')
                return obs
            results = start(create)
            expected = [ReactiveTest.on_completed(206)]
            assert results == expected

    def test_start_with_cold_normal(self):
        if False:
            while True:
                i = 10
        with marbles_testing() as (start, cold, hot, exp):
            obs = cold('12--3-|')
            '           012345678901234567890'

            def create():
                if False:
                    i = 10
                    return i + 15
                return obs
            results = start(create)
            expected = [ReactiveTest.on_next(200.0, 12), ReactiveTest.on_next(204.0, 3), ReactiveTest.on_completed(206.0)]
            assert results == expected

    def test_start_with_cold_no_create_function(self):
        if False:
            for i in range(10):
                print('nop')
        with marbles_testing() as (start, cold, hot, exp):
            obs = cold('12--3-|')
            '           012345678901234567890'
            results = start(obs)
            expected = [ReactiveTest.on_next(200.0, 12), ReactiveTest.on_next(204.0, 3), ReactiveTest.on_completed(206.0)]
            assert results == expected

    def test_start_with_hot_never(self):
        if False:
            i = 10
            return i + 15
        with marbles_testing() as (start, cold, hot, exp):
            obs = hot('------')
            '          012345678901234567890'

            def create():
                if False:
                    print('Hello World!')
                return obs
            results = start(create)
            expected = []
            assert results == expected

    def test_start_with_hot_empty(self):
        if False:
            for i in range(10):
                print('nop')
        with marbles_testing() as (start, cold, hot, exp):
            obs = hot('---|')
            '          012345678901234567890'

            def create():
                if False:
                    return 10
                return obs
            results = start(create)
            expected = [ReactiveTest.on_completed(203.0)]
            assert results == expected

    def test_start_with_hot_normal(self):
        if False:
            for i in range(10):
                print('nop')
        with marbles_testing() as (start, cold, hot, exp):
            obs = hot('-12--3-|')
            '          012345678901234567890'

            def create():
                if False:
                    i = 10
                    return i + 15
                return obs
            results = start(create)
            expected = [ReactiveTest.on_next(201.0, 12), ReactiveTest.on_next(205.0, 3), ReactiveTest.on_completed(207.0)]
            assert results == expected

    def test_exp(self):
        if False:
            i = 10
            return i + 15
        with marbles_testing() as (start, cold, hot, exp):
            results = exp('12--3--4--5-|')
            '              012345678901234567890'
            expected = [ReactiveTest.on_next(200.0, 12), ReactiveTest.on_next(204.0, 3), ReactiveTest.on_next(207.0, 4), ReactiveTest.on_next(210.0, 5), ReactiveTest.on_completed(212.0)]
            assert results == expected

    def test_start_with_hot_and_exp(self):
        if False:
            while True:
                i = 10
        with marbles_testing() as (start, cold, hot, exp):
            obs = hot('     --3--4--5-|')
            expected = exp('--3--4--5-|')
            '               012345678901234567890'

            def create():
                if False:
                    return 10
                return obs
            results = start(create)
            assert results == expected

    def test_start_with_cold_and_exp(self):
        if False:
            print('Hello World!')
        with marbles_testing() as (start, cold, hot, exp):
            obs = cold('     12--3--4--5-|')
            expected = exp(' 12--3--4--5-|')
            '                012345678901234567890'

            def create():
                if False:
                    return 10
                return obs
            results = start(create)
            assert results == expected

    def test_start_with_cold_and_exp_group(self):
        if False:
            print('Hello World!')
        with marbles_testing() as (start, cold, hot, exp):
            obs = cold('     12--(3,6.5)----(5,#)')
            expected = exp(' 12--(3,6.5)----(5,#)')
            '                012345678901234567890'

            def create():
                if False:
                    print('Hello World!')
                return obs
            results = start(create)
            assert results == expected