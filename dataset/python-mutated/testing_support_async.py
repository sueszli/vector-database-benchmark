"""
Unit test support for :mod:`behave.api.async_test` tests.
"""
import inspect

class AsyncStepTheory(object):

    @staticmethod
    def ensure_normal_function(func):
        if False:
            print('Hello World!')
        if hasattr(inspect, 'isawaitable'):
            assert not inspect.isawaitable(func)

    @classmethod
    def validate(cls, func):
        if False:
            while True:
                i = 10
        cls.ensure_normal_function(func)