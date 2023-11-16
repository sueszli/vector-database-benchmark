from django.db import connection, models
from django.test import SimpleTestCase
from .utils import FuncTestMixin

def test_mutation(raises=True):
    if False:
        for i in range(10):
            print('nop')

    def wrapper(mutation_func):
        if False:
            i = 10
            return i + 15

        def test(test_case_instance, *args, **kwargs):
            if False:
                print('Hello World!')

            class TestFunc(models.Func):
                output_field = models.IntegerField()

                def __init__(self):
                    if False:
                        while True:
                            i = 10
                    self.attribute = 'initial'
                    super().__init__('initial', ['initial'])

                def as_sql(self, *args, **kwargs):
                    if False:
                        for i in range(10):
                            print('nop')
                    mutation_func(self)
                    return ('', ())
            if raises:
                msg = 'TestFunc Func was mutated during compilation.'
                with test_case_instance.assertRaisesMessage(AssertionError, msg):
                    getattr(TestFunc(), 'as_' + connection.vendor)(None, None)
            else:
                getattr(TestFunc(), 'as_' + connection.vendor)(None, None)
        return test
    return wrapper

class FuncTestMixinTests(FuncTestMixin, SimpleTestCase):

    @test_mutation()
    def test_mutated_attribute(func):
        if False:
            while True:
                i = 10
        func.attribute = 'mutated'

    @test_mutation()
    def test_mutated_expressions(func):
        if False:
            return 10
        func.source_expressions.clear()

    @test_mutation()
    def test_mutated_expression(func):
        if False:
            print('Hello World!')
        func.source_expressions[0].name = 'mutated'

    @test_mutation()
    def test_mutated_expression_deep(func):
        if False:
            for i in range(10):
                print('nop')
        func.source_expressions[1].value[0] = 'mutated'

    @test_mutation(raises=False)
    def test_not_mutated(func):
        if False:
            while True:
                i = 10
        pass