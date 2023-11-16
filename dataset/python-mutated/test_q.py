from django.core.exceptions import FieldError
from django.db.models import BooleanField, Exists, ExpressionWrapper, F, OuterRef, Q, Value
from django.db.models.expressions import NegatedExpression, RawSQL
from django.db.models.functions import Lower
from django.db.models.sql.where import NothingNode
from django.test import SimpleTestCase, TestCase
from .models import Tag

class QTests(SimpleTestCase):

    def test_combine_and_empty(self):
        if False:
            return 10
        q = Q(x=1)
        self.assertEqual(q & Q(), q)
        self.assertEqual(Q() & q, q)
        q = Q(x__in={}.keys())
        self.assertEqual(q & Q(), q)
        self.assertEqual(Q() & q, q)

    def test_combine_and_both_empty(self):
        if False:
            while True:
                i = 10
        self.assertEqual(Q() & Q(), Q())

    def test_combine_or_empty(self):
        if False:
            i = 10
            return i + 15
        q = Q(x=1)
        self.assertEqual(q | Q(), q)
        self.assertEqual(Q() | q, q)
        q = Q(x__in={}.keys())
        self.assertEqual(q | Q(), q)
        self.assertEqual(Q() | q, q)

    def test_combine_xor_empty(self):
        if False:
            for i in range(10):
                print('nop')
        q = Q(x=1)
        self.assertEqual(q ^ Q(), q)
        self.assertEqual(Q() ^ q, q)
        q = Q(x__in={}.keys())
        self.assertEqual(q ^ Q(), q)
        self.assertEqual(Q() ^ q, q)

    def test_combine_empty_copy(self):
        if False:
            while True:
                i = 10
        base_q = Q(x=1)
        tests = [base_q | Q(), Q() | base_q, base_q & Q(), Q() & base_q, base_q ^ Q(), Q() ^ base_q]
        for (i, q) in enumerate(tests):
            with self.subTest(i=i):
                self.assertEqual(q, base_q)
                self.assertIsNot(q, base_q)

    def test_combine_or_both_empty(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(Q() | Q(), Q())

    def test_combine_xor_both_empty(self):
        if False:
            while True:
                i = 10
        self.assertEqual(Q() ^ Q(), Q())

    def test_combine_not_q_object(self):
        if False:
            for i in range(10):
                print('nop')
        obj = object()
        q = Q(x=1)
        with self.assertRaisesMessage(TypeError, str(obj)):
            q | obj
        with self.assertRaisesMessage(TypeError, str(obj)):
            q & obj
        with self.assertRaisesMessage(TypeError, str(obj)):
            q ^ obj

    def test_combine_negated_boolean_expression(self):
        if False:
            i = 10
            return i + 15
        tagged = Tag.objects.filter(category=OuterRef('pk'))
        tests = [Q() & ~Exists(tagged), Q() | ~Exists(tagged), Q() ^ ~Exists(tagged)]
        for q in tests:
            with self.subTest(q=q):
                self.assertIsInstance(q, NegatedExpression)

    def test_deconstruct(self):
        if False:
            i = 10
            return i + 15
        q = Q(price__gt=F('discounted_price'))
        (path, args, kwargs) = q.deconstruct()
        self.assertEqual(path, 'django.db.models.Q')
        self.assertEqual(args, (('price__gt', F('discounted_price')),))
        self.assertEqual(kwargs, {})

    def test_deconstruct_negated(self):
        if False:
            i = 10
            return i + 15
        q = ~Q(price__gt=F('discounted_price'))
        (path, args, kwargs) = q.deconstruct()
        self.assertEqual(args, (('price__gt', F('discounted_price')),))
        self.assertEqual(kwargs, {'_negated': True})

    def test_deconstruct_or(self):
        if False:
            i = 10
            return i + 15
        q1 = Q(price__gt=F('discounted_price'))
        q2 = Q(price=F('discounted_price'))
        q = q1 | q2
        (path, args, kwargs) = q.deconstruct()
        self.assertEqual(args, (('price__gt', F('discounted_price')), ('price', F('discounted_price'))))
        self.assertEqual(kwargs, {'_connector': Q.OR})

    def test_deconstruct_xor(self):
        if False:
            for i in range(10):
                print('nop')
        q1 = Q(price__gt=F('discounted_price'))
        q2 = Q(price=F('discounted_price'))
        q = q1 ^ q2
        (path, args, kwargs) = q.deconstruct()
        self.assertEqual(args, (('price__gt', F('discounted_price')), ('price', F('discounted_price'))))
        self.assertEqual(kwargs, {'_connector': Q.XOR})

    def test_deconstruct_and(self):
        if False:
            print('Hello World!')
        q1 = Q(price__gt=F('discounted_price'))
        q2 = Q(price=F('discounted_price'))
        q = q1 & q2
        (path, args, kwargs) = q.deconstruct()
        self.assertEqual(args, (('price__gt', F('discounted_price')), ('price', F('discounted_price'))))
        self.assertEqual(kwargs, {})

    def test_deconstruct_multiple_kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        q = Q(price__gt=F('discounted_price'), price=F('discounted_price'))
        (path, args, kwargs) = q.deconstruct()
        self.assertEqual(args, (('price', F('discounted_price')), ('price__gt', F('discounted_price'))))
        self.assertEqual(kwargs, {})

    def test_deconstruct_nested(self):
        if False:
            print('Hello World!')
        q = Q(Q(price__gt=F('discounted_price')))
        (path, args, kwargs) = q.deconstruct()
        self.assertEqual(args, (Q(price__gt=F('discounted_price')),))
        self.assertEqual(kwargs, {})

    def test_deconstruct_boolean_expression(self):
        if False:
            return 10
        expr = RawSQL('1 = 1', BooleanField())
        q = Q(expr)
        (_, args, kwargs) = q.deconstruct()
        self.assertEqual(args, (expr,))
        self.assertEqual(kwargs, {})

    def test_reconstruct(self):
        if False:
            while True:
                i = 10
        q = Q(price__gt=F('discounted_price'))
        (path, args, kwargs) = q.deconstruct()
        self.assertEqual(Q(*args, **kwargs), q)

    def test_reconstruct_negated(self):
        if False:
            for i in range(10):
                print('nop')
        q = ~Q(price__gt=F('discounted_price'))
        (path, args, kwargs) = q.deconstruct()
        self.assertEqual(Q(*args, **kwargs), q)

    def test_reconstruct_or(self):
        if False:
            print('Hello World!')
        q1 = Q(price__gt=F('discounted_price'))
        q2 = Q(price=F('discounted_price'))
        q = q1 | q2
        (path, args, kwargs) = q.deconstruct()
        self.assertEqual(Q(*args, **kwargs), q)

    def test_reconstruct_xor(self):
        if False:
            print('Hello World!')
        q1 = Q(price__gt=F('discounted_price'))
        q2 = Q(price=F('discounted_price'))
        q = q1 ^ q2
        (path, args, kwargs) = q.deconstruct()
        self.assertEqual(Q(*args, **kwargs), q)

    def test_reconstruct_and(self):
        if False:
            i = 10
            return i + 15
        q1 = Q(price__gt=F('discounted_price'))
        q2 = Q(price=F('discounted_price'))
        q = q1 & q2
        (path, args, kwargs) = q.deconstruct()
        self.assertEqual(Q(*args, **kwargs), q)

    def test_equal(self):
        if False:
            return 10
        self.assertEqual(Q(), Q())
        self.assertEqual(Q(('pk__in', (1, 2))), Q(('pk__in', [1, 2])))
        self.assertEqual(Q(('pk__in', (1, 2))), Q(pk__in=[1, 2]))
        self.assertEqual(Q(('pk__in', (1, 2))), Q(('pk__in', {1: 'first', 2: 'second'}.keys())))
        self.assertNotEqual(Q(name__iexact=F('other_name')), Q(name=Lower(F('other_name'))))

    def test_hash(self):
        if False:
            return 10
        self.assertEqual(hash(Q()), hash(Q()))
        self.assertEqual(hash(Q(('pk__in', (1, 2)))), hash(Q(('pk__in', [1, 2]))))
        self.assertEqual(hash(Q(('pk__in', (1, 2)))), hash(Q(pk__in=[1, 2])))
        self.assertEqual(hash(Q(('pk__in', (1, 2)))), hash(Q(('pk__in', {1: 'first', 2: 'second'}.keys()))))
        self.assertNotEqual(hash(Q(name__iexact=F('other_name'))), hash(Q(name=Lower(F('other_name')))))

    def test_flatten(self):
        if False:
            while True:
                i = 10
        q = Q()
        self.assertEqual(list(q.flatten()), [q])
        q = Q(NothingNode())
        self.assertEqual(list(q.flatten()), [q, q.children[0]])
        q = Q(ExpressionWrapper(Q(RawSQL('id = 0', params=(), output_field=BooleanField())) | Q(price=Value('4.55')) | Q(name=Lower('category')), output_field=BooleanField()))
        flatten = list(q.flatten())
        self.assertEqual(len(flatten), 7)

    def test_create_helper(self):
        if False:
            return 10
        items = [('a', 1), ('b', 2), ('c', 3)]
        for connector in [Q.AND, Q.OR, Q.XOR]:
            with self.subTest(connector=connector):
                self.assertEqual(Q.create(items, connector=connector), Q(*items, _connector=connector))

class QCheckTests(TestCase):

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        q = Q(price__gt=20)
        self.assertIs(q.check({'price': 30}), True)
        self.assertIs(q.check({'price': 10}), False)

    def test_expression(self):
        if False:
            for i in range(10):
                print('nop')
        q = Q(name='test')
        self.assertIs(q.check({'name': Lower(Value('TeSt'))}), True)
        self.assertIs(q.check({'name': Value('other')}), False)

    def test_missing_field(self):
        if False:
            while True:
                i = 10
        q = Q(description__startswith='prefix')
        msg = "Cannot resolve keyword 'description' into field."
        with self.assertRaisesMessage(FieldError, msg):
            q.check({'name': 'test'})

    def test_boolean_expression(self):
        if False:
            for i in range(10):
                print('nop')
        q = Q(ExpressionWrapper(Q(price__gt=20), output_field=BooleanField()))
        self.assertIs(q.check({'price': 25}), True)
        self.assertIs(q.check({'price': Value(10)}), False)

    def test_rawsql(self):
        if False:
            while True:
                i = 10
        '\n        RawSQL expressions cause a database error because "price" cannot be\n        replaced by its value. In this case, Q.check() logs a warning and\n        return True.\n        '
        q = Q(RawSQL('price > %s', params=(20,), output_field=BooleanField()))
        with self.assertLogs('django.db.models', 'WARNING') as cm:
            self.assertIs(q.check({'price': 10}), True)
        self.assertIn(f'Got a database error calling check() on {q!r}: ', cm.records[0].getMessage())