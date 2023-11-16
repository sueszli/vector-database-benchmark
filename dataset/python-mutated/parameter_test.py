import datetime
from helpers import with_config, LuigiTestCase, parsing, in_parse, RunOnceTask
from datetime import timedelta
import enum
import mock
import pytest
import luigi
import luigi.date_interval
import luigi.interface
import luigi.notifications
from luigi.mock import MockTarget
from luigi.parameter import ParameterException
from worker_test import email_patch
luigi.notifications.DEBUG = True

class A(luigi.Task):
    p = luigi.IntParameter()

class WithDefault(luigi.Task):
    x = luigi.Parameter(default='xyz')

class WithDefaultTrue(luigi.Task):
    x = luigi.BoolParameter(default=True)

class WithDefaultFalse(luigi.Task):
    x = luigi.BoolParameter(default=False)

class Foo(luigi.Task):
    bar = luigi.Parameter()
    p2 = luigi.IntParameter()
    not_a_param = 'lol'

class Baz(luigi.Task):
    bool = luigi.BoolParameter()
    bool_true = luigi.BoolParameter(default=True)
    bool_explicit = luigi.BoolParameter(parsing=luigi.BoolParameter.EXPLICIT_PARSING)

    def run(self):
        if False:
            print('Hello World!')
        Baz._val = self.bool
        Baz._val_true = self.bool_true
        Baz._val_explicit = self.bool_explicit

class ListFoo(luigi.Task):
    my_list = luigi.ListParameter()

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        ListFoo._val = self.my_list

class TupleFoo(luigi.Task):
    my_tuple = luigi.TupleParameter()

    def run(self):
        if False:
            while True:
                i = 10
        TupleFoo._val = self.my_tuple

class ForgotParam(luigi.Task):
    param = luigi.Parameter()

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class ForgotParamDep(luigi.Task):

    def requires(self):
        if False:
            while True:
                i = 10
        return ForgotParam()

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class BananaDep(luigi.Task):
    x = luigi.Parameter()
    y = luigi.Parameter(default='def')

    def output(self):
        if False:
            i = 10
            return i + 15
        return MockTarget('banana-dep-%s-%s' % (self.x, self.y))

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        self.output().open('w').close()

class Banana(luigi.Task):
    x = luigi.Parameter()
    y = luigi.Parameter()
    style = luigi.Parameter(default=None)

    def requires(self):
        if False:
            print('Hello World!')
        if self.style is None:
            return BananaDep()
        elif self.style == 'x-arg':
            return BananaDep(self.x)
        elif self.style == 'y-kwarg':
            return BananaDep(y=self.y)
        elif self.style == 'x-arg-y-arg':
            return BananaDep(self.x, self.y)
        else:
            raise Exception('unknown style')

    def output(self):
        if False:
            return 10
        return MockTarget('banana-%s-%s' % (self.x, self.y))

    def run(self):
        if False:
            i = 10
            return i + 15
        self.output().open('w').close()

class MyConfig(luigi.Config):
    mc_p = luigi.IntParameter()
    mc_q = luigi.IntParameter(default=73)

class MyConfigWithoutSection(luigi.Config):
    use_cmdline_section = False
    mc_r = luigi.IntParameter()
    mc_s = luigi.IntParameter(default=99)

class NoopTask(luigi.Task):
    pass

class MyEnum(enum.Enum):
    A = 1
    C = 3

def _value(parameter):
    if False:
        for i in range(10):
            print('nop')
    '\n    A hackish way to get the "value" of a parameter.\n\n    Previously Parameter exposed ``param_obj._value``. This is replacement for\n    that so I don\'t need to rewrite all test cases.\n    '

    class DummyLuigiTask(luigi.Task):
        param = parameter
    return DummyLuigiTask().param

class ParameterTest(LuigiTestCase):

    def test_default_param(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(WithDefault().x, 'xyz')

    def test_missing_param(self):
        if False:
            print('Hello World!')

        def create_a():
            if False:
                for i in range(10):
                    print('nop')
            return A()
        self.assertRaises(luigi.parameter.MissingParameterException, create_a)

    def test_unknown_param(self):
        if False:
            i = 10
            return i + 15

        def create_a():
            if False:
                while True:
                    i = 10
            return A(p=5, q=4)
        self.assertRaises(luigi.parameter.UnknownParameterException, create_a)

    def test_unknown_param_2(self):
        if False:
            print('Hello World!')

        def create_a():
            if False:
                i = 10
                return i + 15
            return A(1, 2, 3)
        self.assertRaises(luigi.parameter.UnknownParameterException, create_a)

    def test_duplicated_param(self):
        if False:
            for i in range(10):
                print('nop')

        def create_a():
            if False:
                return 10
            return A(5, p=7)
        self.assertRaises(luigi.parameter.DuplicateParameterException, create_a)

    def test_parameter_registration(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(len(Foo.get_params()), 2)

    def test_task_creation(self):
        if False:
            return 10
        f = Foo('barval', p2=5)
        self.assertEqual(len(f.get_params()), 2)
        self.assertEqual(f.bar, 'barval')
        self.assertEqual(f.p2, 5)
        self.assertEqual(f.not_a_param, 'lol')

    def test_bool_parsing(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_locally(['Baz'])
        self.assertFalse(Baz._val)
        self.assertTrue(Baz._val_true)
        self.assertFalse(Baz._val_explicit)
        self.run_locally(['Baz', '--bool', '--bool-true'])
        self.assertTrue(Baz._val)
        self.assertTrue(Baz._val_true)
        self.run_locally(['Baz', '--bool-explicit', 'true'])
        self.assertTrue(Baz._val_explicit)
        self.run_locally(['Baz', '--bool-explicit', 'false'])
        self.assertFalse(Baz._val_explicit)

    def test_bool_default(self):
        if False:
            while True:
                i = 10
        self.assertTrue(WithDefaultTrue().x)
        self.assertFalse(WithDefaultFalse().x)

    def test_bool_coerce(self):
        if False:
            while True:
                i = 10
        self.assertTrue(WithDefaultTrue(x='true').x)
        self.assertFalse(WithDefaultTrue(x='false').x)

    def test_bool_no_coerce_none(self):
        if False:
            while True:
                i = 10
        self.assertIsNone(WithDefaultTrue(x=None).x)

    def test_forgot_param(self):
        if False:
            print('Hello World!')
        self.assertRaises(luigi.parameter.MissingParameterException, self.run_locally, ['ForgotParam'])

    @email_patch
    def test_forgot_param_in_dep(self, emails):
        if False:
            return 10
        self.run_locally(['ForgotParamDep'])
        self.assertNotEqual(emails, [])

    def test_default_param_cmdline(self):
        if False:
            return 10
        self.assertEqual(WithDefault().x, 'xyz')

    def test_default_param_cmdline_2(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(WithDefault().x, 'xyz')

    def test_insignificant_parameter(self):
        if False:
            print('Hello World!')

        class InsignificantParameterTask(luigi.Task):
            foo = luigi.Parameter(significant=False, default='foo_default')
            bar = luigi.Parameter()
        t1 = InsignificantParameterTask(foo='x', bar='y')
        self.assertEqual(str(t1), 'InsignificantParameterTask(bar=y)')
        t2 = InsignificantParameterTask('u', 'z')
        self.assertEqual(t2.foo, 'u')
        self.assertEqual(t2.bar, 'z')
        self.assertEqual(str(t2), 'InsignificantParameterTask(bar=z)')

    def test_local_significant_param(self):
        if False:
            for i in range(10):
                print('nop')
        ' Obviously, if anything should be positional, so should local\n        significant parameters '

        class MyTask(luigi.Task):
            x = luigi.Parameter(significant=True)
        MyTask('arg')
        self.assertRaises(luigi.parameter.MissingParameterException, lambda : MyTask())

    def test_local_insignificant_param(self):
        if False:
            i = 10
            return i + 15
        ' Ensure we have the same behavior as in before a78338c  '

        class MyTask(luigi.Task):
            x = luigi.Parameter(significant=False)
        MyTask('arg')
        self.assertRaises(luigi.parameter.MissingParameterException, lambda : MyTask())

    def test_nonpositional_param(self):
        if False:
            return 10
        ' Ensure we have the same behavior as in before a78338c  '

        class MyTask(luigi.Task):
            x = luigi.Parameter(significant=False, positional=False)
        MyTask(x='arg')
        self.assertRaises(luigi.parameter.UnknownParameterException, lambda : MyTask('arg'))

    def test_enum_param_valid(self):
        if False:
            print('Hello World!')
        p = luigi.parameter.EnumParameter(enum=MyEnum)
        self.assertEqual(MyEnum.A, p.parse('A'))

    def test_enum_param_invalid(self):
        if False:
            print('Hello World!')
        p = luigi.parameter.EnumParameter(enum=MyEnum)
        self.assertRaises(ValueError, lambda : p.parse('B'))

    def test_enum_param_missing(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(ParameterException, lambda : luigi.parameter.EnumParameter())

    def test_enum_list_param_valid(self):
        if False:
            i = 10
            return i + 15
        p = luigi.parameter.EnumListParameter(enum=MyEnum)
        self.assertEqual((), p.parse(''))
        self.assertEqual((MyEnum.A,), p.parse('A'))
        self.assertEqual((MyEnum.A, MyEnum.C), p.parse('A,C'))

    def test_enum_list_param_invalid(self):
        if False:
            while True:
                i = 10
        p = luigi.parameter.EnumListParameter(enum=MyEnum)
        self.assertRaises(ValueError, lambda : p.parse('A,B'))

    def test_enum_list_param_missing(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(ParameterException, lambda : luigi.parameter.EnumListParameter())

    def test_tuple_serialize_parse(self):
        if False:
            i = 10
            return i + 15
        a = luigi.TupleParameter()
        b_tuple = ((1, 2), (3, 4))
        self.assertEqual(b_tuple, a.parse(a.serialize(b_tuple)))

    def test_parse_list_without_batch_method(self):
        if False:
            i = 10
            return i + 15
        param = luigi.Parameter()
        for xs in ([], ['x'], ['x', 'y']):
            self.assertRaises(NotImplementedError, param._parse_list, xs)

    def test_parse_empty_list_raises_value_error(self):
        if False:
            for i in range(10):
                print('nop')
        for batch_method in (max, min, tuple, ','.join):
            param = luigi.Parameter(batch_method=batch_method)
            self.assertRaises(ValueError, param._parse_list, [])

    def test_parse_int_list_max(self):
        if False:
            return 10
        param = luigi.IntParameter(batch_method=max)
        self.assertEqual(17, param._parse_list(['7', '17', '5']))

    def test_parse_string_list_max(self):
        if False:
            for i in range(10):
                print('nop')
        param = luigi.Parameter(batch_method=max)
        self.assertEqual('7', param._parse_list(['7', '17', '5']))

    def test_parse_list_as_tuple(self):
        if False:
            while True:
                i = 10
        param = luigi.IntParameter(batch_method=tuple)
        self.assertEqual((7, 17, 5), param._parse_list(['7', '17', '5']))

    @mock.patch('luigi.parameter.warnings')
    def test_warn_on_default_none(self, warnings):
        if False:
            print('Hello World!')

        class TestConfig(luigi.Config):
            param = luigi.Parameter(default=None)
        TestConfig()
        warnings.warn.assert_called_once_with('Parameter "param" with value "None" is not of type string.')

    @mock.patch('luigi.parameter.warnings')
    def test_no_warn_on_string(self, warnings):
        if False:
            print('Hello World!')

        class TestConfig(luigi.Config):
            param = luigi.Parameter(default=None)
        TestConfig(param='str')
        warnings.warn.assert_not_called()

    def test_no_warn_on_none_in_optional(self):
        if False:
            while True:
                i = 10

        class TestConfig(luigi.Config):
            param = luigi.OptionalParameter(default=None)
        with mock.patch('luigi.parameter.warnings') as warnings:
            TestConfig()
            warnings.warn.assert_not_called()
        with mock.patch('luigi.parameter.warnings') as warnings:
            TestConfig(param=None)
            warnings.warn.assert_not_called()
        with mock.patch('luigi.parameter.warnings') as warnings:
            TestConfig(param='')
            warnings.warn.assert_not_called()

    @mock.patch('luigi.parameter.warnings')
    def test_no_warn_on_string_in_optional(self, warnings):
        if False:
            while True:
                i = 10

        class TestConfig(luigi.Config):
            param = luigi.OptionalParameter(default=None)
        TestConfig(param='value')
        warnings.warn.assert_not_called()

    @mock.patch('luigi.parameter.warnings')
    def test_warn_on_bad_type_in_optional(self, warnings):
        if False:
            return 10

        class TestConfig(luigi.Config):
            param = luigi.OptionalParameter()
        TestConfig(param=1)
        warnings.warn.assert_called_once_with('OptionalParameter "param" with value "1" is not of type "str" or None.', luigi.parameter.OptionalParameterTypeWarning)

    def test_optional_parameter_parse_none(self):
        if False:
            print('Hello World!')
        self.assertIsNone(luigi.OptionalParameter().parse(''))

    def test_optional_parameter_parse_string(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual('test', luigi.OptionalParameter().parse('test'))

    def test_optional_parameter_serialize_none(self):
        if False:
            return 10
        self.assertEqual('', luigi.OptionalParameter().serialize(None))

    def test_optional_parameter_serialize_string(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual('test', luigi.OptionalParameter().serialize('test'))

class TestParametersHashability(LuigiTestCase):

    def test_date(self):
        if False:
            for i in range(10):
                print('nop')

        class Foo(luigi.Task):
            args = luigi.parameter.DateParameter()
        p = luigi.parameter.DateParameter()
        self.assertEqual(hash(Foo(args=datetime.date(2000, 1, 1)).args), hash(p.parse('2000-1-1')))

    def test_dateminute(self):
        if False:
            for i in range(10):
                print('nop')

        class Foo(luigi.Task):
            args = luigi.parameter.DateMinuteParameter()
        p = luigi.parameter.DateMinuteParameter()
        self.assertEqual(hash(Foo(args=datetime.datetime(2000, 1, 1, 12, 0)).args), hash(p.parse('2000-1-1T1200')))

    def test_dateinterval(self):
        if False:
            i = 10
            return i + 15

        class Foo(luigi.Task):
            args = luigi.parameter.DateIntervalParameter()
        p = luigi.parameter.DateIntervalParameter()
        di = luigi.date_interval.Custom(datetime.date(2000, 1, 1), datetime.date(2000, 2, 12))
        self.assertEqual(hash(Foo(args=di).args), hash(p.parse('2000-01-01-2000-02-12')))

    def test_timedelta(self):
        if False:
            while True:
                i = 10

        class Foo(luigi.Task):
            args = luigi.parameter.TimeDeltaParameter()
        p = luigi.parameter.TimeDeltaParameter()
        self.assertEqual(hash(Foo(args=datetime.timedelta(days=2, hours=3, minutes=2)).args), hash(p.parse('P2DT3H2M')))

    def test_boolean(self):
        if False:
            while True:
                i = 10

        class Foo(luigi.Task):
            args = luigi.parameter.BoolParameter()
        p = luigi.parameter.BoolParameter()
        self.assertEqual(hash(Foo(args=True).args), hash(p.parse('true')))
        self.assertEqual(hash(Foo(args=False).args), hash(p.parse('false')))

    def test_int(self):
        if False:
            return 10

        class Foo(luigi.Task):
            args = luigi.parameter.IntParameter()
        p = luigi.parameter.IntParameter()
        self.assertEqual(hash(Foo(args=1).args), hash(p.parse('1')))

    def test_float(self):
        if False:
            for i in range(10):
                print('nop')

        class Foo(luigi.Task):
            args = luigi.parameter.FloatParameter()
        p = luigi.parameter.FloatParameter()
        self.assertEqual(hash(Foo(args=1.0).args), hash(p.parse('1')))

    def test_enum(self):
        if False:
            for i in range(10):
                print('nop')

        class Foo(luigi.Task):
            args = luigi.parameter.EnumParameter(enum=MyEnum)
        p = luigi.parameter.EnumParameter(enum=MyEnum)
        self.assertEqual(hash(Foo(args=MyEnum.A).args), hash(p.parse('A')))

    def test_enum_list(self):
        if False:
            return 10

        class Foo(luigi.Task):
            args = luigi.parameter.EnumListParameter(enum=MyEnum)
        p = luigi.parameter.EnumListParameter(enum=MyEnum)
        self.assertEqual(hash(Foo(args=(MyEnum.A, MyEnum.C)).args), hash(p.parse('A,C')))

        class FooWithDefault(luigi.Task):
            args = luigi.parameter.EnumListParameter(enum=MyEnum, default=[MyEnum.C])
        self.assertEqual(FooWithDefault().args, p.parse('C'))

    def test_dict(self):
        if False:
            i = 10
            return i + 15

        class Foo(luigi.Task):
            args = luigi.parameter.DictParameter()
        p = luigi.parameter.DictParameter()
        self.assertEqual(hash(Foo(args=dict(foo=1, bar='hello')).args), hash(p.parse('{"foo":1,"bar":"hello"}')))

    def test_list(self):
        if False:
            while True:
                i = 10

        class Foo(luigi.Task):
            args = luigi.parameter.ListParameter()
        p = luigi.parameter.ListParameter()
        self.assertEqual(hash(Foo(args=[1, 'hello']).args), hash(p.normalize(p.parse('[1,"hello"]'))))

    def test_list_param_with_default_none_in_dynamic_req_task(self):
        if False:
            return 10

        class TaskWithDefaultNoneParameter(RunOnceTask):
            args = luigi.parameter.ListParameter(default=None)

        class DynamicTaskCallsDefaultNoneParameter(RunOnceTask):

            def run(self):
                if False:
                    print('Hello World!')
                yield [TaskWithDefaultNoneParameter()]
                self.comp = True
        self.assertTrue(self.run_locally(['DynamicTaskCallsDefaultNoneParameter']))

    def test_list_dict(self):
        if False:
            while True:
                i = 10

        class Foo(luigi.Task):
            args = luigi.parameter.ListParameter()
        p = luigi.parameter.ListParameter()
        self.assertEqual(hash(Foo(args=[{'foo': 'bar'}, {'doge': 'wow'}]).args), hash(p.normalize(p.parse('[{"foo": "bar"}, {"doge": "wow"}]'))))

    def test_list_nested(self):
        if False:
            print('Hello World!')

        class Foo(luigi.Task):
            args = luigi.parameter.ListParameter()
        p = luigi.parameter.ListParameter()
        self.assertEqual(hash(Foo(args=[['foo', 'bar'], ['doge', 'wow']]).args), hash(p.normalize(p.parse('[["foo", "bar"], ["doge", "wow"]]'))))

    def test_tuple(self):
        if False:
            print('Hello World!')

        class Foo(luigi.Task):
            args = luigi.parameter.TupleParameter()
        p = luigi.parameter.TupleParameter()
        self.assertEqual(hash(Foo(args=(1, 'hello')).args), hash(p.parse('(1,"hello")')))

    def test_tuple_dict(self):
        if False:
            return 10

        class Foo(luigi.Task):
            args = luigi.parameter.TupleParameter()
        p = luigi.parameter.TupleParameter()
        self.assertEqual(hash(Foo(args=({'foo': 'bar'}, {'doge': 'wow'})).args), hash(p.normalize(p.parse('({"foo": "bar"}, {"doge": "wow"})'))))

    def test_tuple_nested(self):
        if False:
            i = 10
            return i + 15

        class Foo(luigi.Task):
            args = luigi.parameter.TupleParameter()
        p = luigi.parameter.TupleParameter()
        self.assertEqual(hash(Foo(args=(('foo', 'bar'), ('doge', 'wow'))).args), hash(p.normalize(p.parse('(("foo", "bar"), ("doge", "wow"))'))))

    def test_task(self):
        if False:
            print('Hello World!')

        class Bar(luigi.Task):
            pass

        class Foo(luigi.Task):
            args = luigi.parameter.TaskParameter()
        p = luigi.parameter.TaskParameter()
        self.assertEqual(hash(Foo(args=Bar).args), hash(p.parse('Bar')))

class TestNewStyleGlobalParameters(LuigiTestCase):

    def setUp(self):
        if False:
            return 10
        super(TestNewStyleGlobalParameters, self).setUp()
        MockTarget.fs.clear()

    def expect_keys(self, expected):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(set(MockTarget.fs.get_all_data().keys()), set(expected))

    def test_x_arg(self):
        if False:
            print('Hello World!')
        self.run_locally(['Banana', '--x', 'foo', '--y', 'bar', '--style', 'x-arg'])
        self.expect_keys(['banana-foo-bar', 'banana-dep-foo-def'])

    def test_x_arg_override(self):
        if False:
            return 10
        self.run_locally(['Banana', '--x', 'foo', '--y', 'bar', '--style', 'x-arg', '--BananaDep-y', 'xyz'])
        self.expect_keys(['banana-foo-bar', 'banana-dep-foo-xyz'])

    def test_x_arg_override_stupid(self):
        if False:
            return 10
        self.run_locally(['Banana', '--x', 'foo', '--y', 'bar', '--style', 'x-arg', '--BananaDep-x', 'blabla'])
        self.expect_keys(['banana-foo-bar', 'banana-dep-foo-def'])

    def test_x_arg_y_arg(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_locally(['Banana', '--x', 'foo', '--y', 'bar', '--style', 'x-arg-y-arg'])
        self.expect_keys(['banana-foo-bar', 'banana-dep-foo-bar'])

    def test_x_arg_y_arg_override(self):
        if False:
            i = 10
            return i + 15
        self.run_locally(['Banana', '--x', 'foo', '--y', 'bar', '--style', 'x-arg-y-arg', '--BananaDep-y', 'xyz'])
        self.expect_keys(['banana-foo-bar', 'banana-dep-foo-bar'])

    def test_x_arg_y_arg_override_all(self):
        if False:
            while True:
                i = 10
        self.run_locally(['Banana', '--x', 'foo', '--y', 'bar', '--style', 'x-arg-y-arg', '--BananaDep-y', 'xyz', '--BananaDep-x', 'blabla'])
        self.expect_keys(['banana-foo-bar', 'banana-dep-foo-bar'])

    def test_y_arg_override(self):
        if False:
            while True:
                i = 10
        self.run_locally(['Banana', '--x', 'foo', '--y', 'bar', '--style', 'y-kwarg', '--BananaDep-x', 'xyz'])
        self.expect_keys(['banana-foo-bar', 'banana-dep-xyz-bar'])

    def test_y_arg_override_both(self):
        if False:
            return 10
        self.run_locally(['Banana', '--x', 'foo', '--y', 'bar', '--style', 'y-kwarg', '--BananaDep-x', 'xyz', '--BananaDep-y', 'blah'])
        self.expect_keys(['banana-foo-bar', 'banana-dep-xyz-bar'])

    def test_y_arg_override_banana(self):
        if False:
            while True:
                i = 10
        self.run_locally(['Banana', '--y', 'bar', '--style', 'y-kwarg', '--BananaDep-x', 'xyz', '--Banana-x', 'baz'])
        self.expect_keys(['banana-baz-bar', 'banana-dep-xyz-bar'])

class TestRemoveGlobalParameters(LuigiTestCase):

    def run_and_check(self, args):
        if False:
            for i in range(10):
                print('nop')
        run_exit_status = self.run_locally(args)
        self.assertTrue(run_exit_status)
        return run_exit_status

    @parsing(['--MyConfig-mc-p', '99', '--mc-r', '55', 'NoopTask'])
    def test_use_config_class_1(self):
        if False:
            print('Hello World!')
        self.assertEqual(MyConfig().mc_p, 99)
        self.assertEqual(MyConfig().mc_q, 73)
        self.assertEqual(MyConfigWithoutSection().mc_r, 55)
        self.assertEqual(MyConfigWithoutSection().mc_s, 99)

    @parsing(['NoopTask', '--MyConfig-mc-p', '99', '--mc-r', '55'])
    def test_use_config_class_2(self):
        if False:
            return 10
        self.assertEqual(MyConfig().mc_p, 99)
        self.assertEqual(MyConfig().mc_q, 73)
        self.assertEqual(MyConfigWithoutSection().mc_r, 55)
        self.assertEqual(MyConfigWithoutSection().mc_s, 99)

    @parsing(['--MyConfig-mc-p', '99', '--mc-r', '55', 'NoopTask', '--mc-s', '123', '--MyConfig-mc-q', '42'])
    def test_use_config_class_more_args(self):
        if False:
            print('Hello World!')
        self.assertEqual(MyConfig().mc_p, 99)
        self.assertEqual(MyConfig().mc_q, 42)
        self.assertEqual(MyConfigWithoutSection().mc_r, 55)
        self.assertEqual(MyConfigWithoutSection().mc_s, 123)

    @with_config({'MyConfig': {'mc_p': '666', 'mc_q': '777'}})
    @parsing(['--mc-r', '555', 'NoopTask'])
    def test_use_config_class_with_configuration(self):
        if False:
            print('Hello World!')
        self.assertEqual(MyConfig().mc_p, 666)
        self.assertEqual(MyConfig().mc_q, 777)
        self.assertEqual(MyConfigWithoutSection().mc_r, 555)
        self.assertEqual(MyConfigWithoutSection().mc_s, 99)

    @with_config({'MyConfigWithoutSection': {'mc_r': '999', 'mc_s': '888'}})
    @parsing(['NoopTask', '--MyConfig-mc-p', '222', '--mc-r', '555'])
    def test_use_config_class_with_configuration_2(self):
        if False:
            print('Hello World!')
        self.assertEqual(MyConfig().mc_p, 222)
        self.assertEqual(MyConfig().mc_q, 73)
        self.assertEqual(MyConfigWithoutSection().mc_r, 555)
        self.assertEqual(MyConfigWithoutSection().mc_s, 888)

    @with_config({'MyConfig': {'mc_p': '555', 'mc-p': '666', 'mc-q': '777'}})
    def test_configuration_style(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(MyConfig().mc_p, 555)
        self.assertEqual(MyConfig().mc_q, 777)

    def test_misc_1(self):
        if False:
            while True:
                i = 10

        class Dogs(luigi.Config):
            n_dogs = luigi.IntParameter()

        class CatsWithoutSection(luigi.Config):
            use_cmdline_section = False
            n_cats = luigi.IntParameter()
        with luigi.cmdline_parser.CmdlineParser.global_instance(['--n-cats', '123', '--Dogs-n-dogs', '456', 'WithDefault'], allow_override=True):
            self.assertEqual(Dogs().n_dogs, 456)
            self.assertEqual(CatsWithoutSection().n_cats, 123)
        with luigi.cmdline_parser.CmdlineParser.global_instance(['WithDefault', '--n-cats', '321', '--Dogs-n-dogs', '654'], allow_override=True):
            self.assertEqual(Dogs().n_dogs, 654)
            self.assertEqual(CatsWithoutSection().n_cats, 321)

    def test_global_significant_param_warning(self):
        if False:
            return 10
        " We don't want any kind of global param to be positional "
        with self.assertWarnsRegex(DeprecationWarning, 'is_global support is removed. Assuming positional=False'):

            class MyTask(luigi.Task):
                x_g1 = luigi.Parameter(default='y', is_global=True, significant=True)
        self.assertRaises(luigi.parameter.UnknownParameterException, lambda : MyTask('arg'))

        def test_global_insignificant_param_warning(self):
            if False:
                for i in range(10):
                    print('nop')
            " We don't want any kind of global param to be positional "
            with self.assertWarnsRegex(DeprecationWarning, 'is_global support is removed. Assuming positional=False'):

                class MyTask(luigi.Task):
                    x_g2 = luigi.Parameter(default='y', is_global=True, significant=False)
            self.assertRaises(luigi.parameter.UnknownParameterException, lambda : MyTask('arg'))

class TestParamWithDefaultFromConfig(LuigiTestCase):

    def testNoSection(self):
        if False:
            while True:
                i = 10
        self.assertRaises(ParameterException, lambda : _value(luigi.Parameter(config_path=dict(section='foo', name='bar'))))

    @with_config({'foo': {}})
    def testNoValue(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(ParameterException, lambda : _value(luigi.Parameter(config_path=dict(section='foo', name='bar'))))

    @with_config({'foo': {'bar': 'baz'}})
    def testDefault(self):
        if False:
            print('Hello World!')

        class LocalA(luigi.Task):
            p = luigi.Parameter(config_path=dict(section='foo', name='bar'))
        self.assertEqual('baz', LocalA().p)
        self.assertEqual('boo', LocalA(p='boo').p)

    @with_config({'foo': {'bar': '2001-02-03T04'}})
    def testDateHour(self):
        if False:
            return 10
        p = luigi.DateHourParameter(config_path=dict(section='foo', name='bar'))
        self.assertEqual(datetime.datetime(2001, 2, 3, 4, 0, 0), _value(p))

    @with_config({'foo': {'bar': '2001-02-03T05'}})
    def testDateHourWithInterval(self):
        if False:
            i = 10
            return i + 15
        p = luigi.DateHourParameter(config_path=dict(section='foo', name='bar'), interval=2)
        self.assertEqual(datetime.datetime(2001, 2, 3, 4, 0, 0), _value(p))

    @with_config({'foo': {'bar': '2001-02-03T0430'}})
    def testDateMinute(self):
        if False:
            for i in range(10):
                print('nop')
        p = luigi.DateMinuteParameter(config_path=dict(section='foo', name='bar'))
        self.assertEqual(datetime.datetime(2001, 2, 3, 4, 30, 0), _value(p))

    @with_config({'foo': {'bar': '2001-02-03T0431'}})
    def testDateWithMinuteInterval(self):
        if False:
            i = 10
            return i + 15
        p = luigi.DateMinuteParameter(config_path=dict(section='foo', name='bar'), interval=2)
        self.assertEqual(datetime.datetime(2001, 2, 3, 4, 30, 0), _value(p))

    @with_config({'foo': {'bar': '2001-02-03T04H30'}})
    def testDateMinuteDeprecated(self):
        if False:
            for i in range(10):
                print('nop')
        p = luigi.DateMinuteParameter(config_path=dict(section='foo', name='bar'))
        with self.assertWarnsRegex(DeprecationWarning, 'Using "H" between hours and minutes is deprecated, omit it instead.'):
            self.assertEqual(datetime.datetime(2001, 2, 3, 4, 30, 0), _value(p))

    @with_config({'foo': {'bar': '2001-02-03T040506'}})
    def testDateSecond(self):
        if False:
            i = 10
            return i + 15
        p = luigi.DateSecondParameter(config_path=dict(section='foo', name='bar'))
        self.assertEqual(datetime.datetime(2001, 2, 3, 4, 5, 6), _value(p))

    @with_config({'foo': {'bar': '2001-02-03T040507'}})
    def testDateSecondWithInterval(self):
        if False:
            return 10
        p = luigi.DateSecondParameter(config_path=dict(section='foo', name='bar'), interval=2)
        self.assertEqual(datetime.datetime(2001, 2, 3, 4, 5, 6), _value(p))

    @with_config({'foo': {'bar': '2001-02-03'}})
    def testDate(self):
        if False:
            return 10
        p = luigi.DateParameter(config_path=dict(section='foo', name='bar'))
        self.assertEqual(datetime.date(2001, 2, 3), _value(p))

    @with_config({'foo': {'bar': '2001-02-03'}})
    def testDateWithInterval(self):
        if False:
            while True:
                i = 10
        p = luigi.DateParameter(config_path=dict(section='foo', name='bar'), interval=3, start=datetime.date(2001, 2, 1))
        self.assertEqual(datetime.date(2001, 2, 1), _value(p))

    @with_config({'foo': {'bar': '2015-07'}})
    def testMonthParameter(self):
        if False:
            return 10
        p = luigi.MonthParameter(config_path=dict(section='foo', name='bar'))
        self.assertEqual(datetime.date(2015, 7, 1), _value(p))

    @with_config({'foo': {'bar': '2015-07'}})
    def testMonthWithIntervalParameter(self):
        if False:
            i = 10
            return i + 15
        p = luigi.MonthParameter(config_path=dict(section='foo', name='bar'), interval=13, start=datetime.date(2014, 1, 1))
        self.assertEqual(datetime.date(2015, 2, 1), _value(p))

    @with_config({'foo': {'bar': '2015'}})
    def testYearParameter(self):
        if False:
            return 10
        p = luigi.YearParameter(config_path=dict(section='foo', name='bar'))
        self.assertEqual(datetime.date(2015, 1, 1), _value(p))

    @with_config({'foo': {'bar': '2015'}})
    def testYearWithIntervalParameter(self):
        if False:
            print('Hello World!')
        p = luigi.YearParameter(config_path=dict(section='foo', name='bar'), start=datetime.date(2011, 1, 1), interval=5)
        self.assertEqual(datetime.date(2011, 1, 1), _value(p))

    @with_config({'foo': {'bar': '123'}})
    def testInt(self):
        if False:
            while True:
                i = 10
        p = luigi.IntParameter(config_path=dict(section='foo', name='bar'))
        self.assertEqual(123, _value(p))

    @with_config({'foo': {'bar': 'true'}})
    def testBool(self):
        if False:
            i = 10
            return i + 15
        p = luigi.BoolParameter(config_path=dict(section='foo', name='bar'))
        self.assertEqual(True, _value(p))

    @with_config({'foo': {'bar': 'false'}})
    def testBoolConfigOutranksDefault(self):
        if False:
            return 10
        p = luigi.BoolParameter(default=True, config_path=dict(section='foo', name='bar'))
        self.assertEqual(False, _value(p))

    @with_config({'foo': {'bar': '2001-02-03-2001-02-28'}})
    def testDateInterval(self):
        if False:
            print('Hello World!')
        p = luigi.DateIntervalParameter(config_path=dict(section='foo', name='bar'))
        expected = luigi.date_interval.Custom.parse('2001-02-03-2001-02-28')
        self.assertEqual(expected, _value(p))

    @with_config({'foo': {'bar': '0 seconds'}})
    def testTimeDeltaNoSeconds(self):
        if False:
            i = 10
            return i + 15
        p = luigi.TimeDeltaParameter(config_path=dict(section='foo', name='bar'))
        self.assertEqual(timedelta(seconds=0), _value(p))

    @with_config({'foo': {'bar': '0 d'}})
    def testTimeDeltaNoDays(self):
        if False:
            for i in range(10):
                print('nop')
        p = luigi.TimeDeltaParameter(config_path=dict(section='foo', name='bar'))
        self.assertEqual(timedelta(days=0), _value(p))

    @with_config({'foo': {'bar': '1 day'}})
    def testTimeDelta(self):
        if False:
            return 10
        p = luigi.TimeDeltaParameter(config_path=dict(section='foo', name='bar'))
        self.assertEqual(timedelta(days=1), _value(p))

    @with_config({'foo': {'bar': '2 seconds'}})
    def testTimeDeltaPlural(self):
        if False:
            while True:
                i = 10
        p = luigi.TimeDeltaParameter(config_path=dict(section='foo', name='bar'))
        self.assertEqual(timedelta(seconds=2), _value(p))

    @with_config({'foo': {'bar': '3w 4h 5m'}})
    def testTimeDeltaMultiple(self):
        if False:
            i = 10
            return i + 15
        p = luigi.TimeDeltaParameter(config_path=dict(section='foo', name='bar'))
        self.assertEqual(timedelta(weeks=3, hours=4, minutes=5), _value(p))

    @with_config({'foo': {'bar': 'P4DT12H30M5S'}})
    def testTimeDelta8601(self):
        if False:
            i = 10
            return i + 15
        p = luigi.TimeDeltaParameter(config_path=dict(section='foo', name='bar'))
        self.assertEqual(timedelta(days=4, hours=12, minutes=30, seconds=5), _value(p))

    @with_config({'foo': {'bar': 'P5D'}})
    def testTimeDelta8601NoTimeComponent(self):
        if False:
            while True:
                i = 10
        p = luigi.TimeDeltaParameter(config_path=dict(section='foo', name='bar'))
        self.assertEqual(timedelta(days=5), _value(p))

    @with_config({'foo': {'bar': 'P5W'}})
    def testTimeDelta8601Weeks(self):
        if False:
            print('Hello World!')
        p = luigi.TimeDeltaParameter(config_path=dict(section='foo', name='bar'))
        self.assertEqual(timedelta(weeks=5), _value(p))

    @mock.patch('luigi.parameter.ParameterException')
    @with_config({'foo': {'bar': 'P3Y6M4DT12H30M5S'}})
    def testTimeDelta8601YearMonthNotSupported(self, exc):
        if False:
            i = 10
            return i + 15

        def f():
            if False:
                return 10
            return _value(luigi.TimeDeltaParameter(config_path=dict(section='foo', name='bar')))
        self.assertRaises(ValueError, f)
        exc.assert_called_once_with('Invalid time delta - could not parse P3Y6M4DT12H30M5S')

    @with_config({'foo': {'bar': 'PT6M'}})
    def testTimeDelta8601MAfterT(self):
        if False:
            i = 10
            return i + 15
        p = luigi.TimeDeltaParameter(config_path=dict(section='foo', name='bar'))
        self.assertEqual(timedelta(minutes=6), _value(p))

    @mock.patch('luigi.parameter.ParameterException')
    @with_config({'foo': {'bar': 'P6M'}})
    def testTimeDelta8601MBeforeT(self, exc):
        if False:
            print('Hello World!')

        def f():
            if False:
                for i in range(10):
                    print('nop')
            return _value(luigi.TimeDeltaParameter(config_path=dict(section='foo', name='bar')))
        self.assertRaises(ValueError, f)
        exc.assert_called_once_with('Invalid time delta - could not parse P6M')

    @with_config({'foo': {'bar': '12.34'}})
    def testTimeDeltaFloat(self):
        if False:
            for i in range(10):
                print('nop')
        p = luigi.TimeDeltaParameter(config_path=dict(section='foo', name='bar'))
        self.assertEqual(timedelta(seconds=12.34), _value(p))

    @with_config({'foo': {'bar': '56789'}})
    def testTimeDeltaInt(self):
        if False:
            print('Hello World!')
        p = luigi.TimeDeltaParameter(config_path=dict(section='foo', name='bar'))
        self.assertEqual(timedelta(seconds=56789), _value(p))

    def testHasDefaultNoSection(self):
        if False:
            print('Hello World!')
        self.assertRaises(luigi.parameter.MissingParameterException, lambda : _value(luigi.Parameter(config_path=dict(section='foo', name='bar'))))

    @with_config({'foo': {}})
    def testHasDefaultNoValue(self):
        if False:
            print('Hello World!')
        self.assertRaises(luigi.parameter.MissingParameterException, lambda : _value(luigi.Parameter(config_path=dict(section='foo', name='bar'))))

    @with_config({'foo': {'bar': 'baz'}})
    def testHasDefaultWithBoth(self):
        if False:
            return 10
        self.assertTrue(_value(luigi.Parameter(config_path=dict(section='foo', name='bar'))))

    @with_config({'foo': {'bar': 'baz'}})
    def testWithDefault(self):
        if False:
            while True:
                i = 10
        p = luigi.Parameter(config_path=dict(section='foo', name='bar'), default='blah')
        self.assertEqual('baz', _value(p))

    def testWithDefaultAndMissing(self):
        if False:
            while True:
                i = 10
        p = luigi.Parameter(config_path=dict(section='foo', name='bar'), default='blah')
        self.assertEqual('blah', _value(p))

    @with_config({'LocalA': {'p': 'p_default'}})
    def testDefaultFromTaskName(self):
        if False:
            return 10

        class LocalA(luigi.Task):
            p = luigi.Parameter()
        self.assertEqual('p_default', LocalA().p)
        self.assertEqual('boo', LocalA(p='boo').p)

    @with_config({'LocalA': {'p': '999'}})
    def testDefaultFromTaskNameInt(self):
        if False:
            print('Hello World!')

        class LocalA(luigi.Task):
            p = luigi.IntParameter()
        self.assertEqual(999, LocalA().p)
        self.assertEqual(777, LocalA(p=777).p)

    @with_config({'LocalA': {'p': 'p_default'}, 'foo': {'bar': 'baz'}})
    def testDefaultFromConfigWithTaskNameToo(self):
        if False:
            for i in range(10):
                print('nop')

        class LocalA(luigi.Task):
            p = luigi.Parameter(config_path=dict(section='foo', name='bar'))
        self.assertEqual('p_default', LocalA().p)
        self.assertEqual('boo', LocalA(p='boo').p)

    @with_config({'LocalA': {'p': 'p_default_2'}})
    def testDefaultFromTaskNameWithDefault(self):
        if False:
            i = 10
            return i + 15

        class LocalA(luigi.Task):
            p = luigi.Parameter(default='banana')
        self.assertEqual('p_default_2', LocalA().p)
        self.assertEqual('boo_2', LocalA(p='boo_2').p)

    @with_config({'MyClass': {'p_wohoo': 'p_default_3'}})
    def testWithLongParameterName(self):
        if False:
            return 10

        class MyClass(luigi.Task):
            p_wohoo = luigi.Parameter(default='banana')
        self.assertEqual('p_default_3', MyClass().p_wohoo)
        self.assertEqual('boo_2', MyClass(p_wohoo='boo_2').p_wohoo)

    @with_config({'RangeDaily': {'days_back': '123'}})
    def testSettingOtherMember(self):
        if False:
            return 10

        class LocalA(luigi.Task):
            pass
        self.assertEqual(123, luigi.tools.range.RangeDaily(of=LocalA).days_back)
        self.assertEqual(70, luigi.tools.range.RangeDaily(of=LocalA, days_back=70).days_back)

    @with_config({'MyClass': {'p_not_global': '123'}})
    def testCommandLineWithDefault(self):
        if False:
            return 10
        '\n        Verify that we also read from the config when we build tasks from the\n        command line parsers.\n        '

        class MyClass(luigi.Task):
            p_not_global = luigi.Parameter(default='banana')

            def complete(self):
                if False:
                    while True:
                        i = 10
                import sys
                luigi.configuration.get_config().write(sys.stdout)
                if self.p_not_global != '123':
                    raise ValueError("The parameter didn't get set!!")
                return True

            def run(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass
        self.assertTrue(self.run_locally(['MyClass']))
        self.assertFalse(self.run_locally(['MyClass', '--p-not-global', '124']))
        self.assertFalse(self.run_locally(['MyClass', '--MyClass-p-not-global', '124']))

    @with_config({'MyClass2': {'p_not_global_no_default': '123'}})
    def testCommandLineNoDefault(self):
        if False:
            i = 10
            return i + 15
        '\n        Verify that we also read from the config when we build tasks from the\n        command line parsers.\n        '

        class MyClass2(luigi.Task):
            """ TODO: Make luigi clean it's register for tests. Hate this 2 dance. """
            p_not_global_no_default = luigi.Parameter()

            def complete(self):
                if False:
                    return 10
                import sys
                luigi.configuration.get_config().write(sys.stdout)
                luigi.configuration.get_config().write(sys.stdout)
                if self.p_not_global_no_default != '123':
                    raise ValueError("The parameter didn't get set!!")
                return True

            def run(self):
                if False:
                    return 10
                pass
        self.assertTrue(self.run_locally(['MyClass2']))
        self.assertFalse(self.run_locally(['MyClass2', '--p-not-global-no-default', '124']))
        self.assertFalse(self.run_locally(['MyClass2', '--MyClass2-p-not-global-no-default', '124']))

    @with_config({'mynamespace.A': {'p': '999'}})
    def testWithNamespaceConfig(self):
        if False:
            for i in range(10):
                print('nop')

        class A(luigi.Task):
            task_namespace = 'mynamespace'
            p = luigi.IntParameter()
        self.assertEqual(999, A().p)
        self.assertEqual(777, A(p=777).p)

    def testWithNamespaceCli(self):
        if False:
            return 10

        class A(luigi.Task):
            task_namespace = 'mynamespace'
            p = luigi.IntParameter(default=100)
            expected = luigi.IntParameter()

            def complete(self):
                if False:
                    i = 10
                    return i + 15
                if self.p != self.expected:
                    raise ValueError
                return True
        self.assertTrue(self.run_locally_split('mynamespace.A --expected 100'))
        self.assertTrue(self.run_locally_split('mynamespace.A --mynamespace.A-p 200 --expected 200'))
        self.assertFalse(self.run_locally_split('mynamespace.A --A-p 200 --expected 200'))

    def testListWithNamespaceCli(self):
        if False:
            for i in range(10):
                print('nop')

        class A(luigi.Task):
            task_namespace = 'mynamespace'
            l_param = luigi.ListParameter(default=[1, 2, 3])
            expected = luigi.ListParameter()

            def complete(self):
                if False:
                    while True:
                        i = 10
                if self.l_param != self.expected:
                    raise ValueError
                return True
        self.assertTrue(self.run_locally_split('mynamespace.A --expected [1,2,3]'))
        self.assertTrue(self.run_locally_split('mynamespace.A --mynamespace.A-l [1,2,3] --expected [1,2,3]'))

    def testTupleWithNamespaceCli(self):
        if False:
            print('Hello World!')

        class A(luigi.Task):
            task_namespace = 'mynamespace'
            t = luigi.TupleParameter(default=((1, 2), (3, 4)))
            expected = luigi.TupleParameter()

            def complete(self):
                if False:
                    for i in range(10):
                        print('nop')
                if self.t != self.expected:
                    raise ValueError
                return True
        self.assertTrue(self.run_locally_split('mynamespace.A --expected ((1,2),(3,4))'))
        self.assertTrue(self.run_locally_split('mynamespace.A --mynamespace.A-t ((1,2),(3,4)) --expected ((1,2),(3,4))'))

    @with_config({'foo': {'bar': '[1,2,3]'}})
    def testListConfig(self):
        if False:
            return 10
        self.assertTrue(_value(luigi.ListParameter(config_path=dict(section='foo', name='bar'))))

    @with_config({'foo': {'bar': '((1,2),(3,4))'}})
    def testTupleConfig(self):
        if False:
            print('Hello World!')
        self.assertTrue(_value(luigi.TupleParameter(config_path=dict(section='foo', name='bar'))))

    @with_config({'foo': {'bar': '-3'}})
    def testNumericalParameter(self):
        if False:
            for i in range(10):
                print('nop')
        p = luigi.NumericalParameter(min_value=-3, max_value=7, var_type=int, config_path=dict(section='foo', name='bar'))
        self.assertEqual(-3, _value(p))

    @with_config({'foo': {'bar': '3'}})
    def testChoiceParameter(self):
        if False:
            print('Hello World!')
        p = luigi.ChoiceParameter(var_type=int, choices=[1, 2, 3], config_path=dict(section='foo', name='bar'))
        self.assertEqual(3, _value(p))

class OverrideEnvStuff(LuigiTestCase):

    @with_config({'core': {'default-scheduler-port': '6543'}})
    def testOverrideSchedulerPort(self):
        if False:
            while True:
                i = 10
        with self.assertWarnsRegex(DeprecationWarning, 'default-scheduler-port is deprecated'):
            env_params = luigi.interface.core()
            self.assertEqual(env_params.scheduler_port, 6543)

    @with_config({'core': {'scheduler-port': '6544'}})
    def testOverrideSchedulerPort2(self):
        if False:
            print('Hello World!')
        with self.assertWarnsRegex(DeprecationWarning, 'scheduler-port \\(with dashes\\) should be avoided'):
            env_params = luigi.interface.core()
        self.assertEqual(env_params.scheduler_port, 6544)

    @with_config({'core': {'scheduler_port': '6545'}})
    def testOverrideSchedulerPort3(self):
        if False:
            i = 10
            return i + 15
        env_params = luigi.interface.core()
        self.assertEqual(env_params.scheduler_port, 6545)

class TestSerializeDateParameters(LuigiTestCase):

    def testSerialize(self):
        if False:
            print('Hello World!')
        date = datetime.date(2013, 2, 3)
        self.assertEqual(luigi.DateParameter().serialize(date), '2013-02-03')
        self.assertEqual(luigi.YearParameter().serialize(date), '2013')
        self.assertEqual(luigi.MonthParameter().serialize(date), '2013-02')
        dt = datetime.datetime(2013, 2, 3, 4, 5)
        self.assertEqual(luigi.DateHourParameter().serialize(dt), '2013-02-03T04')

class TestSerializeTimeDeltaParameters(LuigiTestCase):

    def testSerialize(self):
        if False:
            i = 10
            return i + 15
        tdelta = timedelta(weeks=5, days=4, hours=3, minutes=2, seconds=1)
        self.assertEqual(luigi.TimeDeltaParameter().serialize(tdelta), '5 w 4 d 3 h 2 m 1 s')
        tdelta = timedelta(seconds=0)
        self.assertEqual(luigi.TimeDeltaParameter().serialize(tdelta), '0 w 0 d 0 h 0 m 0 s')

class TestTaskParameter(LuigiTestCase):

    def testUsage(self):
        if False:
            return 10

        class MetaTask(luigi.Task):
            task_namespace = 'mynamespace'
            a = luigi.TaskParameter()

            def run(self):
                if False:
                    i = 10
                    return i + 15
                self.__class__.saved_value = self.a

        class OtherTask(luigi.Task):
            task_namespace = 'other_namespace'
        self.assertEqual(MetaTask(a=MetaTask).a, MetaTask)
        self.assertEqual(MetaTask(a=OtherTask).a, OtherTask)
        self.assertRaises(AttributeError, lambda : MetaTask(a='mynamespace.MetaTask'))
        self.assertRaises(luigi.task_register.TaskClassNotFoundException, lambda : self.run_locally_split('mynamespace.MetaTask --a blah'))
        self.assertRaises(luigi.task_register.TaskClassNotFoundException, lambda : self.run_locally_split('mynamespace.MetaTask --a Taskk'))
        self.assertTrue(self.run_locally_split('mynamespace.MetaTask --a mynamespace.MetaTask'))
        self.assertEqual(MetaTask.saved_value, MetaTask)
        self.assertTrue(self.run_locally_split('mynamespace.MetaTask --a other_namespace.OtherTask'))
        self.assertEqual(MetaTask.saved_value, OtherTask)

    def testSerialize(self):
        if False:
            return 10

        class OtherTask(luigi.Task):

            def complete(self):
                if False:
                    return 10
                return True

        class DepTask(luigi.Task):
            dep = luigi.TaskParameter()
            ran = False

            def complete(self):
                if False:
                    i = 10
                    return i + 15
                return self.__class__.ran

            def requires(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.dep()

            def run(self):
                if False:
                    print('Hello World!')
                self.__class__.ran = True

        class MainTask(luigi.Task):

            def run(self):
                if False:
                    while True:
                        i = 10
                yield DepTask(dep=OtherTask)
        self.assertTrue(self.run_locally(['MainTask']))

class TestSerializeTupleParameter(LuigiTestCase):

    def testSerialize(self):
        if False:
            print('Hello World!')
        the_tuple = (1, 2, 3)
        self.assertEqual(luigi.TupleParameter().parse(luigi.TupleParameter().serialize(the_tuple)), the_tuple)

class NewStyleParameters822Test(LuigiTestCase):
    """
    I bet these tests created at 2015-03-08 are reduntant by now (Oct 2015).
    But maintaining them anyway, just in case I have overlooked something.
    """

    def test_subclasses(self):
        if False:
            for i in range(10):
                print('nop')

        class BarBaseClass(luigi.Task):
            x = luigi.Parameter(default='bar_base_default')

        class BarSubClass(BarBaseClass):
            pass
        in_parse(['BarSubClass', '--x', 'xyz', '--BarBaseClass-x', 'xyz'], lambda task: self.assertEqual(task.x, 'xyz'))
        in_parse(['BarBaseClass', '--BarBaseClass-x', 'xyz'], lambda task: self.assertEqual(task.x, 'xyz'))

class LocalParameters1304Test(LuigiTestCase):
    """
    It was discussed and decided that local parameters (--x) should be
    semantically different from global parameters (--MyTask-x).

    The former sets only the parsed root task, and the later sets the parameter
    for all the tasks.

    https://github.com/spotify/luigi/issues/1304#issuecomment-148402284
    """

    def test_local_params(self):
        if False:
            while True:
                i = 10

        class MyTask(RunOnceTask):
            param1 = luigi.IntParameter()
            param2 = luigi.BoolParameter(default=False)

            def requires(self):
                if False:
                    i = 10
                    return i + 15
                if self.param1 > 0:
                    yield MyTask(param1=self.param1 - 1)

            def run(self):
                if False:
                    for i in range(10):
                        print('nop')
                assert self.param1 == 1 or not self.param2
                self.comp = True
        self.assertTrue(self.run_locally_split('MyTask --param1 1 --param2'))

    def test_local_takes_precedence(self):
        if False:
            while True:
                i = 10

        class MyTask(luigi.Task):
            param = luigi.IntParameter()

            def complete(self):
                if False:
                    print('Hello World!')
                return False

            def run(self):
                if False:
                    for i in range(10):
                        print('nop')
                assert self.param == 5
        self.assertTrue(self.run_locally_split('MyTask --param 5 --MyTask-param 6'))

    def test_local_only_affects_root(self):
        if False:
            for i in range(10):
                print('nop')

        class MyTask(RunOnceTask):
            param = luigi.IntParameter(default=3)

            def requires(self):
                if False:
                    i = 10
                    return i + 15
                assert self.param != 3
                if self.param == 5:
                    yield MyTask()
        self.assertTrue(self.run_locally_split('MyTask --param 5 --MyTask-param 6'))

    def test_range_doesnt_propagate_args(self):
        if False:
            return 10
        "\n        Ensure that ``--task Range --of Blah --blah-arg 123`` doesn't work.\n\n        This will of course not work unless support is explicitly added for it.\n        But being a bit paranoid here and adding this test case so that if\n        somebody decides to add it in the future, they'll be redircted to the\n        dicussion in #1304\n        "

        class Blah(RunOnceTask):
            date = luigi.DateParameter()
            blah_arg = luigi.IntParameter()
        self.assertRaises(SystemExit, self.run_locally_split, 'RangeDailyBase --of Blah --start 2015-01-01 --task-limit 1 --blah-arg 123')
        self.assertTrue(self.run_locally_split('RangeDailyBase --of Blah --start 2015-01-01 --task-limit 1 --Blah-blah-arg 123'))

class TaskAsParameterName1335Test(LuigiTestCase):

    def test_parameter_can_be_named_task(self):
        if False:
            while True:
                i = 10

        class MyTask(luigi.Task):
            task = luigi.IntParameter()
        self.assertTrue(self.run_locally_split('MyTask --task 5'))

class TestPathParameter:

    @pytest.fixture(params=[None, 'not_existing_dir'])
    def default(self, request):
        if False:
            while True:
                i = 10
        return request.param

    @pytest.fixture(params=[True, False])
    def absolute(self, request):
        if False:
            i = 10
            return i + 15
        return request.param

    @pytest.fixture(params=[True, False])
    def exists(self, request):
        if False:
            for i in range(10):
                print('nop')
        return request.param

    @pytest.fixture()
    def path_parameter(self, tmpdir, default, absolute, exists):
        if False:
            while True:
                i = 10

        class TaskPathParameter(luigi.Task):
            a = luigi.PathParameter(default=str(tmpdir / default) if default is not None else str(tmpdir), absolute=absolute, exists=exists)
            b = luigi.OptionalPathParameter(default=str(tmpdir / default) if default is not None else str(tmpdir), absolute=absolute, exists=exists)
            c = luigi.OptionalPathParameter(default=None)
            d = luigi.OptionalPathParameter(default='not empty default')

            def run(self):
                if False:
                    print('Hello World!')
                new_file = self.a / 'test.file'
                new_optional_file = self.b / 'test_optional.file'
                if default is not None:
                    new_file.parent.mkdir(parents=True)
                new_file.touch()
                new_optional_file.touch()
                assert new_file.exists()
                assert new_optional_file.exists()
                assert self.c is None
                assert self.d is None

            def output(self):
                if False:
                    while True:
                        i = 10
                return luigi.LocalTarget('not_existing_file')
        return {'tmpdir': tmpdir, 'default': default, 'absolute': absolute, 'exists': exists, 'cls': TaskPathParameter}

    @with_config({'TaskPathParameter': {'d': ''}})
    def test_exists(self, path_parameter):
        if False:
            while True:
                i = 10
        if path_parameter['default'] is not None and path_parameter['exists']:
            with pytest.raises(ValueError, match='The path .* does not exist'):
                luigi.build([path_parameter['cls']()], local_scheduler=True)
        else:
            assert luigi.build([path_parameter['cls']()], local_scheduler=True)