import warnings
import luigi
import mock
from helpers import LuigiTestCase, with_config

class OptionalParameterTest(LuigiTestCase):

    def actual_test(self, cls, default, expected_value, expected_type, bad_data, **kwargs):
        if False:
            i = 10
            return i + 15

        class TestConfig(luigi.Config):
            param = cls(default=default, **kwargs)
            empty_param = cls(default=default, **kwargs)

            def run(self):
                if False:
                    while True:
                        i = 10
                assert self.param == expected_value
                assert self.empty_param is None
        self.assertIsNone(cls(**kwargs).parse(''))
        with mock.patch('luigi.parameter.warnings') as warnings:
            TestConfig()
            warnings.warn.assert_not_called()
        if cls != luigi.OptionalChoiceParameter:
            with mock.patch('luigi.parameter.warnings') as warnings:
                TestConfig(param=None)
                warnings.warn.assert_not_called()
            with mock.patch('luigi.parameter.warnings') as warnings:
                TestConfig(param=bad_data)
                if cls == luigi.OptionalBoolParameter:
                    warnings.warn.assert_not_called()
                else:
                    warnings.warn.assert_called_with('{} "param" with value "{}" is not of type "{}" or None.'.format(cls.__name__, bad_data, expected_type), luigi.parameter.OptionalParameterTypeWarning)
        self.assertTrue(luigi.build([TestConfig()], local_scheduler=True))

    @with_config({'TestConfig': {'param': 'expected value', 'empty_param': ''}})
    def test_optional_parameter(self):
        if False:
            while True:
                i = 10
        self.actual_test(luigi.OptionalParameter, None, 'expected value', 'str', 0)
        self.actual_test(luigi.OptionalParameter, 'default value', 'expected value', 'str', 0)

    @with_config({'TestConfig': {'param': '10', 'empty_param': ''}})
    def test_optional_int_parameter(self):
        if False:
            i = 10
            return i + 15
        self.actual_test(luigi.OptionalIntParameter, None, 10, 'int', 'bad data')
        self.actual_test(luigi.OptionalIntParameter, 1, 10, 'int', 'bad data')

    @with_config({'TestConfig': {'param': 'true', 'empty_param': ''}})
    def test_optional_bool_parameter(self):
        if False:
            while True:
                i = 10
        self.actual_test(luigi.OptionalBoolParameter, None, True, 'bool', 'bad data')
        self.actual_test(luigi.OptionalBoolParameter, False, True, 'bool', 'bad data')

    @with_config({'TestConfig': {'param': '10.5', 'empty_param': ''}})
    def test_optional_float_parameter(self):
        if False:
            print('Hello World!')
        self.actual_test(luigi.OptionalFloatParameter, None, 10.5, 'float', 'bad data')
        self.actual_test(luigi.OptionalFloatParameter, 1.5, 10.5, 'float', 'bad data')

    @with_config({'TestConfig': {'param': '{"a": 10}', 'empty_param': ''}})
    def test_optional_dict_parameter(self):
        if False:
            return 10
        self.actual_test(luigi.OptionalDictParameter, None, {'a': 10}, 'FrozenOrderedDict', 'bad data')
        self.actual_test(luigi.OptionalDictParameter, {'a': 1}, {'a': 10}, 'FrozenOrderedDict', 'bad data')

    @with_config({'TestConfig': {'param': '[10.5]', 'empty_param': ''}})
    def test_optional_list_parameter(self):
        if False:
            print('Hello World!')
        self.actual_test(luigi.OptionalListParameter, None, (10.5,), 'tuple', 'bad data')
        self.actual_test(luigi.OptionalListParameter, (1.5,), (10.5,), 'tuple', 'bad data')

    @with_config({'TestConfig': {'param': '[10.5]', 'empty_param': ''}})
    def test_optional_tuple_parameter(self):
        if False:
            return 10
        self.actual_test(luigi.OptionalTupleParameter, None, (10.5,), 'tuple', 'bad data')
        self.actual_test(luigi.OptionalTupleParameter, (1.5,), (10.5,), 'tuple', 'bad data')

    @with_config({'TestConfig': {'param': '10.5', 'empty_param': ''}})
    def test_optional_numerical_parameter_float(self):
        if False:
            print('Hello World!')
        self.actual_test(luigi.OptionalNumericalParameter, None, 10.5, 'float', 'bad data', var_type=float, min_value=0, max_value=100)
        self.actual_test(luigi.OptionalNumericalParameter, 1.5, 10.5, 'float', 'bad data', var_type=float, min_value=0, max_value=100)

    @with_config({'TestConfig': {'param': '10', 'empty_param': ''}})
    def test_optional_numerical_parameter_int(self):
        if False:
            while True:
                i = 10
        self.actual_test(luigi.OptionalNumericalParameter, None, 10, 'int', 'bad data', var_type=int, min_value=0, max_value=100)
        self.actual_test(luigi.OptionalNumericalParameter, 1, 10, 'int', 'bad data', var_type=int, min_value=0, max_value=100)

    @with_config({'TestConfig': {'param': 'expected value', 'empty_param': ''}})
    def test_optional_choice_parameter(self):
        if False:
            while True:
                i = 10
        choices = ['default value', 'expected value']
        self.actual_test(luigi.OptionalChoiceParameter, None, 'expected value', 'str', 'bad data', choices=choices)
        self.actual_test(luigi.OptionalChoiceParameter, 'default value', 'expected value', 'str', 'bad data', choices=choices)

    @with_config({'TestConfig': {'param': '1', 'empty_param': ''}})
    def test_optional_choice_parameter_int(self):
        if False:
            for i in range(10):
                print('nop')
        choices = [0, 1, 2]
        self.actual_test(luigi.OptionalChoiceParameter, None, 1, 'int', 'bad data', var_type=int, choices=choices)
        self.actual_test(luigi.OptionalChoiceParameter, 'default value', 1, 'int', 'bad data', var_type=int, choices=choices)

    def test_warning(self):
        if False:
            print('Hello World!')

        class TestOptionalFloatParameterSingleType(luigi.parameter.OptionalParameter, luigi.FloatParameter):
            expected_type = float

        class TestOptionalFloatParameterMultiTypes(luigi.parameter.OptionalParameter, luigi.FloatParameter):
            expected_type = (int, float)

        class TestConfig(luigi.Config):
            param_single = TestOptionalFloatParameterSingleType()
            param_multi = TestOptionalFloatParameterMultiTypes()
        with warnings.catch_warnings(record=True) as record:
            TestConfig(param_single=0.0, param_multi=1.0)
        assert len(record) == 0
        with warnings.catch_warnings(record=True) as record:
            warnings.filterwarnings(action='ignore', category=Warning)
            warnings.simplefilter(action='always', category=luigi.parameter.OptionalParameterTypeWarning)
            assert luigi.build([TestConfig(param_single='0', param_multi='1')], local_scheduler=True)
        assert len(record) == 2
        assert issubclass(record[0].category, luigi.parameter.OptionalParameterTypeWarning)
        assert issubclass(record[1].category, luigi.parameter.OptionalParameterTypeWarning)
        assert str(record[0].message) == 'TestOptionalFloatParameterSingleType "param_single" with value "0" is not of type "float" or None.'
        assert str(record[1].message) == 'TestOptionalFloatParameterMultiTypes "param_multi" with value "1" is not of any type in ["int", "float"] or None.'