import inspect
from unittest import mock
from unittest.mock import MagicMock
from faker.config import DEFAULT_LOCALE
from faker.sphinx.docstring import DEFAULT_SAMPLE_SIZE, DEFAULT_SEED, ProviderMethodDocstring, Sample

class TestProviderMethodDocstring:

    def test_what_is_not_method(self):
        if False:
            i = 10
            return i + 15
        docstring = ProviderMethodDocstring(app=MagicMock(), what='not_a_method', name='name', obj=MagicMock, options=MagicMock(), lines=MagicMock())
        assert docstring.skipped

    def test_name_is_not_dotted_path_to_provider_method(self):
        if False:
            print('Hello World!')
        docstring = ProviderMethodDocstring(app=MagicMock(), what='method', name='faker.sphinx.docstring.ProviderMethodDocString._parse', obj=MagicMock, options=MagicMock(), lines=MagicMock())
        assert docstring.skipped

    def test_name_is_dotted_path_to_base_provider_method(self):
        if False:
            while True:
                i = 10
        docstring = ProviderMethodDocstring(app=MagicMock(), what='method', name='faker.providers.BaseProvider.bothify', obj=MagicMock, options=MagicMock(), lines=MagicMock())
        assert not docstring.skipped
        assert docstring._method == 'bothify'
        assert docstring._locale == DEFAULT_LOCALE

    def test_name_is_dotted_path_to_standard_provider_method(self):
        if False:
            print('Hello World!')
        docstring = ProviderMethodDocstring(app=MagicMock(), what='method', name='faker.providers.barcode.Provider.upc_a', obj=MagicMock, options=MagicMock(), lines=MagicMock())
        assert not docstring.skipped
        assert docstring._method == 'upc_a'
        assert docstring._locale == DEFAULT_LOCALE

    def test_name_is_dotted_path_to_localized_provider_method(self):
        if False:
            for i in range(10):
                print('nop')
        docstring = ProviderMethodDocstring(app=MagicMock(), what='method', name='faker.providers.automotive.en_PH.Provider.protocol_license_plate', obj=MagicMock, options=MagicMock(), lines=MagicMock())
        assert not docstring.skipped
        assert docstring._method == 'protocol_license_plate'
        assert docstring._locale == 'en_PH'

    @mock.patch('faker.sphinx.docstring.logger.warning')
    def test_log_warning(self, mock_logger_warning):
        if False:
            while True:
                i = 10
        path = inspect.getfile(MagicMock)
        name = 'faker.providers.color.Provider'
        docstring = ProviderMethodDocstring(app=MagicMock(), what='method', name=name, obj=MagicMock, options=MagicMock(), lines=MagicMock())
        docstring._log_warning('Test Warning 1')
        docstring._log_warning('Test Warning 2')
        assert docstring._log_prefix == f'{path}:docstring of {name}: WARNING:'
        calls = mock_logger_warning.call_args_list
        assert len(calls) == 2
        (args, kwargs) = calls[0]
        assert len(args) == 1
        assert not kwargs
        assert args[0] == f'{path}:docstring of {name}: WARNING: Test Warning 1'
        (args, kwargs) = calls[1]
        assert len(args) == 1
        assert not kwargs
        assert args[0] == f'{path}:docstring of {name}: WARNING: Test Warning 2'

    def test_stringify_results(self, faker):
        if False:
            return 10

        class TestObject:

            def __repr__(self):
                if False:
                    return 10
                return 'abcdefg'
        docstring = ProviderMethodDocstring(app=MagicMock(), what='method', name='faker.providers.BaseProvider.bothify', obj=MagicMock, options=MagicMock(), lines=[])
        results = ['', "'", "'", '"', '"', 'aa\taaaaa\r\n', b'abcdef', True, False, None, [1, 2, 3, 4, 5], (1, 2, 3, 4, 5), {1: 2, 2: 3, 3: 4, 4: 5}, faker.uuid4(cast_to=None), TestObject()]
        output = [docstring._stringify_result(result) for result in results]
        assert output == ["''", '"\'"', '"\'"', '\'"\'', '\'"\'', "'aa\\taaaaa\\r\\n'", "b'abcdef'", 'True', 'False', 'None', '[1, 2, 3, 4, 5]', '(1, 2, 3, 4, 5)', '{1: 2, 2: 3, 3: 4, 4: 5}', "UUID('e3e70682-c209-4cac-a29f-6fbed82c07cd')", 'abcdefg']

    @mock.patch.object(ProviderMethodDocstring, '_log_warning')
    def test_parsing_empty_lines(self, mock_log_warning):
        if False:
            return 10
        docstring = ProviderMethodDocstring(app=MagicMock(), what='method', name='faker.providers.BaseProvider.bothify', obj=MagicMock, options=MagicMock(), lines=[])
        assert not docstring.skipped
        assert len(docstring._samples) == 1
        assert docstring._samples[0] == Sample(DEFAULT_SAMPLE_SIZE, DEFAULT_SEED, '')

    @mock.patch.object(ProviderMethodDocstring, '_log_warning')
    def test_parsing_single_line_non_sample(self, mock_log_warning):
        if False:
            for i in range(10):
                print('nop')
        docstring = ProviderMethodDocstring(app=MagicMock(), what='method', name='faker.providers.BaseProvider.bothify', obj=MagicMock, options=MagicMock(), lines=['lorem'])
        assert not docstring.skipped
        assert len(docstring._samples) == 1
        assert docstring._samples[0] == Sample(DEFAULT_SAMPLE_SIZE, DEFAULT_SEED, '')

    @mock.patch.object(ProviderMethodDocstring, '_log_warning')
    def test_parsing_single_line_valid_sample(self, mock_log_warning):
        if False:
            return 10
        docstring = ProviderMethodDocstring(app=MagicMock(), what='method', name='faker.providers.BaseProvider.bothify', obj=MagicMock, options=MagicMock(), lines=[':sample: a=1'])
        assert not docstring.skipped
        assert docstring._samples == [Sample(5, 0, 'a=1')]

    @mock.patch.object(ProviderMethodDocstring, '_log_warning')
    def test_parsing_multiple_lines(self, mock_log_warning):
        if False:
            i = 10
            return i + 15
        lines = ['lorem', ':sample:', ':sample 10 2000:', ':sample 10 seed=1000:', ':sample size=10 1000:', ':sample size=0:', ':sample size=100:', ':sample size=0100:', ':sampler', ':sample :', ':sample seed=4761:', '', 'ipsum', ':sample sede=123', ':sample size=4 seed=100:', ':sample seed=103 size=104:', ':sample: a=1, b=2', ':sample size=2222: a=2, b=1', ':sample 11 12:', ':sample seed=3333: d=3', ':sample size=3333 seed=2222: c=1', ':sample size=10 seed=10:', '   arg1=1,', '   arg2="val2",arg3="val3",', ' arg4=4   ,    arg5=5,', ' arg6="ar  g6",', "       arg7='   ar  g 7',", '    arg8="aaa,aaa"', ':sample size=20 seed=3456:', 'arg1="val1,val1,val1",', 'arg2="val2",', 'arg3="val3    val3",']
        expected_output = [Sample(DEFAULT_SAMPLE_SIZE, DEFAULT_SEED, ''), Sample(100, DEFAULT_SEED, ''), Sample(DEFAULT_SAMPLE_SIZE, 4761, ''), Sample(5, 100, ''), Sample(DEFAULT_SAMPLE_SIZE, DEFAULT_SEED, 'a=1, b=2'), Sample(2222, DEFAULT_SEED, 'a=2, b=1'), Sample(DEFAULT_SAMPLE_SIZE, 3333, 'd=3'), Sample(3333, 2222, 'c=1'), Sample(10, 10, 'arg1=1, arg2="val2", arg3="val3", arg4=4, arg5=5, arg6="ar  g6", arg7=\'   ar  g 7\', arg8="aaa,aaa"'), Sample(20, 3456, 'arg1="val1,val1,val1", arg2="val2", arg3="val3    val3",')]
        docstring = ProviderMethodDocstring(app=MagicMock(), what='method', name='faker.providers.BaseProvider.bothify', obj=MagicMock, options=MagicMock(), lines=lines)
        assert not docstring.skipped
        assert docstring._samples == expected_output

    @mock.patch.object(ProviderMethodDocstring, '_log_warning')
    def test_end_to_end_sample_generation(self, mock_warning, faker):
        if False:
            for i in range(10):
                print('nop')
        non_sample_lines = ['lorem', 'ipsum', 'dolor', 'sit', 'amet']
        valid_sample_lines = [':sample 1234jdbvhjdbygdvbhxjhx', ":sample: invalid_arg='value'", ":sample size=3 seed=1000: text='???###'", ':sample: number=100**100**100', ":sample seed=3210: letters='abcde'", ":sample size=10 seed=1: abcd='abcd'", ":sample size=20 seed=1234: text='???###', ", "         letters='abcde'"]
        lines = non_sample_lines + valid_sample_lines
        docstring = ProviderMethodDocstring(app=MagicMock(), what='method', name='faker.providers.BaseProvider.bothify', obj=MagicMock, options=MagicMock(), lines=lines)
        output = docstring.lines[len(non_sample_lines):]
        assert output[0] == ':examples:'
        faker.seed_instance(1000)
        assert output[1] == ''
        assert output[2] == '>>> Faker.seed(1000)'
        assert output[3] == '>>> for _ in range(5):'
        assert output[4] == "...     fake.bothify(text='???###')"
        assert output[5] == '...'
        for i in range(6, 11):
            assert output[i] == docstring._stringify_result(faker.bothify(text='???###'))
        faker.seed_instance(3210)
        assert output[11] == ''
        assert output[12] == '>>> Faker.seed(3210)'
        assert output[13] == '>>> for _ in range(5):'
        assert output[14] == "...     fake.bothify(letters='abcde')"
        assert output[15] == '...'
        for i in range(16, 21):
            assert output[i] == docstring._stringify_result(faker.bothify(letters='abcde'))
        faker.seed_instance(1234)
        assert output[21] == ''
        assert output[22] == '>>> Faker.seed(1234)'
        assert output[23] == '>>> for _ in range(20):'
        assert output[24] == "...     fake.bothify(text='???###', letters='abcde')"
        assert output[25] == '...'
        for i in range(26, 46):
            assert output[i] == docstring._stringify_result(faker.bothify(text='???###', letters='abcde'))
        calls = mock_warning.call_args_list
        assert len(calls) == 4
        (args, kwargs) = calls[0]
        assert len(args) == 1
        assert not kwargs
        assert args[0] == 'The section `:sample 1234jdbvhjdbygdvbhxjhx` is malformed and will be discarded.'
        (args, kwargs) = calls[1]
        assert len(args) == 1
        assert not kwargs
        assert args[0] == "Sample generation failed for method `bothify` with arguments `invalid_arg='value'`."
        (args, kwargs) = calls[2]
        assert len(args) == 1
        assert not kwargs
        assert args[0] == 'Invalid code elements detected. Sample generation will be skipped for method `bothify` with arguments `number=100**100**100`.'
        (args, kwargs) = calls[3]
        assert len(args) == 1
        assert not kwargs
        assert args[0] == "Sample generation failed for method `bothify` with arguments `abcd='abcd'`."