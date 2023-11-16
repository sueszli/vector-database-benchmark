import copy
import pickle
import random
from collections import OrderedDict
from unittest.mock import PropertyMock, patch
import pytest
from faker import Faker
from faker.config import DEFAULT_LOCALE
from faker.generator import Generator

class TestFakerProxyClass:
    """Test Faker proxy class"""

    def test_unspecified_locale(self):
        if False:
            print('Hello World!')
        fake = Faker()
        assert len(fake.locales) == 1
        assert len(fake.factories) == 1
        assert fake.locales[0] == DEFAULT_LOCALE

    def test_locale_as_string(self):
        if False:
            print('Hello World!')
        locale = 'en_US'
        fake = Faker(locale)
        assert len(fake.locales) == 1
        assert len(fake.factories) == 1
        assert fake.locales[0] == locale

    def test_locale_as_list(self):
        if False:
            while True:
                i = 10
        locale = ['en-US', 'en_PH', 'ja_JP', 'de-DE']
        expected = ['en_US', 'en_PH', 'ja_JP', 'de_DE']
        fake = Faker(locale)
        assert fake.locales == expected
        assert len(fake.factories) == len(expected)
        locale = ['en-US', 'en_PH', 'ja_JP', 'de-DE', 'ja-JP', 'de_DE', 'en-US'] * 3
        expected = ['en_US', 'en_PH', 'ja_JP', 'de_DE']
        fake = Faker(locale)
        assert fake.locales == expected
        assert len(fake.factories) == len(expected)

    def test_locale_as_list_invalid_value_type(self):
        if False:
            return 10
        locale = [1, 2]
        with pytest.raises(TypeError) as exc:
            Faker(locale)
        assert str(exc.value) == 'The locale "1" must be a string.'

    def test_locale_as_ordereddict(self):
        if False:
            for i in range(10):
                print('nop')
        locale = OrderedDict([('de_DE', 3), ('en-US', 2), ('en-PH', 1), ('ja_JP', 5)])
        fake = Faker(locale)
        assert fake.locales == ['de_DE', 'en_US', 'en_PH', 'ja_JP']
        assert len(fake.factories) == 4
        assert fake.weights == [3, 2, 1, 5]
        locale = OrderedDict([('de_DE', 3), ('en-US', 2), ('en-PH', 1), ('ja_JP', 5), ('de-DE', 4), ('ja-JP', 2), ('en-US', 1)])
        fake = Faker(locale)
        assert fake.locales == ['de_DE', 'en_US', 'en_PH', 'ja_JP']
        assert len(fake.factories) == 4
        assert fake.weights == [4, 1, 1, 2]

    def test_invalid_locale(self):
        if False:
            print('Hello World!')
        with pytest.raises(AttributeError):
            Faker('foo_Bar')
        with pytest.raises(AttributeError):
            Faker(['en_US', 'foo_Bar'])
        with pytest.raises(AttributeError):
            Faker(OrderedDict([('de_DE', 3), ('en-US', 2), ('en-PH', 1), ('foo_Bar', 5)]))

    def test_items(self):
        if False:
            i = 10
            return i + 15
        locale = ['de_DE', 'en-US', 'en-PH', 'ja_JP', 'de-DE', 'ja-JP', 'en-US']
        processed_locale = list({code.replace('-', '_') for code in locale})
        fake = Faker(locale)
        for (locale_name, factory) in fake.items():
            assert locale_name in processed_locale
            assert isinstance(factory, Generator)

    def test_dunder_getitem(self):
        if False:
            return 10
        locale = ['de_DE', 'en-US', 'en-PH', 'ja_JP']
        fake = Faker(locale)
        for code in locale:
            assert isinstance(fake[code], Generator)
        with pytest.raises(KeyError):
            fake['en_GB']

    def test_seed_classmethod(self):
        if False:
            i = 10
            return i + 15
        fake = Faker()
        with pytest.raises(TypeError):
            fake.seed(0)
        with patch('faker.generator.Generator.seed') as mock_seed:
            mock_seed.assert_not_called()
            Faker.seed(0)
            mock_seed.assert_called_once_with(0)

    def test_seed_class_locales(self):
        if False:
            i = 10
            return i + 15
        Faker.seed(2043)
        count = 5
        fake = Faker(['en_GB', 'fr_FR', 'en_IN'])
        first_list = [fake.name() for _ in range(count)]
        assert len(set(first_list)) == count
        Faker.seed(2043)
        fake = Faker(['en_GB', 'fr_FR', 'en_IN'])
        second_list = [fake.name() for _ in range(count)]
        assert first_list == second_list

    def test_seed_instance(self):
        if False:
            for i in range(10):
                print('nop')
        locale = ['de_DE', 'en-US', 'en-PH', 'ja_JP']
        fake = Faker(locale)
        with patch('faker.generator.Generator.seed_instance') as mock_seed_instance:
            mock_seed_instance.assert_not_called()
            fake.seed_instance(0)
            calls = mock_seed_instance.call_args_list
            assert len(calls) == 4
            for call in calls:
                (args, kwargs) = call
                assert args == (0,)
                assert kwargs == {}

    def test_seed_locale(self):
        if False:
            print('Hello World!')
        from faker.generator import random as shared_random_instance
        locale = ['de_DE', 'en-US', 'en-PH', 'ja_JP']
        fake = Faker(locale)
        states = {}
        for (locale, factory) in fake.items():
            states[locale] = factory.random.getstate()
        fake.seed_locale('en_US', 0)
        for (locale, factory) in fake.items():
            if locale == 'en_US':
                assert factory.random != shared_random_instance
                assert factory.random.getstate() != states[locale]
            else:
                assert factory.random == shared_random_instance
                assert factory.random.getstate() == states[locale]

    def test_single_locale_proxy_behavior(self):
        if False:
            i = 10
            return i + 15
        fake = Faker()
        internal_factory = fake.factories[0]
        for attr in fake.generator_attrs:
            assert getattr(fake, attr) == getattr(internal_factory, attr)
        tmp_random = fake.random
        assert internal_factory.random != 1
        fake.random = 1
        assert internal_factory.random == 1
        fake.random = tmp_random
        with patch('faker.proxy.Faker._select_factory') as mock_select_factory:
            mock_select_factory.assert_not_called()
            assert fake.name == internal_factory.name
            fake.name()
            mock_select_factory.assert_not_called()

    def test_multiple_locale_proxy_behavior(self):
        if False:
            return 10
        fake = Faker(['de-DE', 'en-US', 'en-PH', 'ja-JP'])
        for attr in fake.generator_attrs:
            with pytest.raises(NotImplementedError):
                getattr(fake, attr)
        with pytest.raises(NotImplementedError):
            random = fake.random
            random.seed(0)
        with pytest.raises(NotImplementedError):
            fake.random = 1

    def test_multiple_locale_caching_behavior(self):
        if False:
            i = 10
            return i + 15
        fake = Faker(['de_DE', 'en-US', 'en-PH', 'ja_JP'])
        with patch('faker.proxy.Faker._map_provider_method', wraps=fake._map_provider_method) as mock_map_method:
            mock_map_method.assert_not_called()
            assert not hasattr(fake, '_cached_name_mapping')
            fake.name()
            assert hasattr(fake, '_cached_name_mapping')
            mock_map_method.assert_called_once_with('name')
            with patch.object(Faker, '_cached_name_mapping', create=True, new_callable=PropertyMock) as mock_cached_map:
                mock_cached_map.return_value = ([fake['en_US']], [1])
                for _ in range(100):
                    fake.name()
                assert mock_cached_map.call_count == 200

    @patch('faker.proxy.Faker._select_factory_choice')
    @patch('faker.proxy.Faker._select_factory_distribution')
    def test_multiple_locale_factory_selection_no_weights(self, mock_factory_distribution, mock_factory_choice):
        if False:
            while True:
                i = 10
        fake = Faker(['de_DE', 'en-US', 'en-PH', 'ja_JP'])
        with patch('faker.proxy.Faker._select_factory', wraps=fake._select_factory) as mock_select_factory:
            mock_select_factory.assert_not_called()
            mock_factory_distribution.assert_not_called()
            mock_factory_choice.assert_not_called()
            fake.name()
            mock_select_factory.assert_called_once_with('name')
            mock_factory_distribution.assert_not_called()
            mock_factory_choice.assert_called_once_with(fake.factories)
            mock_select_factory.reset_mock()
            mock_factory_distribution.reset_mock()
            mock_factory_choice.reset_mock()
            fake.luzon_province()
            mock_select_factory.assert_called_with('luzon_province')
            mock_factory_distribution.assert_not_called()
            mock_factory_choice.assert_not_called()
            mock_select_factory.reset_mock()
            mock_factory_distribution.reset_mock()
            mock_factory_choice.reset_mock()
            fake.zipcode()
            mock_select_factory.assert_called_once_with('zipcode')
            mock_factory_distribution.assert_not_called()
            mock_factory_choice.assert_called_once_with([fake['en_US'], fake['ja_JP']])

    @patch('faker.proxy.Faker._select_factory_choice')
    @patch('faker.proxy.Faker._select_factory_distribution')
    def test_multiple_locale_factory_selection_with_weights(self, mock_factory_distribution, mock_factory_choice):
        if False:
            i = 10
            return i + 15
        locale = OrderedDict([('de_DE', 3), ('en-US', 2), ('en-PH', 1), ('ja_JP', 5)])
        fake = Faker(locale)
        mock_factory_distribution.assert_not_called()
        mock_factory_choice.assert_not_called()
        with patch('faker.proxy.Faker._select_factory', wraps=fake._select_factory) as mock_select_factory:
            fake.name()
            mock_select_factory.assert_called_once_with('name')
            mock_factory_distribution.assert_called_once_with(fake.factories, fake.weights)
            mock_factory_choice.assert_not_called()

    @patch('faker.proxy.Faker._select_factory_choice')
    @patch('faker.proxy.Faker._select_factory_distribution')
    def test_multiple_locale_factory_selection_single_provider(self, mock_factory_distribution, mock_factory_choice):
        if False:
            while True:
                i = 10
        locale = OrderedDict([('de_DE', 3), ('en-US', 2), ('en-PH', 1), ('ja_JP', 5)])
        fake = Faker(locale)
        with patch('faker.proxy.Faker._select_factory', wraps=fake._select_factory) as mock_select_factory:
            fake.luzon_province()
            mock_select_factory.assert_called_once_with('luzon_province')
            mock_factory_distribution.assert_not_called()
            mock_factory_choice.assert_not_called()

    @patch('faker.proxy.Faker._select_factory_choice')
    @patch('faker.proxy.Faker._select_factory_distribution')
    def test_multiple_locale_factory_selection_shared_providers(self, mock_factory_distribution, mock_factory_choice):
        if False:
            print('Hello World!')
        locale = OrderedDict([('de_DE', 3), ('en-US', 2), ('en-PH', 1), ('ja_JP', 5)])
        fake = Faker(locale)
        with patch('faker.proxy.Faker._select_factory', wraps=fake._select_factory) as mock_select_factory:
            fake.zipcode()
            mock_select_factory.assert_called_once_with('zipcode')
            mock_factory_distribution.assert_called_once_with([fake['en_US'], fake['ja_JP']], [2, 5])
            mock_factory_choice.assert_not_called()

    def test_multiple_locale_factory_selection_unsupported_method(self):
        if False:
            while True:
                i = 10
        fake = Faker(['en_US', 'en_PH'])
        with pytest.raises(AttributeError):
            fake.obviously_invalid_provider_method_a23f()

    @patch('random.Random.choice')
    @patch('random.Random.choices')
    def test_weighting_disabled_single_choice(self, mock_choices_fn, mock_choice_fn):
        if False:
            return 10
        fake = Faker(use_weighting=False)
        fake.first_name()
        mock_choice_fn.assert_called()
        mock_choices_fn.assert_not_called()

    @patch('random.Random.choice')
    @patch('random.Random.choices', wraps=random.Random().choices)
    def test_weighting_disabled_with_locales(self, mock_choices_fn, mock_choice_fn):
        if False:
            for i in range(10):
                print('nop')
        locale = OrderedDict([('de_DE', 3), ('en-US', 2), ('en-PH', 1), ('ja_JP', 5)])
        fake = Faker(locale, use_weighting=False)
        fake.first_name()
        mock_choices_fn.assert_called()
        mock_choice_fn.assert_called()

    @patch('random.Random.choice')
    @patch('random.Random.choices', wraps=random.Random().choices)
    def test_weighting_disabled_multiple_locales(self, mock_choices_fn, mock_choice_fn):
        if False:
            for i in range(10):
                print('nop')
        locale = OrderedDict([('de_DE', 3), ('en-US', 2), ('en-PH', 1), ('ja_JP', 5)])
        fake = Faker(locale, use_weighting=False)
        fake.first_name()
        mock_choices_fn.assert_called()
        mock_choice_fn.assert_called()

    @patch('random.Random.choice')
    @patch('random.Random.choices', wraps=random.Random().choices)
    def test_weighting_disabled_multiple_choices(self, mock_choices_fn, mock_choice_fn):
        if False:
            i = 10
            return i + 15
        fake = Faker(use_weighting=False)
        fake.uri_path(deep=3)
        assert mock_choices_fn.mock_calls[0][2]['k'] == 3
        assert mock_choices_fn.mock_calls[0][2]['weights'] is None
        mock_choice_fn.assert_not_called()

    @patch('random.Random.choice')
    @patch('random.Random.choices', wraps=random.Random().choices)
    def test_weighting_enabled_multiple_choices(self, mock_choices_fn, mock_choice_fn):
        if False:
            print('Hello World!')
        fake = Faker(use_weighting=True)
        fake.uri_path(deep=3)
        assert mock_choices_fn.mock_calls[0][2]['k'] == 3
        assert mock_choices_fn.mock_calls[0][2]['weights'] is None
        mock_choice_fn.assert_not_called()

    def test_dir_include_all_providers_attribute_in_list(self):
        if False:
            i = 10
            return i + 15
        fake = Faker(['en_US', 'en_PH'])
        expected = set(dir(Faker) + ['_factories', '_locales', '_factory_map', '_weights', '_unique_proxy', '_optional_proxy'])
        for factory in fake.factories:
            expected |= {attr for attr in dir(factory) if not attr.startswith('_')}
        expected = sorted(expected)
        attributes = dir(fake)
        assert attributes == expected

    def test_copy(self):
        if False:
            return 10
        fake = Faker('it_IT')
        fake2 = copy.deepcopy(fake)
        assert fake.locales == fake2.locales
        assert fake.locales is not fake2.locales

    def test_pickle(self):
        if False:
            i = 10
            return i + 15
        fake = Faker()
        pickled = pickle.dumps(fake)
        pickle.loads(pickled)