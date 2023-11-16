from unittest.mock import Mock, patch
import pytest
import streamlink.utils.l10n as l10n

class TestLocalization:

    @pytest.mark.parametrize(('locale', 'expected'), [('en_US', 'en_US'), ('ko_KR', 'ko_KR')])
    def test_valid(self, locale, expected):
        if False:
            return 10
        locale = l10n.Localization(locale)
        assert locale.language_code == expected

    @pytest.mark.parametrize('locale', ['enUS', 'eng_US', 'en_USA'])
    def test_invalid(self, locale):
        if False:
            print('Hello World!')
        with pytest.raises(LookupError):
            l10n.Localization(locale)

    @pytest.mark.parametrize('mock_getlocale', [Mock(return_value=(None, None)), Mock(return_value=('en_150', None)), Mock(side_effect=ValueError('unknown locale: foo_bar'))])
    def test_default(self, mock_getlocale: Mock):
        if False:
            i = 10
            return i + 15
        with patch('locale.getlocale', mock_getlocale):
            locale = l10n.Localization()
            assert locale.language_code == 'en_US'
            assert locale.equivalent(language='en', country='US')

    def test_setter(self):
        if False:
            for i in range(10):
                print('nop')
        with patch('locale.getlocale', Mock(return_value=(None, None))):
            locale = l10n.Localization()
            assert locale.language_code == 'en_US'
            assert locale.equivalent(language='en', country='US')
            locale.language_code = 'de_DE'
            assert locale.language_code == 'de_DE'
            assert locale.equivalent(language='de', country='DE')

class TestLocalizationEquality:

    @pytest.mark.parametrize(('language', 'country'), [(None, None), ('eng', None), ('en', None), ('en', 'CA'), ('en', 'CAN'), ('en', 'Canada')])
    def test_equivalent(self, language, country):
        if False:
            i = 10
            return i + 15
        locale = l10n.Localization('en_CA')
        assert locale.equivalent(language, country)

    @pytest.mark.parametrize('language', ['fra', 'fre'])
    def test_equivalent_remap(self, language):
        if False:
            return 10
        locale = l10n.Localization('fr_FR')
        assert locale.equivalent(language)

    @pytest.mark.parametrize(('language', 'country'), [('eng', None), ('en', None), ('en', 'US'), ('en', 'Canada'), ('en', 'ES'), ('en', 'Spain'), ('en', 'UNKNOWN'), ('UNKNOWN', 'Spain')])
    def test_not_equivalent(self, language, country):
        if False:
            for i in range(10):
                print('nop')
        locale = l10n.Localization('es_ES')
        assert not locale.equivalent(language, country)

class TestCountry:

    @pytest.mark.parametrize(('country', 'attr', 'expected'), [('USA', 'alpha2', 'US'), ('GB', 'alpha2', 'GB'), ('Canada', 'name', 'Canada')])
    def test_get_country(self, country, attr, expected):
        if False:
            for i in range(10):
                print('nop')
        assert getattr(l10n.Localization.get_country(country), attr) == expected

    @pytest.mark.parametrize('country', ['XE', 'XEX', 'Nowhere'])
    def test_get_country_miss(self, country):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(LookupError):
            l10n.Localization.get_country(country)

    def test_hash(self):
        if False:
            for i in range(10):
                print('nop')
        one = l10n.Country('DE', 'DEU', '276', 'Germany', 'Federal Republic of Germany')
        two = l10n.Country('DE', 'DEU', '276', 'Germany', 'Federal Republic of Germany')
        mapping = {one: '1', two: '2'}
        assert one is not two
        assert mapping[one] is mapping[two]

    def test_country_compare(self):
        if False:
            return 10
        assert l10n.Country('AA', 'AAA', '001', 'Test') == l10n.Country('AA', 'AAA', '001', 'Test')

    def test_country_str(self):
        if False:
            for i in range(10):
                print('nop')
        assert str(l10n.Localization.get_country('Germany')) == "Country('DE', 'DEU', '276', 'Germany', official_name='Federal Republic of Germany')"

class TestLanguage:

    @pytest.mark.parametrize(('language', 'attr', 'expected'), [('en', 'alpha3', 'eng'), ('fra', 'bibliographic', 'fre'), ('fre', 'alpha3', 'fra'), ('gre', 'bibliographic', 'gre')])
    def test_get_language(self, language, attr, expected):
        if False:
            print('Hello World!')
        assert getattr(l10n.Localization.get_language(language), attr) == expected

    @pytest.mark.parametrize('language', ['00', '000', '0000'])
    def test_get_language_miss(self, language):
        if False:
            print('Hello World!')
        with pytest.raises(LookupError):
            l10n.Localization.get_language(language)

    def test_hash(self):
        if False:
            for i in range(10):
                print('nop')
        one = l10n.Language('de', 'deu', 'German', 'ger')
        two = l10n.Language('de', 'deu', 'German', 'ger')
        mapping = {one: '1', two: '2'}
        assert one is not two
        assert mapping[one] is mapping[two]

    def test_language_compare(self):
        if False:
            while True:
                i = 10
        assert l10n.Language('AA', 'AAA', 'Test') == l10n.Language('AA', None, 'Test')
        assert l10n.Language('BB', 'BBB', 'Test') != l10n.Language('AA', None, 'Test')

    def test_language_str(self):
        if False:
            while True:
                i = 10
        assert str(l10n.Localization.get_language('German')) == "Language('de', 'deu', 'German', bibliographic='ger')"

    def test_language_a3_no_a2(self):
        if False:
            for i in range(10):
                print('nop')
        lang = l10n.Localization.get_language('des')
        assert lang.alpha2 == ''
        assert lang.alpha3 == 'des'
        assert lang.name == 'Desano'
        assert lang.bibliographic == ''

    @pytest.mark.parametrize('language', ['en', 'eng', 'English'])
    def test_language_en(self, language):
        if False:
            return 10
        lang = l10n.Localization.get_language(language)
        assert lang.alpha2 == 'en'
        assert lang.alpha3 == 'eng'
        assert lang.name == 'English'
        assert lang.bibliographic == ''