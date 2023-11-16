from hypothesis import given
from hypothesis.extra.django import TestCase, from_form, register_field_strategy
from hypothesis.strategies import booleans, sampled_from
from tests.django.toystore.forms import BasicFieldForm, BroadBooleanField, ChoiceFieldForm, CustomerForm, DynamicForm, EmailFieldForm, InternetProtocolForm, ManyMultiValueForm, ManyNumericsForm, ManyTimesForm, OddFieldsForm, RegexFieldForm, ShortStringForm, SlugFieldForm, TemporalFieldForm, URLFieldForm, UsernameForm, UUIDFieldForm, WithValidatorsForm
register_field_strategy(BroadBooleanField, booleans() | sampled_from(['1', '0', 'True', 'False']))

class TestGetsBasicForms(TestCase):

    @given(from_form(CustomerForm))
    def test_valid_customer(self, customer_form):
        if False:
            return 10
        self.assertTrue(customer_form.is_valid())

    @given(from_form(ManyNumericsForm))
    def test_valid_numerics(self, numerics_form):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(numerics_form.is_valid())

    @given(from_form(ManyTimesForm))
    def test_valid_times(self, times_form):
        if False:
            while True:
                i = 10
        self.assertTrue(times_form.is_valid())

    @given(from_form(OddFieldsForm))
    def test_valid_odd_fields(self, odd_form):
        if False:
            return 10
        self.assertTrue(odd_form.is_valid())

    def test_dynamic_form(self):
        if False:
            return 10
        for field_count in range(2, 7):

            @given(from_form(DynamicForm, form_kwargs={'field_count': field_count}))
            def _test(dynamic_form):
                if False:
                    i = 10
                    return i + 15
                self.assertTrue(dynamic_form.is_valid())
            _test()

    @given(from_form(BasicFieldForm))
    def test_basic_fields_form(self, basic_field_form):
        if False:
            return 10
        self.assertTrue(basic_field_form.is_valid())

    @given(from_form(TemporalFieldForm))
    def test_temporal_fields_form(self, time_field_form):
        if False:
            i = 10
            return i + 15
        self.assertTrue(time_field_form.is_valid())

    @given(from_form(EmailFieldForm))
    def test_email_field_form(self, email_field_form):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(email_field_form.is_valid())

    @given(from_form(SlugFieldForm))
    def test_slug_field_form(self, slug_field_form):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(slug_field_form.is_valid())

    @given(from_form(URLFieldForm))
    def test_url_field_form(self, url_field_form):
        if False:
            while True:
                i = 10
        self.assertTrue(url_field_form.is_valid())

    @given(from_form(RegexFieldForm))
    def test_regex_field_form(self, regex_field_form):
        if False:
            i = 10
            return i + 15
        self.assertTrue(regex_field_form.is_valid())

    @given(from_form(UUIDFieldForm))
    def test_uuid_field_form(self, uuid_field_form):
        if False:
            while True:
                i = 10
        self.assertTrue(uuid_field_form.is_valid())

    @given(from_form(ChoiceFieldForm))
    def test_choice_fields_form(self, choice_field_form):
        if False:
            print('Hello World!')
        self.assertTrue(choice_field_form.is_valid())

    @given(from_form(InternetProtocolForm))
    def test_ip_fields_form(self, ip_field_form):
        if False:
            print('Hello World!')
        self.assertTrue(ip_field_form.is_valid())

    @given(from_form(ManyMultiValueForm, form_kwargs={'subfield_count': 2}))
    def test_many_values_in_multi_value_field(self, many_multi_value_form):
        if False:
            i = 10
            return i + 15
        self.assertTrue(many_multi_value_form.is_valid())

    @given(from_form(ManyMultiValueForm, form_kwargs={'subfield_count': 105}))
    def test_excessive_values_in_multi_value_field(self, excessive_form):
        if False:
            return 10
        self.assertTrue(excessive_form.is_valid())

    @given(from_form(ShortStringForm))
    def test_short_string_form(self, short_string_form):
        if False:
            while True:
                i = 10
        self.assertTrue(short_string_form.is_valid())

    @given(from_form(WithValidatorsForm))
    def test_tight_validators_form(self, x):
        if False:
            print('Hello World!')
        self.assertTrue(1 <= x.data['_int_one_to_five'] <= 5)
        self.assertTrue(1 <= x.data['_decimal_one_to_five'] <= 5)
        self.assertTrue(1 <= x.data['_float_one_to_five'] <= 5)
        self.assertTrue(5 <= len(x.data['_string_five_to_ten']) <= 10)

    @given(from_form(UsernameForm))
    def test_username_form(self, username_form):
        if False:
            print('Hello World!')
        self.assertTrue(username_form.is_valid())

    @given(from_form(UsernameForm))
    def test_read_only_password_hash_field_form(self, password_form):
        if False:
            while True:
                i = 10
        self.assertTrue(password_form.is_valid())