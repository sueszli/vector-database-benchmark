import sys
import unittest
from functools import partial
from typing import TYPE_CHECKING, Optional, Type, TypeVar, Union
from django import forms as df, test as dt
from django.contrib.staticfiles import testing as dst
from django.core.exceptions import ValidationError
from django.db import IntegrityError, models as dm
from hypothesis import reject, strategies as st
from hypothesis.errors import InvalidArgument
from hypothesis.extra.django._fields import from_field
from hypothesis.strategies._internal.utils import defines_strategy
if sys.version_info >= (3, 10):
    from types import EllipsisType as EllipsisType
elif TYPE_CHECKING:
    from builtins import ellipsis as EllipsisType
else:
    EllipsisType = type(Ellipsis)
ModelT = TypeVar('ModelT', bound=dm.Model)

class HypothesisTestCase:

    def setup_example(self):
        if False:
            return 10
        self._pre_setup()

    def teardown_example(self, example):
        if False:
            return 10
        self._post_teardown()

    def __call__(self, result=None):
        if False:
            while True:
                i = 10
        testMethod = getattr(self, self._testMethodName)
        if getattr(testMethod, 'is_hypothesis_test', False):
            return unittest.TestCase.__call__(self, result)
        else:
            return dt.SimpleTestCase.__call__(self, result)

class TestCase(HypothesisTestCase, dt.TestCase):
    pass

class TransactionTestCase(HypothesisTestCase, dt.TransactionTestCase):
    pass

class LiveServerTestCase(HypothesisTestCase, dt.LiveServerTestCase):
    pass

class StaticLiveServerTestCase(HypothesisTestCase, dst.StaticLiveServerTestCase):
    pass

@defines_strategy()
def from_model(model: Type[ModelT], /, **field_strategies: Union[st.SearchStrategy, EllipsisType]) -> st.SearchStrategy[ModelT]:
    if False:
        for i in range(10):
            print('nop')
    'Return a strategy for examples of ``model``.\n\n    .. warning::\n        Hypothesis creates saved models. This will run inside your testing\n        transaction when using the test runner, but if you use the dev console\n        this will leave debris in your database.\n\n    ``model`` must be an subclass of :class:`~django:django.db.models.Model`.\n    Strategies for fields may be passed as keyword arguments, for example\n    ``is_staff=st.just(False)``.  In order to support models with fields named\n    "model", this is a positional-only parameter.\n\n    Hypothesis can often infer a strategy based the field type and validators,\n    and will attempt to do so for any required fields.  No strategy will be\n    inferred for an :class:`~django:django.db.models.AutoField`, nullable field,\n    foreign key, or field for which a keyword\n    argument is passed to ``from_model()``.  For example,\n    a Shop type with a foreign key to Company could be generated with::\n\n        shop_strategy = from_model(Shop, company=from_model(Company))\n\n    Like for :func:`~hypothesis.strategies.builds`, you can pass\n    ``...`` (:obj:`python:Ellipsis`) as a keyword argument to infer a strategy for\n    a field which has a default value instead of using the default.\n    '
    if not issubclass(model, dm.Model):
        raise InvalidArgument(f'model={model!r} must be a subtype of Model')
    fields_by_name = {f.name: f for f in model._meta.concrete_fields}
    for (name, value) in sorted(field_strategies.items()):
        if value is ...:
            field_strategies[name] = from_field(fields_by_name[name])
    for (name, field) in sorted(fields_by_name.items()):
        if name not in field_strategies and (not field.auto_created) and (field.default is dm.fields.NOT_PROVIDED):
            field_strategies[name] = from_field(field)
    for field in field_strategies:
        if model._meta.get_field(field).primary_key:
            kwargs = {field: field_strategies.pop(field)}
            kwargs['defaults'] = st.fixed_dictionaries(field_strategies)
            return _models_impl(st.builds(model.objects.update_or_create, **kwargs))
    return _models_impl(st.builds(model.objects.get_or_create, **field_strategies))

@st.composite
def _models_impl(draw, strat):
    if False:
        print('Hello World!')
    'Handle the nasty part of drawing a value for models()'
    try:
        return draw(strat)[0]
    except IntegrityError:
        reject()

@defines_strategy()
def from_form(form: Type[df.Form], form_kwargs: Optional[dict]=None, **field_strategies: Union[st.SearchStrategy, EllipsisType]) -> st.SearchStrategy[df.Form]:
    if False:
        return 10
    'Return a strategy for examples of ``form``.\n\n    ``form`` must be an subclass of :class:`~django:django.forms.Form`.\n    Strategies for fields may be passed as keyword arguments, for example\n    ``is_staff=st.just(False)``.\n\n    Hypothesis can often infer a strategy based the field type and validators,\n    and will attempt to do so for any required fields.  No strategy will be\n    inferred for a disabled field or field for which a keyword argument\n    is passed to ``from_form()``.\n\n    This function uses the fields of an unbound ``form`` instance to determine\n    field strategies, any keyword arguments needed to instantiate the unbound\n    ``form`` instance can be passed into ``from_form()`` as a dict with the\n    keyword ``form_kwargs``. E.g.::\n\n        shop_strategy = from_form(Shop, form_kwargs={"company_id": 5})\n\n    Like for :func:`~hypothesis.strategies.builds`, you can pass\n    ``...`` (:obj:`python:Ellipsis`) as a keyword argument to infer a strategy for\n    a field which has a default value instead of using the default.\n    '
    form_kwargs = form_kwargs or {}
    if not issubclass(form, df.BaseForm):
        raise InvalidArgument(f'form={form!r} must be a subtype of Form')
    unbound_form = form(**form_kwargs)
    fields_by_name = {}
    for (name, field) in unbound_form.fields.items():
        if isinstance(field, df.MultiValueField):
            for (i, _field) in enumerate(field.fields):
                fields_by_name[f'{name}_{i}'] = _field
        else:
            fields_by_name[name] = field
    for (name, value) in sorted(field_strategies.items()):
        if value is ...:
            field_strategies[name] = from_field(fields_by_name[name])
    for (name, field) in sorted(fields_by_name.items()):
        if name not in field_strategies and (not field.disabled):
            field_strategies[name] = from_field(field)
    return _forms_impl(st.builds(partial(form, **form_kwargs), data=st.fixed_dictionaries(field_strategies)))

@st.composite
def _forms_impl(draw, strat):
    if False:
        print('Hello World!')
    'Handle the nasty part of drawing a value for from_form()'
    try:
        return draw(strat)
    except ValidationError:
        reject()