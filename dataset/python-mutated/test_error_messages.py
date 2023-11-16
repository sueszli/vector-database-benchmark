import re
import pytest
from dagster import DagsterInvalidDefinitionError, Dict, List, Noneable, Optional, op
from dagster._core.errors import DagsterInvalidConfigDefinitionError

def test_invalid_optional_in_config():
    if False:
        i = 10
        return i + 15
    with pytest.raises(DagsterInvalidDefinitionError, match=re.escape('You have passed an instance of DagsterType Int? to the config system')):

        @op(config_schema=Optional[int])
        def _op(_):
            if False:
                while True:
                    i = 10
            pass

def test_invalid_dict_call():
    if False:
        return 10
    with pytest.raises(TypeError, match=re.escape("'DagsterDictApi' object is not callable")):

        @op(config_schema=Dict({'foo': int}))
        def _op(_):
            if False:
                return 10
            pass

def test_list_in_config():
    if False:
        i = 10
        return i + 15
    with pytest.raises(DagsterInvalidDefinitionError, match=re.escape('Cannot use List in the context of config. Please use a python list (e.g. [int]) or dagster.Array (e.g. Array(int)) instead.')):

        @op(config_schema=List[int])
        def _op(_):
            if False:
                for i in range(10):
                    print('nop')
            pass

def test_invalid_list_element():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(DagsterInvalidDefinitionError, match=re.escape('Invalid type: dagster_type must be an instance of DagsterType or a Python type: ')):
        _ = List[Noneable(int)]

def test_non_scalar_key_map():
    if False:
        while True:
            i = 10
    with pytest.raises(DagsterInvalidConfigDefinitionError, match=re.escape('Map dict must have a scalar type as its only key.')):

        @op(config_schema={Noneable(int): str})
        def _op(_):
            if False:
                i = 10
                return i + 15
            pass