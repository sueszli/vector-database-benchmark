"""CheckResult serialization tests."""
import json
import typing as t
import pandas as pd
import wandb
from hamcrest import all_of, assert_that, calling, contains_exactly, contains_string, equal_to, greater_than, has_entries, has_item, has_length, has_property, instance_of, matches_regexp, only_contains, raises, starts_with
from IPython.display import Image
from ipywidgets import HTML, Tab, VBox
from pandas.io.formats.style import Styler
from plotly.basedatatypes import BaseFigure
from deepchecks.core.check_result import DisplayMap
from deepchecks.core.serialization.check_result.html import CheckResultSerializer as HtmlSerializer
from deepchecks.core.serialization.check_result.html import DisplayItemsHandler as HtmlDisplayItemsHandler
from deepchecks.core.serialization.check_result.ipython import CheckResultSerializer as IPythonSerializer
from deepchecks.core.serialization.check_result.ipython import DisplayItemsHandler as IPythonDisplayItemsHandler
from deepchecks.core.serialization.check_result.json import CheckResultSerializer as JsonSerializer
from deepchecks.core.serialization.check_result.json import DisplayItemsHandler as JsonDisplayItemsHandler
from deepchecks.core.serialization.check_result.junit import CheckResultSerializer as JunitSerializer
from deepchecks.core.serialization.check_result.wandb import CheckResultSerializer as WandbSerializer
from deepchecks.core.serialization.check_result.widget import CheckResultSerializer as WidgetSerializer
from deepchecks.core.serialization.check_result.widget import DisplayItemsHandler as WidgetDisplayItemsHandler
from deepchecks.core.serialization.common import plotlyjs_script
from deepchecks.utils.strings import get_random_string
from tests.common import DummyCheck, create_check_result, create_check_result_display, instance_of_ipython_formatter

def test_html_serializer_initialization():
    if False:
        print('Hello World!')
    serializer = HtmlSerializer(create_check_result())

def test_html_serializer_initialization_with_incorrect_type_of_value():
    if False:
        for i in range(10):
            print('nop')
    assert_that(calling(HtmlSerializer).with_args(5), raises(TypeError, 'Expected "CheckResult" but got "int"'))

def test_html_serialization():
    if False:
        return 10
    result = create_check_result()
    output = HtmlSerializer(result).serialize()
    assert_that(output, all_of(instance_of(str), has_length(greater_than(0)), contains_string(result.get_header()), contains_string(t.cast(str, DummyCheck.__doc__))))

def test_html_serialization_with_empty__check_sections__parameter():
    if False:
        for i in range(10):
            print('nop')
    result = create_check_result()
    assert_that(calling(HtmlSerializer(result).serialize).with_args(check_sections=[]), raises(ValueError, 'include parameter cannot be empty'))

def test_html_serialization_with__output_id__parameter():
    if False:
        for i in range(10):
            print('nop')
    result = create_check_result()
    output_id = get_random_string(n=25)
    output = HtmlSerializer(result).serialize(output_id=output_id)
    assert_that(output, all_of(instance_of(str), has_length(greater_than(0)), contains_string(output_id)))

def test_check_result_without_display_and_conditions_into_html_serialization():
    if False:
        for i in range(10):
            print('nop')
    check_result_without_conditions = create_check_result(include_conditions=False, include_display=False)
    output_without_conditions = HtmlSerializer(check_result_without_conditions).serialize()
    assert_that(output_without_conditions, all_of(instance_of(str), has_length(greater_than(0))))
    full_check_result = create_check_result()
    full_output = HtmlSerializer(full_check_result).serialize()
    assert_that(full_output, all_of(instance_of(str), has_length(greater_than(0))))
    assert_that(len(full_output) > len(output_without_conditions))

def test_html_serialization_with_plotply_activation_script():
    if False:
        while True:
            i = 10
    result = create_check_result()
    output = HtmlSerializer(result).serialize()
    assert_that(output, all_of(instance_of(str), has_length(greater_than(0)), starts_with(plotlyjs_script())))

def test_html_serialization_to_full_html_page():
    if False:
        while True:
            i = 10
    result = create_check_result()
    output = HtmlSerializer(result).serialize(full_html=True)
    whitespace = '[\\s]*'
    anything = '([\\s\\S\\d\\D\\w\\W]*)'
    regexp = f'^{whitespace}{anything}<html>{whitespace}<head><meta charset="utf-8"\\/><\\/head>{whitespace}<body{anything}>{anything}<\\/body>{whitespace}<\\/html>{whitespace}$'
    assert_that(output, all_of(instance_of(str), matches_regexp(regexp)))

def test_display_map_serialization_to_html():
    if False:
        print('Hello World!')
    html_section = HtmlDisplayItemsHandler.handle_display(display=[DisplayMap(a=create_check_result_display())], include_header=False, include_trailing_link=False)
    assert_that(html_section, all_of(instance_of(list), contains_exactly(is_display_map_sections('a'))))

def test_nested_display_map_serialization_to_html():
    if False:
        for i in range(10):
            print('nop')
    html_section = HtmlDisplayItemsHandler.handle_display(display=[DisplayMap(a=create_check_result_display(), b=[DisplayMap(a=create_check_result_display())])], include_header=False, include_trailing_link=False)
    assert_that(html_section, all_of(instance_of(list), contains_exactly(is_display_map_sections('a', 'b'))))

def is_display_map_sections(*section_names):
    if False:
        i = 10
        return i + 15
    assert len(section_names) != 0
    patterns = []
    for name in section_names:
        patterns.append(f'<details>[\\s]*<summary>[\\s]*<strong>{name}<\\/strong>[\\s]*<\\/summary>[\\s]*<div([\\s\\S\\d\\D\\w\\W]*)>([\\s\\S\\d\\D\\w\\W]*)<\\/div>[\\s]*<\\/details>')
    pattern = '[\\s]*'.join(patterns)
    pattern = f'^[\\s]*{pattern}[\\s]*$'
    return all_of(instance_of(str), matches_regexp(pattern))

def test_ipython_serializer_initialization():
    if False:
        while True:
            i = 10
    serializer = IPythonSerializer(create_check_result())

def test_ipython_serializer_initialization_with_incorrect_type_of_value():
    if False:
        print('Hello World!')
    assert_that(calling(IPythonSerializer).with_args([]), raises(TypeError, 'Expected "CheckResult" but got "list"'))

def test_ipython_serialization():
    if False:
        i = 10
        return i + 15
    result = create_check_result()
    output = IPythonSerializer(result).serialize()
    assert_that(output, all_of(instance_of(list), has_length(greater_than(0)), only_contains(instance_of_ipython_formatter()), has_item(instance_of(Image)), has_item(instance_of(BaseFigure))))

def test_ipython_serialization_with_empty__check_sections__parameter():
    if False:
        i = 10
        return i + 15
    result = create_check_result()
    assert_that(calling(IPythonSerializer(result).serialize).with_args(check_sections=[]), raises(ValueError, 'include parameter cannot be empty'))

def test_display_map_serialization_to_list_of_ipython_formatters():
    if False:
        while True:
            i = 10
    formatters = IPythonDisplayItemsHandler.handle_display(display=[DisplayMap(a=create_check_result_display())], include_header=False, include_trailing_link=False)
    assert_that(formatters, all_of(instance_of(list), has_length(equal_to(6)), only_contains(instance_of_ipython_formatter())))

def test_nested_display_map_serialization_to_list_of_ipython_formatters():
    if False:
        while True:
            i = 10
    formatters = IPythonDisplayItemsHandler.handle_display(display=[DisplayMap(a=create_check_result_display(), b=[DisplayMap(a=create_check_result_display())])], include_header=False, include_trailing_link=False)
    assert_that(formatters, all_of(instance_of(list), has_length(equal_to(13)), only_contains(instance_of_ipython_formatter())))

def test_json_serializer_initialization():
    if False:
        return 10
    serializer = JsonSerializer(create_check_result())

def test_json_serializer_initialization_with_incorrect_type_of_value():
    if False:
        while True:
            i = 10
    assert_that(calling(JsonSerializer).with_args(dict()), raises(TypeError, 'Expected "CheckResult" but got "dict"'))

def test_json_serialization():
    if False:
        return 10
    check_result = create_check_result()
    output = JsonSerializer(check_result).serialize()
    assert_json_output(output, check_result)

def test_check_result_without_conditions_and_display_into_json_serialization():
    if False:
        print('Hello World!')
    check_result = create_check_result(include_conditions=False, include_display=False)
    output = JsonSerializer(check_result).serialize()
    assert_json_output(output, check_result, with_conditions_table=False, with_display=False)

def test_display_map_serialization_to_json():
    if False:
        return 10
    output = JsonDisplayItemsHandler.handle_display(display=[DisplayMap(a=create_check_result_display())])
    assert_that(output, all_of(instance_of(list), only_contains(has_entries({'type': equal_to('displaymap'), 'payload': has_entries({'a': all_of(instance_of(list), has_length(equal_to(5)))})}))))

def test_nested_display_map_serialization_to_json():
    if False:
        for i in range(10):
            print('nop')
    output = JsonDisplayItemsHandler.handle_display(display=[DisplayMap(a=create_check_result_display(), b=[DisplayMap(a=create_check_result_display())])])
    assert_that(output, all_of(instance_of(list), contains_exactly(serialized_to_json_displaymap(has_entries({'a': all_of(instance_of(list), has_length(equal_to(5))), 'b': all_of(instance_of(list), contains_exactly(serialized_to_json_displaymap(has_entries({'a': all_of(instance_of(list), has_length(equal_to(5)))}))))})))))

def serialized_to_json_displaymap(entries_matcher):
    if False:
        print('Hello World!')
    return has_entries({'type': equal_to('displaymap'), 'payload': entries_matcher})

def assert_json_output(output, check_result, with_conditions_table=True, with_display=True):
    if False:
        return 10
    assert_that(json.loads(json.dumps(output)) == output)
    assert_that(output, all_of(instance_of(dict), has_entries({'check': instance_of(dict), 'value': equal_to(check_result.value), 'header': equal_to(check_result.header), 'conditions_results': all_of(instance_of(list), has_length(greater_than(0)) if with_conditions_table is True else has_length(equal_to(0))), 'display': all_of(instance_of(list), has_length(greater_than(0)) if with_display is True else has_length(equal_to(0)))})))
    if with_display is True:
        for (index, it) in enumerate(check_result.display):
            assert_that(output['display'][index], is_serialized_to_json_display_item(type(it)))

def is_serialized_to_json_display_item(item_type):
    if False:
        while True:
            i = 10
    if issubclass(item_type, (pd.DataFrame, Styler)):
        return has_entries({'type': equal_to('dataframe'), 'payload': instance_of(list)})
    elif issubclass(item_type, str):
        return has_entries({'type': equal_to('html'), 'payload': instance_of(str)})
    elif issubclass(item_type, BaseFigure):
        return has_entries({'type': equal_to('plotly'), 'payload': instance_of(str)})
    elif issubclass(item_type, t.Callable):
        return has_entries({'type': equal_to('images'), 'payload': all_of(has_length(greater_than(0)))})
    elif issubclass(item_type, DisplayMap):
        return has_entries({'type': equal_to('displaymap'), 'payload': instance_of(dict)})
    else:
        raise TypeError(f'Unknown display item type {type(item_type)}')

def test_junit_serializer_initialization():
    if False:
        for i in range(10):
            print('nop')
    serializer = JunitSerializer(create_check_result())

def test_junit_serializer_initialization_with_incorrect_type_of_value():
    if False:
        i = 10
        return i + 15
    assert_that(calling(JunitSerializer).with_args(dict()), raises(TypeError, 'Expected "CheckResult" but got "dict"'))

def test_junit_serialization():
    if False:
        return 10
    check_result = create_check_result()
    output = JunitSerializer(check_result).serialize()
    assert_that(list(output.attrib.keys()), ['classname', 'name', 'time'])
    assert_that(output.tag, 'testcase')

def test_wandb_serializer_initialization():
    if False:
        i = 10
        return i + 15
    serializer = WandbSerializer(create_check_result())

def test_wandb_serializer_initialization_with_incorrect_type_of_value():
    if False:
        while True:
            i = 10
    assert_that(calling(WandbSerializer).with_args(dict()), raises(TypeError, 'Expected "CheckResult" but got "dict"'))

def test_wandb_serialization():
    if False:
        return 10
    check_result = create_check_result()
    output = WandbSerializer(check_result).serialize()
    assert_that(output, wandb_output_assertion(check_result))

def test_check_result_without_conditions_serialization_to_wandb():
    if False:
        i = 10
        return i + 15
    check_result = create_check_result(include_conditions=False)
    output = WandbSerializer(check_result).serialize()
    assert_that(output, wandb_output_assertion(check_result, with_conditions_table=False))

def test_check_result_without_conditions_and_display_serialization_to_wandb():
    if False:
        for i in range(10):
            print('nop')
    check_result = create_check_result(include_conditions=False, include_display=False)
    output = WandbSerializer(check_result).serialize()
    assert_that(output, wandb_output_assertion(check_result, with_conditions_table=False, with_display=False))

def wandb_output_assertion(check_result, with_conditions_table=True, with_display=True):
    if False:
        while True:
            i = 10
    entries = {f'{check_result.header}/results': instance_of(wandb.Table)}
    if with_display is True:
        for (index, it) in enumerate(check_result.display):
            if isinstance(it, (pd.DataFrame, Styler)):
                entries[f'{check_result.header}/item-{index}-table'] = instance_of(wandb.Table)
            elif isinstance(it, str):
                entries[f'{check_result.header}/item-{index}-html'] = instance_of(wandb.Html)
            elif isinstance(it, BaseFigure):
                entries[f'{check_result.header}/item-{index}-plot'] = instance_of(wandb.Plotly)
            elif callable(it):
                entries[f'{check_result.header}/item-{index}-figure'] = instance_of(wandb.Image)
            elif isinstance(it, DisplayMap):
                pass
            else:
                raise TypeError(f'Unknown display item type {type(it)}')
    if with_conditions_table is True:
        entries[f'{check_result.header}/conditions table'] = instance_of(wandb.Table)
    return all_of(instance_of(dict), has_entries(entries))

def test_widget_serializer_initialization():
    if False:
        for i in range(10):
            print('nop')
    serializer = WandbSerializer(create_check_result())

def test_widget_serializer_initialization_with_incorrect_type_of_value():
    if False:
        i = 10
        return i + 15
    assert_that(calling(WandbSerializer).with_args(dict()), raises(TypeError, 'Expected "CheckResult" but got "dict"'))

def test_widget_serialization():
    if False:
        return 10
    check_result = create_check_result()
    output = WidgetSerializer(check_result).serialize()
    assert_widget_output(output, check_result)

def test_widget_serialization_without_conditions_section():
    if False:
        print('Hello World!')
    check_result = create_check_result()
    output = WidgetSerializer(check_result).serialize(check_sections=['additional-output'])
    assert_widget_output(output, check_result, with_conditions_section=False, with_display_section=True)

def test_widget_serialization_without_display_section():
    if False:
        i = 10
        return i + 15
    check_result = create_check_result()
    output = WidgetSerializer(check_result).serialize(check_sections=['condition-table'])
    assert_widget_output(output, check_result, with_conditions_section=True, with_display_section=False)

def test_widget_serialization_with_empty__check_sections__parameter():
    if False:
        i = 10
        return i + 15
    result = create_check_result()
    assert_that(calling(WidgetSerializer(result).serialize).with_args(check_sections=[]), raises(ValueError, 'include parameter cannot be empty'))

def test_display_map_serialization_to_widget():
    if False:
        i = 10
        return i + 15
    display_items = create_check_result_display()
    displaymap = DisplayMap(a=display_items)
    output = WidgetDisplayItemsHandler.handle_display(display=[displaymap], include_header=False, include_trailing_link=False)
    assert_that(output, contains_exactly(serialized_to_widget_displaymap(displaymap=displaymap, display_items=display_items)))

def test_nested_display_map_serialization_to_widget():
    if False:
        while True:
            i = 10
    display_items = create_check_result_display()
    inner_displaymap = DisplayMap(a=display_items)
    outer_displaymap = DisplayMap(a=display_items, b=[inner_displaymap])
    output = WidgetDisplayItemsHandler.handle_display(display=[outer_displaymap], include_header=False, include_trailing_link=False)
    assert_that(output, contains_exactly(serialized_to_widget_displaymap(displaymap=outer_displaymap, tabs=[has_length(len(display_items)), contains_exactly(serialized_to_widget_displaymap(displaymap=inner_displaymap, display_items=display_items))])))

def serialized_to_widget_displaymap(displaymap: DisplayMap, display_items: t.Optional[t.List[t.Any]]=None, tabs: t.Optional[t.List[t.Any]]=None):
    if False:
        print('Hello World!')
    if display_items is not None:
        tabs_matcher = [all_of(instance_of(VBox), has_property('children', has_length(len(display_items)))) for _ in range(len(displaymap))]
    elif tabs is not None:
        tabs_matcher = [all_of(instance_of(VBox), has_property('children', m)) for m in tabs]
    else:
        raise ValueError('At least one of the parameters must be provided - [display_items, tabs]')
    return all_of(instance_of(VBox), has_property('children', contains_exactly(instance_of(HTML), all_of(instance_of(Tab), has_property('children', all_of(contains_exactly(*tabs_matcher)))))))

def assert_widget_output(output, check_result, with_conditions_section=True, with_display_section=True):
    if False:
        while True:
            i = 10
    children_count = 4
    header_section_index = 0
    summary_section_index = 1
    conditions_section_index = 2
    display_section_index = 3
    if with_conditions_section is False and with_display_section is False:
        children_count = 2
    elif with_conditions_section is False and with_display_section is True:
        children_count = 3
        display_section_index = 2
    elif with_conditions_section is False or with_display_section is False:
        children_count = 3
    assert_that(output, all_of(instance_of(VBox), has_property('children', all_of(instance_of(tuple), has_length(children_count)))))
    assert_that(output.children[header_section_index], all_of(instance_of(HTML), has_property('value', all_of(instance_of(str), contains_string(check_result.header)))))
    assert_that(output.children[summary_section_index], all_of(instance_of(HTML), has_property('value', all_of(instance_of(str), contains_string(t.cast(str, DummyCheck.__doc__))))))
    if with_conditions_section is True:
        assert_that(output.children[conditions_section_index], instance_of(HTML))
    if with_display_section is True:
        assert_that(output.children[display_section_index], all_of(instance_of(VBox), has_property('children', all_of(instance_of(tuple), has_length(equal_to(len(check_result.display) + 1))))))