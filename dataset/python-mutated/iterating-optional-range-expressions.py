import pulumi
import pulumi_range as range
root = range.Root('root')
from_list_of_strings = []

def create_from_list_of_strings(range_body):
    if False:
        for i in range(10):
            print('nop')
    for range in [{'key': k, 'value': v} for [k, v] in enumerate(range_body)]:
        from_list_of_strings.append(range.Example(f"fromListOfStrings-{range['key']}", some_string=range['value']))
root.array_of_string.apply(create_from_list_of_strings)
from_map_of_strings = []

def create_from_map_of_strings(range_body):
    if False:
        i = 10
        return i + 15
    for range in [{'key': k, 'value': v} for [k, v] in enumerate(range_body)]:
        from_map_of_strings.append(range.Example(f"fromMapOfStrings-{range['key']}", some_string=f"{range['key']} {range['value']}"))
root.map_of_string.apply(create_from_map_of_strings)
from_computed_list_of_strings = []

def create_from_computed_list_of_strings(range_body):
    if False:
        while True:
            i = 10
    for range in [{'key': k, 'value': v} for [k, v] in enumerate(range_body)]:
        from_computed_list_of_strings.append(range.Example(f"fromComputedListOfStrings-{range['key']}", some_string=f"{range['key']} {range['value']}"))
pulumi.Output.all(root.map_of_string['hello'], root.map_of_string['world']).apply(create_from_computed_list_of_strings)
from_computed_for_expression = []

def create_from_computed_for_expression(range_body):
    if False:
        print('Hello World!')
    for range in [{'key': k, 'value': v} for [k, v] in enumerate(range_body)]:
        from_computed_for_expression.append(range.Example(f"fromComputedForExpression-{range['key']}", some_string=f"{range['key']} {range['value']}"))
pulumi.Output.all(array_of_string=root.array_of_string, map_of_string=root.map_of_string).apply(lambda resolved_outputs: create_from_computed_for_expression([resolved_outputs['map_of_string'][value] for value in resolved_outputs['array_of_string']]))