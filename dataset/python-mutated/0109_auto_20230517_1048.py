import pint
from django.core.exceptions import ValidationError
from django.db import migrations
import InvenTree.conversion

def update_template_units(apps, schema_editor):
    if False:
        print('Hello World!')
    'Update the units for each parameter template:\n\n    - Check if the units are valid\n    - Attempt to convert to valid units (if possible)\n    '
    PartParameterTemplate = apps.get_model('part', 'PartParameterTemplate')
    n_templates = PartParameterTemplate.objects.count()
    if n_templates == 0:
        return
    ureg = InvenTree.conversion.get_unit_registry()
    n_converted = 0
    invalid_units = set()
    for template in PartParameterTemplate.objects.all():
        if not template.units:
            continue
        if template.units == '%':
            template.units = 'percent'
            template.save()
            n_converted += 1
            continue
        try:
            ureg.Unit(template.units)
            continue
        except Exception:
            pass
        try:
            ureg.Unit(template.units.lower())
            print(f'Found unit match: {template.units} -> {template.units.lower()}')
            template.units = template.units.lower()
            template.save()
            n_converted += 1
            continue
        except Exception:
            pass
        found = False
        for unit in ureg:
            if unit.lower() == template.units.lower():
                print(f'Found unit match: {template.units} -> {unit}')
                template.units = str(unit)
                template.save()
                n_converted += 1
                found = True
                break
        if not found:
            print(f'warning: Could not find unit match for {template.units}')
            invalid_units.add(template.units)
    print(f'Updated units for {n_templates} parameter templates')
    if n_converted > 0:
        print(f' - Converted {n_converted} units')
    if len(invalid_units) > 0:
        print(f' - Found {len(invalid_units)} invalid units:')
        for unit in invalid_units:
            print(f'   - {unit}')

def convert_to_numeric_value(value: str, units: str):
    if False:
        while True:
            i = 10
    'Convert a value (with units) to a numeric value.\n\n    Defaults to zero if the value cannot be converted.\n    '
    result = None
    if units:
        try:
            result = InvenTree.conversion.convert_physical_value(value, units)
            result = float(result.magnitude)
        except Exception:
            pass
    else:
        try:
            result = float(value)
        except Exception:
            pass
    return result

def update_parameter_values(apps, schema_editor):
    if False:
        for i in range(10):
            print('nop')
    "Update the parameter values for all parts:\n\n    - Calculate the 'data_numeric' value for each parameter\n    - If the template has invalid units, we'll ignore\n    "
    PartParameter = apps.get_model('part', 'PartParameter')
    n_params = PartParameter.objects.count()
    for parameter in PartParameter.objects.all():
        try:
            parameter.data_numeric = convert_to_numeric_value(parameter.data, parameter.template.units)
            parameter.save()
        except Exception:
            pass
    if n_params > 0:
        print(f'Updated {n_params} parameter values')

class Migration(migrations.Migration):
    dependencies = [('part', '0108_auto_20230516_1334')]
    operations = [migrations.RunPython(update_template_units, reverse_code=migrations.RunPython.noop), migrations.RunPython(update_parameter_values, reverse_code=migrations.RunPython.noop)]