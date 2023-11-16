"""Variable subcommands."""
from __future__ import annotations
import json
import os
from json import JSONDecodeError
from sqlalchemy import select
from airflow.cli.simple_table import AirflowConsole
from airflow.cli.utils import print_export_output
from airflow.models import Variable
from airflow.utils import cli as cli_utils
from airflow.utils.cli import suppress_logs_and_warning
from airflow.utils.providers_configuration_loader import providers_configuration_loaded
from airflow.utils.session import create_session, provide_session

@suppress_logs_and_warning
@providers_configuration_loaded
def variables_list(args):
    if False:
        for i in range(10):
            print('nop')
    'Display all the variables.'
    with create_session() as session:
        variables = session.scalars(select(Variable)).all()
    AirflowConsole().print_as(data=variables, output=args.output, mapper=lambda x: {'key': x.key})

@suppress_logs_and_warning
@providers_configuration_loaded
def variables_get(args):
    if False:
        return 10
    'Display variable by a given name.'
    try:
        if args.default is None:
            var = Variable.get(args.key, deserialize_json=args.json)
            print(var)
        else:
            var = Variable.get(args.key, deserialize_json=args.json, default_var=args.default)
            print(var)
    except (ValueError, KeyError) as e:
        raise SystemExit(str(e).strip('\'"'))

@cli_utils.action_cli
@providers_configuration_loaded
def variables_set(args):
    if False:
        print('Hello World!')
    'Create new variable with a given name, value and description.'
    Variable.set(args.key, args.value, args.description, serialize_json=args.json)
    print(f'Variable {args.key} created')

@cli_utils.action_cli
@providers_configuration_loaded
def variables_delete(args):
    if False:
        for i in range(10):
            print('nop')
    'Delete variable by a given name.'
    Variable.delete(args.key)
    print(f'Variable {args.key} deleted')

@cli_utils.action_cli
@providers_configuration_loaded
@provide_session
def variables_import(args, session):
    if False:
        print('Hello World!')
    'Import variables from a given file.'
    if not os.path.exists(args.file):
        raise SystemExit('Missing variables file.')
    with open(args.file) as varfile:
        try:
            var_json = json.load(varfile)
        except JSONDecodeError:
            raise SystemExit('Invalid variables file.')
    suc_count = fail_count = 0
    skipped = set()
    action_on_existing = args.action_on_existing_key
    existing_keys = set()
    if action_on_existing != 'overwrite':
        existing_keys = set(session.scalars(select(Variable.key).where(Variable.key.in_(var_json))))
    if action_on_existing == 'fail' and existing_keys:
        raise SystemExit(f'Failed. These keys: {sorted(existing_keys)} already exists.')
    for (k, v) in var_json.items():
        if action_on_existing == 'skip' and k in existing_keys:
            skipped.add(k)
            continue
        try:
            Variable.set(k, v, serialize_json=not isinstance(v, str))
        except Exception as e:
            print(f'Variable import failed: {e!r}')
            fail_count += 1
        else:
            suc_count += 1
    print(f'{suc_count} of {len(var_json)} variables successfully updated.')
    if fail_count:
        print(f'{fail_count} variable(s) failed to be updated.')
    if skipped:
        print(f'The variables with these keys: {list(sorted(skipped))} were skipped because they already exists')

@providers_configuration_loaded
def variables_export(args):
    if False:
        return 10
    'Export all the variables to the file.'
    var_dict = {}
    with create_session() as session:
        qry = session.scalars(select(Variable))
        data = json.JSONDecoder()
        for var in qry:
            try:
                val = data.decode(var.val)
            except Exception:
                val = var.val
            var_dict[var.key] = val
    with args.file as varfile:
        json.dump(var_dict, varfile, sort_keys=True, indent=4)
        print_export_output('Variables', var_dict, varfile)