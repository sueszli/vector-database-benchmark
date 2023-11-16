import sys
from itertools import chain
from metaflow import namespace
from metaflow.client import Flow, Run
from metaflow.current import current
from metaflow.util import resolve_identity
from metaflow.exception import CommandException, MetaflowNotFound, MetaflowInternalError
from metaflow.exception import MetaflowNamespaceMismatch
from metaflow._vendor import click

def _print_tags_for_runs_by_groups(obj, system_tags_by_group, all_tags_by_group, by_tag):
    if False:
        for i in range(10):
            print('nop')
    header = ['', 'System tags and *tags*; only *tags* can be modified', '--------------------------------------------------------', '']
    obj.echo('\n'.join(header), err=False)
    groups = sorted(set(chain(all_tags_by_group.keys(), system_tags_by_group.keys())))
    for group in groups:
        all_tags_in_group = all_tags_by_group.get(group, set())
        system_tags_in_group = system_tags_by_group.get(group, set())
        if by_tag:
            if all_tags_in_group:
                _print_tags_for_group_by_tag(obj, group, all_tags_in_group, is_system=False)
            if system_tags_in_group:
                _print_tags_for_group_by_tag(obj, group, system_tags_in_group, is_system=True)
        else:
            _print_tags_for_group(obj, group, system_tags_in_group, all_tags_in_group, skip_per_group_header=len(all_tags_by_group) == 1)

def _print_tags_for_group(obj, group, system_tags, all_tags, skip_per_group_header=False):
    if False:
        while True:
            i = 10
    if not obj.is_quiet:
        max_length = max([len(t) for t in all_tags] if all_tags else [0])
        all_tags = sorted(all_tags)
        if sys.version_info[0] < 3:
            all_tags = [t if t in system_tags else '*%s*' % t.encode('utf-8') for t in all_tags]
        else:
            all_tags = [t if t in system_tags else '*%s*' % t for t in all_tags]
        num_tags = len(all_tags)
        column_count = 124 // (max_length + 4)
        if column_count == 0:
            column_count = 1
            words_per_column = num_tags
        else:
            words_per_column = (num_tags + column_count - 1) // column_count
        addl_tags = words_per_column * column_count - num_tags
        if addl_tags > 0:
            all_tags.extend([' ' for _ in range(addl_tags)])
        lines = []
        if not skip_per_group_header:
            lines.append('For run %s' % group)
        for i in range(words_per_column):
            line_values = [all_tags[i + j * words_per_column] for j in range(column_count)]
            length_values = [max_length + 2 if l[0] == '*' else max_length for l in line_values]
            formatter = '    '.join(['{:<%d}' % l for l in length_values])
            lines.append(formatter.format(*line_values))
        lines.append('')
        obj.echo('\n'.join(lines), err=False)
    else:
        obj.echo_always(' ; '.join([group, ','.join(sorted(system_tags)), ','.join(sorted(all_tags.difference(system_tags)))]))

def _print_tags_for_group_by_tag(obj, group, runs, is_system):
    if False:
        while True:
            i = 10
    if not obj.is_quiet:
        max_length = max([len(t) for t in runs] if runs else [0])
        all_runs = sorted(runs)
        num_runs = len(all_runs)
        column_count = 124 // (max_length + 4)
        if column_count == 0:
            column_count = 1
            words_per_column = num_runs
        else:
            words_per_column = (num_runs + column_count - 1) // column_count
        addl_runs = words_per_column * column_count - num_runs
        if addl_runs > 0:
            all_runs.extend([' ' for _ in range(addl_runs)])
        lines = []
        lines.append('For tag %s' % (group if is_system else '*%s*' % group))
        for i in range(words_per_column):
            line_values = [all_runs[i + j * words_per_column] for j in range(column_count)]
            length_values = [max_length for l in line_values]
            formatter = '    '.join(['{:<%d}' % l for l in length_values])
            lines.append(formatter.format(*line_values))
        lines.append('')
        obj.echo('\n'.join(lines), err=False)
    else:
        obj.echo_always(' ; '.join(['system:%s' % group if is_system else 'user:%s' % group, ','.join(runs)]))

def _print_tags_for_one_run(obj, run):
    if False:
        return 10
    system_tags = {run.pathspec: run.system_tags}
    all_tags = {run.pathspec: run.tags}
    return _print_tags_for_runs_by_groups(obj, system_tags, all_tags, by_tag=False)

def _get_client_run_obj(obj, run_id, user_namespace):
    if False:
        print('Hello World!')
    flow_name = obj.flow.name
    try:
        namespace(user_namespace)
        Flow(pathspec=flow_name)
    except MetaflowNotFound:
        raise CommandException('No run found for *%s*. Please run the flow before tagging.' % flow_name)
    except MetaflowNamespaceMismatch:
        raise CommandException('No run found for *%s* in namespace *%s*. You can switch the namespace using --namespace' % (flow_name, user_namespace))
    if run_id is None:
        latest_run_id = Flow(pathspec=flow_name).latest_run.id
        msg = "Please specify a run-id using --run-id.\n*%s*'s latest run in namespace *%s* has id *%s*." % (flow_name, user_namespace, latest_run_id)
        raise CommandException(msg)
    run_id_parts = run_id.split('/')
    if len(run_id_parts) == 1:
        path_spec = '%s/%s' % (flow_name, run_id)
    else:
        raise CommandException('Run-id *%s* is not a valid run-id' % run_id)
    try:
        namespace(user_namespace)
        run = Run(pathspec=path_spec)
    except MetaflowNotFound:
        raise CommandException('No run *%s* found for flow *%s*' % (path_spec, flow_name))
    except MetaflowNamespaceMismatch:
        msg = 'Run *%s* for flow *%s* does not belong to namespace *%s*\n' % (path_spec, flow_name, user_namespace)
        raise CommandException(msg)
    return run

def _set_current(obj):
    if False:
        return 10
    current._set_env(metadata_str='%s@%s' % (obj.metadata.__class__.TYPE, obj.metadata.__class__.INFO))

@click.group()
def cli():
    if False:
        i = 10
        return i + 15
    pass

@cli.group(help='Commands related to tagging.')
def tag():
    if False:
        return 10
    pass

@tag.command('add', help='Add tags to a run.')
@click.option('--run-id', required=False, default=None, type=str, help='Run ID of the specific run to tag. [required]')
@click.option('--namespace', 'user_namespace', required=False, default=None, type=str, help='Change namespace from the default (your username) to the one specified.')
@click.argument('tags', required=True, type=str, nargs=-1)
@click.pass_obj
def add(obj, run_id, user_namespace, tags):
    if False:
        i = 10
        return i + 15
    _set_current(obj)
    user_namespace = resolve_identity() if user_namespace is None else user_namespace
    run = _get_client_run_obj(obj, run_id, user_namespace)
    run.add_tags(tags)
    obj.echo('Operation successful. New tags:', err=False)
    _print_tags_for_one_run(obj, run)

@tag.command('remove', help='Remove tags from a run.')
@click.option('--run-id', required=False, default=None, type=str, help='Run ID of the specific run to tag. [required]')
@click.option('--namespace', 'user_namespace', required=False, default=None, type=str, help='Change namespace from the default (your username) to the one specified.')
@click.argument('tags', required=True, type=str, nargs=-1)
@click.pass_obj
def remove(obj, run_id, user_namespace, tags):
    if False:
        for i in range(10):
            print('nop')
    _set_current(obj)
    user_namespace = resolve_identity() if user_namespace is None else user_namespace
    run = _get_client_run_obj(obj, run_id, user_namespace)
    run.remove_tags(tags)
    obj.echo('Operation successful. New tags:')
    _print_tags_for_one_run(obj, run)

@tag.command('replace', help='Replace one or more tags of a run atomically. Removals are applied first, then additions.')
@click.option('--run-id', required=False, default=None, type=str, help='Run ID of the specific run to tag. [required]')
@click.option('--namespace', 'user_namespace', required=False, default=None, type=str, help='Change namespace from the default (your username) to the one specified.')
@click.option('--add', 'tags_to_add', multiple=True, default=None, help='Add this tag to a run. Must specify one or more tags to add.')
@click.option('--remove', 'tags_to_remove', multiple=True, default=None, help='Remove this tag from a run. Must specify one or more tags to remove.')
@click.pass_obj
def replace(obj, run_id, user_namespace, tags_to_add=None, tags_to_remove=None):
    if False:
        return 10
    _set_current(obj)
    if not tags_to_add and (not tags_to_remove):
        raise CommandException('Specify at least one tag to add (--add) and one tag to remove (--remove)')
    if not tags_to_remove:
        raise CommandException('Specify at least one tag to remove; else please use *tag add*.')
    if not tags_to_add:
        raise CommandException('Specify at least one tag to add, else please use *tag remove*.')
    user_namespace = resolve_identity() if user_namespace is None else user_namespace
    run = _get_client_run_obj(obj, run_id, user_namespace)
    run.replace_tags(tags_to_remove, tags_to_add)
    obj.echo('Operation successful. New tags:')
    _print_tags_for_one_run(obj, run)

@tag.command('list', help='List tags of a run.')
@click.option('--run-id', required=False, default=None, type=str, help='Run ID of the specific run to list.')
@click.option('--all', 'list_all', required=False, is_flag=True, default=False, help='List tags across all runs of this flow.')
@click.option('--my-runs', 'my_runs', required=False, is_flag=True, default=False, help='List tags across all runs of the flow under the default namespace.')
@click.option('--hide-system-tags', required=False, is_flag=True, default=False, help='Hide system tags.')
@click.option('--group-by-tag', required=False, is_flag=True, default=False, help='Display results by showing runs grouped by tags')
@click.option('--group-by-run', required=False, is_flag=True, default=False, help='Display tags grouped by run')
@click.option('--flat', required=False, is_flag=True, default=False, help='List tags, one line per tag, no groupings', hidden=True)
@click.argument('arg_run_id', required=False, default=None, type=str)
@click.pass_obj
def tag_list(obj, run_id, hide_system_tags, list_all, my_runs, group_by_tag, group_by_run, flat, arg_run_id):
    if False:
        while True:
            i = 10
    _set_current(obj)
    if run_id is None and arg_run_id is None and (not list_all) and (not my_runs):
        list_all = True
    if list_all and my_runs:
        raise CommandException('Option --all cannot be used together with --my-runs.')
    if run_id is not None and arg_run_id is not None:
        raise CommandException('Specify a run either using --run-id or as an argument but not both')
    if arg_run_id is not None:
        run_id = arg_run_id
    if group_by_run and group_by_tag:
        raise CommandException('Option --group-by-tag cannot be used with --group-by-run')
    if flat and (group_by_run or group_by_tag):
        raise CommandException('Option --flat cannot be used with any --group-by-* option')
    system_tags_by_some_grouping = dict()
    all_tags_by_some_grouping = dict()

    def _populate_tag_groups_from_run(_run):
        if False:
            print('Hello World!')
        if group_by_run:
            if hide_system_tags:
                all_tags_by_some_grouping[_run.pathspec] = _run.tags - _run.system_tags
            else:
                system_tags_by_some_grouping[_run.pathspec] = _run.system_tags
                all_tags_by_some_grouping[_run.pathspec] = _run.tags
        elif group_by_tag:
            for t in _run.tags - _run.system_tags:
                all_tags_by_some_grouping.setdefault(t, []).append(_run.pathspec)
            if not hide_system_tags:
                for t in _run.system_tags:
                    system_tags_by_some_grouping.setdefault(t, []).append(_run.pathspec)
        elif hide_system_tags:
            all_tags_by_some_grouping.setdefault('_', set()).update(_run.tags.difference(_run.system_tags))
        else:
            system_tags_by_some_grouping.setdefault('_', set()).update(_run.system_tags)
            all_tags_by_some_grouping.setdefault('_', set()).update(_run.tags)
    pathspecs = []
    if list_all or my_runs:
        user_namespace = resolve_identity() if my_runs else None
        namespace(user_namespace)
        try:
            flow = Flow(pathspec=obj.flow.name)
        except MetaflowNotFound:
            raise CommandException('Cannot list tags because the flow %s has never been run.' % (obj.flow.name,))
        for run in flow.runs():
            _populate_tag_groups_from_run(run)
            pathspecs.append(run.pathspec)
    else:
        run = _get_client_run_obj(obj, run_id, None)
        _populate_tag_groups_from_run(run)
        pathspecs.append(run.pathspec)
    if not group_by_run and (not group_by_tag):
        system_tags_by_some_grouping[','.join(pathspecs)] = system_tags_by_some_grouping.get('_', set())
        all_tags_by_some_grouping[','.join(pathspecs)] = all_tags_by_some_grouping.get('_', set())
        if '_' in system_tags_by_some_grouping:
            del system_tags_by_some_grouping['_']
        if '_' in all_tags_by_some_grouping:
            del all_tags_by_some_grouping['_']
    if flat:
        if len(all_tags_by_some_grouping) != 1:
            raise MetaflowInternalError('Failed to flatten tag set')
        for v in all_tags_by_some_grouping.values():
            for tag in v:
                obj.echo(tag)
            return
    _print_tags_for_runs_by_groups(obj, system_tags_by_some_grouping, all_tags_by_some_grouping, group_by_tag)