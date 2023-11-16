import os
import shlex
import textwrap
import SCons
from SCons.Subst import SUBST_CMD
from SCons.Tool.ninja import NINJA_CUSTOM_HANDLERS, NINJA_RULES, NINJA_POOLS
from SCons.Tool.ninja.Globals import __NINJA_RULE_MAPPING
from SCons.Tool.ninja.Utils import get_targets_sources, get_dependencies, get_order_only, get_outputs, get_inputs, get_rule, get_path, generate_command, get_command_env, get_comstr

def register_custom_handler(env, name, handler):
    if False:
        return 10
    'Register a custom handler for SCons function actions.'
    env[NINJA_CUSTOM_HANDLERS][name] = handler

def register_custom_rule_mapping(env, pre_subst_string, rule):
    if False:
        for i in range(10):
            print('nop')
    'Register a function to call for a given rule.'
    SCons.Tool.ninja.Globals.__NINJA_RULE_MAPPING[pre_subst_string] = rule

def register_custom_rule(env, rule, command, description='', deps=None, pool=None, use_depfile=False, use_response_file=False, response_file_content='$rspc'):
    if False:
        for i in range(10):
            print('nop')
    'Allows specification of Ninja rules from inside SCons files.'
    rule_obj = {'command': command, 'description': description if description else '{} $out'.format(rule)}
    if use_depfile:
        rule_obj['depfile'] = os.path.join(get_path(env['NINJA_DIR']), '$out.depfile')
    if deps is not None:
        rule_obj['deps'] = deps
    if pool is not None:
        rule_obj['pool'] = pool
    if use_response_file:
        rule_obj['rspfile'] = '$out.rsp'
        rule_obj['rspfile_content'] = response_file_content
    env[NINJA_RULES][rule] = rule_obj

def register_custom_pool(env, pool, size):
    if False:
        return 10
    'Allows the creation of custom Ninja pools'
    env[NINJA_POOLS][pool] = size

def set_build_node_callback(env, node, callback):
    if False:
        i = 10
        return i + 15
    if not node.is_conftest():
        node.attributes.ninja_build_callback = callback

def get_generic_shell_command(env, node, action, targets, sources, executor=None):
    if False:
        while True:
            i = 10
    return ('GENERATED_CMD', {'cmd': generate_command(env, node, action, targets, sources, executor=executor), 'env': get_command_env(env, targets, sources)}, [])

def CheckNinjaCompdbExpand(env, context):
    if False:
        for i in range(10):
            print('nop')
    " Configure check testing if ninja's compdb can expand response files"
    context.Message('Checking if ninja compdb can expand response files... ')
    (ret, output) = context.TryAction(action='ninja -f $SOURCE -t compdb -x CMD_RSP > $TARGET', extension='.ninja', text=textwrap.dedent('\n            rule CMD_RSP\n              command = $cmd @$out.rsp > fake_output.txt\n              description = Building $out\n              rspfile = $out.rsp\n              rspfile_content = $rspc\n            build fake_output.txt: CMD_RSP fake_input.txt\n              cmd = echo\n              pool = console\n              rspc = "test"\n            '))
    result = '@fake_output.txt.rsp' not in output
    context.Result(result)
    return result

def get_command(env, node, action):
    if False:
        i = 10
        return i + 15
    'Get the command to execute for node.'
    if node.env:
        sub_env = node.env
    else:
        sub_env = env
    executor = node.get_executor()
    (tlist, slist) = get_targets_sources(node)
    if isinstance(action, SCons.Action.CommandGeneratorAction):
        action = action._generate(tlist, slist, sub_env, SUBST_CMD, executor=executor)
    variables = {}
    comstr = str(get_comstr(sub_env, action, tlist, slist))
    if not comstr:
        return None
    provider = __NINJA_RULE_MAPPING.get(comstr, get_generic_shell_command)
    (rule, variables, provider_deps) = provider(sub_env, node, action, tlist, slist, executor=executor)
    if node.get_env().get('NINJA_FORCE_SCONS_BUILD'):
        rule = 'TEMPLATE'
    implicit = list({dep for tgt in tlist for dep in get_dependencies(tgt)})
    for provider_dep in provider_deps:
        provider_dep = sub_env.subst(provider_dep)
        if not provider_dep:
            continue
        if isinstance(provider_dep, SCons.Node.Node) or os.path.exists(provider_dep):
            implicit.append(provider_dep)
            continue
        prog_suffix = sub_env.get('PROGSUFFIX', '')
        provider_dep_ext = provider_dep if provider_dep.endswith(prog_suffix) else provider_dep + prog_suffix
        if os.path.exists(provider_dep_ext):
            implicit.append(provider_dep_ext)
            continue
        provider_dep_abspath = sub_env.WhereIs(provider_dep) or sub_env.WhereIs(provider_dep, path=os.environ['PATH'])
        if provider_dep_abspath:
            implicit.append(provider_dep_abspath)
            continue
        raise Exception("Could not resolve path for %s dependency on node '%s'" % (provider_dep, node))
    ninja_build = {'order_only': get_order_only(node), 'outputs': get_outputs(node), 'inputs': get_inputs(node), 'implicit': implicit, 'rule': get_rule(node, rule), 'variables': variables}
    if node.env and node.env.get('NINJA_POOL', None) is not None:
        ninja_build['pool'] = node.env['NINJA_POOL']
    return ninja_build

def gen_get_response_file_command(env, rule, tool, tool_is_dynamic=False, custom_env={}):
    if False:
        while True:
            i = 10
    'Generate a response file command provider for rule name.'
    use_command_env = not env['PLATFORM'] == 'win32'
    if '$' in tool:
        tool_is_dynamic = True

    def get_response_file_command(env, node, action, targets, sources, executor=None):
        if False:
            for i in range(10):
                print('nop')
        if hasattr(action, 'process'):
            (cmd_list, _, _) = action.process(targets, sources, env, executor=executor)
            cmd_list = [str(c).replace('$', '$$') for c in cmd_list[0]]
        else:
            command = generate_command(env, node, action, targets, sources, executor=executor)
            cmd_list = shlex.split(command)
        if tool_is_dynamic:
            tool_command = env.subst(tool, target=targets, source=sources, executor=executor)
        else:
            tool_command = tool
        try:
            tool_idx = cmd_list.index(tool_command) + 1
        except ValueError:
            raise Exception('Could not find tool {} in {} generated from {}'.format(tool, cmd_list, get_comstr(env, action, targets, sources)))
        (cmd, rsp_content) = (cmd_list[:tool_idx], cmd_list[tool_idx:])
        if os.altsep:
            rsp_content = [rsp_content_item.replace(os.sep, os.altsep) for rsp_content_item in rsp_content]
        rsp_content = ['"' + rsp_content_item + '"' for rsp_content_item in rsp_content]
        rsp_content = ' '.join(rsp_content)
        variables = {'rspc': rsp_content, rule: cmd}
        if use_command_env:
            variables['env'] = get_command_env(env, targets, sources)
            for (key, value) in custom_env.items():
                variables['env'] += env.subst('export %s=%s;' % (key, value), target=targets, source=sources, executor=executor) + ' '
        if node.get_env().get('NINJA_FORCE_SCONS_BUILD'):
            ret_rule = 'TEMPLATE'
        elif len(' '.join(cmd_list)) < env.get('MAXLINELENGTH', 2048):
            ret_rule = rule
        else:
            ret_rule = rule + '_RSP'
        return (ret_rule, variables, [tool_command])
    return get_response_file_command