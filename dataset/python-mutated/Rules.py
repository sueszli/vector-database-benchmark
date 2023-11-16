from .Utils import get_outputs, get_rule, get_inputs, get_dependencies

def _install_action_function(_env, node):
    if False:
        return 10
    'Install files using the install or copy commands'
    return {'outputs': get_outputs(node), 'rule': get_rule(node, 'INSTALL'), 'inputs': get_inputs(node), 'implicit': get_dependencies(node)}

def _mkdir_action_function(env, node):
    if False:
        return 10
    return {'outputs': get_outputs(node), 'rule': get_rule(node, 'GENERATED_CMD'), 'variables': {'cmd': 'mkdir {args}'.format(args=' '.join(get_outputs(node)) + ' & exit /b 0' if env['PLATFORM'] == 'win32' else '-p ' + ' '.join(get_outputs(node)))}}

def _copy_action_function(env, node):
    if False:
        i = 10
        return i + 15
    return {'outputs': get_outputs(node), 'inputs': get_inputs(node), 'rule': get_rule(node, 'CMD'), 'variables': {'cmd': '$COPY'}}

def _lib_symlink_action_function(_env, node):
    if False:
        for i in range(10):
            print('nop')
    'Create shared object symlinks if any need to be created'
    symlinks = node.check_attributes('shliblinks')
    if not symlinks or symlinks is None:
        return None
    outputs = [link.get_dir().rel_path(linktgt) for (link, linktgt) in symlinks]
    inputs = [link.get_path() for (link, _) in symlinks]
    return {'outputs': outputs, 'inputs': inputs, 'rule': get_rule(node, 'SYMLINK'), 'implicit': get_dependencies(node)}