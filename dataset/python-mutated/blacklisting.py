import ast
import fnmatch
from bandit.core import issue

def report_issue(check, name):
    if False:
        return 10
    return issue.Issue(severity=check.get('level', 'MEDIUM'), confidence='HIGH', cwe=check.get('cwe', issue.Cwe.NOTSET), text=check['message'].replace('{name}', name), ident=name, test_id=check.get('id', 'LEGACY'))

def blacklist(context, config):
    if False:
        return 10
    "Generic blacklist test, B001.\n\n    This generic blacklist test will be called for any encountered node with\n    defined blacklist data available. This data is loaded via plugins using\n    the 'bandit.blacklists' entry point. Please see the documentation for more\n    details. Each blacklist datum has a unique bandit ID that may be used for\n    filtering purposes, or alternatively all blacklisting can be filtered using\n    the id of this built in test, 'B001'.\n    "
    blacklists = config
    node_type = context.node.__class__.__name__
    if node_type == 'Call':
        func = context.node.func
        if isinstance(func, ast.Name) and func.id == '__import__':
            if len(context.node.args):
                if isinstance(context.node.args[0], ast.Str):
                    name = context.node.args[0].s
                else:
                    name = 'UNKNOWN'
            else:
                name = ''
        else:
            name = context.call_function_name_qual
            if name in ['importlib.import_module', 'importlib.__import__']:
                if context.call_args_count > 0:
                    name = context.call_args[0]
                else:
                    name = context.call_keywords['name']
        for check in blacklists[node_type]:
            for qn in check['qualnames']:
                if name is not None and fnmatch.fnmatch(name, qn):
                    return report_issue(check, name)
    if node_type.startswith('Import'):
        prefix = ''
        if node_type == 'ImportFrom':
            if context.node.module is not None:
                prefix = context.node.module + '.'
        for check in blacklists[node_type]:
            for name in context.node.names:
                for qn in check['qualnames']:
                    if (prefix + name.name).startswith(qn):
                        return report_issue(check, name.name)