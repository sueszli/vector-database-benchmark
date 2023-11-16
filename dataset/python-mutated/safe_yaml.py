import re
import yaml
__all__ = ['safe_dump', 'SafeLoader']

class SafeStringDumper(yaml.SafeDumper):

    def represent_data(self, value):
        if False:
            return 10
        if isinstance(value, str):
            return self.represent_scalar('!unsafe', value)
        return super(SafeStringDumper, self).represent_data(value)

class SafeLoader(yaml.Loader):

    def construct_yaml_unsafe(self, node):
        if False:
            for i in range(10):
                print('nop')

        class UnsafeText(str):
            __UNSAFE__ = True
        node = UnsafeText(self.construct_scalar(node))
        return node
SafeLoader.add_constructor(u'!unsafe', SafeLoader.construct_yaml_unsafe)

def safe_dump(x, safe_dict=None):
    if False:
        while True:
            i = 10
    '\n    Used to serialize an extra_vars dict to YAML\n\n    By default, extra vars are marked as `!unsafe` in the generated yaml\n    _unless_ they\'ve been deemed "trusted" (meaning, they likely were set/added\n    by a user with a high level of privilege).\n\n    This function allows you to pass in a trusted `safe_dict` to allow\n    certain extra vars so that they are _not_ marked as `!unsafe` in the\n    resulting YAML.  Anything _not_ in this dict will automatically be\n    `!unsafe`.\n\n    safe_dump({\'a\': \'b\', \'c\': \'d\'}) ->\n    !unsafe \'a\': !unsafe \'b\'\n    !unsafe \'c\': !unsafe \'d\'\n\n    safe_dump({\'a\': \'b\', \'c\': \'d\'}, safe_dict={\'a\': \'b\'})\n    a: b\n    !unsafe \'c\': !unsafe \'d\'\n    '
    if isinstance(x, dict):
        yamls = []
        safe_dict = safe_dict or {}
        for (k, v) in x.items():
            dumper = yaml.SafeDumper
            if k not in safe_dict or safe_dict.get(k) != v:
                dumper = SafeStringDumper
            yamls.append(yaml.dump_all([{k: v}], None, Dumper=dumper, default_flow_style=False))
        return ''.join(yamls)
    else:
        return yaml.dump_all([x], None, Dumper=SafeStringDumper, default_flow_style=False)

def sanitize_jinja(arg):
    if False:
        while True:
            i = 10
    '\n    For some string, prevent usage of Jinja-like flags\n    '
    if isinstance(arg, str):
        if re.search('\\{\\{[^}]+}}', arg) is not None:
            raise ValueError('Inline Jinja variables are not allowed.')
        if re.search('\\{%[^%]+%}', arg) is not None:
            raise ValueError('Inline Jinja variables are not allowed.')
    return arg