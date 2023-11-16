"""
    sphinx.domains.modbus
    ~~~~~~~~~~~~~~~~~~~~~

    The Modbus domain.

    :copyright: Copyright 2007-2010 by the Sphinx team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""
import re
from docutils import nodes
from docutils.parsers.rst import directives
from sphinx import addnodes
from sphinx.roles import XRefRole
from sphinx.locale import l_, _
from sphinx.domains import Domain, ObjType, Index
from sphinx.directives import ObjectDescription
from sphinx.util.nodes import make_refnode
from sphinx.util.compat import Directive
from sphinx.util.docfields import Field, GroupedField, TypedField
from sphinxextra.utils import fixup_index_entry
trans = {'en_US': {'Value': 'Value', 'Function ID': 'Function ID', 'Request': 'Request', 'Response': 'Response'}, 'de_DE': {'Value': 'Wert', 'Function ID': 'Funktions ID', 'Request': 'Anfrage', 'Response': 'Antwort'}}

class Translatable:
    app_config = None

    def __init__(self, key):
        if False:
            for i in range(10):
                print('nop')
        self.key = key

    def __str__(self):
        if False:
            while True:
                i = 10
        return trans[self.app_config.language][self.key]
modbus_sig_re = re.compile('^ ([\\w.]*\\.)?            # class name(s)\n          (\\w+)  \\s*             # thing name\n          (?: \\((.*)\\)           # optional: arguments\n           (?:\\s* -> \\s* (.*))?  #           return annotation\n          )? $                   # and nothing more\n          ', re.VERBOSE)
modbus_paramlist_re = re.compile('([\\[\\],])')

class ModbusObject(ObjectDescription):
    """
    Description of a general Modbus object.
    """
    option_spec = {'noindex': directives.flag, 'module': directives.unchanged}
    doc_field_types = [Field('value', label=Translatable('Value'), has_arg=False, names=('value',)), Field('functionid', label=Translatable('Function ID'), has_arg=False, names=('functionid',)), Field('emptyrequest', label=Translatable('Request'), has_arg=False, names=('emptyrequest',)), Field('emptyresponse', label=Translatable('Response'), has_arg=False, names=('emptyresponse',)), Field('noresponse', label=Translatable('Response'), has_arg=False, names=('noresponse',)), TypedField('parameter', label=Translatable('Request'), names=('param', 'parameter', 'arg', 'argument', 'keyword', 'kwarg', 'kwparam', 'request'), typerolename='obj', typenames=('paramtype', 'type')), TypedField('returnvalue', label=Translatable('Response'), names=('returns', 'return', 'response'), typerolename='obj', typenames=('paramtype', 'type'))]

    def get_signature_prefix(self, sig):
        if False:
            i = 10
            return i + 15
        '\n        May return a prefix to put before the object name in the signature.\n        '
        return ''

    def needs_arglist(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        May return true if an empty argument list is to be generated even if\n        the document contains none.\n        '
        return False

    def handle_signature(self, sig, signode):
        if False:
            while True:
                i = 10
        '\n        Transform a Modbus signature into RST nodes.\n        Returns (fully qualified name of the thing, classname if any).\n\n        If inside a class, the current class name is handled intelligently:\n        * it is stripped from the displayed name if present\n        * it is added to the full name (return value) if not present\n        '
        m = modbus_sig_re.match(sig)
        if m is None:
            raise ValueError
        (name_prefix, name, arglist, retann) = m.groups()
        modname = self.options.get('module', self.env.temp_data.get('modbus:module'))
        classname = self.env.temp_data.get('modbus:class')
        if classname:
            add_module = False
            if name_prefix and name_prefix.startswith(classname):
                fullname = name_prefix + name
                name_prefix = name_prefix[len(classname):].lstrip('.')
            elif name_prefix:
                fullname = classname + '.' + name_prefix + name
            else:
                fullname = classname + '.' + name
        else:
            add_module = True
            if name_prefix:
                classname = name_prefix.rstrip('.')
                fullname = name_prefix + name
            else:
                classname = ''
                fullname = name
        signode['module'] = modname
        signode['class'] = classname
        signode['fullname'] = fullname
        sig_prefix = self.get_signature_prefix(sig)
        if sig_prefix:
            signode += addnodes.desc_annotation(sig_prefix, sig_prefix)
        if name_prefix:
            signode += addnodes.desc_addname(name_prefix, name_prefix)
        elif add_module and self.env.config.add_module_names:
            modname = self.options.get('module', self.env.temp_data.get('modbus:module'))
            if modname and modname != 'exceptions':
                nodetext = modname + '.'
                signode += addnodes.desc_addname(nodetext, nodetext)
        signode += addnodes.desc_name(name, name)
        if not arglist:
            if self.needs_arglist():
                signode += addnodes.desc_parameterlist()
            if retann:
                signode += addnodes.desc_returns(retann, retann)
            return (fullname, name_prefix)
        signode += addnodes.desc_parameterlist()
        stack = [signode[-1]]
        for token in modbus_paramlist_re.split(arglist):
            if token == '[':
                opt = addnodes.desc_optional()
                stack[-1] += opt
                stack.append(opt)
            elif token == ']':
                try:
                    stack.pop()
                except IndexError:
                    raise ValueError
            elif not token or token == ',' or token.isspace():
                pass
            else:
                token = token.strip()
                stack[-1] += addnodes.desc_parameter(token, token)
        if len(stack) != 1:
            raise ValueError
        if retann:
            signode += addnodes.desc_returns(retann, retann)
        return (fullname, name_prefix)

    def get_index_text(self, modname, name):
        if False:
            while True:
                i = 10
        '\n        Return the text for the index entry of the object.\n        '
        raise NotImplementedError('must be implemented in subclasses')

    def add_target_and_index(self, name_cls, sig, signode):
        if False:
            return 10
        modname = self.options.get('module', self.env.temp_data.get('modbus:module'))
        fullname = (modname and modname + '.' or '') + name_cls[0]
        if fullname not in self.state.document.ids:
            signode['names'].append(fullname)
            signode['ids'].append(fullname)
            signode['first'] = not self.names
            self.state.document.note_explicit_target(signode)
            objects = self.env.domaindata['modbus']['objects']
            if fullname in objects:
                self.env.warn(self.env.docname, 'duplicate object description of %s, ' % fullname + 'other instance in ' + self.env.doc2path(objects[fullname][0]) + ', use :noindex: for one of them', self.lineno)
            objects[fullname] = (self.env.docname, self.objtype)
        indextext = self.get_index_text(modname, name_cls)
        if indextext:
            self.indexnode['entries'].append(fixup_index_entry(('single', indextext, fullname, fullname, 'foobar')))

    def before_content(self):
        if False:
            for i in range(10):
                print('nop')
        self.clsname_set = False

    def after_content(self):
        if False:
            while True:
                i = 10
        if self.clsname_set:
            self.env.temp_data['modbus:class'] = None

class ModbusModulelevel(ModbusObject):
    """
    Description of an object on module level (functions, data).
    """

    def needs_arglist(self):
        if False:
            print('Hello World!')
        return False

    def get_index_text(self, modname, name_cls):
        if False:
            return 10
        if self.objtype == 'function':
            if not modname:
                return _('%s (built-in function)') % name_cls[0]
            return _('%s (in module %s)') % (name_cls[0], modname)
        elif self.objtype == 'data':
            if not modname:
                return _('%s (built-in variable)') % name_cls[0]
            return _('%s (in module %s)') % (name_cls[0], modname)
        else:
            return ''

class ModbusClasslike(ModbusObject):
    """
    Description of a class-like object (classes, interfaces, exceptions).
    """

    def get_signature_prefix(self, sig):
        if False:
            i = 10
            return i + 15
        return self.objtype + ' '

    def get_index_text(self, modname, name_cls):
        if False:
            print('Hello World!')
        if self.objtype == 'class':
            if not modname:
                return _('%s (built-in class)') % name_cls[0]
            return _('%s (class in %s)') % (name_cls[0], modname)
        elif self.objtype == 'exception':
            return name_cls[0]
        else:
            return ''

    def before_content(self):
        if False:
            while True:
                i = 10
        ModbusObject.before_content(self)
        if self.names:
            self.env.temp_data['modbus:class'] = self.names[0][0]
            self.clsname_set = True

class ModbusClassmember(ModbusObject):
    """
    Description of a class member (methods, attributes).
    """

    def needs_arglist(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def get_signature_prefix(self, sig):
        if False:
            while True:
                i = 10
        if self.objtype == 'staticmethod':
            return 'static '
        elif self.objtype == 'classmethod':
            return 'classmethod '
        return ''

    def get_index_text(self, modname, name_cls):
        if False:
            while True:
                i = 10
        (name, cls) = name_cls
        add_modules = self.env.config.add_module_names
        if self.objtype == 'method':
            try:
                (clsname, methname) = name.rsplit('.', 1)
            except ValueError:
                if modname:
                    return _('%s() (in module %s)') % (name, modname)
                else:
                    return '%s()' % name
            if modname and add_modules:
                return _('%s() (%s.%s method)') % (methname, modname, clsname)
            else:
                return _('%s() (%s method)') % (methname, clsname)
        elif self.objtype == 'staticmethod':
            try:
                (clsname, methname) = name.rsplit('.', 1)
            except ValueError:
                if modname:
                    return _('%s() (in module %s)') % (name, modname)
                else:
                    return '%s()' % name
            if modname and add_modules:
                return _('%s() (%s.%s static method)') % (methname, modname, clsname)
            else:
                return _('%s() (%s static method)') % (methname, clsname)
        elif self.objtype == 'classmethod':
            try:
                (clsname, methname) = name.rsplit('.', 1)
            except ValueError:
                if modname:
                    return _('%s() (in module %s)') % (name, modname)
                else:
                    return '%s()' % name
            if modname:
                return _('%s() (%s.%s class method)') % (methname, modname, clsname)
            else:
                return _('%s() (%s class method)') % (methname, clsname)
        elif self.objtype == 'attribute':
            try:
                (clsname, attrname) = name.rsplit('.', 1)
            except ValueError:
                if modname:
                    return _('%s (in module %s)') % (name, modname)
                else:
                    return name
            if modname and add_modules:
                return _('%s (%s.%s attribute)') % (attrname, modname, clsname)
            else:
                return _('%s (%s attribute)') % (attrname, clsname)
        else:
            return ''

    def before_content(self):
        if False:
            i = 10
            return i + 15
        ModbusObject.before_content(self)
        lastname = self.names and self.names[-1][1]
        if lastname and (not self.env.temp_data.get('modbus:class')):
            self.env.temp_data['modbus:class'] = lastname.strip('.')
            self.clsname_set = True

class ModbusModule(Directive):
    """
    Directive to mark description of a new module.
    """
    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {'platform': lambda x: x, 'synopsis': lambda x: x, 'noindex': directives.flag, 'deprecated': directives.flag}

    def run(self):
        if False:
            while True:
                i = 10
        env = self.state.document.settings.env
        modname = self.arguments[0].strip()
        noindex = 'noindex' in self.options
        env.temp_data['modbus:module'] = modname
        env.domaindata['modbus']['modules'][modname] = (env.docname, self.options.get('synopsis', ''), self.options.get('platform', ''), 'deprecated' in self.options)
        targetnode = nodes.target('', '', ids=['module-' + modname], ismod=True)
        self.state.document.note_explicit_target(targetnode)
        ret = [targetnode]
        if 'platform' in self.options:
            platform = self.options['platform']
            node = nodes.paragraph()
            node += nodes.emphasis('', _('Platforms: '))
            node += nodes.Text(platform, platform)
            ret.append(node)
        if not noindex:
            indextext = _('%s (module)') % modname
            inode = addnodes.index(entries=[('single', indextext, 'module-' + modname, modname, 'foobar')])
            ret.append(inode)
        return ret

class ModbusCurrentModule(Directive):
    """
    This directive is just to tell Sphinx that we're documenting
    stuff in module foo, but links to module foo won't lead here.
    """
    has_content = False
    required_arguments = 1
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {}

    def run(self):
        if False:
            i = 10
            return i + 15
        env = self.state.document.settings.env
        modname = self.arguments[0].strip()
        if modname == 'None':
            env.temp_data['modbus:module'] = None
        else:
            env.temp_data['modbus:module'] = modname
        return []

class ModbusXRefRole(XRefRole):

    def _fix_parens(self, env, has_explicit_title, title, target):
        if False:
            for i in range(10):
                print('nop')
        if not has_explicit_title:
            if title.endswith('()'):
                title = title[:-2]
            if env.config.add_function_parentheses:
                pass
        if target.endswith('()'):
            target = target[:-2]
        return (title, target)

    def process_link(self, env, refnode, has_explicit_title, title, target):
        if False:
            while True:
                i = 10
        refnode['modbus:module'] = env.temp_data.get('modbus:module')
        refnode['modbus:class'] = env.temp_data.get('modbus:class')
        if not has_explicit_title:
            title = title.lstrip('.')
            target = target.lstrip('~')
            if title[0:1] == '~':
                title = title[1:]
                dot = title.rfind('.')
                if dot != -1:
                    title = title[dot + 1:]
        if target[0:1] == '.':
            target = target[1:]
            refnode['refspecific'] = True
        return (title, target)

class ModbusModuleIndex(Index):
    """
    Index subclass to provide the Modbus module index.
    """
    name = 'modindex'
    localname = l_('Modbus Module Index')
    shortname = l_('modules')

    def generate(self, docnames=None):
        if False:
            while True:
                i = 10
        content = {}
        ignores = self.domain.env.config['modindex_common_prefix']
        ignores = sorted(ignores, key=len, reverse=True)
        modules = sorted(self.domain.data['modules'].iteritems(), key=lambda x: x[0].lower())
        prev_modname = ''
        num_toplevels = 0
        for (modname, (docname, synopsis, platforms, deprecated)) in modules:
            if docnames and docname not in docnames:
                continue
            for ignore in ignores:
                if modname.startswith(ignore):
                    modname = modname[len(ignore):]
                    stripped = ignore
                    break
            else:
                stripped = ''
            if not modname:
                (modname, stripped) = (stripped, '')
            entries = content.setdefault(modname[0].lower(), [])
            package = modname.split('.')[0]
            if package != modname:
                if prev_modname == package:
                    entries[-1][1] = 1
                elif not prev_modname.startswith(package):
                    entries.append([stripped + package, 1, '', '', '', '', ''])
                subtype = 2
            else:
                num_toplevels += 1
                subtype = 0
            qualifier = deprecated and _('Deprecated') or ''
            entries.append([stripped + modname, subtype, docname, 'module-' + stripped + modname, platforms, qualifier, synopsis])
            prev_modname = modname
        collapse = len(modules) - num_toplevels < num_toplevels
        content = sorted(content.iteritems())
        return (content, collapse)

class ModbusDomain(Domain):
    """Modbus language domain."""
    name = 'modbus'
    label = 'Modbus'
    object_types = {'function': ObjType(l_('function'), 'func', 'obj'), 'data': ObjType(l_('data'), 'data', 'obj'), 'class': ObjType(l_('class'), 'class', 'obj'), 'exception': ObjType(l_('exception'), 'exc', 'obj'), 'method': ObjType(l_('method'), 'meth', 'obj'), 'classmethod': ObjType(l_('class method'), 'meth', 'obj'), 'staticmethod': ObjType(l_('static method'), 'meth', 'obj'), 'attribute': ObjType(l_('attribute'), 'attr', 'obj'), 'module': ObjType(l_('module'), 'mod', 'obj')}
    directives = {'function': ModbusModulelevel, 'data': ModbusModulelevel, 'class': ModbusClasslike, 'exception': ModbusClasslike, 'method': ModbusClassmember, 'classmethod': ModbusClassmember, 'staticmethod': ModbusClassmember, 'attribute': ModbusClassmember, 'module': ModbusModule, 'currentmodule': ModbusCurrentModule}
    roles = {'data': ModbusXRefRole(), 'exc': ModbusXRefRole(), 'func': ModbusXRefRole(fix_parens=True), 'class': ModbusXRefRole(), 'const': ModbusXRefRole(), 'attr': ModbusXRefRole(), 'meth': ModbusXRefRole(fix_parens=True), 'mod': ModbusXRefRole(), 'obj': ModbusXRefRole()}
    initial_data = {'objects': {}, 'modules': {}}
    indices = [ModbusModuleIndex]

    def clear_doc(self, docname):
        if False:
            for i in range(10):
                print('nop')
        for (fullname, (fn, _)) in self.data['objects'].items():
            if fn == docname:
                del self.data['objects'][fullname]
        for (modname, (fn, _, _, _)) in self.data['modules'].items():
            if fn == docname:
                del self.data['modules'][modname]

    def find_obj(self, env, modname, classname, name, type, searchorder=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Find a Modbus object for "name", perhaps using the given module and/or\n        classname.  Returns a list of (name, object entry) tuples.\n        '
        if name[-2:] == '()':
            name = name[:-2]
        if not name:
            return (None, None)
        objects = self.data['objects']
        matches = []
        newname = None
        if searchorder == 1:
            if modname and classname and (modname + '.' + classname + '.' + name in objects):
                newname = modname + '.' + classname + '.' + name
            elif modname and modname + '.' + name in objects:
                newname = modname + '.' + name
            elif name in objects:
                newname = name
            else:
                searchname = '.' + name
                matches = [(name, objects[name]) for name in objects if name.endswith(searchname)]
        elif name in objects:
            newname = name
        elif classname and classname + '.' + name in objects:
            newname = classname + '.' + name
        elif modname and modname + '.' + name in objects:
            newname = modname + '.' + name
        elif modname and classname and (modname + '.' + classname + '.' + name in objects):
            newname = modname + '.' + classname + '.' + name
        elif type == 'exc' and '.' not in name and ('exceptions.' + name in objects):
            newname = 'exceptions.' + name
        elif type in ('func', 'meth') and '.' not in name and ('object.' + name in objects):
            newname = 'object.' + name
        if newname is not None:
            matches.append((newname, objects[newname]))
        return matches

    def resolve_xref(self, env, fromdocname, builder, type, target, node, contnode):
        if False:
            for i in range(10):
                print('nop')
        if type == 'mod' or (type == 'obj' and target in self.data['modules']):
            (docname, synopsis, platform, deprecated) = self.data['modules'].get(target, ('', '', '', ''))
            if not docname:
                return None
            else:
                title = '%s%s%s' % (platform and '(%s) ' % platform, synopsis, deprecated and ' (deprecated)' or '')
                return make_refnode(builder, fromdocname, docname, 'module-' + target, contnode, title)
        else:
            modname = node.get('modbus:module')
            clsname = node.get('modbus:class')
            searchorder = node.hasattr('refspecific') and 1 or 0
            matches = self.find_obj(env, modname, clsname, target, type, searchorder)
            if not matches:
                return None
            elif len(matches) > 1:
                env.warn(fromdocname, 'more than one target found for cross-reference %r: %s' % (target, ', '.join((match[0] for match in matches))), node.line)
            (name, obj) = matches[0]
            return make_refnode(builder, fromdocname, obj[0], name, contnode, name)

    def get_objects(self):
        if False:
            return 10
        for (modname, info) in self.data['modules'].iteritems():
            yield (modname, modname, 'module', info[0], 'module-' + modname, 0)
        for (refname, (docname, type)) in self.data['objects'].iteritems():
            yield (refname, refname, type, docname, refname, 1)

def setup(app):
    if False:
        for i in range(10):
            print('nop')
    Translatable.app_config = app.config
    app.add_domain(ModbusDomain)