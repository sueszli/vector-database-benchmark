"""
    sphinx.domains.javascript
    ~~~~~~~~~~~~~~~~~~~~~~~~~

    The JavaScript domain.

    :copyright: Copyright 2007-2013 by the Sphinx team, see AUTHORS.
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
javascript_sig_re = re.compile('^ ((?:new)[ ])?\\s*\n          ([\\w.]*\\.)?            # class name(s)\n          (\\w+)  \\s*             # thing name\n          (?: \\((.*)\\)           # optional: arguments\n           (?:\\s* -> \\s* (.*))?  #           return annotation\n          )? $                   # and nothing more\n          ', re.VERBOSE)

def _pseudo_parse_arglist(signode, arglist):
    if False:
        while True:
            i = 10
    '"Parse" a list of arguments separated by commas.\n\n    Arguments can have "optional" annotations given by enclosing them in\n    brackets.  Currently, this will split at any comma, even if it\'s inside a\n    string literal (e.g. default argument value).\n    '
    paramlist = addnodes.desc_parameterlist()
    stack = [paramlist]
    try:
        for argument in arglist.split(','):
            argument = argument.strip()
            ends_open = ends_close = 0
            while argument.startswith('['):
                stack.append(addnodes.desc_optional())
                stack[-2] += stack[-1]
                argument = argument[1:].strip()
            while argument.startswith(']'):
                stack.pop()
                argument = argument[1:].strip()
            while argument.endswith(']'):
                ends_close += 1
                argument = argument[:-1].strip()
            while argument.endswith('['):
                ends_open += 1
                argument = argument[:-1].strip()
            if argument:
                stack[-1] += addnodes.desc_parameter(argument, argument)
            while ends_open:
                stack.append(addnodes.desc_optional())
                stack[-2] += stack[-1]
                ends_open -= 1
            while ends_close:
                stack.pop()
                ends_close -= 1
        if len(stack) != 1:
            raise IndexError
    except IndexError:
        signode += addnodes.desc_parameterlist()
        signode[-1] += addnodes.desc_parameter(arglist, arglist)
    else:
        signode += paramlist

class JavaScriptObject(ObjectDescription):
    """
    Description of a general JavaScript object.
    """
    option_spec = {'noindex': directives.flag, 'module': directives.unchanged, 'annotation': directives.unchanged}
    doc_field_types = [TypedField('parameter', label=l_('Parameters'), names=('param', 'parameter', 'arg', 'argument', 'keyword', 'kwarg', 'kwparam', 'request'), typerolename='obj', typenames=('paramtype', 'type')), TypedField('returnvalue', label=l_('Callback'), names=('returns', 'return', 'response'), typerolename='obj', typenames=('paramtype', 'type')), Field('noreturnvalue', label=l_('Callback'), has_arg=False, names=('noreturn',)), Field('returntype', label=l_('Return type'), has_arg=False, names=('rtype',))]

    def get_signature_prefix(self, sig):
        if False:
            for i in range(10):
                print('nop')
        'May return a prefix to put before the object name in the\n        signature.\n        '
        return ''

    def needs_arglist(self):
        if False:
            while True:
                i = 10
        'May return true if an empty argument list is to be generated even if\n        the document contains none.\n        '
        return False

    def handle_signature(self, sig, signode):
        if False:
            return 10
        'Transform a JavaScript signature into RST nodes.\n\n        Return (fully qualified name of the thing, classname if any).\n\n        If inside a class, the current class name is handled intelligently:\n        * it is stripped from the displayed name if present\n        * it is added to the full name (return value) if not present\n        '
        m = javascript_sig_re.match(sig)
        if m is None:
            raise ValueError
        (kind, name_prefix, name, arglist, retann) = m.groups()
        modname = self.options.get('module', self.env.temp_data.get('javascript:module'))
        classname = self.env.temp_data.get('javascript:class')
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
        if kind:
            signode += addnodes.desc_annotation(kind, kind)
        sig_prefix = self.get_signature_prefix(sig)
        if sig_prefix:
            signode += addnodes.desc_annotation(sig_prefix, sig_prefix)
        if name_prefix:
            signode += addnodes.desc_addname(name_prefix, name_prefix)
        elif add_module and self.env.config.add_module_names:
            modname = self.options.get('module', self.env.temp_data.get('javascript:module'))
            if modname and modname != 'exceptions':
                nodetext = modname + '.'
                signode += addnodes.desc_addname(nodetext, nodetext)
        anno = self.options.get('annotation')
        signode += addnodes.desc_name(name, name)
        if not arglist:
            if self.needs_arglist():
                signode += addnodes.desc_parameterlist()
            if retann:
                signode += addnodes.desc_returns(retann, retann)
            if anno:
                signode += addnodes.desc_annotation(' ' + anno, ' ' + anno)
            return (fullname, name_prefix)
        _pseudo_parse_arglist(signode, arglist)
        if retann:
            signode += addnodes.desc_returns(retann, retann)
        if anno:
            signode += addnodes.desc_annotation(' ' + anno, ' ' + anno)
        return (fullname, name_prefix)

    def get_index_text(self, modname, name):
        if False:
            print('Hello World!')
        'Return the text for the index entry of the object.'
        raise NotImplementedError('must be implemented in subclasses')

    def add_target_and_index(self, name_cls, sig, signode):
        if False:
            return 10
        modname = self.options.get('module', self.env.temp_data.get('javascript:module'))
        fullname = (modname and modname + '.' or '') + name_cls[0]
        if fullname not in self.state.document.ids:
            signode['names'].append(fullname)
            signode['ids'].append(fullname)
            signode['first'] = not self.names
            self.state.document.note_explicit_target(signode)
            objects = self.env.domaindata['javascript']['objects']
            if fullname in objects:
                self.state_machine.reporter.warning('duplicate object description of %s, ' % fullname + 'other instance in ' + self.env.doc2path(objects[fullname][0]) + ', use :noindex: for one of them', line=self.lineno)
            objects[fullname] = (self.env.docname, self.objtype)
        indextext = self.get_index_text(modname, name_cls)
        if indextext:
            self.indexnode['entries'].append(fixup_index_entry(('single', indextext, fullname, '', 'foobar')))

    def before_content(self):
        if False:
            i = 10
            return i + 15
        self.clsname_set = False

    def after_content(self):
        if False:
            print('Hello World!')
        if self.clsname_set:
            self.env.temp_data['javascript:class'] = None

class JavaScriptModulelevel(JavaScriptObject):
    """
    Description of an object on module level (functions, data).
    """

    def needs_arglist(self):
        if False:
            i = 10
            return i + 15
        return self.objtype == 'function'

    def get_index_text(self, modname, name_cls):
        if False:
            print('Hello World!')
        if self.objtype == 'function':
            if not modname:
                return _('%s() (built-in function)') % name_cls[0]
            return _('%s() (in module %s)') % (name_cls[0], modname)
        elif self.objtype == 'data':
            if not modname:
                return _('%s (built-in variable)') % name_cls[0]
            return _('%s (in module %s)') % (name_cls[0], modname)
        else:
            return ''

class JavaScriptClasslike(JavaScriptObject):
    """
    Description of a class-like object (classes, interfaces, exceptions).
    """

    def get_signature_prefix(self, sig):
        if False:
            for i in range(10):
                print('nop')
        return self.objtype + ' '

    def get_index_text(self, modname, name_cls):
        if False:
            while True:
                i = 10
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
            i = 10
            return i + 15
        JavaScriptObject.before_content(self)
        if self.names:
            self.env.temp_data['javascript:class'] = self.names[0][0]
            self.clsname_set = True

class JavaScriptClassmember(JavaScriptObject):
    """
    Description of a class member (methods, attributes).
    """

    def needs_arglist(self):
        if False:
            while True:
                i = 10
        return self.objtype.endswith('method')

    def get_signature_prefix(self, sig):
        if False:
            for i in range(10):
                print('nop')
        if self.objtype == 'staticmethod':
            return 'static '
        elif self.objtype == 'classmethod':
            return 'classmethod '
        return ''

    def get_index_text(self, modname, name_cls):
        if False:
            for i in range(10):
                print('nop')
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
            print('Hello World!')
        JavaScriptObject.before_content(self)
        lastname = self.names and self.names[-1][1]
        if lastname and (not self.env.temp_data.get('javascript:class')):
            self.env.temp_data['javascript:class'] = lastname.strip('.')
            self.clsname_set = True

class JavaScriptDecoratorMixin(object):
    """
    Mixin for decorator directives.
    """

    def handle_signature(self, sig, signode):
        if False:
            i = 10
            return i + 15
        ret = super(JavaScriptDecoratorMixin, self).handle_signature(sig, signode)
        signode.insert(0, addnodes.desc_addname('@', '@'))
        return ret

    def needs_arglist(self):
        if False:
            for i in range(10):
                print('nop')
        return False

class JavaScriptDecoratorFunction(JavaScriptDecoratorMixin, JavaScriptModulelevel):
    """
    Directive to mark functions meant to be used as decorators.
    """

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        self.name = 'javascript:function'
        return JavaScriptModulelevel.run(self)

class JavaScriptDecoratorMethod(JavaScriptDecoratorMixin, JavaScriptClassmember):
    """
    Directive to mark methods meant to be used as decorators.
    """

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        self.name = 'javascript:method'
        return JavaScriptClassmember.run(self)

class JavaScriptModule(Directive):
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
        env.temp_data['javascript:module'] = modname
        ret = []
        if not noindex:
            env.domaindata['javascript']['modules'][modname] = (env.docname, self.options.get('synopsis', ''), self.options.get('platform', ''), 'deprecated' in self.options)
            env.domaindata['javascript']['objects'][modname] = (env.docname, 'module')
            targetnode = nodes.target('', '', ids=['module-' + modname], ismod=True)
            self.state.document.note_explicit_target(targetnode)
            ret.append(targetnode)
            indextext = _('%s (module)') % modname
            inode = addnodes.index(entries=[('single', indextext, 'module-' + modname, '', 'foobar')])
            ret.append(inode)
        return ret

class JavaScriptCurrentModule(Directive):
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
            print('Hello World!')
        env = self.state.document.settings.env
        modname = self.arguments[0].strip()
        if modname == 'None':
            env.temp_data['javascript:module'] = None
        else:
            env.temp_data['javascript:module'] = modname
        return []

class JavaScriptXRefRole(XRefRole):

    def process_link(self, env, refnode, has_explicit_title, title, target):
        if False:
            for i in range(10):
                print('nop')
        refnode['javascript:module'] = env.temp_data.get('javascript:module')
        refnode['javascript:class'] = env.temp_data.get('javascript:class')
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

class JavaScriptModuleIndex(Index):
    """
    Index subclass to provide the JavaScript module index.
    """
    name = 'modindex'
    localname = l_('JavaScript Module Index')
    shortname = l_('modules')

    def generate(self, docnames=None):
        if False:
            print('Hello World!')
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
                    if entries:
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

class JavaScriptDomain(Domain):
    """JavaScript language domain."""
    name = 'javascript'
    label = 'JavaScript'
    object_types = {'function': ObjType(l_('function'), 'func', 'obj'), 'data': ObjType(l_('data'), 'data', 'obj'), 'class': ObjType(l_('class'), 'class', 'obj'), 'exception': ObjType(l_('exception'), 'exc', 'obj'), 'method': ObjType(l_('method'), 'meth', 'obj'), 'classmethod': ObjType(l_('class method'), 'meth', 'obj'), 'staticmethod': ObjType(l_('static method'), 'meth', 'obj'), 'attribute': ObjType(l_('attribute'), 'attr', 'obj'), 'module': ObjType(l_('module'), 'mod', 'obj')}
    directives = {'function': JavaScriptModulelevel, 'data': JavaScriptModulelevel, 'class': JavaScriptClasslike, 'exception': JavaScriptClasslike, 'method': JavaScriptClassmember, 'classmethod': JavaScriptClassmember, 'staticmethod': JavaScriptClassmember, 'attribute': JavaScriptClassmember, 'module': JavaScriptModule, 'currentmodule': JavaScriptCurrentModule, 'decorator': JavaScriptDecoratorFunction, 'decoratormethod': JavaScriptDecoratorMethod}
    roles = {'data': JavaScriptXRefRole(), 'exc': JavaScriptXRefRole(), 'func': JavaScriptXRefRole(fix_parens=True), 'class': JavaScriptXRefRole(), 'const': JavaScriptXRefRole(), 'attr': JavaScriptXRefRole(), 'meth': JavaScriptXRefRole(fix_parens=True), 'mod': JavaScriptXRefRole(), 'obj': JavaScriptXRefRole()}
    initial_data = {'objects': {}, 'modules': {}}
    indices = [JavaScriptModuleIndex]

    def clear_doc(self, docname):
        if False:
            i = 10
            return i + 15
        for (fullname, (fn, _)) in self.data['objects'].items():
            if fn == docname:
                del self.data['objects'][fullname]
        for (modname, (fn, _, _, _)) in self.data['modules'].items():
            if fn == docname:
                del self.data['modules'][modname]

    def find_obj(self, env, modname, classname, name, type, searchmode=0):
        if False:
            for i in range(10):
                print('nop')
        'Find a JavaScript object for "name", perhaps using the given module\n        and/or classname.  Returns a list of (name, object entry) tuples.\n        '
        if name[-2:] == '()':
            name = name[:-2]
        if not name:
            return []
        objects = self.data['objects']
        matches = []
        newname = None
        if searchmode == 1:
            objtypes = self.objtypes_for_role(type)
            if objtypes is not None:
                if modname and classname:
                    fullname = modname + '.' + classname + '.' + name
                    if fullname in objects and objects[fullname][1] in objtypes:
                        newname = fullname
                if not newname:
                    if modname and modname + '.' + name in objects and (objects[modname + '.' + name][1] in objtypes):
                        newname = modname + '.' + name
                    elif name in objects and objects[name][1] in objtypes:
                        newname = name
                    else:
                        searchname = '.' + name
                        matches = [(oname, objects[oname]) for oname in objects if oname.endswith(searchname) and objects[oname][1] in objtypes]
        elif name in objects:
            newname = name
        elif type == 'mod':
            return []
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
            while True:
                i = 10
        modname = node.get('javascript:module')
        clsname = node.get('javascript:class')
        searchmode = node.hasattr('refspecific') and 1 or 0
        matches = self.find_obj(env, modname, clsname, target, type, searchmode)
        if not matches:
            return None
        elif len(matches) > 1:
            env.warn_node('more than one target found for cross-reference %r: %s' % (target, ', '.join((match[0] for match in matches))), node)
        (name, obj) = matches[0]
        if obj[1] == 'module':
            (docname, synopsis, platform, deprecated) = self.data['modules'][name]
            assert docname == obj[0]
            title = name
            if synopsis:
                title += ': ' + synopsis
            if deprecated:
                title += _(' (deprecated)')
            if platform:
                title += ' (' + platform + ')'
            return make_refnode(builder, fromdocname, docname, 'module-' + name, contnode, title)
        else:
            return make_refnode(builder, fromdocname, obj[0], name, contnode, name)

    def get_objects(self):
        if False:
            print('Hello World!')
        for (modname, info) in self.data['modules'].iteritems():
            yield (modname, modname, 'module', info[0], 'module-' + modname, 0)
        for (refname, (docname, type)) in self.data['objects'].iteritems():
            yield (refname, refname, type, docname, refname, 1)

def setup(app):
    if False:
        for i in range(10):
            print('nop')
    app.add_domain(JavaScriptDomain)