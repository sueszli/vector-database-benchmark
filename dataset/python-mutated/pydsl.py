"""
:maintainer: Jack Kuan <kjkuan@gmail.com>
:maturity: new
:platform: all

A Python DSL for generating Salt's highstate data structure.

This module is intended to be used with the `pydsl` renderer,
but can also be used on its own. Here's what you can do with
Salt PyDSL::

    # Example translated from the online salt tutorial

    apache = state('apache')
    apache.pkg.installed()
    apache.service.running() \\
                  .watch(pkg='apache',
                         file='/etc/httpd/conf/httpd.conf',
                         user='apache')

    if __grains__['os'] == 'RedHat':
        apache.pkg.installed(name='httpd')
        apache.service.running(name='httpd')

    apache.group.present(gid=87).require(apache.pkg)
    apache.user.present(uid=87, gid=87,
                        home='/var/www/html',
                        shell='/bin/nologin') \\
               .require(apache.group)

    state('/etc/httpd/conf/httpd.conf').file.managed(
        source='salt://apache/httpd.conf',
        user='root',
        group='root',
        mode=644)


Example with ``include`` and ``extend``, translated from
the online salt tutorial::

    include('http', 'ssh')
    extend(
        state('apache').file(
            name='/etc/httpd/conf/httpd.conf',
            source='salt://http/httpd2.conf'
        ),
        state('ssh-server').service.watch(file='/etc/ssh/banner')
    )
    state('/etc/ssh/banner').file.managed(source='salt://ssh/banner')


Example of a ``cmd`` state calling a python function::

    def hello(s):
        s = "hello world, %s" % s
        return dict(result=True, changes=dict(changed=True, output=s))

    state('hello').cmd.call(hello, 'pydsl!')

"""
from uuid import uuid4 as _uuid
from salt.state import HighState
from salt.utils.odict import OrderedDict
REQUISITES = set('listen require watch prereq use listen_in require_in watch_in prereq_in use_in onchanges onfail'.split())

class PyDslError(Exception):
    pass

class Options(dict):

    def __getattr__(self, name):
        if False:
            for i in range(10):
                print('nop')
        return self.get(name)
SLS_MATCHES = None

class Sls:

    def __init__(self, sls, saltenv, rendered_sls):
        if False:
            while True:
                i = 10
        self.name = sls
        self.saltenv = saltenv
        self.includes = []
        self.included_highstate = HighState.get_active().building_highstate
        self.extends = []
        self.decls = []
        self.options = Options()
        self.funcs = []
        self.rendered_sls = rendered_sls
        if not HighState.get_active():
            raise PyDslError('PyDSL only works with a running high state!')

    @classmethod
    def get_all_decls(cls):
        if False:
            for i in range(10):
                print('nop')
        return HighState.get_active()._pydsl_all_decls

    @classmethod
    def get_render_stack(cls):
        if False:
            while True:
                i = 10
        return HighState.get_active()._pydsl_render_stack

    def set(self, **options):
        if False:
            return 10
        self.options.update(options)

    def include(self, *sls_names, **kws):
        if False:
            print('Hello World!')
        if 'env' in kws:
            kws.pop('env')
        saltenv = kws.get('saltenv', self.saltenv)
        if kws.get('delayed', False):
            for incl in sls_names:
                self.includes.append((saltenv, incl))
            return
        HIGHSTATE = HighState.get_active()
        global SLS_MATCHES
        if SLS_MATCHES is None:
            SLS_MATCHES = HIGHSTATE.top_matches(HIGHSTATE.get_top())
        highstate = self.included_highstate
        slsmods = []
        for sls in sls_names:
            r_env = '{}:{}'.format(saltenv, sls)
            if r_env not in self.rendered_sls:
                self.rendered_sls.add(sls)
                (histates, errors) = HIGHSTATE.render_state(sls, saltenv, self.rendered_sls, SLS_MATCHES)
                HIGHSTATE.merge_included_states(highstate, histates, errors)
                if errors:
                    raise PyDslError('\n'.join(errors))
                HIGHSTATE.clean_duplicate_extends(highstate)
            state_id = '_slsmod_{}'.format(sls)
            if state_id not in highstate:
                slsmods.append(None)
            else:
                for arg in highstate[state_id]['stateconf']:
                    if isinstance(arg, dict) and next(iter(arg)) == 'slsmod':
                        slsmods.append(arg['slsmod'])
                        break
        if not slsmods:
            return None
        return slsmods[0] if len(slsmods) == 1 else slsmods

    def extend(self, *state_funcs):
        if False:
            for i in range(10):
                print('nop')
        if self.options.ordered or self.last_func():
            raise PyDslError('Cannot extend() after the ordered option was turned on!')
        for f in state_funcs:
            state_id = f.mod._state_id
            self.extends.append(self.get_all_decls().pop(state_id))
            i = len(self.decls)
            for decl in reversed(self.decls):
                i -= 1
                if decl._id == state_id:
                    del self.decls[i]
                    break

    def state(self, id=None):
        if False:
            i = 10
            return i + 15
        if not id:
            id = '.{}'.format(_uuid())
        try:
            return self.get_all_decls()[id]
        except KeyError:
            self.get_all_decls()[id] = s = StateDeclaration(id)
            self.decls.append(s)
            return s

    def last_func(self):
        if False:
            print('Hello World!')
        return self.funcs[-1] if self.funcs else None

    def track_func(self, statefunc):
        if False:
            return 10
        self.funcs.append(statefunc)

    def to_highstate(self, slsmod):
        if False:
            i = 10
            return i + 15
        slsmod_id = '_slsmod_' + self.name
        self.state(slsmod_id).stateconf.set(slsmod=slsmod)
        del self.get_all_decls()[slsmod_id]
        highstate = OrderedDict()
        if self.includes:
            highstate['include'] = [{t[0]: t[1]} for t in self.includes]
        if self.extends:
            highstate['extend'] = extend = OrderedDict()
            for ext in self.extends:
                extend[ext._id] = ext._repr(context='extend')
        for decl in self.decls:
            highstate[decl._id] = decl._repr()
        if self.included_highstate:
            errors = []
            HighState.get_active().merge_included_states(highstate, self.included_highstate, errors)
            if errors:
                raise PyDslError('\n'.join(errors))
        return highstate

    def load_highstate(self, highstate):
        if False:
            for i in range(10):
                print('nop')
        for (sid, decl) in highstate.items():
            s = self.state(sid)
            for (modname, args) in decl.items():
                if '.' in modname:
                    (modname, funcname) = modname.rsplit('.', 1)
                else:
                    funcname = next((x for x in args if isinstance(x, str)))
                    args.remove(funcname)
                mod = getattr(s, modname)
                named_args = {}
                for x in args:
                    if isinstance(x, dict):
                        (k, v) = next(iter(x.items()))
                        named_args[k] = v
                mod(funcname, **named_args)

class StateDeclaration:

    def __init__(self, id):
        if False:
            while True:
                i = 10
        self._id = id
        self._mods = []

    def __getattr__(self, name):
        if False:
            i = 10
            return i + 15
        for m in self._mods:
            if m._name == name:
                return m
        m = StateModule(name, self._id)
        self._mods.append(m)
        return m
    __getitem__ = __getattr__

    def __str__(self):
        if False:
            return 10
        return self._id

    def __iter__(self):
        if False:
            print('Hello World!')
        return iter(self._mods)

    def _repr(self, context=None):
        if False:
            print('Hello World!')
        return OrderedDict((m._repr(context) for m in self))

    def __call__(self, check=True):
        if False:
            return 10
        sls = Sls.get_render_stack()[-1]
        if self._id in sls.get_all_decls():
            last_func = sls.last_func()
            if last_func and self._mods[-1]._func is not last_func:
                raise PyDslError('Cannot run state({}: {}) that is required by a runtime state({}: {}), at compile time.'.format(self._mods[-1]._name, self._id, last_func.mod, last_func.mod._state_id))
            sls.get_all_decls().pop(self._id)
            sls.decls.remove(self)
            self._mods[0]._func._remove_auto_require()
            for m in self._mods:
                try:
                    sls.funcs.remove(m._func)
                except ValueError:
                    pass
        result = HighState.get_active().state.functions['state.high']({self._id: self._repr()})
        if not isinstance(result, dict):
            raise PyDslError('An error occurred while running highstate: {}'.format('; '.join(result)))
        result = sorted(result.items(), key=lambda t: t[1]['__run_num__'])
        if check:
            for (k, v) in result:
                if not v['result']:
                    import pprint
                    raise PyDslError('Failed executing low state at compile time:\n{}'.format(pprint.pformat({k: v})))
        return result

class StateModule:

    def __init__(self, name, parent_decl):
        if False:
            return 10
        self._state_id = parent_decl
        self._name = name
        self._func = None

    def __getattr__(self, name):
        if False:
            print('Hello World!')
        if self._func:
            if name == self._func.name:
                return self._func
            else:
                if name not in REQUISITES:
                    if self._func.name:
                        raise PyDslError('Multiple state functions({}) not allowed in a state module({})!'.format(name, self._name))
                    self._func.name = name
                    return self._func
                return getattr(self._func, name)
        if name in REQUISITES:
            self._func = f = StateFunction(None, self)
            return getattr(f, name)
        else:
            self._func = f = StateFunction(name, self)
            return f

    def __call__(self, _fname, *args, **kws):
        if False:
            for i in range(10):
                print('nop')
        return getattr(self, _fname).configure(args, kws)

    def __str__(self):
        if False:
            return 10
        return self._name

    def _repr(self, context=None):
        if False:
            while True:
                i = 10
        return (self._name, self._func._repr(context))

def _generate_requsite_method(t):
    if False:
        i = 10
        return i + 15

    def req(self, *args, **kws):
        if False:
            print('Hello World!')
        for mod in args:
            self.reference(t, mod, None)
        for mod_ref in kws.items():
            self.reference(t, *mod_ref)
        return self
    return req

class StateFunction:

    def __init__(self, name, parent_mod):
        if False:
            print('Hello World!')
        self.mod = parent_mod
        self.name = name
        self.args = []
        self.require_index = None
        sls = Sls.get_render_stack()[-1]
        if sls.options.ordered:
            last_f = sls.last_func()
            if last_f:
                self.require(last_f.mod)
                self.require_index = len(self.args) - 1
            sls.track_func(self)

    def _remove_auto_require(self):
        if False:
            for i in range(10):
                print('nop')
        if self.require_index is not None:
            del self.args[self.require_index]
            self.require_index = None

    def __call__(self, *args, **kws):
        if False:
            while True:
                i = 10
        self.configure(args, kws)
        return self

    def _repr(self, context=None):
        if False:
            while True:
                i = 10
        if not self.name and context != 'extend':
            raise PyDslError('No state function specified for module: {}'.format(self.mod._name))
        if not self.name and context == 'extend':
            return self.args
        return [self.name] + self.args

    def configure(self, args, kws):
        if False:
            while True:
                i = 10
        args = list(args)
        if args:
            first = args[0]
            if self.mod._name == 'cmd' and self.name in ('call', 'wait_call') and callable(first):
                args[0] = first.__name__
                kws = dict(func=first, args=args[1:], kws=kws)
                del args[1:]
            args[0] = dict(name=args[0])
        for (k, v) in kws.items():
            args.append({k: v})
        self.args.extend(args)
        return self

    def reference(self, req_type, mod, ref):
        if False:
            while True:
                i = 10
        if isinstance(mod, StateModule):
            ref = mod._state_id
        elif not (mod and ref):
            raise PyDslError('Invalid a requisite reference declaration! {}: {}'.format(mod, ref))
        self.args.append({req_type: [{str(mod): str(ref)}]})
    ns = locals()
    for req_type in REQUISITES:
        ns[req_type] = _generate_requsite_method(req_type)
    del ns
    del req_type