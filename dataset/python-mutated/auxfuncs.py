"""

Auxiliary functions for f2py2e.

Copyright 1999,2000 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@ioc.ee>
Permission to use, modify, and distribute this software is given under the
terms of the NumPy (BSD style) LICENSE.


NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Date: 2005/07/24 19:01:55 $
Pearu Peterson

"""
import pprint
import sys
import re
import types
from functools import reduce
from copy import deepcopy
from . import __version__
from . import cfuncs
__all__ = ['applyrules', 'debugcapi', 'dictappend', 'errmess', 'gentitle', 'getargs2', 'getcallprotoargument', 'getcallstatement', 'getfortranname', 'getpymethoddef', 'getrestdoc', 'getusercode', 'getusercode1', 'getdimension', 'hasbody', 'hascallstatement', 'hascommon', 'hasexternals', 'hasinitvalue', 'hasnote', 'hasresultnote', 'isallocatable', 'isarray', 'isarrayofstrings', 'ischaracter', 'ischaracterarray', 'ischaracter_or_characterarray', 'iscomplex', 'iscomplexarray', 'iscomplexfunction', 'iscomplexfunction_warn', 'isdouble', 'isdummyroutine', 'isexternal', 'isfunction', 'isfunction_wrap', 'isint1', 'isint1array', 'isinteger', 'isintent_aux', 'isintent_c', 'isintent_callback', 'isintent_copy', 'isintent_dict', 'isintent_hide', 'isintent_in', 'isintent_inout', 'isintent_inplace', 'isintent_nothide', 'isintent_out', 'isintent_overwrite', 'islogical', 'islogicalfunction', 'islong_complex', 'islong_double', 'islong_doublefunction', 'islong_long', 'islong_longfunction', 'ismodule', 'ismoduleroutine', 'isoptional', 'isprivate', 'isrequired', 'isroutine', 'isscalar', 'issigned_long_longarray', 'isstring', 'isstringarray', 'isstring_or_stringarray', 'isstringfunction', 'issubroutine', 'get_f2py_modulename', 'issubroutine_wrap', 'isthreadsafe', 'isunsigned', 'isunsigned_char', 'isunsigned_chararray', 'isunsigned_long_long', 'isunsigned_long_longarray', 'isunsigned_short', 'isunsigned_shortarray', 'l_and', 'l_not', 'l_or', 'outmess', 'replace', 'show', 'stripcomma', 'throw_error', 'isattr_value', 'deep_merge']
f2py_version = __version__.version
errmess = sys.stderr.write
show = pprint.pprint
options = {}
debugoptions = []
wrapfuncs = 1

def outmess(t):
    if False:
        print('Hello World!')
    if options.get('verbose', 1):
        sys.stdout.write(t)

def debugcapi(var):
    if False:
        i = 10
        return i + 15
    return 'capi' in debugoptions

def _ischaracter(var):
    if False:
        while True:
            i = 10
    return 'typespec' in var and var['typespec'] == 'character' and (not isexternal(var))

def _isstring(var):
    if False:
        i = 10
        return i + 15
    return 'typespec' in var and var['typespec'] == 'character' and (not isexternal(var))

def ischaracter_or_characterarray(var):
    if False:
        for i in range(10):
            print('nop')
    return _ischaracter(var) and 'charselector' not in var

def ischaracter(var):
    if False:
        while True:
            i = 10
    return ischaracter_or_characterarray(var) and (not isarray(var))

def ischaracterarray(var):
    if False:
        return 10
    return ischaracter_or_characterarray(var) and isarray(var)

def isstring_or_stringarray(var):
    if False:
        return 10
    return _ischaracter(var) and 'charselector' in var

def isstring(var):
    if False:
        while True:
            i = 10
    return isstring_or_stringarray(var) and (not isarray(var))

def isstringarray(var):
    if False:
        return 10
    return isstring_or_stringarray(var) and isarray(var)

def isarrayofstrings(var):
    if False:
        while True:
            i = 10
    return isstringarray(var) and var['dimension'][-1] == '(*)'

def isarray(var):
    if False:
        i = 10
        return i + 15
    return 'dimension' in var and (not isexternal(var))

def isscalar(var):
    if False:
        for i in range(10):
            print('nop')
    return not (isarray(var) or isstring(var) or isexternal(var))

def iscomplex(var):
    if False:
        return 10
    return isscalar(var) and var.get('typespec') in ['complex', 'double complex']

def islogical(var):
    if False:
        i = 10
        return i + 15
    return isscalar(var) and var.get('typespec') == 'logical'

def isinteger(var):
    if False:
        i = 10
        return i + 15
    return isscalar(var) and var.get('typespec') == 'integer'

def isreal(var):
    if False:
        print('Hello World!')
    return isscalar(var) and var.get('typespec') == 'real'

def get_kind(var):
    if False:
        for i in range(10):
            print('nop')
    try:
        return var['kindselector']['*']
    except KeyError:
        try:
            return var['kindselector']['kind']
        except KeyError:
            pass

def isint1(var):
    if False:
        for i in range(10):
            print('nop')
    return var.get('typespec') == 'integer' and get_kind(var) == '1' and (not isarray(var))

def islong_long(var):
    if False:
        print('Hello World!')
    if not isscalar(var):
        return 0
    if var.get('typespec') not in ['integer', 'logical']:
        return 0
    return get_kind(var) == '8'

def isunsigned_char(var):
    if False:
        while True:
            i = 10
    if not isscalar(var):
        return 0
    if var.get('typespec') != 'integer':
        return 0
    return get_kind(var) == '-1'

def isunsigned_short(var):
    if False:
        for i in range(10):
            print('nop')
    if not isscalar(var):
        return 0
    if var.get('typespec') != 'integer':
        return 0
    return get_kind(var) == '-2'

def isunsigned(var):
    if False:
        while True:
            i = 10
    if not isscalar(var):
        return 0
    if var.get('typespec') != 'integer':
        return 0
    return get_kind(var) == '-4'

def isunsigned_long_long(var):
    if False:
        print('Hello World!')
    if not isscalar(var):
        return 0
    if var.get('typespec') != 'integer':
        return 0
    return get_kind(var) == '-8'

def isdouble(var):
    if False:
        while True:
            i = 10
    if not isscalar(var):
        return 0
    if not var.get('typespec') == 'real':
        return 0
    return get_kind(var) == '8'

def islong_double(var):
    if False:
        return 10
    if not isscalar(var):
        return 0
    if not var.get('typespec') == 'real':
        return 0
    return get_kind(var) == '16'

def islong_complex(var):
    if False:
        while True:
            i = 10
    if not iscomplex(var):
        return 0
    return get_kind(var) == '32'

def iscomplexarray(var):
    if False:
        while True:
            i = 10
    return isarray(var) and var.get('typespec') in ['complex', 'double complex']

def isint1array(var):
    if False:
        i = 10
        return i + 15
    return isarray(var) and var.get('typespec') == 'integer' and (get_kind(var) == '1')

def isunsigned_chararray(var):
    if False:
        return 10
    return isarray(var) and var.get('typespec') in ['integer', 'logical'] and (get_kind(var) == '-1')

def isunsigned_shortarray(var):
    if False:
        while True:
            i = 10
    return isarray(var) and var.get('typespec') in ['integer', 'logical'] and (get_kind(var) == '-2')

def isunsignedarray(var):
    if False:
        i = 10
        return i + 15
    return isarray(var) and var.get('typespec') in ['integer', 'logical'] and (get_kind(var) == '-4')

def isunsigned_long_longarray(var):
    if False:
        while True:
            i = 10
    return isarray(var) and var.get('typespec') in ['integer', 'logical'] and (get_kind(var) == '-8')

def issigned_chararray(var):
    if False:
        while True:
            i = 10
    return isarray(var) and var.get('typespec') in ['integer', 'logical'] and (get_kind(var) == '1')

def issigned_shortarray(var):
    if False:
        for i in range(10):
            print('nop')
    return isarray(var) and var.get('typespec') in ['integer', 'logical'] and (get_kind(var) == '2')

def issigned_array(var):
    if False:
        print('Hello World!')
    return isarray(var) and var.get('typespec') in ['integer', 'logical'] and (get_kind(var) == '4')

def issigned_long_longarray(var):
    if False:
        i = 10
        return i + 15
    return isarray(var) and var.get('typespec') in ['integer', 'logical'] and (get_kind(var) == '8')

def isallocatable(var):
    if False:
        return 10
    return 'attrspec' in var and 'allocatable' in var['attrspec']

def ismutable(var):
    if False:
        return 10
    return not ('dimension' not in var or isstring(var))

def ismoduleroutine(rout):
    if False:
        return 10
    return 'modulename' in rout

def ismodule(rout):
    if False:
        while True:
            i = 10
    return 'block' in rout and 'module' == rout['block']

def isfunction(rout):
    if False:
        i = 10
        return i + 15
    return 'block' in rout and 'function' == rout['block']

def isfunction_wrap(rout):
    if False:
        print('Hello World!')
    if isintent_c(rout):
        return 0
    return wrapfuncs and isfunction(rout) and (not isexternal(rout))

def issubroutine(rout):
    if False:
        for i in range(10):
            print('nop')
    return 'block' in rout and 'subroutine' == rout['block']

def issubroutine_wrap(rout):
    if False:
        i = 10
        return i + 15
    if isintent_c(rout):
        return 0
    return issubroutine(rout) and hasassumedshape(rout)

def isattr_value(var):
    if False:
        print('Hello World!')
    return 'value' in var.get('attrspec', [])

def hasassumedshape(rout):
    if False:
        return 10
    if rout.get('hasassumedshape'):
        return True
    for a in rout['args']:
        for d in rout['vars'].get(a, {}).get('dimension', []):
            if d == ':':
                rout['hasassumedshape'] = True
                return True
    return False

def requiresf90wrapper(rout):
    if False:
        while True:
            i = 10
    return ismoduleroutine(rout) or hasassumedshape(rout)

def isroutine(rout):
    if False:
        return 10
    return isfunction(rout) or issubroutine(rout)

def islogicalfunction(rout):
    if False:
        for i in range(10):
            print('nop')
    if not isfunction(rout):
        return 0
    if 'result' in rout:
        a = rout['result']
    else:
        a = rout['name']
    if a in rout['vars']:
        return islogical(rout['vars'][a])
    return 0

def islong_longfunction(rout):
    if False:
        for i in range(10):
            print('nop')
    if not isfunction(rout):
        return 0
    if 'result' in rout:
        a = rout['result']
    else:
        a = rout['name']
    if a in rout['vars']:
        return islong_long(rout['vars'][a])
    return 0

def islong_doublefunction(rout):
    if False:
        for i in range(10):
            print('nop')
    if not isfunction(rout):
        return 0
    if 'result' in rout:
        a = rout['result']
    else:
        a = rout['name']
    if a in rout['vars']:
        return islong_double(rout['vars'][a])
    return 0

def iscomplexfunction(rout):
    if False:
        i = 10
        return i + 15
    if not isfunction(rout):
        return 0
    if 'result' in rout:
        a = rout['result']
    else:
        a = rout['name']
    if a in rout['vars']:
        return iscomplex(rout['vars'][a])
    return 0

def iscomplexfunction_warn(rout):
    if False:
        while True:
            i = 10
    if iscomplexfunction(rout):
        outmess('    **************************************************************\n        Warning: code with a function returning complex value\n        may not work correctly with your Fortran compiler.\n        When using GNU gcc/g77 compilers, codes should work\n        correctly for callbacks with:\n        f2py -c -DF2PY_CB_RETURNCOMPLEX\n    **************************************************************\n')
        return 1
    return 0

def isstringfunction(rout):
    if False:
        return 10
    if not isfunction(rout):
        return 0
    if 'result' in rout:
        a = rout['result']
    else:
        a = rout['name']
    if a in rout['vars']:
        return isstring(rout['vars'][a])
    return 0

def hasexternals(rout):
    if False:
        i = 10
        return i + 15
    return 'externals' in rout and rout['externals']

def isthreadsafe(rout):
    if False:
        print('Hello World!')
    return 'f2pyenhancements' in rout and 'threadsafe' in rout['f2pyenhancements']

def hasvariables(rout):
    if False:
        while True:
            i = 10
    return 'vars' in rout and rout['vars']

def isoptional(var):
    if False:
        print('Hello World!')
    return ('attrspec' in var and 'optional' in var['attrspec'] and ('required' not in var['attrspec'])) and isintent_nothide(var)

def isexternal(var):
    if False:
        print('Hello World!')
    return 'attrspec' in var and 'external' in var['attrspec']

def getdimension(var):
    if False:
        return 10
    dimpattern = '\\((.*?)\\)'
    if 'attrspec' in var.keys():
        if any(('dimension' in s for s in var['attrspec'])):
            return [re.findall(dimpattern, v) for v in var['attrspec']][0]

def isrequired(var):
    if False:
        print('Hello World!')
    return not isoptional(var) and isintent_nothide(var)

def isintent_in(var):
    if False:
        return 10
    if 'intent' not in var:
        return 1
    if 'hide' in var['intent']:
        return 0
    if 'inplace' in var['intent']:
        return 0
    if 'in' in var['intent']:
        return 1
    if 'out' in var['intent']:
        return 0
    if 'inout' in var['intent']:
        return 0
    if 'outin' in var['intent']:
        return 0
    return 1

def isintent_inout(var):
    if False:
        print('Hello World!')
    return 'intent' in var and ('inout' in var['intent'] or 'outin' in var['intent']) and ('in' not in var['intent']) and ('hide' not in var['intent']) and ('inplace' not in var['intent'])

def isintent_out(var):
    if False:
        while True:
            i = 10
    return 'out' in var.get('intent', [])

def isintent_hide(var):
    if False:
        while True:
            i = 10
    return 'intent' in var and ('hide' in var['intent'] or ('out' in var['intent'] and 'in' not in var['intent'] and (not l_or(isintent_inout, isintent_inplace)(var))))

def isintent_nothide(var):
    if False:
        while True:
            i = 10
    return not isintent_hide(var)

def isintent_c(var):
    if False:
        for i in range(10):
            print('nop')
    return 'c' in var.get('intent', [])

def isintent_cache(var):
    if False:
        while True:
            i = 10
    return 'cache' in var.get('intent', [])

def isintent_copy(var):
    if False:
        print('Hello World!')
    return 'copy' in var.get('intent', [])

def isintent_overwrite(var):
    if False:
        while True:
            i = 10
    return 'overwrite' in var.get('intent', [])

def isintent_callback(var):
    if False:
        print('Hello World!')
    return 'callback' in var.get('intent', [])

def isintent_inplace(var):
    if False:
        for i in range(10):
            print('nop')
    return 'inplace' in var.get('intent', [])

def isintent_aux(var):
    if False:
        i = 10
        return i + 15
    return 'aux' in var.get('intent', [])

def isintent_aligned4(var):
    if False:
        i = 10
        return i + 15
    return 'aligned4' in var.get('intent', [])

def isintent_aligned8(var):
    if False:
        return 10
    return 'aligned8' in var.get('intent', [])

def isintent_aligned16(var):
    if False:
        while True:
            i = 10
    return 'aligned16' in var.get('intent', [])
isintent_dict = {isintent_in: 'INTENT_IN', isintent_inout: 'INTENT_INOUT', isintent_out: 'INTENT_OUT', isintent_hide: 'INTENT_HIDE', isintent_cache: 'INTENT_CACHE', isintent_c: 'INTENT_C', isoptional: 'OPTIONAL', isintent_inplace: 'INTENT_INPLACE', isintent_aligned4: 'INTENT_ALIGNED4', isintent_aligned8: 'INTENT_ALIGNED8', isintent_aligned16: 'INTENT_ALIGNED16'}

def isprivate(var):
    if False:
        print('Hello World!')
    return 'attrspec' in var and 'private' in var['attrspec']

def hasinitvalue(var):
    if False:
        return 10
    return '=' in var

def hasinitvalueasstring(var):
    if False:
        for i in range(10):
            print('nop')
    if not hasinitvalue(var):
        return 0
    return var['='][0] in ['"', "'"]

def hasnote(var):
    if False:
        print('Hello World!')
    return 'note' in var

def hasresultnote(rout):
    if False:
        for i in range(10):
            print('nop')
    if not isfunction(rout):
        return 0
    if 'result' in rout:
        a = rout['result']
    else:
        a = rout['name']
    if a in rout['vars']:
        return hasnote(rout['vars'][a])
    return 0

def hascommon(rout):
    if False:
        print('Hello World!')
    return 'common' in rout

def containscommon(rout):
    if False:
        return 10
    if hascommon(rout):
        return 1
    if hasbody(rout):
        for b in rout['body']:
            if containscommon(b):
                return 1
    return 0

def containsmodule(block):
    if False:
        for i in range(10):
            print('nop')
    if ismodule(block):
        return 1
    if not hasbody(block):
        return 0
    for b in block['body']:
        if containsmodule(b):
            return 1
    return 0

def hasbody(rout):
    if False:
        while True:
            i = 10
    return 'body' in rout

def hascallstatement(rout):
    if False:
        while True:
            i = 10
    return getcallstatement(rout) is not None

def istrue(var):
    if False:
        return 10
    return 1

def isfalse(var):
    if False:
        return 10
    return 0

class F2PYError(Exception):
    pass

class throw_error:

    def __init__(self, mess):
        if False:
            while True:
                i = 10
        self.mess = mess

    def __call__(self, var):
        if False:
            print('Hello World!')
        mess = '\n\n  var = %s\n  Message: %s\n' % (var, self.mess)
        raise F2PYError(mess)

def l_and(*f):
    if False:
        return 10
    (l1, l2) = ('lambda v', [])
    for i in range(len(f)):
        l1 = '%s,f%d=f[%d]' % (l1, i, i)
        l2.append('f%d(v)' % i)
    return eval('%s:%s' % (l1, ' and '.join(l2)))

def l_or(*f):
    if False:
        i = 10
        return i + 15
    (l1, l2) = ('lambda v', [])
    for i in range(len(f)):
        l1 = '%s,f%d=f[%d]' % (l1, i, i)
        l2.append('f%d(v)' % i)
    return eval('%s:%s' % (l1, ' or '.join(l2)))

def l_not(f):
    if False:
        print('Hello World!')
    return eval('lambda v,f=f:not f(v)')

def isdummyroutine(rout):
    if False:
        return 10
    try:
        return rout['f2pyenhancements']['fortranname'] == ''
    except KeyError:
        return 0

def getfortranname(rout):
    if False:
        while True:
            i = 10
    try:
        name = rout['f2pyenhancements']['fortranname']
        if name == '':
            raise KeyError
        if not name:
            errmess('Failed to use fortranname from %s\n' % rout['f2pyenhancements'])
            raise KeyError
    except KeyError:
        name = rout['name']
    return name

def getmultilineblock(rout, blockname, comment=1, counter=0):
    if False:
        while True:
            i = 10
    try:
        r = rout['f2pyenhancements'].get(blockname)
    except KeyError:
        return
    if not r:
        return
    if counter > 0 and isinstance(r, str):
        return
    if isinstance(r, list):
        if counter >= len(r):
            return
        r = r[counter]
    if r[:3] == "'''":
        if comment:
            r = '\t/* start ' + blockname + ' multiline (' + repr(counter) + ') */\n' + r[3:]
        else:
            r = r[3:]
        if r[-3:] == "'''":
            if comment:
                r = r[:-3] + '\n\t/* end multiline (' + repr(counter) + ')*/'
            else:
                r = r[:-3]
        else:
            errmess("%s multiline block should end with `'''`: %s\n" % (blockname, repr(r)))
    return r

def getcallstatement(rout):
    if False:
        while True:
            i = 10
    return getmultilineblock(rout, 'callstatement')

def getcallprotoargument(rout, cb_map={}):
    if False:
        print('Hello World!')
    r = getmultilineblock(rout, 'callprotoargument', comment=0)
    if r:
        return r
    if hascallstatement(rout):
        outmess('warning: callstatement is defined without callprotoargument\n')
        return
    from .capi_maps import getctype
    (arg_types, arg_types2) = ([], [])
    if l_and(isstringfunction, l_not(isfunction_wrap))(rout):
        arg_types.extend(['char*', 'size_t'])
    for n in rout['args']:
        var = rout['vars'][n]
        if isintent_callback(var):
            continue
        if n in cb_map:
            ctype = cb_map[n] + '_typedef'
        else:
            ctype = getctype(var)
            if l_and(isintent_c, l_or(isscalar, iscomplex))(var):
                pass
            elif isstring(var):
                pass
            elif not isattr_value(var):
                ctype = ctype + '*'
            if isstring(var) or isarrayofstrings(var) or isstringarray(var):
                arg_types2.append('size_t')
        arg_types.append(ctype)
    proto_args = ','.join(arg_types + arg_types2)
    if not proto_args:
        proto_args = 'void'
    return proto_args

def getusercode(rout):
    if False:
        for i in range(10):
            print('nop')
    return getmultilineblock(rout, 'usercode')

def getusercode1(rout):
    if False:
        return 10
    return getmultilineblock(rout, 'usercode', counter=1)

def getpymethoddef(rout):
    if False:
        print('Hello World!')
    return getmultilineblock(rout, 'pymethoddef')

def getargs(rout):
    if False:
        return 10
    (sortargs, args) = ([], [])
    if 'args' in rout:
        args = rout['args']
        if 'sortvars' in rout:
            for a in rout['sortvars']:
                if a in args:
                    sortargs.append(a)
            for a in args:
                if a not in sortargs:
                    sortargs.append(a)
        else:
            sortargs = rout['args']
    return (args, sortargs)

def getargs2(rout):
    if False:
        print('Hello World!')
    (sortargs, args) = ([], rout.get('args', []))
    auxvars = [a for a in rout['vars'].keys() if isintent_aux(rout['vars'][a]) and a not in args]
    args = auxvars + args
    if 'sortvars' in rout:
        for a in rout['sortvars']:
            if a in args:
                sortargs.append(a)
        for a in args:
            if a not in sortargs:
                sortargs.append(a)
    else:
        sortargs = auxvars + rout['args']
    return (args, sortargs)

def getrestdoc(rout):
    if False:
        i = 10
        return i + 15
    if 'f2pymultilines' not in rout:
        return None
    k = None
    if rout['block'] == 'python module':
        k = (rout['block'], rout['name'])
    return rout['f2pymultilines'].get(k, None)

def gentitle(name):
    if False:
        while True:
            i = 10
    ln = (80 - len(name) - 6) // 2
    return '/*%s %s %s*/' % (ln * '*', name, ln * '*')

def flatlist(lst):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(lst, list):
        return reduce(lambda x, y, f=flatlist: x + f(y), lst, [])
    return [lst]

def stripcomma(s):
    if False:
        return 10
    if s and s[-1] == ',':
        return s[:-1]
    return s

def replace(str, d, defaultsep=''):
    if False:
        return 10
    if isinstance(d, list):
        return [replace(str, _m, defaultsep) for _m in d]
    if isinstance(str, list):
        return [replace(_m, d, defaultsep) for _m in str]
    for k in 2 * list(d.keys()):
        if k == 'separatorsfor':
            continue
        if 'separatorsfor' in d and k in d['separatorsfor']:
            sep = d['separatorsfor'][k]
        else:
            sep = defaultsep
        if isinstance(d[k], list):
            str = str.replace('#%s#' % k, sep.join(flatlist(d[k])))
        else:
            str = str.replace('#%s#' % k, d[k])
    return str

def dictappend(rd, ar):
    if False:
        print('Hello World!')
    if isinstance(ar, list):
        for a in ar:
            rd = dictappend(rd, a)
        return rd
    for k in ar.keys():
        if k[0] == '_':
            continue
        if k in rd:
            if isinstance(rd[k], str):
                rd[k] = [rd[k]]
            if isinstance(rd[k], list):
                if isinstance(ar[k], list):
                    rd[k] = rd[k] + ar[k]
                else:
                    rd[k].append(ar[k])
            elif isinstance(rd[k], dict):
                if isinstance(ar[k], dict):
                    if k == 'separatorsfor':
                        for k1 in ar[k].keys():
                            if k1 not in rd[k]:
                                rd[k][k1] = ar[k][k1]
                    else:
                        rd[k] = dictappend(rd[k], ar[k])
        else:
            rd[k] = ar[k]
    return rd

def applyrules(rules, d, var={}):
    if False:
        return 10
    ret = {}
    if isinstance(rules, list):
        for r in rules:
            rr = applyrules(r, d, var)
            ret = dictappend(ret, rr)
            if '_break' in rr:
                break
        return ret
    if '_check' in rules and (not rules['_check'](var)):
        return ret
    if 'need' in rules:
        res = applyrules({'needs': rules['need']}, d, var)
        if 'needs' in res:
            cfuncs.append_needs(res['needs'])
    for k in rules.keys():
        if k == 'separatorsfor':
            ret[k] = rules[k]
            continue
        if isinstance(rules[k], str):
            ret[k] = replace(rules[k], d)
        elif isinstance(rules[k], list):
            ret[k] = []
            for i in rules[k]:
                ar = applyrules({k: i}, d, var)
                if k in ar:
                    ret[k].append(ar[k])
        elif k[0] == '_':
            continue
        elif isinstance(rules[k], dict):
            ret[k] = []
            for k1 in rules[k].keys():
                if isinstance(k1, types.FunctionType) and k1(var):
                    if isinstance(rules[k][k1], list):
                        for i in rules[k][k1]:
                            if isinstance(i, dict):
                                res = applyrules({'supertext': i}, d, var)
                                if 'supertext' in res:
                                    i = res['supertext']
                                else:
                                    i = ''
                            ret[k].append(replace(i, d))
                    else:
                        i = rules[k][k1]
                        if isinstance(i, dict):
                            res = applyrules({'supertext': i}, d)
                            if 'supertext' in res:
                                i = res['supertext']
                            else:
                                i = ''
                        ret[k].append(replace(i, d))
        else:
            errmess('applyrules: ignoring rule %s.\n' % repr(rules[k]))
        if isinstance(ret[k], list):
            if len(ret[k]) == 1:
                ret[k] = ret[k][0]
            if ret[k] == []:
                del ret[k]
    return ret

def deep_merge(dict1, dict2):
    if False:
        for i in range(10):
            print('nop')
    'Recursively merge two dictionaries into a new dictionary.\n\n    Parameters:\n    - dict1: The base dictionary.\n    - dict2: The dictionary to merge into a copy of dict1.\n             If a key exists in both, the dict2 value will take precedence.\n\n    Returns:\n    - A new merged dictionary.\n    '
    merged_dict = deepcopy(dict1)
    for (key, value) in dict2.items():
        if key in merged_dict:
            if isinstance(merged_dict[key], dict) and isinstance(value, dict):
                merged_dict[key] = deep_merge(merged_dict[key], value)
            else:
                merged_dict[key] = value
        else:
            merged_dict[key] = value
    return merged_dict
_f2py_module_name_match = re.compile('\\s*python\\s*module\\s*(?P<name>[\\w_]+)', re.I).match
_f2py_user_module_name_match = re.compile('\\s*python\\s*module\\s*(?P<name>[\\w_]*?__user__[\\w_]*)', re.I).match

def get_f2py_modulename(source):
    if False:
        i = 10
        return i + 15
    name = None
    with open(source) as f:
        for line in f:
            m = _f2py_module_name_match(line)
            if m:
                if _f2py_user_module_name_match(line):
                    continue
                name = m.group('name')
                break
    return name