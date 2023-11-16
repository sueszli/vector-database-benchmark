"""More comprehensive traceback formatting for Python scripts.
To enable this module, do:
    import tracebackplus; tracebackplus.enable()
at the top of your script.  The optional arguments to enable() are:
    logdir      - if set, tracebacks are written to files in this directory
    context     - number of lines of source code to show for each stack frame
By default, tracebacks are displayed but not saved and the context is 5 lines.
Alternatively, if you have caught an exception and want tracebackplus to display it
for you, call tracebackplus.handler().  The optional argument to handler() is a
3-item tuple (etype, evalue, etb) just like the value of sys.exc_info().
"""
'\ntracebackplus was derived from the cgitb standard library module. As cgitb is being\ndeprecated, this simplified version of cgitb was created.\n\nhttps://github.com/python/cpython/blob/3.8/Lib/cgitb.py\n\n"Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,\n2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020 Python Software Foundation;\nAll Rights Reserved"\n\nPYTHON SOFTWARE FOUNDATION LICENSE VERSION 2\n--------------------------------------------\n\n1. This LICENSE AGREEMENT is between the Python Software Foundation\n("PSF"), and the Individual or Organization ("Licensee") accessing and\notherwise using this software ("Python") in source or binary form and\nits associated documentation.\n\n2. Subject to the terms and conditions of this License Agreement, PSF hereby\ngrants Licensee a nonexclusive, royalty-free, world-wide license to reproduce,\nanalyze, test, perform and/or display publicly, prepare derivative works,\ndistribute, and otherwise use Python alone or in any derivative version,\nprovided, however, that PSF\'s License Agreement and PSF\'s notice of copyright,\ni.e., "Copyright (c) 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,\n2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020 Python Software Foundation;\nAll Rights Reserved" are retained in Python alone or in any derivative version\nprepared by Licensee.\n\n3. In the event Licensee prepares a derivative work that is based on\nor incorporates Python or any part thereof, and wants to make\nthe derivative work available to others as provided herein, then\nLicensee hereby agrees to include in any such work a brief summary of\nthe changes made to Python.\n\n4. PSF is making Python available to Licensee on an "AS IS"\nbasis.  PSF MAKES NO REPRESENTATIONS OR WARRANTIES, EXPRESS OR\nIMPLIED.  BY WAY OF EXAMPLE, BUT NOT LIMITATION, PSF MAKES NO AND\nDISCLAIMS ANY REPRESENTATION OR WARRANTY OF MERCHANTABILITY OR FITNESS\nFOR ANY PARTICULAR PURPOSE OR THAT THE USE OF PYTHON WILL NOT\nINFRINGE ANY THIRD PARTY RIGHTS.\n\n5. PSF SHALL NOT BE LIABLE TO LICENSEE OR ANY OTHER USERS OF PYTHON\nFOR ANY INCIDENTAL, SPECIAL, OR CONSEQUENTIAL DAMAGES OR LOSS AS\nA RESULT OF MODIFYING, DISTRIBUTING, OR OTHERWISE USING PYTHON,\nOR ANY DERIVATIVE THEREOF, EVEN IF ADVISED OF THE POSSIBILITY THEREOF.\n\n6. This License Agreement will automatically terminate upon a material\nbreach of its terms and conditions.\n\n7. Nothing in this License Agreement shall be deemed to create any\nrelationship of agency, partnership, or joint venture between PSF and\nLicensee.  This License Agreement does not grant permission to use PSF\ntrademarks or trade name in a trademark sense to endorse or promote\nproducts or services of Licensee, or any third party.\n\n8. By copying, installing or otherwise using Python, Licensee\nagrees to be bound by the terms and conditions of this License\nAgreement.\n'
import inspect
import keyword
import linecache
import os
import pydoc
import sys
import tempfile
import time
import tokenize
import traceback
__UNDEF__ = []

def lookup(name, frame, locals):
    if False:
        i = 10
        return i + 15
    'Find the value for a given name in the given environment.'
    if name in locals:
        return ('local', locals[name])
    if name in frame.f_globals:
        return ('global', frame.f_globals[name])
    if '__builtins__' in frame.f_globals:
        builtins = frame.f_globals['__builtins__']
        if isinstance(builtins, dict):
            if name in builtins:
                return ('builtin', builtins[name])
        elif hasattr(builtins, name):
            return ('builtin', getattr(builtins, name))
    return (None, __UNDEF__)

def scanvars(reader, frame, locals):
    if False:
        while True:
            i = 10
    'Scan one logical line of Python and look up values of variables used.'
    (vars, lasttoken, parent, prefix, value) = ([], None, None, '', __UNDEF__)
    for (ttype, token, start, end, line) in tokenize.generate_tokens(reader):
        if ttype == tokenize.NEWLINE:
            break
        if ttype == tokenize.NAME and token not in keyword.kwlist:
            if lasttoken == '.':
                if parent is not __UNDEF__:
                    value = getattr(parent, token, __UNDEF__)
                    vars.append((prefix + token, prefix, value))
            else:
                (where, value) = lookup(token, frame, locals)
                vars.append((token, where, value))
        elif token == '.':
            prefix += lasttoken + '.'
            parent = value
        else:
            (parent, prefix) = (None, '')
        lasttoken = token
    return vars

def text(einfo, context=5):
    if False:
        print('Hello World!')
    'Return a plain text document describing a given traceback.'
    (etype, evalue, etb) = einfo
    if isinstance(etype, type):
        etype = etype.__name__
    pyver = 'Python ' + sys.version.split()[0] + ': ' + sys.executable
    date = time.ctime(time.time())
    head = '%s\n%s\n%s\n' % (str(etype), pyver, date) + '\nA problem occurred in a Python script.  Here is the sequence of\nfunction calls leading up to the error, in the order they occurred.\n'
    frames = []
    records = inspect.getinnerframes(etb, context)
    for (frame, file, lnum, func, lines, index) in records:
        file = file and os.path.abspath(file) or '?'
        (args, varargs, varkw, locals) = inspect.getargvalues(frame)
        call = ''
        if func != '?':
            call = 'in ' + func + inspect.formatargvalues(args, varargs, varkw, locals, formatvalue=lambda value: '=' + pydoc.text.repr(value))
        highlight = {}

        def reader(lnum=[lnum]):
            if False:
                for i in range(10):
                    print('nop')
            highlight[lnum[0]] = 1
            try:
                return linecache.getline(file, lnum[0])
            finally:
                lnum[0] += 1
        vars = scanvars(reader, frame, locals)
        rows = [' %s %s' % (file, call)]
        if index is not None:
            i = lnum - index
            for line in lines:
                num = '%5d ' % i
                rows.append(num + line.rstrip())
                i += 1
        (done, dump) = ({}, [])
        for (name, where, value) in vars:
            if name in done:
                continue
            done[name] = 1
            if value is not __UNDEF__:
                if where == 'global':
                    name = 'global ' + name
                elif where != 'local':
                    name = where + name.split('.')[-1]
                dump.append('%s = %s' % (name, pydoc.text.repr(value)))
            else:
                dump.append(name + ' undefined')
        rows.append('\n'.join(dump))
        frames.append('\n%s\n' % '\n'.join(rows))
    exception = ['%s: %s' % (str(etype), str(evalue))]
    for name in dir(evalue):
        value = pydoc.text.repr(getattr(evalue, name))
        exception.append('\n%s%s = %s' % (' ' * 4, name, value))
    return head + ''.join(frames) + ''.join(exception) + '\n\nThe above is a description of an error in a Python program.  Here is\nthe original traceback:\n\n%s\n' % ''.join(traceback.format_exception(etype, evalue, etb))

class Hook:
    """A hook to replace sys.excepthook"""

    def __init__(self, logdir=None, context=5, file=None):
        if False:
            print('Hello World!')
        self.logdir = logdir
        self.context = context
        self.file = file or sys.stdout

    def __call__(self, etype, evalue, etb):
        if False:
            i = 10
            return i + 15
        self.handle((etype, evalue, etb))

    def handle(self, info=None):
        if False:
            i = 10
            return i + 15
        info = info or sys.exc_info()
        formatter = text
        try:
            doc = formatter(info, self.context)
        except:
            doc = ''.join(traceback.format_exception(*info))
        self.file.write(doc + '\n')
        if self.logdir is not None:
            suffix = '.txt'
            (fd, path) = tempfile.mkstemp(suffix=suffix, dir=self.logdir)
            try:
                with os.fdopen(fd, 'w') as file:
                    file.write(doc)
                msg = '%s contains the description of this error.' % path
            except:
                msg = 'Tried to save traceback to %s, but failed.' % path
            self.file.write(msg + '\n')
        try:
            self.file.flush()
        except:
            pass
handler = Hook().handle

def enable(logdir=None, context=5):
    if False:
        return 10
    'Install an exception handler that sends verbose tracebacks to STDOUT.'
    sys.excepthook = Hook(logdir=logdir, context=context)