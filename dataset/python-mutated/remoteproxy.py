import os
import sys
import threading
import time
import traceback
import warnings
import weakref
import builtins
import pickle
import numpy as np
from ..util import cprint

class ClosedError(Exception):
    """Raised when an event handler receives a request to close the connection
    or discovers that the connection has been closed."""
    pass

class NoResultError(Exception):
    """Raised when a request for the return value of a remote call fails
    because the call has not yet returned."""
    pass

class RemoteExceptionWarning(UserWarning):
    """Emitted when a request to a remote object results in an Exception """
    pass

class RemoteEventHandler(object):
    """
    This class handles communication between two processes. One instance is present on 
    each process and listens for communication from the other process. This enables
    (amongst other things) ObjectProxy instances to look up their attributes and call 
    their methods.
    
    This class is responsible for carrying out actions on behalf of the remote process.
    Each instance holds one end of a Connection which allows python
    objects to be passed between processes.
    
    For the most common operations, see _import(), close(), and transfer()
    
    To handle and respond to incoming requests, RemoteEventHandler requires that its
    processRequests method is called repeatedly (this is usually handled by the Process
    classes defined in multiprocess.processes).
    
    
    
    
    """
    handlers = {}

    def __init__(self, connection, name, pid, debug=False):
        if False:
            return 10
        self.debug = debug
        self.conn = connection
        self.name = name
        self.results = {}
        self.resultLock = threading.RLock()
        self.proxies = {}
        self.proxyLock = threading.RLock()
        self.proxyOptions = {'callSync': 'sync', 'timeout': 10, 'returnType': 'auto', 'autoProxy': False, 'deferGetattr': False, 'noProxyTypes': [type(None), str, bytes, int, float, tuple, list, dict, LocalObjectProxy, ObjectProxy]}
        self.optsLock = threading.RLock()
        self.nextRequestId = 0
        self.exited = False
        self.processLock = threading.RLock()
        self.sendLock = threading.RLock()
        if pid is None:
            connection.send(os.getpid())
            pid = connection.recv()
        RemoteEventHandler.handlers[pid] = self

    @classmethod
    def getHandler(cls, pid):
        if False:
            while True:
                i = 10
        try:
            return cls.handlers[pid]
        except:
            print(pid, cls.handlers)
            raise

    def debugMsg(self, msg, *args):
        if False:
            return 10
        if not self.debug:
            return
        cprint.cout(self.debug, '[%d] %s\n' % (os.getpid(), str(msg) % args), -1)

    def getProxyOption(self, opt):
        if False:
            print('Hello World!')
        with self.optsLock:
            return self.proxyOptions[opt]

    def setProxyOptions(self, **kwds):
        if False:
            i = 10
            return i + 15
        '\n        Set the default behavior options for object proxies.\n        See ObjectProxy._setProxyOptions for more info.\n        '
        with self.optsLock:
            self.proxyOptions.update(kwds)

    def processRequests(self):
        if False:
            i = 10
            return i + 15
        'Process all pending requests from the pipe, return\n        after no more events are immediately available. (non-blocking)\n        Returns the number of events processed.\n        '
        with self.processLock:
            if self.exited:
                self.debugMsg('  processRequests: exited already; raise ClosedError.')
                raise ClosedError()
            numProcessed = 0
            while self.conn.poll():
                try:
                    self.handleRequest()
                    numProcessed += 1
                except ClosedError:
                    self.debugMsg('processRequests: got ClosedError from handleRequest; setting exited=True.')
                    self.exited = True
                    raise
                except:
                    print('Error in process %s' % self.name)
                    sys.excepthook(*sys.exc_info())
            if numProcessed > 0:
                self.debugMsg('processRequests: finished %d requests', numProcessed)
            return numProcessed

    def handleRequest(self):
        if False:
            i = 10
            return i + 15
        'Handle a single request from the remote process. \n        Blocks until a request is available.'
        result = None
        while True:
            try:
                (cmd, reqId, nByteMsgs, optStr) = self.conn.recv()
                break
            except EOFError:
                self.debugMsg('  handleRequest: got EOFError from recv; raise ClosedError.')
                raise ClosedError()
            except IOError as err:
                if err.errno == 4:
                    self.debugMsg('  handleRequest: got IOError 4 from recv; try again.')
                    continue
                else:
                    self.debugMsg('  handleRequest: got IOError %d from recv (%s); raise ClosedError.', err.errno, err.strerror)
                    raise ClosedError()
        self.debugMsg('  handleRequest: received %s %s', cmd, reqId)
        byteData = []
        if nByteMsgs > 0:
            self.debugMsg('    handleRequest: reading %d byte messages', nByteMsgs)
        for i in range(nByteMsgs):
            while True:
                try:
                    byteData.append(self.conn.recv_bytes())
                    break
                except EOFError:
                    self.debugMsg('    handleRequest: got EOF while reading byte messages; raise ClosedError.')
                    raise ClosedError()
                except IOError as err:
                    if err.errno == 4:
                        self.debugMsg('    handleRequest: got IOError 4 while reading byte messages; try again.')
                        continue
                    else:
                        self.debugMsg('    handleRequest: got IOError while reading byte messages; raise ClosedError.')
                        raise ClosedError()
        try:
            if cmd == 'result' or cmd == 'error':
                resultId = reqId
                reqId = None
            opts = pickle.loads(optStr)
            self.debugMsg('    handleRequest: id=%s opts=%s', reqId, opts)
            returnType = opts.get('returnType', 'auto')
            if cmd == 'result':
                with self.resultLock:
                    self.results[resultId] = ('result', opts['result'])
            elif cmd == 'error':
                with self.resultLock:
                    self.results[resultId] = ('error', (opts['exception'], opts['excString']))
            elif cmd == 'getObjAttr':
                result = getattr(opts['obj'], opts['attr'])
            elif cmd == 'callObj':
                obj = opts['obj']
                fnargs = opts['args']
                fnkwds = opts['kwds']
                if len(byteData) > 0:
                    for (i, arg) in enumerate(fnargs):
                        if isinstance(arg, tuple) and len(arg) > 0 and (arg[0] == '__byte_message__'):
                            ind = arg[1]
                            (dtype, shape) = arg[2]
                            fnargs[i] = np.frombuffer(byteData[ind], dtype=dtype).reshape(shape)
                    for (k, arg) in fnkwds.items():
                        if isinstance(arg, tuple) and len(arg) > 0 and (arg[0] == '__byte_message__'):
                            ind = arg[1]
                            (dtype, shape) = arg[2]
                            fnkwds[k] = np.frombuffer(byteData[ind], dtype=dtype).reshape(shape)
                if len(fnkwds) == 0:
                    try:
                        result = obj(*fnargs)
                    except:
                        print('Failed to call object %s: %d, %s' % (obj, len(fnargs), fnargs[1:]))
                        raise
                else:
                    result = obj(*fnargs, **fnkwds)
            elif cmd == 'getObjValue':
                result = opts['obj']
                returnType = 'value'
            elif cmd == 'transfer':
                result = opts['obj']
                returnType = 'proxy'
            elif cmd == 'transferArray':
                result = np.frombuffer(byteData[0], dtype=opts['dtype']).reshape(opts['shape'])
                returnType = 'proxy'
            elif cmd == 'import':
                name = opts['module']
                fromlist = opts.get('fromlist', [])
                mod = builtins.__import__(name, fromlist=fromlist)
                if len(fromlist) == 0:
                    parts = name.lstrip('.').split('.')
                    result = mod
                    for part in parts[1:]:
                        result = getattr(result, part)
                else:
                    result = map(mod.__getattr__, fromlist)
            elif cmd == 'del':
                LocalObjectProxy.releaseProxyId(opts['proxyId'])
            elif cmd == 'close':
                if reqId is not None:
                    result = True
                    returnType = 'value'
            exc = None
        except:
            exc = sys.exc_info()
        if reqId is not None:
            if exc is None:
                self.debugMsg('    handleRequest: sending return value for %d: %s', reqId, result)
                if returnType == 'auto':
                    with self.optsLock:
                        noProxyTypes = self.proxyOptions['noProxyTypes']
                    result = self.autoProxy(result, noProxyTypes)
                elif returnType == 'proxy':
                    result = LocalObjectProxy(result)
                try:
                    self.replyResult(reqId, result)
                except:
                    sys.excepthook(*sys.exc_info())
                    self.replyError(reqId, *sys.exc_info())
            else:
                self.debugMsg('    handleRequest: returning exception for %d', reqId)
                self.replyError(reqId, *exc)
        elif exc is not None:
            sys.excepthook(*exc)
        if cmd == 'close':
            if opts.get('noCleanup', False) is True:
                os._exit(0)
            else:
                raise ClosedError()

    def replyResult(self, reqId, result):
        if False:
            return 10
        self.send(request='result', reqId=reqId, callSync='off', opts=dict(result=result))

    def replyError(self, reqId, *exc):
        if False:
            print('Hello World!')
        excStr = traceback.format_exception(*exc)
        try:
            self.send(request='error', reqId=reqId, callSync='off', opts=dict(exception=exc[1], excString=excStr))
        except:
            self.send(request='error', reqId=reqId, callSync='off', opts=dict(exception=None, excString=excStr))

    def send(self, request, opts=None, reqId=None, callSync='sync', timeout=10, returnType=None, byteData=None, **kwds):
        if False:
            return 10
        "Send a request or return packet to the remote process.\n        Generally it is not necessary to call this method directly; it is for internal use.\n        (The docstring has information that is nevertheless useful to the programmer\n        as it describes the internal protocol used to communicate between processes)\n        \n        ==============  ====================================================================\n        **Arguments:**\n        request         String describing the type of request being sent (see below)\n        reqId           Integer uniquely linking a result back to the request that generated\n                        it. (most requests leave this blank)\n        callSync        'sync':  return the actual result of the request\n                        'async': return a Request object which can be used to look up the\n                                result later\n                        'off':   return no result\n        timeout         Time in seconds to wait for a response when callSync=='sync'\n        opts            Extra arguments sent to the remote process that determine the way\n                        the request will be handled (see below)\n        returnType      'proxy', 'value', or 'auto'\n        byteData        If specified, this is a list of objects to be sent as byte messages\n                        to the remote process.\n                        This is used to send large arrays without the cost of pickling.\n        ==============  ====================================================================\n        \n        Description of request strings and options allowed for each:\n        \n        =============  =============  ========================================================\n        request        option         description\n        -------------  -------------  --------------------------------------------------------\n        getObjAttr                    Request the remote process return (proxy to) an\n                                      attribute of an object.\n                       obj            reference to object whose attribute should be \n                                      returned\n                       attr           string name of attribute to return\n                       returnValue    bool or 'auto' indicating whether to return a proxy or\n                                      the actual value. \n                       \n        callObj                       Request the remote process call a function or \n                                      method. If a request ID is given, then the call's\n                                      return value will be sent back (or information\n                                      about the error that occurred while running the\n                                      function)\n                       obj            the (reference to) object to call\n                       args           tuple of arguments to pass to callable\n                       kwds           dict of keyword arguments to pass to callable\n                       returnValue    bool or 'auto' indicating whether to return a proxy or\n                                      the actual value. \n                       \n        getObjValue                   Request the remote process return the value of\n                                      a proxied object (must be picklable)\n                       obj            reference to object whose value should be returned\n                       \n        transfer                      Copy an object to the remote process and request\n                                      it return a proxy for the new object.\n                       obj            The object to transfer.\n                       \n        import                        Request the remote process import new symbols\n                                      and return proxy(ies) to the imported objects\n                       module         the string name of the module to import\n                       fromlist       optional list of string names to import from module\n                       \n        del                           Inform the remote process that a proxy has been \n                                      released (thus the remote process may be able to \n                                      release the original object)\n                       proxyId        id of proxy which is no longer referenced by \n                                      remote host\n                                      \n        close                         Instruct the remote process to stop its event loop\n                                      and exit. Optionally, this request may return a \n                                      confirmation.\n            \n        result                        Inform the remote process that its request has \n                                      been processed                        \n                       result         return value of a request\n                       \n        error                         Inform the remote process that its request failed\n                       exception      the Exception that was raised (or None if the \n                                      exception could not be pickled)\n                       excString      string-formatted version of the exception and \n                                      traceback\n        =============  =====================================================================\n        "
        if self.exited:
            self.debugMsg('  send: exited already; raise ClosedError.')
            raise ClosedError()
        with self.sendLock:
            if opts is None:
                opts = {}
            assert callSync in ['off', 'sync', 'async'], 'callSync must be one of "off", "sync", or "async" (got %r)' % callSync
            if reqId is None:
                if callSync != 'off':
                    reqId = self.nextRequestId
                    self.nextRequestId += 1
            else:
                assert request in ['result', 'error']
            if returnType is not None:
                opts['returnType'] = returnType
            try:
                optStr = pickle.dumps(opts)
            except:
                print('====  Error pickling this object:  ====')
                print(opts)
                print('=======================================')
                raise
            nByteMsgs = 0
            if byteData is not None:
                nByteMsgs = len(byteData)
            request = (request, reqId, nByteMsgs, optStr)
            self.debugMsg('send request: cmd=%s nByteMsgs=%d id=%s opts=%s', request[0], nByteMsgs, reqId, opts)
            self.conn.send(request)
            if byteData is not None:
                for obj in byteData:
                    self.conn.send_bytes(bytes(obj))
                self.debugMsg('  sent %d byte messages', len(byteData))
            self.debugMsg('  call sync: %s', callSync)
            if callSync == 'off':
                return
        req = Request(self, reqId, description=str(request), timeout=timeout)
        if callSync == 'async':
            return req
        if callSync == 'sync':
            return req.result()

    def close(self, callSync='off', noCleanup=False, **kwds):
        if False:
            print('Hello World!')
        try:
            self.send(request='close', opts=dict(noCleanup=noCleanup), callSync=callSync, **kwds)
            self.exited = True
        except ClosedError:
            pass

    def getResult(self, reqId):
        if False:
            i = 10
            return i + 15
        with self.resultLock:
            haveResult = reqId in self.results
        if not haveResult:
            try:
                self.processRequests()
            except ClosedError:
                pass
        with self.resultLock:
            if reqId not in self.results:
                raise NoResultError()
            (status, result) = self.results.pop(reqId)
        if status == 'result':
            return result
        elif status == 'error':
            (exc, excStr) = result
            if exc is not None:
                normal = ['AttributeError']
                if not any((excStr[-1].startswith(x) for x in normal)):
                    warnings.warn('===== Remote process raised exception on request: =====', RemoteExceptionWarning)
                    warnings.warn(''.join(excStr), RemoteExceptionWarning)
                    warnings.warn('===== Local Traceback to request follows: =====', RemoteExceptionWarning)
                raise exc
            else:
                print(''.join(excStr))
                raise Exception('Error getting result. See above for exception from remote process.')
        else:
            raise Exception('Internal error.')

    def _import(self, mod, **kwds):
        if False:
            print('Hello World!')
        "\n        Request the remote process import a module (or symbols from a module)\n        and return the proxied results. Uses built-in __import__() function, but \n        adds a bit more processing:\n        \n            _import('module')  =>  returns module\n            _import('module.submodule')  =>  returns submodule \n                                             (note this differs from behavior of __import__)\n            _import('module', fromlist=[name1, name2, ...])  =>  returns [module.name1, module.name2, ...]\n                                             (this also differs from behavior of __import__)\n            \n        "
        return self.send(request='import', callSync='sync', opts=dict(module=mod), **kwds)

    def getObjAttr(self, obj, attr, **kwds):
        if False:
            i = 10
            return i + 15
        return self.send(request='getObjAttr', opts=dict(obj=obj, attr=attr), **kwds)

    def getObjValue(self, obj, **kwds):
        if False:
            i = 10
            return i + 15
        return self.send(request='getObjValue', opts=dict(obj=obj), **kwds)

    def callObj(self, obj, args, kwds, **opts):
        if False:
            while True:
                i = 10
        opts = opts.copy()
        args = list(args)
        with self.optsLock:
            noProxyTypes = opts.pop('noProxyTypes', None)
            if noProxyTypes is None:
                noProxyTypes = self.proxyOptions['noProxyTypes']
            autoProxy = opts.pop('autoProxy', self.proxyOptions['autoProxy'])
        if autoProxy is True:
            args = [self.autoProxy(v, noProxyTypes) for v in args]
            for (k, v) in kwds.items():
                opts[k] = self.autoProxy(v, noProxyTypes)
        byteMsgs = []
        for (i, arg) in enumerate(args):
            if arg.__class__ == np.ndarray:
                args[i] = ('__byte_message__', len(byteMsgs), (arg.dtype, arg.shape))
                byteMsgs.append(arg)
        for (k, v) in kwds.items():
            if v.__class__ == np.ndarray:
                kwds[k] = ('__byte_message__', len(byteMsgs), (v.dtype, v.shape))
                byteMsgs.append(v)
        return self.send(request='callObj', opts=dict(obj=obj, args=args, kwds=kwds), byteData=byteMsgs, **opts)

    def registerProxy(self, proxy):
        if False:
            i = 10
            return i + 15
        with self.proxyLock:
            ref = weakref.ref(proxy, self.deleteProxy)
            self.proxies[ref] = proxy._proxyId

    def deleteProxy(self, ref):
        if False:
            print('Hello World!')
        if self.send is None:
            return
        with self.proxyLock:
            proxyId = self.proxies.pop(ref)
        try:
            self.send(request='del', opts=dict(proxyId=proxyId), callSync='off')
        except ClosedError:
            pass

    def transfer(self, obj, **kwds):
        if False:
            for i in range(10):
                print('nop')
        '\n        Transfer an object by value to the remote host (the object must be picklable) \n        and return a proxy for the new remote object.\n        '
        if obj.__class__ is np.ndarray:
            opts = {'dtype': obj.dtype, 'shape': obj.shape}
            return self.send(request='transferArray', opts=opts, byteData=[obj], **kwds)
        else:
            return self.send(request='transfer', opts=dict(obj=obj), **kwds)

    def autoProxy(self, obj, noProxyTypes):
        if False:
            print('Hello World!')
        for typ in noProxyTypes:
            if isinstance(obj, typ):
                return obj
        return LocalObjectProxy(obj)

class Request(object):
    """
    Request objects are returned when calling an ObjectProxy in asynchronous mode
    or if a synchronous call has timed out. Use hasResult() to ask whether
    the result of the call has been returned yet. Use result() to get
    the returned value.
    """

    def __init__(self, process, reqId, description=None, timeout=10):
        if False:
            print('Hello World!')
        self.proc = process
        self.description = description
        self.reqId = reqId
        self.gotResult = False
        self._result = None
        self.timeout = timeout

    def result(self, block=True, timeout=None):
        if False:
            i = 10
            return i + 15
        "\n        Return the result for this request. \n        \n        If block is True, wait until the result has arrived or *timeout* seconds passes.\n        If the timeout is reached, raise NoResultError. (use timeout=None to disable)\n        If block is False, raise NoResultError immediately if the result has not arrived yet.\n        \n        If the process's connection has closed before the result arrives, raise ClosedError.\n        "
        if self.gotResult:
            return self._result
        if timeout is None:
            timeout = self.timeout
        if block:
            start = time.time()
            while not self.hasResult():
                if self.proc.exited:
                    raise ClosedError()
                time.sleep(0.005)
                if timeout >= 0 and time.time() - start > timeout:
                    print('Request timed out: %s' % self.description)
                    import traceback
                    traceback.print_stack()
                    raise NoResultError()
            return self._result
        else:
            self._result = self.proc.getResult(self.reqId)
            self.gotResult = True
            return self._result

    def hasResult(self):
        if False:
            return 10
        'Returns True if the result for this request has arrived.'
        try:
            self.result(block=False)
        except NoResultError:
            pass
        return self.gotResult

class LocalObjectProxy(object):
    """
    Used for wrapping local objects to ensure that they are send by proxy to a remote host.
    Note that 'proxy' is just a shorter alias for LocalObjectProxy.
    
    For example::
    
        data = [1,2,3,4,5]
        remotePlot.plot(data)         ## by default, lists are pickled and sent by value
        remotePlot.plot(proxy(data))  ## force the object to be sent by proxy
    
    """
    nextProxyId = 0
    proxiedObjects = {}

    @classmethod
    def registerObject(cls, obj):
        if False:
            return 10
        pid = cls.nextProxyId
        cls.nextProxyId += 1
        cls.proxiedObjects[pid] = obj
        return pid

    @classmethod
    def lookupProxyId(cls, pid):
        if False:
            print('Hello World!')
        return cls.proxiedObjects[pid]

    @classmethod
    def releaseProxyId(cls, pid):
        if False:
            for i in range(10):
                print('nop')
        del cls.proxiedObjects[pid]

    def __init__(self, obj, **opts):
        if False:
            while True:
                i = 10
        "\n        Create a 'local' proxy object that, when sent to a remote host,\n        will appear as a normal ObjectProxy to *obj*. \n        Any extra keyword arguments are passed to proxy._setProxyOptions()\n        on the remote side.\n        "
        self.processId = os.getpid()
        self.typeStr = repr(obj)
        self.obj = obj
        self.opts = opts

    def __reduce__(self):
        if False:
            return 10
        pid = LocalObjectProxy.registerObject(self.obj)
        return (unpickleObjectProxy, (self.processId, pid, self.typeStr, None, self.opts))
proxy = LocalObjectProxy

def unpickleObjectProxy(processId, proxyId, typeStr, attributes=None, opts=None):
    if False:
        for i in range(10):
            print('nop')
    if processId == os.getpid():
        obj = LocalObjectProxy.lookupProxyId(proxyId)
        if attributes is not None:
            for attr in attributes:
                obj = getattr(obj, attr)
        return obj
    else:
        proxy = ObjectProxy(processId, proxyId=proxyId, typeStr=typeStr)
        if opts is not None:
            proxy._setProxyOptions(**opts)
        return proxy

class ObjectProxy(object):
    """
    Proxy to an object stored by the remote process. Proxies are created
    by calling Process._import(), Process.transfer(), or by requesting/calling
    attributes on existing proxy objects.
    
    For the most part, this object can be used exactly as if it
    were a local object::
    
        rsys = proc._import('sys')   # returns proxy to sys module on remote process
        rsys.stdout                  # proxy to remote sys.stdout
        rsys.stdout.write            # proxy to remote sys.stdout.write
        rsys.stdout.write('hello')   # calls sys.stdout.write('hello') on remote machine
                                     # and returns the result (None)
    
    When calling a proxy to a remote function, the call can be made synchronous
    (result of call is returned immediately), asynchronous (result is returned later),
    or return can be disabled entirely::
    
        ros = proc._import('os')
        
        ## synchronous call; result is returned immediately
        pid = ros.getpid()
        
        ## asynchronous call
        request = ros.getpid(_callSync='async')
        while not request.hasResult():
            time.sleep(0.01)
        pid = request.result()
        
        ## disable return when we know it isn't needed
        rsys.stdout.write('hello', _callSync='off')
    
    Additionally, values returned from a remote function call are automatically
    returned either by value (must be picklable) or by proxy. 
    This behavior can be forced::
    
        rnp = proc._import('numpy')
        arrProxy = rnp.array([1,2,3,4], _returnType='proxy')
        arrValue = rnp.array([1,2,3,4], _returnType='value')
    
    The default callSync and returnType behaviors (as well as others) can be set 
    for each proxy individually using ObjectProxy._setProxyOptions() or globally using 
    proc.setProxyOptions(). 
    
    """

    def __init__(self, processId, proxyId, typeStr='', parent=None):
        if False:
            print('Hello World!')
        object.__init__(self)
        self.__dict__['_processId'] = processId
        self.__dict__['_typeStr'] = typeStr
        self.__dict__['_proxyId'] = proxyId
        self.__dict__['_attributes'] = ()
        self.__dict__['_proxyOptions'] = {'callSync': None, 'timeout': None, 'returnType': None, 'deferGetattr': None, 'noProxyTypes': None, 'autoProxy': None}
        self.__dict__['_handler'] = RemoteEventHandler.getHandler(processId)
        self.__dict__['_handler'].registerProxy(self)

    def _setProxyOptions(self, **kwds):
        if False:
            print('Hello World!')
        "\n        Change the behavior of this proxy. For all options, a value of None\n        will cause the proxy to instead use the default behavior defined\n        by its parent Process.\n        \n        Options are:\n        \n        =============  =============================================================\n        callSync       'sync', 'async', 'off', or None. \n                       If 'async', then calling methods will return a Request object\n                       which can be used to inquire later about the result of the \n                       method call.\n                       If 'sync', then calling a method\n                       will block until the remote process has returned its result\n                       or the timeout has elapsed (in this case, a Request object\n                       is returned instead).\n                       If 'off', then the remote process is instructed _not_ to \n                       reply and the method call will return None immediately.\n        returnType     'auto', 'proxy', 'value', or None. \n                       If 'proxy', then the value returned when calling a method\n                       will be a proxy to the object on the remote process.\n                       If 'value', then attempt to pickle the returned object and\n                       send it back.\n                       If 'auto', then the decision is made by consulting the\n                       'noProxyTypes' option.\n        autoProxy      bool or None. If True, arguments to __call__ are \n                       automatically converted to proxy unless their type is \n                       listed in noProxyTypes (see below). If False, arguments\n                       are left untouched. Use proxy(obj) to manually convert\n                       arguments before sending. \n        timeout        float or None. Length of time to wait during synchronous \n                       requests before returning a Request object instead.\n        deferGetattr   True, False, or None. \n                       If False, all attribute requests will be sent to the remote \n                       process immediately and will block until a response is\n                       received (or timeout has elapsed).\n                       If True, requesting an attribute from the proxy returns a\n                       new proxy immediately. The remote process is _not_ contacted\n                       to make this request. This is faster, but it is possible to \n                       request an attribute that does not exist on the proxied\n                       object. In this case, AttributeError will not be raised\n                       until an attempt is made to look up the attribute on the\n                       remote process.\n        noProxyTypes   List of object types that should _not_ be proxied when\n                       sent to the remote process.\n        =============  =============================================================\n        "
        for k in kwds:
            if k not in self._proxyOptions:
                raise KeyError("Unrecognized proxy option '%s'" % k)
        self._proxyOptions.update(kwds)

    def _getValue(self):
        if False:
            print('Hello World!')
        '\n        Return the value of the proxied object\n        (the remote object must be picklable)\n        '
        return self._handler.getObjValue(self)

    def _getProxyOption(self, opt):
        if False:
            while True:
                i = 10
        val = self._proxyOptions[opt]
        if val is None:
            return self._handler.getProxyOption(opt)
        return val

    def _getProxyOptions(self):
        if False:
            i = 10
            return i + 15
        return dict([(k, self._getProxyOption(k)) for k in self._proxyOptions])

    def __reduce__(self):
        if False:
            for i in range(10):
                print('nop')
        return (unpickleObjectProxy, (self._processId, self._proxyId, self._typeStr, self._attributes))

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<ObjectProxy for process %d, object 0x%x: %s >' % (self._processId, self._proxyId, self._typeStr)

    def __getattr__(self, attr, **kwds):
        if False:
            while True:
                i = 10
        "\n        Calls __getattr__ on the remote object and returns the attribute\n        by value or by proxy depending on the options set (see\n        ObjectProxy._setProxyOptions and RemoteEventHandler.setProxyOptions)\n        \n        If the option 'deferGetattr' is True for this proxy, then a new proxy object\n        is returned _without_ asking the remote object whether the named attribute exists.\n        This can save time when making multiple chained attribute requests,\n        but may also defer a possible AttributeError until later, making\n        them more difficult to debug.\n        "
        opts = self._getProxyOptions()
        for k in opts:
            if '_' + k in kwds:
                opts[k] = kwds.pop('_' + k)
        if opts['deferGetattr'] is True:
            return self._deferredAttr(attr)
        else:
            return self._handler.getObjAttr(self, attr, **opts)

    def _deferredAttr(self, attr):
        if False:
            return 10
        return DeferredObjectProxy(self, attr)

    def __call__(self, *args, **kwds):
        if False:
            for i in range(10):
                print('nop')
        "\n        Attempts to call the proxied object from the remote process.\n        Accepts extra keyword arguments:\n        \n            _callSync    'off', 'sync', or 'async'\n            _returnType   'value', 'proxy', or 'auto'\n        \n        If the remote call raises an exception on the remote process,\n        it will be re-raised on the local process.\n        \n        "
        opts = self._getProxyOptions()
        for k in opts:
            if '_' + k in kwds:
                opts[k] = kwds.pop('_' + k)
        return self._handler.callObj(obj=self, args=args, kwds=kwds, **opts)

    def _getSpecialAttr(self, attr):
        if False:
            i = 10
            return i + 15
        return self._deferredAttr(attr)

    def __getitem__(self, *args):
        if False:
            print('Hello World!')
        return self._getSpecialAttr('__getitem__')(*args)

    def __setitem__(self, *args):
        if False:
            while True:
                i = 10
        return self._getSpecialAttr('__setitem__')(*args, _callSync='off')

    def __setattr__(self, *args):
        if False:
            return 10
        return self._getSpecialAttr('__setattr__')(*args, _callSync='off')

    def __str__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        return self._getSpecialAttr('__str__')(*args, _returnType='value')

    def __len__(self, *args):
        if False:
            return 10
        return self._getSpecialAttr('__len__')(*args)

    def __add__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        return self._getSpecialAttr('__add__')(*args)

    def __sub__(self, *args):
        if False:
            return 10
        return self._getSpecialAttr('__sub__')(*args)

    def __div__(self, *args):
        if False:
            print('Hello World!')
        return self._getSpecialAttr('__div__')(*args)

    def __truediv__(self, *args):
        if False:
            return 10
        return self._getSpecialAttr('__truediv__')(*args)

    def __floordiv__(self, *args):
        if False:
            print('Hello World!')
        return self._getSpecialAttr('__floordiv__')(*args)

    def __mul__(self, *args):
        if False:
            while True:
                i = 10
        return self._getSpecialAttr('__mul__')(*args)

    def __pow__(self, *args):
        if False:
            print('Hello World!')
        return self._getSpecialAttr('__pow__')(*args)

    def __iadd__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        return self._getSpecialAttr('__iadd__')(*args, _callSync='off')

    def __isub__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        return self._getSpecialAttr('__isub__')(*args, _callSync='off')

    def __idiv__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        return self._getSpecialAttr('__idiv__')(*args, _callSync='off')

    def __itruediv__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        return self._getSpecialAttr('__itruediv__')(*args, _callSync='off')

    def __ifloordiv__(self, *args):
        if False:
            return 10
        return self._getSpecialAttr('__ifloordiv__')(*args, _callSync='off')

    def __imul__(self, *args):
        if False:
            while True:
                i = 10
        return self._getSpecialAttr('__imul__')(*args, _callSync='off')

    def __ipow__(self, *args):
        if False:
            i = 10
            return i + 15
        return self._getSpecialAttr('__ipow__')(*args, _callSync='off')

    def __rshift__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        return self._getSpecialAttr('__rshift__')(*args)

    def __lshift__(self, *args):
        if False:
            return 10
        return self._getSpecialAttr('__lshift__')(*args)

    def __irshift__(self, *args):
        if False:
            return 10
        return self._getSpecialAttr('__irshift__')(*args, _callSync='off')

    def __ilshift__(self, *args):
        if False:
            return 10
        return self._getSpecialAttr('__ilshift__')(*args, _callSync='off')

    def __eq__(self, *args):
        if False:
            i = 10
            return i + 15
        return self._getSpecialAttr('__eq__')(*args)

    def __ne__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        return self._getSpecialAttr('__ne__')(*args)

    def __lt__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        return self._getSpecialAttr('__lt__')(*args)

    def __gt__(self, *args):
        if False:
            print('Hello World!')
        return self._getSpecialAttr('__gt__')(*args)

    def __le__(self, *args):
        if False:
            i = 10
            return i + 15
        return self._getSpecialAttr('__le__')(*args)

    def __ge__(self, *args):
        if False:
            print('Hello World!')
        return self._getSpecialAttr('__ge__')(*args)

    def __and__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        return self._getSpecialAttr('__and__')(*args)

    def __or__(self, *args):
        if False:
            while True:
                i = 10
        return self._getSpecialAttr('__or__')(*args)

    def __xor__(self, *args):
        if False:
            while True:
                i = 10
        return self._getSpecialAttr('__xor__')(*args)

    def __iand__(self, *args):
        if False:
            return 10
        return self._getSpecialAttr('__iand__')(*args, _callSync='off')

    def __ior__(self, *args):
        if False:
            return 10
        return self._getSpecialAttr('__ior__')(*args, _callSync='off')

    def __ixor__(self, *args):
        if False:
            while True:
                i = 10
        return self._getSpecialAttr('__ixor__')(*args, _callSync='off')

    def __mod__(self, *args):
        if False:
            while True:
                i = 10
        return self._getSpecialAttr('__mod__')(*args)

    def __radd__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        return self._getSpecialAttr('__radd__')(*args)

    def __rsub__(self, *args):
        if False:
            while True:
                i = 10
        return self._getSpecialAttr('__rsub__')(*args)

    def __rdiv__(self, *args):
        if False:
            while True:
                i = 10
        return self._getSpecialAttr('__rdiv__')(*args)

    def __rfloordiv__(self, *args):
        if False:
            while True:
                i = 10
        return self._getSpecialAttr('__rfloordiv__')(*args)

    def __rtruediv__(self, *args):
        if False:
            while True:
                i = 10
        return self._getSpecialAttr('__rtruediv__')(*args)

    def __rmul__(self, *args):
        if False:
            while True:
                i = 10
        return self._getSpecialAttr('__rmul__')(*args)

    def __rpow__(self, *args):
        if False:
            while True:
                i = 10
        return self._getSpecialAttr('__rpow__')(*args)

    def __rrshift__(self, *args):
        if False:
            i = 10
            return i + 15
        return self._getSpecialAttr('__rrshift__')(*args)

    def __rlshift__(self, *args):
        if False:
            while True:
                i = 10
        return self._getSpecialAttr('__rlshift__')(*args)

    def __rand__(self, *args):
        if False:
            i = 10
            return i + 15
        return self._getSpecialAttr('__rand__')(*args)

    def __ror__(self, *args):
        if False:
            i = 10
            return i + 15
        return self._getSpecialAttr('__ror__')(*args)

    def __rxor__(self, *args):
        if False:
            i = 10
            return i + 15
        return self._getSpecialAttr('__ror__')(*args)

    def __rmod__(self, *args):
        if False:
            print('Hello World!')
        return self._getSpecialAttr('__rmod__')(*args)

    def __hash__(self):
        if False:
            while True:
                i = 10
        return id(self)

class DeferredObjectProxy(ObjectProxy):
    """
    This class represents an attribute (or sub-attribute) of a proxied object.
    It is used to speed up attribute requests. Take the following scenario::
    
        rsys = proc._import('sys')
        rsys.stdout.write('hello')
        
    For this simple example, a total of 4 synchronous requests are made to 
    the remote process: 
    
    1) import sys
    2) getattr(sys, 'stdout')
    3) getattr(stdout, 'write')
    4) write('hello')
    
    This takes a lot longer than running the equivalent code locally. To
    speed things up, we can 'defer' the two attribute lookups so they are
    only carried out when neccessary::
    
        rsys = proc._import('sys')
        rsys._setProxyOptions(deferGetattr=True)
        rsys.stdout.write('hello')
        
    This example only makes two requests to the remote process; the two 
    attribute lookups immediately return DeferredObjectProxy instances 
    immediately without contacting the remote process. When the call 
    to write() is made, all attribute requests are processed at the same time.
    
    Note that if the attributes requested do not exist on the remote object, 
    making the call to write() will raise an AttributeError.
    """

    def __init__(self, parentProxy, attribute):
        if False:
            for i in range(10):
                print('nop')
        for k in ['_processId', '_typeStr', '_proxyId', '_handler']:
            self.__dict__[k] = getattr(parentProxy, k)
        self.__dict__['_parent'] = parentProxy
        self.__dict__['_attributes'] = parentProxy._attributes + (attribute,)
        self.__dict__['_proxyOptions'] = parentProxy._proxyOptions.copy()

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return ObjectProxy.__repr__(self) + '.' + '.'.join(self._attributes)

    def _undefer(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a non-deferred ObjectProxy referencing the same object\n        '
        return self._parent.__getattr__(self._attributes[-1], _deferGetattr=False)