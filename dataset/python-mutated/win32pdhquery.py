"""
Performance Data Helper (PDH) Query Classes

Wrapper classes for end-users and high-level access to the PDH query
mechanisms.  PDH is a win32-specific mechanism for accessing the
performance data made available by the system.  The Python for Windows
PDH module does not implement the "Registry" interface, implementing
the more straightforward Query-based mechanism.

The basic idea of a PDH Query is an object which can query the system
about the status of any number of "counters."  The counters are paths
to a particular piece of performance data.  For instance, the path 
'\\Memory\\Available Bytes' describes just about exactly what it says
it does, the amount of free memory on the default computer expressed 
in Bytes.  These paths can be considerably more complex than this, 
but part of the point of this wrapper module is to hide that
complexity from the end-user/programmer.

EXAMPLE: A more complex Path
	'\\\\RAISTLIN\\PhysicalDisk(_Total)\\Avg. Disk Bytes/Read'
	Raistlin --> Computer Name
	PhysicalDisk --> Object Name
	_Total --> The particular Instance (in this case, all instances, i.e. all drives)
	Avg. Disk Bytes/Read --> The piece of data being monitored.

EXAMPLE: Collecting Data with a Query
	As an example, the following code implements a logger which allows the
	user to choose what counters they would like to log, and logs those
	counters for 30 seconds, at two-second intervals.
	
	query = Query()
	query.addcounterbybrowsing()
	query.collectdatafor(30,2)
	
	The data is now stored in a list of lists as:
	query.curresults
	
	The counters(paths) which were used to collect the data are:
	query.curpaths
	
	You can use the win32pdh.ParseCounterPath(path) utility function
	to turn the paths into more easily read values for your task, or
	write the data to a file, or do whatever you want with it.

OTHER NOTABLE METHODS:
	query.collectdatawhile(period) # start a logging thread for collecting data
	query.collectdatawhile_stop() # signal the logging thread to stop logging
	query.collectdata() # run the query only once
	query.addperfcounter(object, counter, machine=None) # add a standard performance counter
	query.addinstcounter(object, counter,machine=None,objtype = 'Process',volatile=1,format = win32pdh.PDH_FMT_LONG) # add a possibly volatile counter

### Known bugs and limitations ###
Due to a problem with threading under the PythonWin interpreter, there
will be no data logged if the PythonWin window is not the foreground
application.  Workaround: scripts using threading should be run in the
python.exe interpreter.

The volatile-counter handlers are possibly buggy, they haven't been
tested to any extent.  The wrapper Query makes it safe to pass invalid
paths (a -1 will be returned, or the Query will be totally ignored,
depending on the missing element), so you should be able to work around
the error by including all possible paths and filtering out the -1's.

There is no way I know of to stop a thread which is currently sleeping,
so you have to wait until the thread in collectdatawhile is activated
again.  This might become a problem in situations where the collection
period is multiple minutes (or hours, or whatever).

Should make the win32pdh.ParseCounter function available to the Query
classes as a method or something similar, so that it can be accessed
by programmes that have just picked up an instance from somewhere.

Should explicitly mention where QueryErrors can be raised, and create a
full test set to see if there are any uncaught win32api.error's still
hanging around.

When using the python.exe interpreter, the addcounterbybrowsing-
generated browser window is often hidden behind other windows.  No known
workaround other than Alt-tabing to reach the browser window.

### Other References ###
The win32pdhutil module (which should be in the %pythonroot%/win32/lib 
directory) provides quick-and-dirty utilities for one-off access to
variables from the PDH.  Almost everything in that module can be done
with a Query object, but it provides task-oriented functions for a
number of common one-off tasks.

If you can access the MS Developers Network Library, you can find
information about the PDH API as MS describes it.  For a background article,
try:
http://msdn.microsoft.com/library/en-us/dnperfmo/html/msdn_pdhlib.asp

The reference guide for the PDH API was last spotted at:
http://msdn.microsoft.com/library/en-us/perfmon/base/using_the_pdh_interface.asp


In general the Python version of the API is just a wrapper around the
Query-based version of this API (as far as I can see), so you can learn what
you need to from there.  From what I understand, the MSDN Online 
resources are available for the price of signing up for them.  I can't
guarantee how long that's supposed to last. (Or anything for that
matter).
http://premium.microsoft.com/isapi/devonly/prodinfo/msdnprod/msdnlib.idc?theURL=/msdn/library/sdkdoc/perfdata_4982.htm

The eventual plan is for my (Mike Fletcher's) Starship account to include
a section on NT Administration, and the Query is the first project
in this plan.  There should be an article describing the creation of
a simple logger there, but the example above is 90% of the work of
that project, so don't sweat it if you don't find anything there.
(currently the account hasn't been set up).
http://starship.skyport.net/crew/mcfletch/

If you need to contact me immediately, (why I can't imagine), you can
email me at mcfletch@golden.net, or just post your question to the
Python newsgroup with a catchy subject line.
news:comp.lang.python

### Other Stuff ###
The Query classes are by Mike Fletcher, with the working code
being corruptions of Mark Hammonds win32pdhutil module.

Use at your own risk, no warranties, no guarantees, no assurances,
if you use it, you accept the risk of using it, etceteras.

"""
import _thread
import copy
import time
import win32api
import win32pdh

class BaseQuery:
    """
    Provides wrapped access to the Performance Data Helper query
    objects, generally you should use the child class Query
    unless you have need of doing weird things :)

    This class supports two major working paradigms.  In the first,
    you open the query, and run it as many times as you need, closing
    the query when you're done with it.  This is suitable for static
    queries (ones where processes being monitored don't disappear).

    In the second, you allow the query to be opened each time and
    closed afterward.  This causes the base query object to be
    destroyed after each call.  Suitable for dynamic queries (ones
    which watch processes which might be closed while watching.)
    """

    def __init__(self, paths=None):
        if False:
            return 10
        "\n        The PDH Query object is initialised with a single, optional\n        list argument, that must be properly formatted PDH Counter\n        paths.  Generally this list will only be provided by the class\n        when it is being unpickled (removed from storage).  Normal\n        use is to call the class with no arguments and use the various\n        addcounter functions (particularly, for end user's, the use of\n        addcounterbybrowsing is the most common approach)  You might\n        want to provide the list directly if you want to hard-code the\n        elements with which your query deals (and thereby avoid the\n        overhead of unpickling the class).\n        "
        self.counters = []
        if paths:
            self.paths = paths
        else:
            self.paths = []
        self._base = None
        self.active = 0
        self.curpaths = []

    def addcounterbybrowsing(self, flags=win32pdh.PERF_DETAIL_WIZARD, windowtitle='Python Browser'):
        if False:
            while True:
                i = 10
        '\n        Adds possibly multiple paths to the paths attribute of the query,\n        does this by calling the standard counter browsing dialogue.  Within\n        this dialogue, find the counter you want to log, and click: Add,\n        repeat for every path you want to log, then click on close.  The\n        paths are appended to the non-volatile paths list for this class,\n        subclasses may create a function which parses the paths and decides\n        (via heuristics) whether to add the path to the volatile or non-volatile\n        path list.\n        e.g.:\n                query.addcounter()\n        '
        win32pdh.BrowseCounters(None, 0, self.paths.append, flags, windowtitle)

    def rawaddcounter(self, object, counter, instance=None, inum=-1, machine=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Adds a single counter path, without catching any exceptions.\n\n        See addcounter for details.\n        '
        path = win32pdh.MakeCounterPath((machine, object, instance, None, inum, counter))
        self.paths.append(path)

    def addcounter(self, object, counter, instance=None, inum=-1, machine=None):
        if False:
            print('Hello World!')
        "\n        Adds a single counter path to the paths attribute.  Normally\n        this will be called by a child class' speciality functions,\n        rather than being called directly by the user. (Though it isn't\n        hard to call manually, since almost everything is given a default)\n        This method is only functional when the query is closed (or hasn't\n        yet been opened).  This is to prevent conflict in multi-threaded\n        query applications).\n        e.g.:\n                query.addcounter('Memory','Available Bytes')\n        "
        if not self.active:
            try:
                self.rawaddcounter(object, counter, instance, inum, machine)
                return 0
            except win32api.error:
                return -1
        else:
            return -1

    def open(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Build the base query object for this wrapper,\n        then add all of the counters required for the query.\n        Raise a QueryError if we can't complete the functions.\n        If we are already open, then do nothing.\n        "
        if not self.active:
            self.curpaths = copy.copy(self.paths)
            try:
                base = win32pdh.OpenQuery()
                for path in self.paths:
                    try:
                        self.counters.append(win32pdh.AddCounter(base, path))
                    except win32api.error:
                        self.counters.append(0)
                        pass
                self._base = base
                self.active = 1
                return 0
            except:
                try:
                    self.killbase(base)
                except NameError:
                    pass
                self.active = 0
                self.curpaths = []
                raise QueryError(self)
        return 1

    def killbase(self, base=None):
        if False:
            return 10
        "\n        ### This is not a public method\n        Mission critical function to kill the win32pdh objects held\n        by this object.  User's should generally use the close method\n        instead of this method, in case a sub-class has overridden\n        close to provide some special functionality.\n        "
        self._base = None
        counters = self.counters
        self.counters = []
        self.active = 0
        try:
            map(win32pdh.RemoveCounter, counters)
        except:
            pass
        try:
            win32pdh.CloseQuery(base)
        except:
            pass
        del counters
        del base

    def close(self):
        if False:
            return 10
        '\n        Makes certain that the underlying query object has been closed,\n        and that all counters have been removed from it.  This is\n        important for reference counting.\n        You should only need to call close if you have previously called\n        open.  The collectdata methods all can handle opening and\n        closing the query.  Calling close multiple times is acceptable.\n        '
        try:
            self.killbase(self._base)
        except AttributeError:
            self.killbase()
    __del__ = close

    def collectdata(self, format=win32pdh.PDH_FMT_LONG):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the formatted current values for the Query\n        '
        if self._base:
            return self.collectdataslave(format)
        else:
            self.open()
            temp = self.collectdataslave(format)
            self.close()
            return temp

    def collectdataslave(self, format=win32pdh.PDH_FMT_LONG):
        if False:
            return 10
        '\n        ### Not a public method\n        Called only when the Query is known to be open, runs over\n        the whole set of counters, appending results to the temp,\n        returns the values as a list.\n        '
        try:
            win32pdh.CollectQueryData(self._base)
            temp = []
            for counter in self.counters:
                ok = 0
                try:
                    if counter:
                        temp.append(win32pdh.GetFormattedCounterValue(counter, format)[1])
                        ok = 1
                except win32api.error:
                    pass
                if not ok:
                    temp.append(-1)
            return temp
        except win32api.error:
            return [-1] * len(self.counters)

    def __getinitargs__(self):
        if False:
            print('Hello World!')
        '\n        ### Not a public method\n        '
        return (self.paths,)

class Query(BaseQuery):
    """
    Performance Data Helper(PDH) Query object:

    Provides a wrapper around the native PDH query object which
    allows for query reuse, query storage, and general maintenance
    functions (adding counter paths in various ways being the most
    obvious ones).
    """

    def __init__(self, *args, **namedargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        The PDH Query object is initialised with a single, optional\n        list argument, that must be properly formatted PDH Counter\n        paths.  Generally this list will only be provided by the class\n        when it is being unpickled (removed from storage).  Normal\n        use is to call the class with no arguments and use the various\n        addcounter functions (particularly, for end user's, the use of\n        addcounterbybrowsing is the most common approach)  You might\n        want to provide the list directly if you want to hard-code the\n        elements with which your query deals (and thereby avoid the\n        overhead of unpickling the class).\n        "
        self.volatilecounters = []
        BaseQuery.__init__(*(self,) + args, **namedargs)

    def addperfcounter(self, object, counter, machine=None):
        if False:
            return 10
        '\n        A "Performance Counter" is a stable, known, common counter,\n        such as Memory, or Processor.  The use of addperfcounter by\n        end-users is deprecated, since the use of\n        addcounterbybrowsing is considerably more flexible and general.\n        It is provided here to allow the easy development of scripts\n        which need to access variables so common we know them by name\n        (such as Memory|Available Bytes), and to provide symmetry with\n        the add inst counter method.\n        usage:\n                query.addperfcounter(\'Memory\', \'Available Bytes\')\n        It is just as easy to access addcounter directly, the following\n        has an identicle effect.\n                query.addcounter(\'Memory\', \'Available Bytes\')\n        '
        BaseQuery.addcounter(self, object=object, counter=counter, machine=machine)

    def addinstcounter(self, object, counter, machine=None, objtype='Process', volatile=1, format=win32pdh.PDH_FMT_LONG):
        if False:
            for i in range(10):
                print('nop')
        "\n        The purpose of using an instcounter is to track particular\n        instances of a counter object (e.g. a single processor, a single\n        running copy of a process).  For instance, to track all python.exe\n        instances, you would need merely to ask:\n                query.addinstcounter('python','Virtual Bytes')\n        You can find the names of the objects and their available counters\n        by doing an addcounterbybrowsing() call on a query object (or by\n        looking in performance monitor's add dialog.)\n\n        Beyond merely rearranging the call arguments to make more sense,\n        if the volatile flag is true, the instcounters also recalculate\n        the paths of the available instances on every call to open the\n        query.\n        "
        if volatile:
            self.volatilecounters.append((object, counter, machine, objtype, format))
        else:
            self.paths[len(self.paths):] = self.getinstpaths(object, counter, machine, objtype, format)

    def getinstpaths(self, object, counter, machine=None, objtype='Process', format=win32pdh.PDH_FMT_LONG):
        if False:
            return 10
        '\n        ### Not an end-user function\n        Calculate the paths for an instance object. Should alter\n        to allow processing for lists of object-counter pairs.\n        '
        (items, instances) = win32pdh.EnumObjectItems(None, None, objtype, -1)
        instances.sort()
        try:
            cur = instances.index(object)
        except ValueError:
            return []
        temp = [object]
        try:
            while instances[cur + 1] == object:
                temp.append(object)
                cur = cur + 1
        except IndexError:
            pass
        paths = []
        for ind in range(len(temp)):
            paths.append(win32pdh.MakeCounterPath((machine, 'Process', object, None, ind, counter)))
        return paths

    def open(self, *args, **namedargs):
        if False:
            while True:
                i = 10
        '\n        Explicitly open a query:\n        When you are needing to make multiple calls to the same query,\n        it is most efficient to open the query, run all of the calls,\n        then close the query, instead of having the collectdata method\n        automatically open and close the query each time it runs.\n        There are currently no arguments to open.\n        '
        BaseQuery.open(*(self,) + args, **namedargs)
        paths = []
        for tup in self.volatilecounters:
            paths[len(paths):] = self.getinstpaths(*tup)
        for path in paths:
            try:
                self.counters.append(win32pdh.AddCounter(self._base, path))
                self.curpaths.append(path)
            except win32api.error:
                pass

    def collectdatafor(self, totalperiod, period=1):
        if False:
            i = 10
            return i + 15
        '\n        Non-threaded collection of performance data:\n        This method allows you to specify the total period for which you would\n        like to run the Query, and the time interval between individual\n        runs.  The collected data is stored in query.curresults at the\n        _end_ of the run.  The pathnames for the query are stored in\n        query.curpaths.\n        e.g.:\n                query.collectdatafor(30,2)\n        Will collect data for 30seconds at 2 second intervals\n        '
        tempresults = []
        try:
            self.open()
            for ind in range(totalperiod / period):
                tempresults.append(self.collectdata())
                time.sleep(period)
            self.curresults = tempresults
        finally:
            self.close()

    def collectdatawhile(self, period=1):
        if False:
            for i in range(10):
                print('nop')
        '\n        Threaded collection of performance data:\n        This method sets up a simple semaphor system for signalling\n        when you would like to start and stop a threaded data collection\n        method.  The collection runs every period seconds until the\n        semaphor attribute is set to a non-true value (which normally\n        should be done by calling query.collectdatawhile_stop() .)\n        e.g.:\n                query.collectdatawhile(2)\n                # starts the query running, returns control to the caller immediately\n                # is collecting data every two seconds.\n                # do whatever you want to do while the thread runs, then call:\n                query.collectdatawhile_stop()\n                # when you want to deal with the data.  It is generally a good idea\n                # to sleep for period seconds yourself, since the query will not copy\n                # the required data until the next iteration:\n                time.sleep(2)\n                # now you can access the data from the attributes of the query\n                query.curresults\n                query.curpaths\n        '
        self.collectdatawhile_active = 1
        _thread.start_new_thread(self.collectdatawhile_slave, (period,))

    def collectdatawhile_stop(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Signals the collectdatawhile slave thread to stop collecting data\n        on the next logging iteration.\n        '
        self.collectdatawhile_active = 0

    def collectdatawhile_slave(self, period):
        if False:
            print('Hello World!')
        '\n        ### Not a public function\n        Does the threaded work of collecting the data and storing it\n        in an attribute of the class.\n        '
        tempresults = []
        try:
            self.open()
            while self.collectdatawhile_active:
                tempresults.append(self.collectdata())
                time.sleep(period)
            self.curresults = tempresults
        finally:
            self.close()

    def __getinitargs__(self):
        if False:
            i = 10
            return i + 15
        return (self.paths,)

    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        return self.volatilecounters

    def __setstate__(self, volatilecounters):
        if False:
            print('Hello World!')
        self.volatilecounters = volatilecounters

class QueryError:

    def __init__(self, query):
        if False:
            print('Hello World!')
        self.query = query

    def __repr__(self):
        if False:
            print('Hello World!')
        return '<Query Error in %s>' % repr(self.query)
    __str__ = __repr__