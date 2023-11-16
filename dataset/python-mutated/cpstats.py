"""CPStats, a package for collecting and reporting on program statistics.

Overview
========

Statistics about program operation are an invaluable monitoring and debugging
tool. Unfortunately, the gathering and reporting of these critical values is
usually ad-hoc. This package aims to add a centralized place for gathering
statistical performance data, a structure for recording that data which
provides for extrapolation of that data into more useful information,
and a method of serving that data to both human investigators and
monitoring software. Let's examine each of those in more detail.

Data Gathering
--------------

Just as Python's `logging` module provides a common importable for gathering
and sending messages, performance statistics would benefit from a similar
common mechanism, and one that does *not* require each package which wishes
to collect stats to import a third-party module. Therefore, we choose to
re-use the `logging` module by adding a `statistics` object to it.

That `logging.statistics` object is a nested dict. It is not a custom class,
because that would:

 1. require libraries and applications to import a third-party module in
    order to participate
 2. inhibit innovation in extrapolation approaches and in reporting tools, and
 3. be slow.

There are, however, some specifications regarding the structure of the dict.::

   {
     +----"SQLAlchemy": {
     |        "Inserts": 4389745,
     |        "Inserts per Second":
     |            lambda s: s["Inserts"] / (time() - s["Start"]),
     |  C +---"Table Statistics": {
     |  o |        "widgets": {-----------+
   N |  l |            "Rows": 1.3M,      | Record
   a |  l |            "Inserts": 400,    |
   m |  e |        },---------------------+
   e |  c |        "froobles": {
   s |  t |            "Rows": 7845,
   p |  i |            "Inserts": 0,
   a |  o |        },
   c |  n +---},
   e |        "Slow Queries":
     |            [{"Query": "SELECT * FROM widgets;",
     |              "Processing Time": 47.840923343,
     |              },
     |             ],
     +----},
   }

The `logging.statistics` dict has four levels. The topmost level is nothing
more than a set of names to introduce modularity, usually along the lines of
package names. If the SQLAlchemy project wanted to participate, for example,
it might populate the item `logging.statistics['SQLAlchemy']`, whose value
would be a second-layer dict we call a "namespace". Namespaces help multiple
packages to avoid collisions over key names, and make reports easier to read,
to boot. The maintainers of SQLAlchemy should feel free to use more than one
namespace if needed (such as 'SQLAlchemy ORM'). Note that there are no case
or other syntax constraints on the namespace names; they should be chosen
to be maximally readable by humans (neither too short nor too long).

Each namespace, then, is a dict of named statistical values, such as
'Requests/sec' or 'Uptime'. You should choose names which will look
good on a report: spaces and capitalization are just fine.

In addition to scalars, values in a namespace MAY be a (third-layer)
dict, or a list, called a "collection". For example, the CherryPy
:class:`StatsTool` keeps track of what each request is doing (or has most
recently done) in a 'Requests' collection, where each key is a thread ID; each
value in the subdict MUST be a fourth dict (whew!) of statistical data about
each thread. We call each subdict in the collection a "record". Similarly,
the :class:`StatsTool` also keeps a list of slow queries, where each record
contains data about each slow query, in order.

Values in a namespace or record may also be functions, which brings us to:

Extrapolation
-------------

The collection of statistical data needs to be fast, as close to unnoticeable
as possible to the host program. That requires us to minimize I/O, for example,
but in Python it also means we need to minimize function calls. So when you
are designing your namespace and record values, try to insert the most basic
scalar values you already have on hand.

When it comes time to report on the gathered data, however, we usually have
much more freedom in what we can calculate. Therefore, whenever reporting
tools (like the provided :class:`StatsPage` CherryPy class) fetch the contents
of `logging.statistics` for reporting, they first call
`extrapolate_statistics` (passing the whole `statistics` dict as the only
argument). This makes a deep copy of the statistics dict so that the
reporting tool can both iterate over it and even change it without harming
the original. But it also expands any functions in the dict by calling them.
For example, you might have a 'Current Time' entry in the namespace with the
value "lambda scope: time.time()". The "scope" parameter is the current
namespace dict (or record, if we're currently expanding one of those
instead), allowing you access to existing static entries. If you're truly
evil, you can even modify more than one entry at a time.

However, don't try to calculate an entry and then use its value in further
extrapolations; the order in which the functions are called is not guaranteed.
This can lead to a certain amount of duplicated work (or a redesign of your
schema), but that's better than complicating the spec.

After the whole thing has been extrapolated, it's time for:

Reporting
---------

The :class:`StatsPage` class grabs the `logging.statistics` dict, extrapolates
it all, and then transforms it to HTML for easy viewing. Each namespace gets
its own header and attribute table, plus an extra table for each collection.
This is NOT part of the statistics specification; other tools can format how
they like.

You can control which columns are output and how they are formatted by updating
StatsPage.formatting, which is a dict that mirrors the keys and nesting of
`logging.statistics`. The difference is that, instead of data values, it has
formatting values. Use None for a given key to indicate to the StatsPage that a
given column should not be output. Use a string with formatting
(such as '%.3f') to interpolate the value(s), or use a callable (such as
lambda v: v.isoformat()) for more advanced formatting. Any entry which is not
mentioned in the formatting dict is output unchanged.

Monitoring
----------

Although the HTML output takes pains to assign unique id's to each <td> with
statistical data, you're probably better off fetching /cpstats/data, which
outputs the whole (extrapolated) `logging.statistics` dict in JSON format.
That is probably easier to parse, and doesn't have any formatting controls,
so you get the "original" data in a consistently-serialized format.
Note: there's no treatment yet for datetime objects. Try time.time() instead
for now if you can. Nagios will probably thank you.

Turning Collection Off
----------------------

It is recommended each namespace have an "Enabled" item which, if False,
stops collection (but not reporting) of statistical data. Applications
SHOULD provide controls to pause and resume collection by setting these
entries to False or True, if present.


Usage
=====

To collect statistics on CherryPy applications::

    from cherrypy.lib import cpstats
    appconfig['/']['tools.cpstats.on'] = True

To collect statistics on your own code::

    import logging
    # Initialize the repository
    if not hasattr(logging, 'statistics'): logging.statistics = {}
    # Initialize my namespace
    mystats = logging.statistics.setdefault('My Stuff', {})
    # Initialize my namespace's scalars and collections
    mystats.update({
        'Enabled': True,
        'Start Time': time.time(),
        'Important Events': 0,
        'Events/Second': lambda s: (
            (s['Important Events'] / (time.time() - s['Start Time']))),
        })
    ...
    for event in events:
        ...
        # Collect stats
        if mystats.get('Enabled', False):
            mystats['Important Events'] += 1

To report statistics::

    root.cpstats = cpstats.StatsPage()

To format statistics reports::

    See 'Reporting', above.

"""
import logging
import os
import sys
import threading
import time
import cherrypy
from cherrypy._json import json
if not hasattr(logging, 'statistics'):
    logging.statistics = {}

def extrapolate_statistics(scope):
    if False:
        return 10
    'Return an extrapolated copy of the given scope.'
    c = {}
    for (k, v) in scope.copy().items():
        if isinstance(v, dict):
            v = extrapolate_statistics(v)
        elif isinstance(v, (list, tuple)):
            v = [extrapolate_statistics(record) for record in v]
        elif hasattr(v, '__call__'):
            v = v(scope)
        c[k] = v
    return c
appstats = logging.statistics.setdefault('CherryPy Applications', {})
appstats.update({'Enabled': True, 'Bytes Read/Request': lambda s: s['Total Requests'] and s['Total Bytes Read'] / float(s['Total Requests']) or 0.0, 'Bytes Read/Second': lambda s: s['Total Bytes Read'] / s['Uptime'](s), 'Bytes Written/Request': lambda s: s['Total Requests'] and s['Total Bytes Written'] / float(s['Total Requests']) or 0.0, 'Bytes Written/Second': lambda s: s['Total Bytes Written'] / s['Uptime'](s), 'Current Time': lambda s: time.time(), 'Current Requests': 0, 'Requests/Second': lambda s: float(s['Total Requests']) / s['Uptime'](s), 'Server Version': cherrypy.__version__, 'Start Time': time.time(), 'Total Bytes Read': 0, 'Total Bytes Written': 0, 'Total Requests': 0, 'Total Time': 0, 'Uptime': lambda s: time.time() - s['Start Time'], 'Requests': {}})

def proc_time(s):
    if False:
        print('Hello World!')
    return time.time() - s['Start Time']

class ByteCountWrapper(object):
    """Wraps a file-like object, counting the number of bytes read."""

    def __init__(self, rfile):
        if False:
            return 10
        self.rfile = rfile
        self.bytes_read = 0

    def read(self, size=-1):
        if False:
            return 10
        data = self.rfile.read(size)
        self.bytes_read += len(data)
        return data

    def readline(self, size=-1):
        if False:
            while True:
                i = 10
        data = self.rfile.readline(size)
        self.bytes_read += len(data)
        return data

    def readlines(self, sizehint=0):
        if False:
            while True:
                i = 10
        total = 0
        lines = []
        line = self.readline()
        while line:
            lines.append(line)
            total += len(line)
            if 0 < sizehint <= total:
                break
            line = self.readline()
        return lines

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        self.rfile.close()

    def __iter__(self):
        if False:
            print('Hello World!')
        return self

    def next(self):
        if False:
            return 10
        data = self.rfile.next()
        self.bytes_read += len(data)
        return data

def average_uriset_time(s):
    if False:
        while True:
            i = 10
    return s['Count'] and s['Sum'] / s['Count'] or 0

def _get_threading_ident():
    if False:
        print('Hello World!')
    if sys.version_info >= (3, 3):
        return threading.get_ident()
    return threading._get_ident()

class StatsTool(cherrypy.Tool):
    """Record various information about the current request."""

    def __init__(self):
        if False:
            print('Hello World!')
        cherrypy.Tool.__init__(self, 'on_end_request', self.record_stop)

    def _setup(self):
        if False:
            print('Hello World!')
        'Hook this tool into cherrypy.request.\n\n        The standard CherryPy request object will automatically call this\n        method when the tool is "turned on" in config.\n        '
        if appstats.get('Enabled', False):
            cherrypy.Tool._setup(self)
            self.record_start()

    def record_start(self):
        if False:
            while True:
                i = 10
        'Record the beginning of a request.'
        request = cherrypy.serving.request
        if not hasattr(request.rfile, 'bytes_read'):
            request.rfile = ByteCountWrapper(request.rfile)
            request.body.fp = request.rfile
        r = request.remote
        appstats['Current Requests'] += 1
        appstats['Total Requests'] += 1
        appstats['Requests'][_get_threading_ident()] = {'Bytes Read': None, 'Bytes Written': None, 'Client': lambda s: '%s:%s' % (r.ip, r.port), 'End Time': None, 'Processing Time': proc_time, 'Request-Line': request.request_line, 'Response Status': None, 'Start Time': time.time()}

    def record_stop(self, uriset=None, slow_queries=1.0, slow_queries_count=100, debug=False, **kwargs):
        if False:
            return 10
        'Record the end of a request.'
        resp = cherrypy.serving.response
        w = appstats['Requests'][_get_threading_ident()]
        r = cherrypy.request.rfile.bytes_read
        w['Bytes Read'] = r
        appstats['Total Bytes Read'] += r
        if resp.stream:
            w['Bytes Written'] = 'chunked'
        else:
            cl = int(resp.headers.get('Content-Length', 0))
            w['Bytes Written'] = cl
            appstats['Total Bytes Written'] += cl
        w['Response Status'] = getattr(resp, 'output_status', resp.status).decode()
        w['End Time'] = time.time()
        p = w['End Time'] - w['Start Time']
        w['Processing Time'] = p
        appstats['Total Time'] += p
        appstats['Current Requests'] -= 1
        if debug:
            cherrypy.log('Stats recorded: %s' % repr(w), 'TOOLS.CPSTATS')
        if uriset:
            rs = appstats.setdefault('URI Set Tracking', {})
            r = rs.setdefault(uriset, {'Min': None, 'Max': None, 'Count': 0, 'Sum': 0, 'Avg': average_uriset_time})
            if r['Min'] is None or p < r['Min']:
                r['Min'] = p
            if r['Max'] is None or p > r['Max']:
                r['Max'] = p
            r['Count'] += 1
            r['Sum'] += p
        if slow_queries and p > slow_queries:
            sq = appstats.setdefault('Slow Queries', [])
            sq.append(w.copy())
            if len(sq) > slow_queries_count:
                sq.pop(0)
cherrypy.tools.cpstats = StatsTool()
thisdir = os.path.abspath(os.path.dirname(__file__))
missing = object()

def locale_date(v):
    if False:
        while True:
            i = 10
    return time.strftime('%c', time.gmtime(v))

def iso_format(v):
    if False:
        while True:
            i = 10
    return time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(v))

def pause_resume(ns):
    if False:
        while True:
            i = 10

    def _pause_resume(enabled):
        if False:
            while True:
                i = 10
        pause_disabled = ''
        resume_disabled = ''
        if enabled:
            resume_disabled = 'disabled="disabled" '
        else:
            pause_disabled = 'disabled="disabled" '
        return '\n            <form action="pause" method="POST" style="display:inline">\n            <input type="hidden" name="namespace" value="%s" />\n            <input type="submit" value="Pause" %s/>\n            </form>\n            <form action="resume" method="POST" style="display:inline">\n            <input type="hidden" name="namespace" value="%s" />\n            <input type="submit" value="Resume" %s/>\n            </form>\n            ' % (ns, pause_disabled, ns, resume_disabled)
    return _pause_resume

class StatsPage(object):
    formatting = {'CherryPy Applications': {'Enabled': pause_resume('CherryPy Applications'), 'Bytes Read/Request': '%.3f', 'Bytes Read/Second': '%.3f', 'Bytes Written/Request': '%.3f', 'Bytes Written/Second': '%.3f', 'Current Time': iso_format, 'Requests/Second': '%.3f', 'Start Time': iso_format, 'Total Time': '%.3f', 'Uptime': '%.3f', 'Slow Queries': {'End Time': None, 'Processing Time': '%.3f', 'Start Time': iso_format}, 'URI Set Tracking': {'Avg': '%.3f', 'Max': '%.3f', 'Min': '%.3f', 'Sum': '%.3f'}, 'Requests': {'Bytes Read': '%s', 'Bytes Written': '%s', 'End Time': None, 'Processing Time': '%.3f', 'Start Time': None}}, 'CherryPy WSGIServer': {'Enabled': pause_resume('CherryPy WSGIServer'), 'Connections/second': '%.3f', 'Start time': iso_format}}

    @cherrypy.expose
    def index(self):
        if False:
            print('Hello World!')
        yield '\n<html>\n<head>\n    <title>Statistics</title>\n<style>\n\nth, td {\n    padding: 0.25em 0.5em;\n    border: 1px solid #666699;\n}\n\ntable {\n    border-collapse: collapse;\n}\n\ntable.stats1 {\n    width: 100%;\n}\n\ntable.stats1 th {\n    font-weight: bold;\n    text-align: right;\n    background-color: #CCD5DD;\n}\n\ntable.stats2, h2 {\n    margin-left: 50px;\n}\n\ntable.stats2 th {\n    font-weight: bold;\n    text-align: center;\n    background-color: #CCD5DD;\n}\n\n</style>\n</head>\n<body>\n'
        for (title, scalars, collections) in self.get_namespaces():
            yield ("\n<h1>%s</h1>\n\n<table class='stats1'>\n    <tbody>\n" % title)
            for (i, (key, value)) in enumerate(scalars):
                colnum = i % 3
                if colnum == 0:
                    yield '\n        <tr>'
                yield ("\n            <th>%(key)s</th><td id='%(title)s-%(key)s'>%(value)s</td>" % vars())
                if colnum == 2:
                    yield '\n        </tr>'
            if colnum == 0:
                yield '\n            <th></th><td></td>\n            <th></th><td></td>\n        </tr>'
            elif colnum == 1:
                yield '\n            <th></th><td></td>\n        </tr>'
            yield '\n    </tbody>\n</table>'
            for (subtitle, headers, subrows) in collections:
                yield ("\n<h2>%s</h2>\n<table class='stats2'>\n    <thead>\n        <tr>" % subtitle)
                for key in headers:
                    yield ('\n            <th>%s</th>' % key)
                yield '\n        </tr>\n    </thead>\n    <tbody>'
                for subrow in subrows:
                    yield '\n        <tr>'
                    for value in subrow:
                        yield ('\n            <td>%s</td>' % value)
                    yield '\n        </tr>'
                yield '\n    </tbody>\n</table>'
        yield '\n</body>\n</html>\n'

    def get_namespaces(self):
        if False:
            i = 10
            return i + 15
        'Yield (title, scalars, collections) for each namespace.'
        s = extrapolate_statistics(logging.statistics)
        for (title, ns) in sorted(s.items()):
            scalars = []
            collections = []
            ns_fmt = self.formatting.get(title, {})
            for (k, v) in sorted(ns.items()):
                fmt = ns_fmt.get(k, {})
                if isinstance(v, dict):
                    (headers, subrows) = self.get_dict_collection(v, fmt)
                    collections.append((k, ['ID'] + headers, subrows))
                elif isinstance(v, (list, tuple)):
                    (headers, subrows) = self.get_list_collection(v, fmt)
                    collections.append((k, headers, subrows))
                else:
                    format = ns_fmt.get(k, missing)
                    if format is None:
                        continue
                    if hasattr(format, '__call__'):
                        v = format(v)
                    elif format is not missing:
                        v = format % v
                    scalars.append((k, v))
            yield (title, scalars, collections)

    def get_dict_collection(self, v, formatting):
        if False:
            while True:
                i = 10
        'Return ([headers], [rows]) for the given collection.'
        headers = []
        vals = v.values()
        for record in vals:
            for k3 in record:
                format = formatting.get(k3, missing)
                if format is None:
                    continue
                if k3 not in headers:
                    headers.append(k3)
        headers.sort()
        subrows = []
        for (k2, record) in sorted(v.items()):
            subrow = [k2]
            for k3 in headers:
                v3 = record.get(k3, '')
                format = formatting.get(k3, missing)
                if format is None:
                    continue
                if hasattr(format, '__call__'):
                    v3 = format(v3)
                elif format is not missing:
                    v3 = format % v3
                subrow.append(v3)
            subrows.append(subrow)
        return (headers, subrows)

    def get_list_collection(self, v, formatting):
        if False:
            return 10
        'Return ([headers], [subrows]) for the given collection.'
        headers = []
        for record in v:
            for k3 in record:
                format = formatting.get(k3, missing)
                if format is None:
                    continue
                if k3 not in headers:
                    headers.append(k3)
        headers.sort()
        subrows = []
        for record in v:
            subrow = []
            for k3 in headers:
                v3 = record.get(k3, '')
                format = formatting.get(k3, missing)
                if format is None:
                    continue
                if hasattr(format, '__call__'):
                    v3 = format(v3)
                elif format is not missing:
                    v3 = format % v3
                subrow.append(v3)
            subrows.append(subrow)
        return (headers, subrows)
    if json is not None:

        @cherrypy.expose
        def data(self):
            if False:
                i = 10
                return i + 15
            s = extrapolate_statistics(logging.statistics)
            cherrypy.response.headers['Content-Type'] = 'application/json'
            return json.dumps(s, sort_keys=True, indent=4).encode('utf-8')

    @cherrypy.expose
    def pause(self, namespace):
        if False:
            i = 10
            return i + 15
        logging.statistics.get(namespace, {})['Enabled'] = False
        raise cherrypy.HTTPRedirect('./')
    pause.cp_config = {'tools.allow.on': True, 'tools.allow.methods': ['POST']}

    @cherrypy.expose
    def resume(self, namespace):
        if False:
            print('Hello World!')
        logging.statistics.get(namespace, {})['Enabled'] = True
        raise cherrypy.HTTPRedirect('./')
    resume.cp_config = {'tools.allow.on': True, 'tools.allow.methods': ['POST']}