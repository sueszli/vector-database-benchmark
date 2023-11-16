from __future__ import print_function
import sys
import select
import socket
import re
from struct import *
if sys.version_info > (3,):
    long = int
    text_type = str
else:
    text_type = unicode
SEARCHD_COMMAND_SEARCH = 0
SEARCHD_COMMAND_EXCERPT = 1
SEARCHD_COMMAND_UPDATE = 2
SEARCHD_COMMAND_KEYWORDS = 3
SEARCHD_COMMAND_PERSIST = 4
SEARCHD_COMMAND_STATUS = 5
SEARCHD_COMMAND_FLUSHATTRS = 7
VER_COMMAND_SEARCH = 288
VER_COMMAND_EXCERPT = 260
VER_COMMAND_UPDATE = 259
VER_COMMAND_KEYWORDS = 256
VER_COMMAND_STATUS = 257
VER_COMMAND_FLUSHATTRS = 256
SEARCHD_OK = 0
SEARCHD_ERROR = 1
SEARCHD_RETRY = 2
SEARCHD_WARNING = 3
SPH_RANK_PROXIMITY_BM15 = 0
SPH_RANK_BM15 = 1
SPH_RANK_NONE = 2
SPH_RANK_WORDCOUNT = 3
SPH_RANK_PROXIMITY = 4
SPH_RANK_MATCHANY = 5
SPH_RANK_FIELDMASK = 6
SPH_RANK_SPH04 = 7
SPH_RANK_EXPR = 8
SPH_RANK_TOTAL = 9
SPH_RANK_PROXIMITY_BM25 = 0
SPH_RANK_BM25 = 1
SPH_SORT_RELEVANCE = 0
SPH_SORT_ATTR_DESC = 1
SPH_SORT_ATTR_ASC = 2
SPH_SORT_TIME_SEGMENTS = 3
SPH_SORT_EXTENDED = 4
SPH_FILTER_VALUES = 0
SPH_FILTER_RANGE = 1
SPH_FILTER_FLOATRANGE = 2
SPH_FILTER_STRING = 3
SPH_FILTER_STRING_LIST = 6
SPH_ATTR_NONE = 0
SPH_ATTR_INTEGER = 1
SPH_ATTR_TIMESTAMP = 2
SPH_ATTR_ORDINAL = 3
SPH_ATTR_BOOL = 4
SPH_ATTR_FLOAT = 5
SPH_ATTR_BIGINT = 6
SPH_ATTR_STRING = 7
SPH_ATTR_FACTORS = 1001
SPH_ATTR_MULTI = long(1073741825)
SPH_ATTR_MULTI64 = long(1073741826)
SPH_ATTR_TYPES = (SPH_ATTR_NONE, SPH_ATTR_INTEGER, SPH_ATTR_TIMESTAMP, SPH_ATTR_ORDINAL, SPH_ATTR_BOOL, SPH_ATTR_FLOAT, SPH_ATTR_BIGINT, SPH_ATTR_STRING, SPH_ATTR_MULTI, SPH_ATTR_MULTI64)
SPH_GROUPBY_DAY = 0
SPH_GROUPBY_WEEK = 1
SPH_GROUPBY_MONTH = 2
SPH_GROUPBY_YEAR = 3
SPH_GROUPBY_ATTR = 4
SPH_GROUPBY_ATTRPAIR = 5

class SphinxClient:

    def __init__(self):
        if False:
            return 10
        '\n\t\tCreate a new client object, and fill defaults.\n\t\t'
        self._host = 'localhost'
        self._port = 9312
        self._path = None
        self._socket = None
        self._offset = 0
        self._limit = 20
        self._weights = []
        self._sort = SPH_SORT_RELEVANCE
        self._sortby = bytearray()
        self._min_id = 0
        self._max_id = 0
        self._filters = []
        self._groupby = bytearray()
        self._groupfunc = SPH_GROUPBY_DAY
        self._groupsort = str_bytes('@group desc')
        self._groupdistinct = bytearray()
        self._maxmatches = 1000
        self._cutoff = 0
        self._retrycount = 0
        self._retrydelay = 0
        self._indexweights = {}
        self._ranker = SPH_RANK_PROXIMITY_BM15
        self._rankexpr = bytearray()
        self._maxquerytime = 0
        self._timeout = 1.0
        self._fieldweights = {}
        self._select = str_bytes('*')
        self._query_flags = SetBit(0, 6, True)
        self._predictedtime = 0
        self._outerorderby = bytearray()
        self._outeroffset = 0
        self._outerlimit = 0
        self._hasouter = False
        self._tokenfilterlibrary = bytearray()
        self._tokenfiltername = bytearray()
        self._tokenfilteropts = bytearray()
        self._error = ''
        self._warning = ''
        self._reqs = []

    def __del__(self):
        if False:
            print('Hello World!')
        if self._socket:
            self._socket.close()

    def GetLastError(self):
        if False:
            while True:
                i = 10
        '\n\t\tGet last error message (string).\n\t\t'
        return self._error

    def GetLastWarning(self):
        if False:
            return 10
        '\n\t\tGet last warning message (string).\n\t\t'
        return self._warning

    def SetServer(self, host, port=None):
        if False:
            i = 10
            return i + 15
        '\n\t\tSet searchd server host and port.\n\t\t'
        assert isinstance(host, str)
        if host.startswith('/'):
            self._path = host
            return
        elif host.startswith('unix://'):
            self._path = host[7:]
            return
        self._host = host
        if isinstance(port, int):
            assert port > 0 and port < 65536
            self._port = port
        self._path = None

    def SetConnectTimeout(self, timeout):
        if False:
            for i in range(10):
                print('nop')
        '\n\t\tSet connection timeout ( float second )\n\t\t'
        assert isinstance(timeout, float)
        self._timeout = max(0.001, timeout)

    def _Connect(self):
        if False:
            for i in range(10):
                print('nop')
        '\n\t\tINTERNAL METHOD, DO NOT CALL. Connects to searchd server.\n\t\t'
        if self._socket:
            (sr, sw, _) = select.select([self._socket], [self._socket], [], 0)
            if len(sr) == 0 and len(sw) == 1:
                return self._socket
            self._socket.close()
            self._socket = None
        try:
            if self._path:
                af = socket.AF_UNIX
                addr = self._path
                desc = self._path
            else:
                af = socket.AF_INET
                addr = (self._host, self._port)
                desc = '%s;%s' % addr
            sock = socket.socket(af, socket.SOCK_STREAM)
            sock.settimeout(self._timeout)
            sock.connect(addr)
        except socket.error as msg:
            if sock:
                sock.close()
            self._error = 'connection to %s failed (%s)' % (desc, msg)
            return
        v = unpack('>L', sock.recv(4))[0]
        if v < 1:
            sock.close()
            self._error = 'expected searchd protocol version, got %s' % v
            return
        sock.send(pack('>L', 1))
        return sock

    def _GetResponse(self, sock, client_ver):
        if False:
            i = 10
            return i + 15
        '\n\t\tINTERNAL METHOD, DO NOT CALL. Gets and checks response packet from searchd server.\n\t\t'
        (status, ver, length) = unpack('>2HL', sock.recv(8))
        response = bytearray()
        left = length
        while left > 0:
            chunk = sock.recv(left)
            if chunk:
                response += chunk
                left -= len(chunk)
            else:
                break
        if not self._socket:
            sock.close()
        read = len(response)
        if not response or read != length:
            if length:
                self._error = 'failed to read searchd response (status=%s, ver=%s, len=%s, read=%s)' % (status, ver, length, read)
            else:
                self._error = 'received zero-sized searchd response'
            return None
        if status == SEARCHD_WARNING:
            wend = 4 + unpack('>L', response[0:4])[0]
            self._warning = bytes_str(response[4:wend])
            return response[wend:]
        if status == SEARCHD_ERROR:
            self._error = 'searchd error: ' + bytes_str(response[4:])
            return None
        if status == SEARCHD_RETRY:
            self._error = 'temporary searchd error: ' + bytes_str(response[4:])
            return None
        if status != SEARCHD_OK:
            self._error = 'unknown status code %d' % status
            return None
        if ver < client_ver:
            self._warning = "searchd command v.%d.%d older than client's v.%d.%d, some options might not work" % (ver >> 8, ver & 255, client_ver >> 8, client_ver & 255)
        return response

    def _Send(self, sock, req):
        if False:
            for i in range(10):
                print('nop')
        '\n\t\tINTERNAL METHOD, DO NOT CALL. send request to searchd server.\n\t\t'
        total = 0
        while True:
            sent = sock.send(req[total:])
            if sent <= 0:
                break
            total = total + sent
        return total

    def SetLimits(self, offset, limit, maxmatches=0, cutoff=0):
        if False:
            for i in range(10):
                print('nop')
        '\n\t\tSet offset and count into result set, and optionally set max-matches and cutoff limits.\n\t\t'
        assert type(offset) in [int, long] and 0 <= offset < 16777216
        assert type(limit) in [int, long] and 0 < limit < 16777216
        assert maxmatches >= 0
        self._offset = offset
        self._limit = limit
        if maxmatches > 0:
            self._maxmatches = maxmatches
        if cutoff >= 0:
            self._cutoff = cutoff

    def SetMaxQueryTime(self, maxquerytime):
        if False:
            print('Hello World!')
        "\n\t\tSet maximum query time, in milliseconds, per-index. 0 means 'do not limit'.\n\t\t"
        assert isinstance(maxquerytime, int) and maxquerytime > 0
        self._maxquerytime = maxquerytime

    def SetRankingMode(self, ranker, rankexpr=''):
        if False:
            print('Hello World!')
        '\n\t\tSet ranking mode.\n\t\t'
        assert ranker >= 0 and ranker < SPH_RANK_TOTAL
        self._ranker = ranker
        self._rankexpr = str_bytes(rankexpr)

    def SetSortMode(self, mode, clause=''):
        if False:
            return 10
        '\n\t\tSet sorting mode.\n\t\t'
        assert mode in [SPH_SORT_RELEVANCE, SPH_SORT_ATTR_DESC, SPH_SORT_ATTR_ASC, SPH_SORT_TIME_SEGMENTS, SPH_SORT_EXTENDED]
        assert isinstance(clause, (str, text_type))
        self._sort = mode
        self._sortby = str_bytes(clause)

    def SetFieldWeights(self, weights):
        if False:
            print('Hello World!')
        '\n\t\tBind per-field weights by name; expects (name,field_weight) dictionary as argument.\n\t\t'
        assert isinstance(weights, dict)
        for (key, val) in list(weights.items()):
            assert isinstance(key, str)
            AssertUInt32(val)
        self._fieldweights = weights

    def SetIndexWeights(self, weights):
        if False:
            while True:
                i = 10
        '\n\t\tBind per-index weights by name; expects (name,index_weight) dictionary as argument.\n\t\t'
        assert isinstance(weights, dict)
        for (key, val) in list(weights.items()):
            assert isinstance(key, str)
            AssertUInt32(val)
        self._indexweights = weights

    def SetIDRange(self, minid, maxid):
        if False:
            for i in range(10):
                print('nop')
        '\n\t\tSet IDs range to match.\n\t\tOnly match records if document ID is beetwen $min and $max (inclusive).\n\t\t'
        assert isinstance(minid, (int, long))
        assert isinstance(maxid, (int, long))
        assert minid <= maxid
        self._min_id = minid
        self._max_id = maxid

    def SetFilter(self, attribute, values, exclude=0):
        if False:
            print('Hello World!')
        "\n\t\tSet values set filter.\n\t\tOnly match records where 'attribute' value is in given 'values' set.\n\t\t"
        assert isinstance(attribute, str)
        assert iter(values)
        for value in values:
            AssertInt32(value)
        self._filters.append({'type': SPH_FILTER_VALUES, 'attr': attribute, 'exclude': exclude, 'values': values})

    def SetFilterString(self, attribute, value, exclude=0):
        if False:
            i = 10
            return i + 15
        "\n\t\tSet string filter.\n\t\tOnly match records where 'attribute' value is equal\n\t\t"
        assert isinstance(attribute, str)
        assert isinstance(value, str)
        self._filters.append({'type': SPH_FILTER_STRING, 'attr': attribute, 'exclude': exclude, 'value': value})

    def SetFilterStringList(self, attribute, value, exclude=0):
        if False:
            while True:
                i = 10
        '\n\t\tSet string list filter.\n\t\t'
        assert isinstance(attribute, str)
        assert iter(value)
        for v in value:
            assert isinstance(v, str)
        self._filters.append({'type': SPH_FILTER_STRING_LIST, 'attr': attribute, 'exclude': exclude, 'values': value})

    def SetFilterRange(self, attribute, min_, max_, exclude=0):
        if False:
            i = 10
            return i + 15
        "\n\t\tSet range filter.\n\t\tOnly match records if 'attribute' value is beetwen 'min_' and 'max_' (inclusive).\n\t\t"
        assert isinstance(attribute, str)
        AssertInt32(min_)
        AssertInt32(max_)
        assert min_ <= max_
        self._filters.append({'type': SPH_FILTER_RANGE, 'attr': attribute, 'exclude': exclude, 'min': min_, 'max': max_})

    def SetFilterFloatRange(self, attribute, min_, max_, exclude=0):
        if False:
            i = 10
            return i + 15
        assert isinstance(attribute, str)
        assert isinstance(min_, float)
        assert isinstance(max_, float)
        assert min_ <= max_
        self._filters.append({'type': SPH_FILTER_FLOATRANGE, 'attr': attribute, 'exclude': exclude, 'min': min_, 'max': max_})

    def SetGroupBy(self, attribute, func, groupsort='@group desc'):
        if False:
            i = 10
            return i + 15
        '\n\t\tSet grouping attribute and function.\n\t\t'
        assert isinstance(attribute, (str, text_type))
        assert func in [SPH_GROUPBY_DAY, SPH_GROUPBY_WEEK, SPH_GROUPBY_MONTH, SPH_GROUPBY_YEAR, SPH_GROUPBY_ATTR, SPH_GROUPBY_ATTRPAIR]
        assert isinstance(groupsort, (str, text_type))
        self._groupby = str_bytes(attribute)
        self._groupfunc = func
        self._groupsort = str_bytes(groupsort)

    def SetGroupDistinct(self, attribute):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(attribute, (str, text_type))
        self._groupdistinct = str_bytes(attribute)

    def SetRetries(self, count, delay=0):
        if False:
            while True:
                i = 10
        assert isinstance(count, int) and count >= 0
        assert isinstance(delay, int) and delay >= 0
        self._retrycount = count
        self._retrydelay = delay

    def SetSelect(self, select):
        if False:
            print('Hello World!')
        assert isinstance(select, (str, text_type))
        self._select = str_bytes(select)

    def SetQueryFlag(self, name, value):
        if False:
            return 10
        known_names = ['reverse_scan', 'sort_method', 'max_predicted_time', 'boolean_simplify', 'idf', 'global_idf']
        flags = {'reverse_scan': [0, 1], 'sort_method': ['pq', 'kbuffer'], 'max_predicted_time': [0], 'boolean_simplify': [True, False], 'idf': ['normalized', 'plain', 'tfidf_normalized', 'tfidf_unnormalized'], 'global_idf': [True, False]}
        assert name in known_names
        assert value in flags[name] or (name == 'max_predicted_time' and isinstance(value, (int, long)) and (value >= 0))
        if name == 'reverse_scan':
            self._query_flags = SetBit(self._query_flags, 0, value == 1)
        if name == 'sort_method':
            self._query_flags = SetBit(self._query_flags, 1, value == 'kbuffer')
        if name == 'max_predicted_time':
            self._query_flags = SetBit(self._query_flags, 2, value > 0)
            self._predictedtime = int(value)
        if name == 'boolean_simplify':
            self._query_flags = SetBit(self._query_flags, 3, value)
        if name == 'idf' and (value == 'plain' or value == 'normalized'):
            self._query_flags = SetBit(self._query_flags, 4, value == 'plain')
        if name == 'global_idf':
            self._query_flags = SetBit(self._query_flags, 5, value)
        if name == 'idf' and (value == 'tfidf_normalized' or value == 'tfidf_unnormalized'):
            self._query_flags = SetBit(self._query_flags, 6, value == 'tfidf_normalized')

    def SetOuterSelect(self, orderby, offset, limit):
        if False:
            i = 10
            return i + 15
        assert isinstance(orderby, (str, text_type))
        assert isinstance(offset, (int, long))
        assert isinstance(limit, (int, long))
        assert offset >= 0
        assert limit > 0
        self._outerorderby = str_bytes(orderby)
        self._outeroffset = offset
        self._outerlimit = limit
        self._hasouter = True

    def SetTokenFilter(self, library, name, opts=''):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(library, str)
        assert isinstance(name, str)
        assert isinstance(opts, str)
        self._tokenfilterlibrary = str_bytes(library)
        self._tokenfiltername = str_bytes(name)
        self._tokenfilteropts = str_bytes(opts)

    def ResetFilters(self):
        if False:
            print('Hello World!')
        '\n\t\tClear all filters (for multi-queries).\n\t\t'
        self._filters = []

    def ResetGroupBy(self):
        if False:
            return 10
        '\n\t\tClear groupby settings (for multi-queries).\n\t\t'
        self._groupby = bytearray()
        self._groupfunc = SPH_GROUPBY_DAY
        self._groupsort = str_bytes('@group desc')
        self._groupdistinct = bytearray()

    def ResetQueryFlag(self):
        if False:
            for i in range(10):
                print('nop')
        self._query_flags = SetBit(0, 6, True)
        self._predictedtime = 0

    def ResetOuterSelect(self):
        if False:
            print('Hello World!')
        self._outerorderby = bytearray()
        self._outeroffset = 0
        self._outerlimit = 0
        self._hasouter = False

    def Query(self, query, index='*', comment=''):
        if False:
            return 10
        '\n\t\tConnect to searchd server and run given search query.\n\t\tReturns None on failure; result set hash on success (see documentation for details).\n\t\t'
        assert len(self._reqs) == 0
        self.AddQuery(query, index, comment)
        results = self.RunQueries()
        self._reqs = []
        if not results or len(results) == 0:
            return None
        self._error = results[0]['error']
        self._warning = results[0]['warning']
        if results[0]['status'] == SEARCHD_ERROR:
            return None
        return results[0]

    def AddQuery(self, query, index='*', comment=''):
        if False:
            i = 10
            return i + 15
        '\n\t\tAdd query to batch.\n\t\t'
        req = bytearray()
        req.extend(pack('>5L', self._query_flags, self._offset, self._limit, 6, self._ranker))
        if self._ranker == SPH_RANK_EXPR:
            req.extend(pack('>L', len(self._rankexpr)))
            req.extend(self._rankexpr)
        req.extend(pack('>L', self._sort))
        req.extend(pack('>L', len(self._sortby)))
        req.extend(self._sortby)
        query = str_bytes(query)
        assert isinstance(query, bytearray)
        req.extend(pack('>L', len(query)))
        req.extend(query)
        req.extend(pack('>L', len(self._weights)))
        for w in self._weights:
            req.extend(pack('>L', w))
        index = str_bytes(index)
        assert isinstance(index, bytearray)
        req.extend(pack('>L', len(index)))
        req.extend(index)
        req.extend(pack('>L', 1))
        req.extend(pack('>Q', self._min_id))
        req.extend(pack('>Q', self._max_id))
        req.extend(pack('>L', len(self._filters)))
        for f in self._filters:
            attr = str_bytes(f['attr'])
            req.extend(pack('>L', len(f['attr'])) + attr)
            filtertype = f['type']
            req.extend(pack('>L', filtertype))
            if filtertype == SPH_FILTER_VALUES:
                req.extend(pack('>L', len(f['values'])))
                for val in f['values']:
                    req.extend(pack('>q', val))
            elif filtertype == SPH_FILTER_RANGE:
                req.extend(pack('>2q', f['min'], f['max']))
            elif filtertype == SPH_FILTER_FLOATRANGE:
                req.extend(pack('>2f', f['min'], f['max']))
            elif filtertype == SPH_FILTER_STRING:
                val = str_bytes(f['value'])
                req.extend(pack('>L', len(val)))
                req.extend(val)
            elif filtertype == SPH_FILTER_STRING_LIST:
                req.extend(pack('>L', len(f['values'])))
                for sval in f['values']:
                    val = str_bytes(sval)
                    req.extend(pack('>L', len(val)))
                    req.extend(val)
            req.extend(pack('>L', f['exclude']))
        req.extend(pack('>2L', self._groupfunc, len(self._groupby)))
        req.extend(self._groupby)
        req.extend(pack('>2L', self._maxmatches, len(self._groupsort)))
        req.extend(self._groupsort)
        req.extend(pack('>LLL', self._cutoff, self._retrycount, self._retrydelay))
        req.extend(pack('>L', len(self._groupdistinct)))
        req.extend(self._groupdistinct)
        req.extend(pack('>L', 0))
        req.extend(pack('>L', len(self._indexweights)))
        for (indx, weight) in list(self._indexweights.items()):
            indx = str_bytes(indx)
            req.extend(pack('>L', len(indx)) + indx + pack('>L', weight))
        req.extend(pack('>L', self._maxquerytime))
        req.extend(pack('>L', len(self._fieldweights)))
        for (field, weight) in list(self._fieldweights.items()):
            field = str_bytes(field)
            req.extend(pack('>L', len(field)) + field + pack('>L', weight))
        comment = str_bytes(comment)
        req.extend(pack('>L', len(comment)) + comment)
        req.extend(pack('>L', 0))
        req.extend(pack('>L', len(self._select)))
        req.extend(self._select)
        if self._predictedtime > 0:
            req.extend(pack('>L', self._predictedtime))
        req.extend(pack('>L', len(self._outerorderby)) + self._outerorderby)
        req.extend(pack('>2L', self._outeroffset, self._outerlimit))
        if self._hasouter:
            req.extend(pack('>L', 1))
        else:
            req.extend(pack('>L', 0))
        req.extend(pack('>L', len(self._tokenfilterlibrary)) + self._tokenfilterlibrary)
        req.extend(pack('>L', len(self._tokenfiltername)) + self._tokenfiltername)
        req.extend(pack('>L', len(self._tokenfilteropts)) + self._tokenfilteropts)
        self._reqs.append(req)
        return

    def RunQueries(self):
        if False:
            while True:
                i = 10
        '\n\t\tRun queries batch.\n\t\tReturns None on network IO failure; or an array of result set hashes on success.\n\t\t'
        if len(self._reqs) == 0:
            self._error = 'no queries defined, issue AddQuery() first'
            return None
        sock = self._Connect()
        if not sock:
            return None
        req = bytearray()
        for r in self._reqs:
            req.extend(r)
        length = len(req) + 8
        req_all = bytearray()
        req_all.extend(pack('>HHLLL', SEARCHD_COMMAND_SEARCH, VER_COMMAND_SEARCH, length, 0, len(self._reqs)))
        req_all.extend(req)
        self._Send(sock, req_all)
        response = self._GetResponse(sock, VER_COMMAND_SEARCH)
        if not response:
            return None
        nreqs = len(self._reqs)
        max_ = len(response)
        p = 0
        results = []
        for i in range(0, nreqs, 1):
            result = {}
            results.append(result)
            result['error'] = ''
            result['warning'] = ''
            status = unpack('>L', response[p:p + 4])[0]
            p += 4
            result['status'] = status
            if status != SEARCHD_OK:
                length = unpack('>L', response[p:p + 4])[0]
                p += 4
                message = bytes_str(response[p:p + length])
                p += length
                if status == SEARCHD_WARNING:
                    result['warning'] = message
                else:
                    result['error'] = message
                    continue
            fields = []
            attrs = []
            nfields = unpack('>L', response[p:p + 4])[0]
            p += 4
            while nfields > 0 and p < max_:
                nfields -= 1
                length = unpack('>L', response[p:p + 4])[0]
                p += 4
                fields.append(bytes_str(response[p:p + length]))
                p += length
            result['fields'] = fields
            nattrs = unpack('>L', response[p:p + 4])[0]
            p += 4
            while nattrs > 0 and p < max_:
                nattrs -= 1
                length = unpack('>L', response[p:p + 4])[0]
                p += 4
                attr = bytes_str(response[p:p + length])
                p += length
                type_ = unpack('>L', response[p:p + 4])[0]
                p += 4
                attrs.append([attr, type_])
            result['attrs'] = attrs
            count = unpack('>L', response[p:p + 4])[0]
            p += 4
            id64 = unpack('>L', response[p:p + 4])[0]
            p += 4
            result['matches'] = []
            while count > 0 and p < max_:
                count -= 1
                if id64:
                    (doc, weight) = unpack('>QL', response[p:p + 12])
                    p += 12
                else:
                    (doc, weight) = unpack('>2L', response[p:p + 8])
                    p += 8
                match = {'id': doc, 'weight': weight, 'attrs': {}}
                for i in range(len(attrs)):
                    if attrs[i][1] == SPH_ATTR_FLOAT:
                        match['attrs'][attrs[i][0]] = unpack('>f', response[p:p + 4])[0]
                    elif attrs[i][1] == SPH_ATTR_BIGINT:
                        match['attrs'][attrs[i][0]] = unpack('>q', response[p:p + 8])[0]
                        p += 4
                    elif attrs[i][1] == SPH_ATTR_STRING:
                        slen = unpack('>L', response[p:p + 4])[0]
                        p += 4
                        match['attrs'][attrs[i][0]] = ''
                        if slen > 0:
                            match['attrs'][attrs[i][0]] = bytes_str(response[p:p + slen])
                        p += slen - 4
                    elif attrs[i][1] == SPH_ATTR_FACTORS:
                        slen = unpack('>L', response[p:p + 4])[0]
                        p += 4
                        match['attrs'][attrs[i][0]] = ''
                        if slen > 0:
                            match['attrs'][attrs[i][0]] = response[p:p + slen - 4]
                            p += slen - 4
                        p -= 4
                    elif attrs[i][1] == SPH_ATTR_MULTI:
                        match['attrs'][attrs[i][0]] = []
                        nvals = unpack('>L', response[p:p + 4])[0]
                        p += 4
                        for n in range(0, nvals, 1):
                            match['attrs'][attrs[i][0]].append(unpack('>L', response[p:p + 4])[0])
                            p += 4
                        p -= 4
                    elif attrs[i][1] == SPH_ATTR_MULTI64:
                        match['attrs'][attrs[i][0]] = []
                        nvals = unpack('>L', response[p:p + 4])[0]
                        nvals = nvals / 2
                        p += 4
                        for n in range(0, nvals, 1):
                            match['attrs'][attrs[i][0]].append(unpack('>q', response[p:p + 8])[0])
                            p += 8
                        p -= 4
                    else:
                        match['attrs'][attrs[i][0]] = unpack('>L', response[p:p + 4])[0]
                    p += 4
                result['matches'].append(match)
            (result['total'], result['total_found'], result['time'], words) = unpack('>4L', response[p:p + 16])
            result['time'] = '%.3f' % (result['time'] / 1000.0)
            p += 16
            result['words'] = []
            while words > 0:
                words -= 1
                length = unpack('>L', response[p:p + 4])[0]
                p += 4
                word = bytes_str(response[p:p + length])
                p += length
                (docs, hits) = unpack('>2L', response[p:p + 8])
                p += 8
                result['words'].append({'word': word, 'docs': docs, 'hits': hits})
        self._reqs = []
        return results

    def BuildExcerpts(self, docs, index, words, opts=None):
        if False:
            i = 10
            return i + 15
        '\n\t\tConnect to searchd server and generate exceprts from given documents.\n\t\t'
        if not opts:
            opts = {}
        assert isinstance(docs, list)
        assert isinstance(index, (str, text_type))
        assert isinstance(words, (str, text_type))
        assert isinstance(opts, dict)
        sock = self._Connect()
        if not sock:
            return None
        opts.setdefault('before_match', '<b>')
        opts.setdefault('after_match', '</b>')
        opts.setdefault('chunk_separator', ' ... ')
        opts.setdefault('html_strip_mode', 'index')
        opts.setdefault('limit', 256)
        opts.setdefault('limit_passages', 0)
        opts.setdefault('limit_words', 0)
        opts.setdefault('around', 5)
        opts.setdefault('start_passage_id', 1)
        opts.setdefault('passage_boundary', 'none')
        flags = 1
        if opts.get('exact_phrase'):
            flags |= 2
        if opts.get('single_passage'):
            flags |= 4
        if opts.get('use_boundaries'):
            flags |= 8
        if opts.get('weight_order'):
            flags |= 16
        if opts.get('query_mode'):
            flags |= 32
        if opts.get('force_all_words'):
            flags |= 64
        if opts.get('load_files'):
            flags |= 128
        if opts.get('allow_empty'):
            flags |= 256
        if opts.get('emit_zones'):
            flags |= 512
        if opts.get('load_files_scattered'):
            flags |= 1024
        req = bytearray()
        req.extend(pack('>2L', 0, flags))
        index = str_bytes(index)
        req.extend(pack('>L', len(index)))
        req.extend(index)
        words = str_bytes(words)
        req.extend(pack('>L', len(words)))
        req.extend(words)
        opts_before_match = str_bytes(opts['before_match'])
        req.extend(pack('>L', len(opts_before_match)))
        req.extend(opts_before_match)
        opts_after_match = str_bytes(opts['after_match'])
        req.extend(pack('>L', len(opts_after_match)))
        req.extend(opts_after_match)
        opts_chunk_separator = str_bytes(opts['chunk_separator'])
        req.extend(pack('>L', len(opts_chunk_separator)))
        req.extend(opts_chunk_separator)
        req.extend(pack('>L', int(opts['limit'])))
        req.extend(pack('>L', int(opts['around'])))
        req.extend(pack('>L', int(opts['limit_passages'])))
        req.extend(pack('>L', int(opts['limit_words'])))
        req.extend(pack('>L', int(opts['start_passage_id'])))
        opts_html_strip_mode = str_bytes(opts['html_strip_mode'])
        req.extend(pack('>L', len(opts_html_strip_mode)))
        req.extend(opts_html_strip_mode)
        opts_passage_boundary = str_bytes(opts['passage_boundary'])
        req.extend(pack('>L', len(opts_passage_boundary)))
        req.extend(opts_passage_boundary)
        req.extend(pack('>L', len(docs)))
        for doc in docs:
            doc = str_bytes(doc)
            req.extend(pack('>L', len(doc)))
            req.extend(doc)
        length = len(req)
        req_head = bytearray()
        req_head.extend(pack('>2HL', SEARCHD_COMMAND_EXCERPT, VER_COMMAND_EXCERPT, length))
        req_all = req_head + req
        self._Send(sock, req_all)
        response = self._GetResponse(sock, VER_COMMAND_EXCERPT)
        if not response:
            return []
        pos = 0
        res = []
        rlen = len(response)
        for i in range(len(docs)):
            length = unpack('>L', response[pos:pos + 4])[0]
            pos += 4
            if pos + length > rlen:
                self._error = 'incomplete reply'
                return []
            res.append(bytes_str(response[pos:pos + length]))
            pos += length
        return res

    def UpdateAttributes(self, index, attrs, values, mva=False, ignorenonexistent=False):
        if False:
            print('Hello World!')
        "\n\t\tUpdate given attribute values on given documents in given indexes.\n\t\tReturns amount of updated documents (0 or more) on success, or -1 on failure.\n\n\t\t'attrs' must be a list of strings.\n\t\t'values' must be a dict with int key (document ID) and list of int values (new attribute values).\n\t\toptional boolean parameter 'mva' points that there is update of MVA attributes.\n\t\tIn this case the 'values' must be a dict with int key (document ID) and list of lists of int values\n\t\t(new MVA attribute values).\n\t\tOptional boolean parameter 'ignorenonexistent' points that the update will silently ignore any warnings about\n\t\ttrying to update a column which is not exists in current index schema.\n\n\t\tExample:\n\t\t\tres = cl.UpdateAttributes ( 'test1', [ 'group_id', 'date_added' ], { 2:[123,1000000000], 4:[456,1234567890] } )\n\t\t"
        assert isinstance(index, str)
        assert isinstance(attrs, list)
        assert isinstance(values, dict)
        for attr in attrs:
            assert isinstance(attr, str)
        for (docid, entry) in list(values.items()):
            AssertUInt32(docid)
            assert isinstance(entry, list)
            assert len(attrs) == len(entry)
            for val in entry:
                if mva:
                    assert isinstance(val, list)
                    for vals in val:
                        AssertInt32(vals)
                else:
                    AssertInt32(val)
        req = bytearray()
        index = str_bytes(index)
        req.extend(pack('>L', len(index)) + index)
        req.extend(pack('>L', len(attrs)))
        ignore_absent = 0
        if ignorenonexistent:
            ignore_absent = 1
        req.extend(pack('>L', ignore_absent))
        mva_attr = 0
        if mva:
            mva_attr = 1
        for attr in attrs:
            attr = str_bytes(attr)
            req.extend(pack('>L', len(attr)) + attr)
            req.extend(pack('>L', mva_attr))
        req.extend(pack('>L', len(values)))
        for (docid, entry) in list(values.items()):
            req.extend(pack('>Q', docid))
            for val in entry:
                val_len = val
                if mva:
                    val_len = len(val)
                req.extend(pack('>L', val_len))
                if mva:
                    for vals in val:
                        req.extend(pack('>L', vals))
        sock = self._Connect()
        if not sock:
            return None
        length = len(req)
        req_all = bytearray()
        req_all.extend(pack('>2HL', SEARCHD_COMMAND_UPDATE, VER_COMMAND_UPDATE, length))
        req_all.extend(req)
        self._Send(sock, req_all)
        response = self._GetResponse(sock, VER_COMMAND_UPDATE)
        if not response:
            return -1
        updated = unpack('>L', response[0:4])[0]
        return updated

    def BuildKeywords(self, query, index, hits):
        if False:
            i = 10
            return i + 15
        '\n\t\tConnect to searchd server, and generate keywords list for a given query.\n\t\tReturns None on failure, or a list of keywords on success.\n\t\t'
        assert isinstance(query, str)
        assert isinstance(index, str)
        assert isinstance(hits, int)
        req = bytearray()
        query = str_bytes(query)
        req.extend(pack('>L', len(query)) + query)
        index = str_bytes(index)
        req.extend(pack('>L', len(index)) + index)
        req.extend(pack('>L', hits))
        sock = self._Connect()
        if not sock:
            return None
        length = len(req)
        req_all = bytearray()
        req_all.extend(pack('>2HL', SEARCHD_COMMAND_KEYWORDS, VER_COMMAND_KEYWORDS, length))
        req_all.extend(req)
        self._Send(sock, req_all)
        response = self._GetResponse(sock, VER_COMMAND_KEYWORDS)
        if not response:
            return None
        res = []
        nwords = unpack('>L', response[0:4])[0]
        p = 4
        max_ = len(response)
        while nwords > 0 and p < max_:
            nwords -= 1
            length = unpack('>L', response[p:p + 4])[0]
            p += 4
            tokenized = response[p:p + length]
            p += length
            length = unpack('>L', response[p:p + 4])[0]
            p += 4
            normalized = response[p:p + length]
            p += length
            entry = {'tokenized': bytes_str(tokenized), 'normalized': bytes_str(normalized)}
            if hits:
                (entry['docs'], entry['hits']) = unpack('>2L', response[p:p + 8])
                p += 8
            res.append(entry)
        if nwords > 0 or p > max_:
            self._error = 'incomplete reply'
            return None
        return res

    def Status(self, session=False):
        if False:
            for i in range(10):
                print('nop')
        '\n\t\tGet the status\n\t\t'
        sock = self._Connect()
        if not sock:
            return None
        sess = 1
        if session:
            sess = 0
        req = pack('>2HLL', SEARCHD_COMMAND_STATUS, VER_COMMAND_STATUS, 4, sess)
        self._Send(sock, req)
        response = self._GetResponse(sock, VER_COMMAND_STATUS)
        if not response:
            return None
        res = []
        p = 8
        max_ = len(response)
        while p < max_:
            length = unpack('>L', response[p:p + 4])[0]
            k = response[p + 4:p + length + 4]
            p += 4 + length
            length = unpack('>L', response[p:p + 4])[0]
            v = response[p + 4:p + length + 4]
            p += 4 + length
            res += [[bytes_str(k), bytes_str(v)]]
        return res

    def Open(self):
        if False:
            for i in range(10):
                print('nop')
        if self._socket:
            self._error = 'already connected'
            return None
        server = self._Connect()
        if not server:
            return None
        request = pack('>hhII', SEARCHD_COMMAND_PERSIST, 0, 4, 1)
        self._Send(server, request)
        self._socket = server
        return True

    def Close(self):
        if False:
            i = 10
            return i + 15
        if not self._socket:
            self._error = 'not connected'
            return
        self._socket.close()
        self._socket = None

    def EscapeString(self, string):
        if False:
            print('Hello World!')
        return re.sub('([=\\(\\)|\\-!@~\\"&/\\\\\\^\\$\\=\\<])', '\\\\\\1', string)

    def FlushAttributes(self):
        if False:
            while True:
                i = 10
        sock = self._Connect()
        if not sock:
            return -1
        request = pack('>hhI', SEARCHD_COMMAND_FLUSHATTRS, VER_COMMAND_FLUSHATTRS, 0)
        self._Send(sock, request)
        response = self._GetResponse(sock, VER_COMMAND_FLUSHATTRS)
        if not response or len(response) != 4:
            self._error = 'unexpected response length'
            return -1
        tag = unpack('>L', response[0:4])[0]
        return tag

def AssertInt32(value):
    if False:
        for i in range(10):
            print('nop')
    assert isinstance(value, (int, long))
    assert value >= -2 ** 32 - 1 and value <= 2 ** 32 - 1

def AssertUInt32(value):
    if False:
        i = 10
        return i + 15
    assert isinstance(value, (int, long))
    assert value >= 0 and value <= 2 ** 32 - 1

def SetBit(flag, bit, on):
    if False:
        i = 10
        return i + 15
    if on:
        flag += 1 << bit
    else:
        reset = 255 ^ 1 << bit
        flag = flag & reset
    return flag
if sys.version_info > (3,):

    def str_bytes(x):
        if False:
            for i in range(10):
                print('nop')
        return bytearray(x, 'utf-8')
else:

    def str_bytes(x):
        if False:
            while True:
                i = 10
        if isinstance(x, unicode):
            return bytearray(x, 'utf-8')
        else:
            return bytearray(x)

def bytes_str(x):
    if False:
        for i in range(10):
            print('nop')
    assert isinstance(x, bytearray)
    return x.decode('utf-8')