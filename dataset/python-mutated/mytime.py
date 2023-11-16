__author__ = 'Cyril Jaquier'
__copyright__ = 'Copyright (c) 2004 Cyril Jaquier'
__license__ = 'GPL'
import datetime
import re
import time

class MyTime:
    """A wrapper around time module primarily for testing purposes

	This class is a wrapper around time.time()  and time.gmtime(). When
	performing unit test, it is very useful to get a fixed value from
	these functions.  Thus, time.time() and time.gmtime() should never
	be called directly.  This wrapper should be called instead. The API
	are equivalent.
	"""
    myTime = None
    alternateNowTime = None
    alternateNow = None

    @staticmethod
    def setAlternateNow(t):
        if False:
            print('Hello World!')
        'Set current time.\n\n\t\tUse None in order to always get the real current time.\n\n\t\t@param t the time to set or None\n\t\t'
        MyTime.alternateNowTime = t
        MyTime.alternateNow = datetime.datetime.fromtimestamp(t) if t is not None else None

    @staticmethod
    def setTime(t):
        if False:
            for i in range(10):
                print('nop')
        'Set current time.\n\n\t\tUse None in order to always get the real current time.\n\n\t\t@param t the time to set or None\n\t\t'
        MyTime.myTime = t

    @staticmethod
    def time():
        if False:
            print('Hello World!')
        'Decorate time.time() for the purpose of testing mocking\n\n\t\t@return time.time() if setTime was called with None\n\t\t'
        if MyTime.myTime is None:
            return time.time()
        else:
            return MyTime.myTime

    @staticmethod
    def gmtime():
        if False:
            print('Hello World!')
        'Decorate time.gmtime() for the purpose of testing mocking\n\n\t\t@return time.gmtime() if setTime was called with None\n\t\t'
        if MyTime.myTime is None:
            return time.gmtime()
        else:
            return time.gmtime(MyTime.myTime)

    @staticmethod
    def now():
        if False:
            return 10
        'Decorate datetime.now() for the purpose of testing mocking\n\n\t\t@return datetime.now() if setTime was called with None\n\t\t'
        if MyTime.myTime is None:
            return datetime.datetime.now()
        if MyTime.myTime == MyTime.alternateNowTime:
            return MyTime.alternateNow
        return datetime.datetime.fromtimestamp(MyTime.myTime)

    @staticmethod
    def localtime(x=None):
        if False:
            while True:
                i = 10
        'Decorate time.localtime() for the purpose of testing mocking\n\n\t\t@return time.localtime() if setTime was called with None\n\t\t'
        if MyTime.myTime is None or x is not None:
            return time.localtime(x)
        else:
            return time.localtime(MyTime.myTime)

    @staticmethod
    def time2str(unixTime, format='%Y-%m-%d %H:%M:%S'):
        if False:
            print('Hello World!')
        'Convert time to a string representing as date and time using given format.\n\t\tDefault format is ISO 8601, YYYY-MM-DD HH:MM:SS without microseconds.\n\n\t\t@return ISO-capable string representation of given unixTime\n\t\t'
        dt = datetime.datetime.fromtimestamp(unixTime).replace(microsecond=0) if unixTime < 253402214400 else datetime.datetime(9999, 12, 31, 23, 59, 59)
        return dt.strftime(format)
    _str2sec_prep = re.compile('(?i)(?<=[a-z])(\\d)')
    _str2sec_fini = re.compile('(\\d)\\s+(\\d)')
    _str2sec_subpart = '(?i)(?<=[\\d\\s])(%s)\\b'
    _str2sec_parts = ((re.compile(_str2sec_subpart % 'days?|da|dd?'), '*' + str(24 * 60 * 60)), (re.compile(_str2sec_subpart % 'weeks?|wee?|ww?'), '*' + str(7 * 24 * 60 * 60)), (re.compile(_str2sec_subpart % 'months?|mon?'), '*' + str((365 * 3 + 366) * 24 * 60 * 60 / 4 / 12)), (re.compile(_str2sec_subpart % 'years?|yea?|yy?'), '*' + str((365 * 3 + 366) * 24 * 60 * 60 / 4)), (re.compile(_str2sec_subpart % 'seconds?|sec?|ss?'), '*' + str(1)), (re.compile(_str2sec_subpart % 'minutes?|min?|mm?'), '*' + str(60)), (re.compile(_str2sec_subpart % 'hours?|hou?|hh?'), '*' + str(60 * 60)))

    @staticmethod
    def str2seconds(val):
        if False:
            for i in range(10):
                print('nop')
        'Wraps string expression like "1h 2m 3s" into number contains seconds (3723).\n\t\tThe string expression will be evaluated as mathematical expression, spaces between each groups \n\t\t  will be wrapped to "+" operand (only if any operand does not specified between).\n\t\tBecause of case insensitivity and overwriting with minutes ("m" or "mm"), the short replacement for month\n\t\t  are "mo" or "mon".\n\t\tEx: 1hour+30min = 5400\n\t\t    0d 1h 30m   = 5400\n\t\t    1year-6mo   = 15778800\n\t\t    6 months    = 15778800\n\t\twarn: month is not 30 days, it is a year in seconds / 12, the leap years will be respected also:\n\t\t      >>>> float(str2seconds("1month")) / 60 / 60 / 24\n\t\t      30.4375\n\t\t      >>>> float(str2seconds("1year")) / 60 / 60 / 24\n\t\t      365.25\t\n\t\t\n\t\t@returns number (calculated seconds from expression "val")\n\t\t'
        if isinstance(val, (int, float, complex)):
            return val
        val = MyTime._str2sec_prep.sub(' \\1', val)
        for (rexp, rpl) in MyTime._str2sec_parts:
            val = rexp.sub(rpl, val)
        val = MyTime._str2sec_fini.sub('\\1+\\2', val)
        return eval(val)

    class seconds2str:
        """Converts seconds to string on demand (if string representation needed).
		Ex: seconds2str(86400*390)            = 1y 3w 4d
		    seconds2str(86400*368)            = 1y 3d
		    seconds2str(86400*365.5)          = 1y
		    seconds2str(86400*2+3600*7+60*15) = 2d 7h 15m
		    seconds2str(86400*2+3599)         = 2d 1h
		    seconds2str(3600-5)               = 1h
		    seconds2str(3600-10)              = 59m 50s
		    seconds2str(59)                   = 59s
		"""

        def __init__(self, sec):
            if False:
                for i in range(10):
                    print('nop')
            self.sec = sec

        def __str__(self):
            if False:
                print('Hello World!')
            s = self.sec
            c = 3
            if s >= 31536000:
                s = int(round(float(s) / 86400))
                r = str(s // 365) + 'y '
                s %= 365
                if s >= 7:
                    r += str(s // 7) + 'w '
                    s %= 7
                if s:
                    r += str(s) + 'd '
                return r[:-1]
            if s >= 604800:
                s = int(round(float(s) / 3600))
                r = str(s // 168) + 'w '
                s %= 168
                if s >= 24:
                    r += str(s // 24) + 'd '
                    s %= 24
                if s:
                    r += str(s) + 'h '
                return r[:-1]
            if s >= 86400:
                s = int(round(float(s) / 60))
                r = str(s // 1440) + 'd '
                s %= 1440
                if s >= 60:
                    r += str(s // 60) + 'h '
                    s %= 60
                if s:
                    r += str(s) + 'm '
                return r[:-1]
            if s >= 3595:
                s = int(round(float(s) / 10))
                r = str(s // 360) + 'h '
                s %= 360
                if s >= 6:
                    r += str(s // 6) + 'm '
                    s %= 6
                return r[:-1]
            r = ''
            if s >= 60:
                r += str(s // 60) + 'm '
                s %= 60
            if s:
                r += str(s) + 's '
            elif not self.sec:
                r = '0 '
            return r[:-1]

        def __repr__(self):
            if False:
                print('Hello World!')
            return self.__str__()