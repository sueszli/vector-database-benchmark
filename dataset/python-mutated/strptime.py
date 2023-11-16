import re
import time
import calendar
import datetime
from _strptime import LocaleTime, TimeRE, _calc_julian_from_U_or_W
from .mytime import MyTime
locale_time = LocaleTime()
TZ_ABBR_RE = '[A-Z](?:[A-Z]{2,4})?'
FIXED_OFFSET_TZ_RE = re.compile('(%s)?([+-][01]\\d(?::?\\d{2})?)?$' % (TZ_ABBR_RE,))
timeRE = TimeRE()
timeRE['k'] = ' ?(?P<H>[0-2]?\\d)'
timeRE['l'] = ' ?(?P<I>1?\\d)'
timeRE['Z'] = '(?P<Z>Z|[A-Z]{3,5})'
timeRE['z'] = '(?P<z>Z|UTC|GMT|[+-][01]\\d(?::?\\d{2})?)'
timeRE['ExZ'] = '(?P<Z>%s)' % (TZ_ABBR_RE,)
timeRE['Exz'] = '(?P<z>(?:%s)?[+-][01]\\d(?::?\\d{2})?|%s)' % (TZ_ABBR_RE, TZ_ABBR_RE)
timeRE['d'] = '(?P<d>[1-2]\\d|[0 ]?[1-9]|3[0-1])'
timeRE['m'] = '(?P<m>0?[1-9]|1[0-2])'
timeRE['Y'] = '(?P<Y>\\d{4})'
timeRE['H'] = '(?P<H>[0-1]?\\d|2[0-3])'
timeRE['M'] = '(?P<M>[0-5]?\\d)'
timeRE['S'] = '(?P<S>[0-5]?\\d|6[0-1])'
timeRE['Exd'] = '(?P<d>[1-2]\\d|0[1-9]|3[0-1])'
timeRE['Exm'] = '(?P<m>0[1-9]|1[0-2])'
timeRE['ExH'] = '(?P<H>[0-1]\\d|2[0-3])'
timeRE['Exk'] = ' ?(?P<H>[0-1]?\\d|2[0-3])'
timeRE['Exl'] = ' ?(?P<I>1[0-2]|\\d)'
timeRE['ExM'] = '(?P<M>[0-5]\\d)'
timeRE['ExS'] = '(?P<S>[0-5]\\d|6[0-1])'

def _updateTimeRE():
    if False:
        for i in range(10):
            print('nop')

    def _getYearCentRE(cent=(0, 3), distance=3, now=(MyTime.now(), MyTime.alternateNow)):
        if False:
            i = 10
            return i + 15
        ' Build century regex for last year and the next years (distance).\n\t\t\t\n\t\tThereby respect possible run in the test-cases (alternate date used there)\n\t\t'
        cent = lambda year, f=cent[0], t=cent[1]: str(year)[f:t]

        def grp(exprset):
            if False:
                print('Hello World!')
            c = None
            if len(exprset) > 1:
                for i in exprset:
                    if c is None or i[0:-1] == c:
                        c = i[0:-1]
                    else:
                        c = None
                        break
                if not c:
                    for i in exprset:
                        if c is None or i[0] == c:
                            c = i[0]
                        else:
                            c = None
                            break
                if c:
                    return '%s%s' % (c, grp([i[len(c):] for i in exprset]))
            return ('(?:%s)' % '|'.join(exprset) if len(exprset[0]) > 1 else '[%s]' % ''.join(exprset)) if len(exprset) > 1 else ''.join(exprset)
        exprset = set((cent(now[0].year + i) for i in (-1, distance)))
        if len(now) > 1 and now[1]:
            exprset |= set((cent(now[1].year + i) for i in range(-1, now[0].year - now[1].year + 1, distance)))
        return grp(sorted(list(exprset)))
    timeRE['ExY'] = '(?P<Y>%s\\d)' % _getYearCentRE(cent=(0, 3), distance=3, now=(datetime.datetime.now(), datetime.datetime.fromtimestamp(min(MyTime.alternateNowTime or 978393600, 978393600))))
    timeRE['Exy'] = '(?P<y>\\d{2})'
_updateTimeRE()

def getTimePatternRE():
    if False:
        while True:
            i = 10
    keys = list(timeRE.keys())
    patt = '%%(%%|%s|[%s])' % ('|'.join([k for k in keys if len(k) > 1]), ''.join([k for k in keys if len(k) == 1]))
    names = {'a': 'DAY', 'A': 'DAYNAME', 'b': 'MON', 'B': 'MONTH', 'd': 'Day', 'H': '24hour', 'I': '12hour', 'j': 'Yearday', 'm': 'Month', 'M': 'Minute', 'p': 'AMPM', 'S': 'Second', 'U': 'Yearweek', 'w': 'Weekday', 'W': 'Yearweek', 'y': 'Year2', 'Y': 'Year', '%': '%', 'z': 'Zone offset', 'f': 'Microseconds', 'Z': 'Zone name'}
    for key in set(keys) - set(names):
        if key.startswith('Ex'):
            kn = names.get(key[2:])
            if kn:
                names[key] = 'Ex' + kn
                continue
        names[key] = '%%%s' % key
    return (patt, names)

def validateTimeZone(tz):
    if False:
        while True:
            i = 10
    'Validate a timezone and convert it to offset if it can (offset-based TZ).\n\n\tFor now this accepts the UTC[+-]hhmm format (UTC has aliases GMT/Z and optional).\n\tAdditionally it accepts all zone abbreviations mentioned below in TZ_STR.\n\tNote that currently this zone abbreviations are offset-based and used fixed\n\toffset without automatically DST-switch (if CET used then no automatically CEST-switch).\n\t\n\tIn the future, it may be extended for named time zones (such as Europe/Paris)\n\tpresent on the system, if a suitable tz library is present (pytz).\n\t'
    if tz is None:
        return None
    m = FIXED_OFFSET_TZ_RE.match(tz)
    if m is None:
        raise ValueError('Unknown or unsupported time zone: %r' % tz)
    tz = m.groups()
    return zone2offset(tz, 0)

def zone2offset(tz, dt):
    if False:
        i = 10
        return i + 15
    "Return the proper offset, in minutes according to given timezone at a given time.\n\n\tParameters\n\t----------\n\ttz: symbolic timezone or offset (for now only TZA?([+-]hh:?mm?)? is supported,\n\t\tas value are accepted:\n\t\t  int offset;\n\t\t  string in form like 'CET+0100' or 'UTC' or '-0400';\n\t\t  tuple (or list) in form (zone name, zone offset);\n\tdt: datetime instance for offset computation (currently unused)\n\t"
    if isinstance(tz, int):
        return tz
    if isinstance(tz, str):
        return validateTimeZone(tz)
    (tz, tzo) = tz
    if tzo is None or tzo == '':
        return TZ_ABBR_OFFS[tz]
    if len(tzo) <= 3:
        return TZ_ABBR_OFFS[tz] + int(tzo) * 60
    if tzo[3] != ':':
        return TZ_ABBR_OFFS[tz] + (-1 if tzo[0] == '-' else 1) * (int(tzo[1:3]) * 60 + int(tzo[3:5]))
    else:
        return TZ_ABBR_OFFS[tz] + (-1 if tzo[0] == '-' else 1) * (int(tzo[1:3]) * 60 + int(tzo[4:6]))

def reGroupDictStrptime(found_dict, msec=False, default_tz=None):
    if False:
        while True:
            i = 10
    'Return time from dictionary of strptime fields\n\n\tThis is tweaked from python built-in _strptime.\n\n\tParameters\n\t----------\n\tfound_dict : dict\n\t\tDictionary where keys represent the strptime fields, and values the\n\t\trespective value.\n\tdefault_tz : default timezone to apply if nothing relevant is in found_dict\n                     (may be a non-fixed one in the future)\n\tReturns\n\t-------\n\tfloat\n\t\tUnix time stamp.\n\t'
    now = year = month = day = tzoffset = weekday = julian = week_of_year = None
    hour = minute = second = fraction = 0
    for (key, val) in found_dict.items():
        if val is None:
            continue
        if key == 'y':
            year = int(val)
            if year <= 2000:
                year += 2000
        elif key == 'Y':
            year = int(val)
        elif key == 'm':
            month = int(val)
        elif key == 'B':
            month = locale_time.f_month.index(val.lower())
        elif key == 'b':
            month = locale_time.a_month.index(val.lower())
        elif key == 'd':
            day = int(val)
        elif key == 'H':
            hour = int(val)
        elif key == 'I':
            hour = int(val)
            ampm = found_dict.get('p', '').lower()
            if ampm in ('', locale_time.am_pm[0]):
                if hour == 12:
                    hour = 0
            elif ampm == locale_time.am_pm[1]:
                if hour != 12:
                    hour += 12
        elif key == 'M':
            minute = int(val)
        elif key == 'S':
            second = int(val)
        elif key == 'f':
            if msec:
                s = val
                s += '0' * (6 - len(s))
                fraction = int(s)
        elif key == 'A':
            weekday = locale_time.f_weekday.index(val.lower())
        elif key == 'a':
            weekday = locale_time.a_weekday.index(val.lower())
        elif key == 'w':
            weekday = int(val) - 1
            if weekday < 0:
                weekday = 6
        elif key == 'j':
            julian = int(val)
        elif key in ('U', 'W'):
            week_of_year = int(val)
            week_of_year_start = 6 if key == 'U' else 0
        elif key in ('z', 'Z'):
            z = val
            if z in ('Z', 'UTC', 'GMT'):
                tzoffset = 0
            else:
                tzoffset = zone2offset(z, 0)
    assume_year = False
    if year is None:
        if not now:
            now = MyTime.now()
        year = now.year
        assume_year = True
    if month is None or day is None:
        if julian is None and week_of_year is not None and (weekday is not None):
            julian = _calc_julian_from_U_or_W(year, week_of_year, weekday, week_of_year_start == 0)
        if julian is not None:
            datetime_result = datetime.datetime.fromordinal(julian - 1 + datetime.datetime(year, 1, 1).toordinal())
            year = datetime_result.year
            month = datetime_result.month
            day = datetime_result.day
    assume_today = False
    if month is None and day is None:
        if not now:
            now = MyTime.now()
        month = now.month
        day = now.day
        assume_today = True
    date_result = datetime.datetime(year, month, day, hour, minute, second, fraction)
    if tzoffset is None and default_tz is not None:
        tzoffset = zone2offset(default_tz, date_result)
    if tzoffset is not None:
        date_result -= datetime.timedelta(seconds=tzoffset * 60)
    if assume_today:
        if not now:
            now = MyTime.now()
        if date_result > now:
            date_result -= datetime.timedelta(days=1)
    if assume_year:
        if not now:
            now = MyTime.now()
        if date_result > now + datetime.timedelta(days=1):
            date_result = date_result.replace(year=year - 1, month=month, day=day)
    if tzoffset is not None:
        tm = calendar.timegm(date_result.utctimetuple())
    else:
        tm = time.mktime(date_result.timetuple())
    if msec:
        tm += fraction / 1000000.0
    return tm
TZ_ABBR_OFFS = {'': 0, None: 0}
TZ_STR = '\n\t-12 Y\n\t-11 X NUT SST\n\t-10 W CKT HAST HST TAHT TKT\n\t-9 V AKST GAMT GIT HADT HNY\n\t-8 U AKDT CIST HAY HNP PST PT\n\t-7 T HAP HNR MST PDT\n\t-6 S CST EAST GALT HAR HNC MDT\n\t-5 R CDT COT EASST ECT EST ET HAC HNE PET\n\t-4 Q AST BOT CLT COST EDT FKT GYT HAE HNA PYT\n\t-3 P ADT ART BRT CLST FKST GFT HAA PMST PYST SRT UYT WGT\n\t-2 O BRST FNT PMDT UYST WGST\n\t-1 N AZOT CVT EGT\n\t0 Z EGST GMT UTC WET WT\n\t1 A CET DFT WAT WEDT WEST\n\t2 B CAT CEDT CEST EET SAST WAST\n\t3 C EAT EEDT EEST IDT MSK\n\t4 D AMT AZT GET GST KUYT MSD MUT RET SAMT SCT\n\t5 E AMST AQTT AZST HMT MAWT MVT PKT TFT TJT TMT UZT YEKT\n\t6 F ALMT BIOT BTT IOT KGT NOVT OMST YEKST\n\t7 G CXT DAVT HOVT ICT KRAT NOVST OMSST THA WIB\n\t8 H ACT AWST BDT BNT CAST HKT IRKT KRAST MYT PHT SGT ULAT WITA WST\n\t9 I AWDT IRKST JST KST PWT TLT WDT WIT YAKT\n\t10 K AEST ChST PGT VLAT YAKST YAPT\n\t11 L AEDT LHDT MAGT NCT PONT SBT VLAST VUT\n\t12 M ANAST ANAT FJT GILT MAGST MHT NZST PETST PETT TVT WFT\n\t13 FJST NZDT\n\t11.5 NFT\n\t10.5 ACDT LHST\n\t9.5 ACST\n\t6.5 CCT MMT\n\t5.75 NPT\n\t5.5 SLT\n\t4.5 AFT IRDT\n\t3.5 IRST\n\t-2.5 HAT NDT\n\t-3.5 HNT NST NT\n\t-4.5 HLV VET\n\t-9.5 MART MIT\n'

def _init_TZ_ABBR():
    if False:
        i = 10
        return i + 15
    'Initialized TZ_ABBR_OFFS dictionary (TZ -> offset in minutes)'
    for tzline in map(str.split, TZ_STR.split('\n')):
        if not len(tzline):
            continue
        tzoffset = int(float(tzline[0]) * 60)
        for tz in tzline[1:]:
            TZ_ABBR_OFFS[tz] = tzoffset
_init_TZ_ABBR()