"""Parse the log4j syslog format used by Hadoop."""
import re
from collections import namedtuple
_HADOOP_LOG4J_LINE_RE = re.compile('^\\s*(?P<timestamp>\\d\\d\\/\\d\\d\\/\\d\\d \\d\\d\\:\\d\\d\\:\\d\\d)\\s+(?P<level>[A-Z]+)\\s+(?P<logger>\\S+)(\\s+\\((?P<thread>.*?)\\))?( - ?|: ?)(?P<message>.*?)$')
_HADOOP_LOG4J_LINE_ALTERNATE_RE = re.compile('^\\s*(?P<timestamp>\\d\\d\\/\\d\\d\\/\\d\\d \\d\\d\\:\\d\\d\\:\\d\\d)\\s+(?P<level>[A-Z]+)(\\s+\\[(?P<thread>.*?)\\])\\s+(?P<logger>\\S+)(\\s+\\((?P<caller_location>\\S+)\\))?( - ?|: ?)(?P<message>.*?)$')

class Log4jRecord(namedtuple('_Log4jRecord', 'caller_location level logger message num_lines start_line thread timestamp')):
    """Represents a Log4J log record.

    caller_location -- e.g. 'YarnClientImpl.java:submitApplication(251)'
    level -- e.g. 'INFO'
    logger -- e.g. 'amazon.emr.metrics.MetricsSaver'
    message -- the actual message. If this is a multi-line message (e.g.
        for counters), the lines will be joined by '\\n'
    num_lines -- how many lines made up the message
    start_line -- which line the message started on (0-indexed)
    thread -- e.g. 'main'. Defaults to ''
    timestamp -- unparsed timestamp, e.g. '15/12/07 20:49:28'
    """

    def __new__(cls, caller_location, level, logger, message, num_lines, start_line, thread, timestamp):
        if False:
            print('Hello World!')
        return super(Log4jRecord, cls).__new__(cls, caller_location, level, logger, message, num_lines, start_line, thread, timestamp)

    @staticmethod
    def fake_record(line, line_num):
        if False:
            return 10
        "Used to represent a leading Log4J line that doesn't conform to the regular expressions we\n        expect.\n        "
        return Log4jRecord(caller_location='', level='', logger='', message=line, num_lines=1, start_line=line_num, thread='', timestamp='')

def parse_hadoop_log4j_records(lines):
    if False:
        print('Hello World!')
    'Parse lines from a hadoop log into log4j records.\n\n    Yield Log4jRecords.\n\n    Lines will be converted to unicode, and trailing \\r and \\n will be stripped\n    from lines.\n\n    Also yields fake records for leading non-log4j lines (trailing non-log4j\n    lines are assumed to be part of a multiline message if not pre-filtered).\n    '
    last_record = None
    line_num = 0
    for (line_num, raw_line) in enumerate(lines.split('\n')):
        line = raw_line.rstrip('\r\n')
        m = _HADOOP_LOG4J_LINE_RE.match(line) or _HADOOP_LOG4J_LINE_ALTERNATE_RE.match(line)
        if m:
            if last_record:
                last_record = last_record._replace(num_lines=line_num - last_record.start_line)
                yield last_record
            matches = m.groupdict()
            last_record = Log4jRecord(caller_location=matches.get('caller_location', ''), level=matches['level'], logger=matches['logger'], message=matches['message'], num_lines=1, start_line=line_num, thread=matches.get('thread', ''), timestamp=matches['timestamp'])
        elif last_record:
            last_record = last_record._replace(message=last_record.message + '\n' + line)
        else:
            yield Log4jRecord.fake_record(line, line_num)
    if last_record:
        last_record = last_record._replace(num_lines=line_num + 1 - last_record.start_line)
        yield last_record