import time
from mrjob.job import MRJob
from mrjob.protocol import RawValueProtocol, ReprProtocol
import re

class MrS3LogParser(MRJob):
    """Parses the logs from S3 based on the S3 logging format:
    http://docs.aws.amazon.com/AmazonS3/latest/dev/LogFormat.html
    
    Aggregates a user's daily requests by user agent and operation
    
    Outputs date_time, requester, user_agent, operation, count
    """
    LOGPATS = '(\\S+) (\\S+) \\[(.*?)\\] (\\S+) (\\S+) (\\S+) (\\S+) (\\S+) ("([^"]+)"|-) (\\S+) (\\S+) (\\S+) (\\S+) (\\S+) (\\S+) ("([^"]+)"|-) ("([^"]+)"|-)'
    NUM_ENTRIES_PER_LINE = 17
    logpat = re.compile(LOGPATS)
    (S3_LOG_BUCKET_OWNER, S3_LOG_BUCKET, S3_LOG_DATE_TIME, S3_LOG_IP, S3_LOG_REQUESTER_ID, S3_LOG_REQUEST_ID, S3_LOG_OPERATION, S3_LOG_KEY, S3_LOG_HTTP_METHOD, S3_LOG_HTTP_STATUS, S3_LOG_S3_ERROR, S3_LOG_BYTES_SENT, S3_LOG_OBJECT_SIZE, S3_LOG_TOTAL_TIME, S3_LOG_TURN_AROUND_TIME, S3_LOG_REFERER, S3_LOG_USER_AGENT) = range(NUM_ENTRIES_PER_LINE)
    DELIMITER = '\t'
    INPUT_PROTOCOL = RawValueProtocol
    OUTPUT_PROTOCOL = RawValueProtocol
    INTERNAL_PROTOCOL = ReprProtocol

    def clean_date_time_zone(self, raw_date_time_zone):
        if False:
            i = 10
            return i + 15
        'Converts entry 22/Jul/2013:21:04:17 +0000 to the format\n        \'YYYY-MM-DD HH:MM:SS\' which is more suitable for loading into\n        a database such as Redshift or RDS\n\n        Note: requires the chars "[ ]" to be stripped prior to input\n        Returns the converted datetime annd timezone\n        or None for both values if failed\n\n        TODO: Needs to combine timezone with date as one field\n        '
        date_time = None
        time_zone_parsed = None
        date_parsed = raw_date_time_zone[:raw_date_time_zone.find(':')]
        time_parsed = raw_date_time_zone[raw_date_time_zone.find(':') + 1:raw_date_time_zone.find('+') - 1]
        time_zone_parsed = raw_date_time_zone[raw_date_time_zone.find('+'):]
        try:
            date_struct = time.strptime(date_parsed, '%d/%b/%Y')
            converted_date = time.strftime('%Y-%m-%d', date_struct)
            date_time = converted_date + ' ' + time_parsed
        except ValueError as error:
            raise ValueError(error)
        else:
            return (converted_date, date_time, time_zone_parsed)

    def mapper(self, _, line):
        if False:
            i = 10
            return i + 15
        line = line.strip()
        match = self.logpat.search(line)
        date_time = None
        requester = None
        user_agent = None
        operation = None
        try:
            for n in range(self.NUM_ENTRIES_PER_LINE):
                group = match.group(1 + n)
                if n == self.S3_LOG_DATE_TIME:
                    (date, date_time, time_zone_parsed) = self.clean_date_time_zone(group)
                    date_time = date + ' 00:00:00'
                elif n == self.S3_LOG_REQUESTER_ID:
                    requester = group
                elif n == self.S3_LOG_USER_AGENT:
                    user_agent = group
                elif n == self.S3_LOG_OPERATION:
                    operation = group
                else:
                    pass
        except Exception:
            yield (('Error while parsing line: %s', line), 1)
        else:
            yield ((date_time, requester, user_agent, operation), 1)

    def reducer(self, key, values):
        if False:
            print('Hello World!')
        output = list(key)
        output = self.DELIMITER.join(output) + self.DELIMITER + str(sum(values))
        yield (None, output)

    def steps(self):
        if False:
            print('Hello World!')
        return [self.mr(mapper=self.mapper, reducer=self.reducer)]
if __name__ == '__main__':
    MrS3LogParser.run()