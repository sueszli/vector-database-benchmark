import syslog
from .NotifyBase import NotifyBase
from ..common import NotifyType
from ..utils import parse_bool
from ..AppriseLocale import gettext_lazy as _

class SyslogFacility:
    """
    All of the supported facilities
    """
    KERN = 'kern'
    USER = 'user'
    MAIL = 'mail'
    DAEMON = 'daemon'
    AUTH = 'auth'
    SYSLOG = 'syslog'
    LPR = 'lpr'
    NEWS = 'news'
    UUCP = 'uucp'
    CRON = 'cron'
    LOCAL0 = 'local0'
    LOCAL1 = 'local1'
    LOCAL2 = 'local2'
    LOCAL3 = 'local3'
    LOCAL4 = 'local4'
    LOCAL5 = 'local5'
    LOCAL6 = 'local6'
    LOCAL7 = 'local7'
SYSLOG_FACILITY_MAP = {SyslogFacility.KERN: syslog.LOG_KERN, SyslogFacility.USER: syslog.LOG_USER, SyslogFacility.MAIL: syslog.LOG_MAIL, SyslogFacility.DAEMON: syslog.LOG_DAEMON, SyslogFacility.AUTH: syslog.LOG_AUTH, SyslogFacility.SYSLOG: syslog.LOG_SYSLOG, SyslogFacility.LPR: syslog.LOG_LPR, SyslogFacility.NEWS: syslog.LOG_NEWS, SyslogFacility.UUCP: syslog.LOG_UUCP, SyslogFacility.CRON: syslog.LOG_CRON, SyslogFacility.LOCAL0: syslog.LOG_LOCAL0, SyslogFacility.LOCAL1: syslog.LOG_LOCAL1, SyslogFacility.LOCAL2: syslog.LOG_LOCAL2, SyslogFacility.LOCAL3: syslog.LOG_LOCAL3, SyslogFacility.LOCAL4: syslog.LOG_LOCAL4, SyslogFacility.LOCAL5: syslog.LOG_LOCAL5, SyslogFacility.LOCAL6: syslog.LOG_LOCAL6, SyslogFacility.LOCAL7: syslog.LOG_LOCAL7}
SYSLOG_FACILITY_RMAP = {syslog.LOG_KERN: SyslogFacility.KERN, syslog.LOG_USER: SyslogFacility.USER, syslog.LOG_MAIL: SyslogFacility.MAIL, syslog.LOG_DAEMON: SyslogFacility.DAEMON, syslog.LOG_AUTH: SyslogFacility.AUTH, syslog.LOG_SYSLOG: SyslogFacility.SYSLOG, syslog.LOG_LPR: SyslogFacility.LPR, syslog.LOG_NEWS: SyslogFacility.NEWS, syslog.LOG_UUCP: SyslogFacility.UUCP, syslog.LOG_CRON: SyslogFacility.CRON, syslog.LOG_LOCAL0: SyslogFacility.LOCAL0, syslog.LOG_LOCAL1: SyslogFacility.LOCAL1, syslog.LOG_LOCAL2: SyslogFacility.LOCAL2, syslog.LOG_LOCAL3: SyslogFacility.LOCAL3, syslog.LOG_LOCAL4: SyslogFacility.LOCAL4, syslog.LOG_LOCAL5: SyslogFacility.LOCAL5, syslog.LOG_LOCAL6: SyslogFacility.LOCAL6, syslog.LOG_LOCAL7: SyslogFacility.LOCAL7}
SYSLOG_PUBLISH_MAP = {NotifyType.INFO: syslog.LOG_INFO, NotifyType.SUCCESS: syslog.LOG_NOTICE, NotifyType.FAILURE: syslog.LOG_CRIT, NotifyType.WARNING: syslog.LOG_WARNING}

class NotifySyslog(NotifyBase):
    """
    A wrapper for Syslog Notifications
    """
    service_name = 'Syslog'
    service_url = 'https://tools.ietf.org/html/rfc5424'
    protocol = 'syslog'
    setup_url = 'https://github.com/caronc/apprise/wiki/Notify_syslog'
    request_rate_per_sec = 0
    templates = ('{schema}://', '{schema}://{facility}')
    template_tokens = dict(NotifyBase.template_tokens, **{'facility': {'name': _('Facility'), 'type': 'choice:string', 'values': [k for k in SYSLOG_FACILITY_MAP.keys()], 'default': SyslogFacility.USER}})
    template_args = dict(NotifyBase.template_args, **{'facility': {'alias_of': 'facility'}, 'logpid': {'name': _('Log PID'), 'type': 'bool', 'default': True, 'map_to': 'log_pid'}, 'logperror': {'name': _('Log to STDERR'), 'type': 'bool', 'default': False, 'map_to': 'log_perror'}})

    def __init__(self, facility=None, log_pid=True, log_perror=False, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Initialize Syslog Object\n        '
        super().__init__(**kwargs)
        if facility:
            try:
                self.facility = SYSLOG_FACILITY_MAP[facility]
            except KeyError:
                msg = 'An invalid syslog facility ({}) was specified.'.format(facility)
                self.logger.warning(msg)
                raise TypeError(msg)
        else:
            self.facility = SYSLOG_FACILITY_MAP[self.template_tokens['facility']['default']]
        self.logoptions = 0
        self.log_pid = log_pid
        self.log_perror = log_perror
        if log_pid:
            self.logoptions |= syslog.LOG_PID
        if log_perror:
            self.logoptions |= syslog.LOG_PERROR
        syslog.openlog(self.app_id, logoption=self.logoptions, facility=self.facility)
        return

    def send(self, body, title='', notify_type=NotifyType.INFO, **kwargs):
        if False:
            return 10
        '\n        Perform Syslog Notification\n        '
        SYSLOG_PUBLISH_MAP = {NotifyType.INFO: syslog.LOG_INFO, NotifyType.SUCCESS: syslog.LOG_NOTICE, NotifyType.FAILURE: syslog.LOG_CRIT, NotifyType.WARNING: syslog.LOG_WARNING}
        if title:
            body = '{}: {}'.format(title, body)
        self.throttle()
        try:
            syslog.syslog(SYSLOG_PUBLISH_MAP[notify_type], body)
        except KeyError:
            self.logger.warning('An invalid notification type ({}) was specified.'.format(notify_type))
            return False
        self.logger.info('Sent Syslog notification.')
        return True

    def url(self, privacy=False, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Returns the URL built dynamically based on specified arguments.\n        '
        params = {'logperror': 'yes' if self.log_perror else 'no', 'logpid': 'yes' if self.log_pid else 'no'}
        params.update(self.url_parameters(*args, privacy=privacy, **kwargs))
        return '{schema}://{facility}/?{params}'.format(facility=self.template_tokens['facility']['default'] if self.facility not in SYSLOG_FACILITY_RMAP else SYSLOG_FACILITY_RMAP[self.facility], schema=self.protocol, params=NotifySyslog.urlencode(params))

    @staticmethod
    def parse_url(url):
        if False:
            while True:
                i = 10
        '\n        Parses the URL and returns enough arguments that can allow\n        us to re-instantiate this object.\n\n        '
        results = NotifyBase.parse_url(url, verify_host=False)
        if not results:
            return results
        tokens = []
        if results['host']:
            tokens.append(NotifySyslog.unquote(results['host']))
        tokens.extend(NotifySyslog.split_path(results['fullpath']))
        facility = None
        if tokens:
            facility = tokens[-1].lower()
        if 'facility' in results['qsd'] and len(results['qsd']['facility']):
            facility = results['qsd']['facility'].lower()
        if facility and facility not in SYSLOG_FACILITY_MAP:
            facility = next((f for f in SYSLOG_FACILITY_MAP.keys() if f.startswith(facility)), facility)
        if facility:
            results['facility'] = facility
        results['log_pid'] = parse_bool(results['qsd'].get('logpid', NotifySyslog.template_args['logpid']['default']))
        results['log_perror'] = parse_bool(results['qsd'].get('logperror', NotifySyslog.template_args['logperror']['default']))
        return results