"""The stats manager."""
import re
from glances.stats import GlancesStats
from glances.globals import iteritems
from glances.logger import logger
oid_to_short_system_name = {'.*Linux.*': 'linux', '.*Darwin.*': 'mac', '.*BSD.*': 'bsd', '.*Windows.*': 'windows', '.*Cisco.*': 'cisco', '.*VMware ESXi.*': 'esxi', '.*NetApp.*': 'netapp'}

class GlancesStatsClientSNMP(GlancesStats):
    """This class stores, updates and gives stats for the SNMP client."""

    def __init__(self, config=None, args=None):
        if False:
            print('Hello World!')
        super(GlancesStatsClientSNMP, self).__init__()
        self.config = config
        self.args = args
        self.os_name = None
        self.load_modules(self.args)

    def check_snmp(self):
        if False:
            i = 10
            return i + 15
        'Check if SNMP is available on the server.'
        from glances.snmp import GlancesSNMPClient
        snmp_client = GlancesSNMPClient(host=self.args.client, port=self.args.snmp_port, version=self.args.snmp_version, community=self.args.snmp_community, user=self.args.snmp_user, auth=self.args.snmp_auth)
        ret = snmp_client.get_by_oid('1.3.6.1.2.1.1.5.0') != {}
        if ret:
            oid_os_name = snmp_client.get_by_oid('1.3.6.1.2.1.1.1.0')
            try:
                self.system_name = self.get_system_name(oid_os_name['1.3.6.1.2.1.1.1.0'])
                logger.info('SNMP system name detected: {}'.format(self.system_name))
            except KeyError:
                self.system_name = None
                logger.warning('Cannot detect SNMP system name')
        return ret

    def get_system_name(self, oid_system_name):
        if False:
            print('Hello World!')
        'Get the short os name from the OS name OID string.'
        short_system_name = None
        if oid_system_name == '':
            return short_system_name
        for (r, v) in iteritems(oid_to_short_system_name):
            if re.search(r, oid_system_name):
                short_system_name = v
                break
        return short_system_name

    def update(self):
        if False:
            i = 10
            return i + 15
        'Update the stats using SNMP.'
        for p in self._plugins:
            if self._plugins[p].is_disabled():
                continue
            self._plugins[p].input_method = 'snmp'
            self._plugins[p].short_system_name = self.system_name
            try:
                self._plugins[p].update()
            except Exception as e:
                logger.error('Update {} failed: {}'.format(p, e))
            else:
                self._plugins[p].update_stats_history()
                self._plugins[p].update_views()