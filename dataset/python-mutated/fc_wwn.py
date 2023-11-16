from __future__ import annotations
import sys
import glob
import ansible.module_utils.compat.typing as t
from ansible.module_utils.facts.utils import get_file_lines
from ansible.module_utils.facts.collector import BaseFactCollector

class FcWwnInitiatorFactCollector(BaseFactCollector):
    name = 'fibre_channel_wwn'
    _fact_ids = set()

    def collect(self, module=None, collected_facts=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Example contents /sys/class/fc_host/*/port_name:\n\n        0x21000014ff52a9bb\n\n        '
        fc_facts = {}
        fc_facts['fibre_channel_wwn'] = []
        if sys.platform.startswith('linux'):
            for fcfile in glob.glob('/sys/class/fc_host/*/port_name'):
                for line in get_file_lines(fcfile):
                    fc_facts['fibre_channel_wwn'].append(line.rstrip()[2:])
        elif sys.platform.startswith('sunos'):
            cmd = module.get_bin_path('fcinfo')
            if cmd:
                cmd = cmd + ' hba-port'
                (rc, fcinfo_out, err) = module.run_command(cmd)
                if rc == 0 and fcinfo_out:
                    for line in fcinfo_out.splitlines():
                        if 'Port WWN' in line:
                            data = line.split(' ')
                            fc_facts['fibre_channel_wwn'].append(data[-1].rstrip())
        elif sys.platform.startswith('aix'):
            cmd = module.get_bin_path('lsdev')
            lscfg_cmd = module.get_bin_path('lscfg')
            if cmd and lscfg_cmd:
                cmd = cmd + ' -Cc adapter -l fcs*'
                (rc, lsdev_out, err) = module.run_command(cmd)
                if rc == 0 and lsdev_out:
                    for line in lsdev_out.splitlines():
                        if 'Available' in line:
                            data = line.split(' ')
                            cmd = lscfg_cmd + ' -vl %s' % data[0]
                            (rc, lscfg_out, err) = module.run_command(cmd)
                            if rc == 0 and lscfg_out:
                                for line in lscfg_out.splitlines():
                                    if 'Network Address' in line:
                                        data = line.split('.')
                                        fc_facts['fibre_channel_wwn'].append(data[-1].rstrip())
        elif sys.platform.startswith('hp-ux'):
            cmd = module.get_bin_path('ioscan')
            fcmsu_cmd = module.get_bin_path('fcmsutil', opt_dirs=['/opt/fcms/bin'])
            if cmd and fcmsu_cmd:
                cmd = cmd + ' -fnC FC'
                (rc, ioscan_out, err) = module.run_command(cmd)
                if rc == 0 and ioscan_out:
                    for line in ioscan_out.splitlines():
                        line = line.strip()
                        if '/dev/fcd' in line:
                            dev = line.split(' ')
                            cmd = fcmsu_cmd + ' %s' % dev[0]
                            (rc, fcmsutil_out, err) = module.run_command(cmd)
                            if rc == 0 and fcmsutil_out:
                                for line in fcmsutil_out.splitlines():
                                    if 'N_Port Port World Wide Name' in line:
                                        data = line.split('=')
                                        fc_facts['fibre_channel_wwn'].append(data[-1].strip())
        return fc_facts