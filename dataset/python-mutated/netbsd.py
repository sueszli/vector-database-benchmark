from __future__ import annotations
import os
import re
import time
from ansible.module_utils.six.moves import reduce
from ansible.module_utils.facts.hardware.base import Hardware, HardwareCollector
from ansible.module_utils.facts.timeout import TimeoutError, timeout
from ansible.module_utils.facts.utils import get_file_content, get_file_lines, get_mount_size
from ansible.module_utils.facts.sysctl import get_sysctl

class NetBSDHardware(Hardware):
    """
    NetBSD-specific subclass of Hardware.  Defines memory and CPU facts:
    - memfree_mb
    - memtotal_mb
    - swapfree_mb
    - swaptotal_mb
    - processor (a list)
    - processor_cores
    - processor_count
    - devices
    - uptime_seconds
    """
    platform = 'NetBSD'
    MEMORY_FACTS = ['MemTotal', 'SwapTotal', 'MemFree', 'SwapFree']

    def populate(self, collected_facts=None):
        if False:
            i = 10
            return i + 15
        hardware_facts = {}
        self.sysctl = get_sysctl(self.module, ['machdep'])
        cpu_facts = self.get_cpu_facts()
        memory_facts = self.get_memory_facts()
        mount_facts = {}
        try:
            mount_facts = self.get_mount_facts()
        except TimeoutError:
            pass
        dmi_facts = self.get_dmi_facts()
        uptime_facts = self.get_uptime_facts()
        hardware_facts.update(cpu_facts)
        hardware_facts.update(memory_facts)
        hardware_facts.update(mount_facts)
        hardware_facts.update(dmi_facts)
        hardware_facts.update(uptime_facts)
        return hardware_facts

    def get_cpu_facts(self):
        if False:
            for i in range(10):
                print('nop')
        cpu_facts = {}
        i = 0
        physid = 0
        sockets = {}
        if not os.access('/proc/cpuinfo', os.R_OK):
            return cpu_facts
        cpu_facts['processor'] = []
        for line in get_file_lines('/proc/cpuinfo'):
            data = line.split(':', 1)
            key = data[0].strip()
            if key == 'model name' or key == 'Processor':
                if 'processor' not in cpu_facts:
                    cpu_facts['processor'] = []
                cpu_facts['processor'].append(data[1].strip())
                i += 1
            elif key == 'physical id':
                physid = data[1].strip()
                if physid not in sockets:
                    sockets[physid] = 1
            elif key == 'cpu cores':
                sockets[physid] = int(data[1].strip())
        if len(sockets) > 0:
            cpu_facts['processor_count'] = len(sockets)
            cpu_facts['processor_cores'] = reduce(lambda x, y: x + y, sockets.values())
        else:
            cpu_facts['processor_count'] = i
            cpu_facts['processor_cores'] = 'NA'
        return cpu_facts

    def get_memory_facts(self):
        if False:
            while True:
                i = 10
        memory_facts = {}
        if not os.access('/proc/meminfo', os.R_OK):
            return memory_facts
        for line in get_file_lines('/proc/meminfo'):
            data = line.split(':', 1)
            key = data[0]
            if key in NetBSDHardware.MEMORY_FACTS:
                val = data[1].strip().split(' ')[0]
                memory_facts['%s_mb' % key.lower()] = int(val) // 1024
        return memory_facts

    @timeout()
    def get_mount_facts(self):
        if False:
            while True:
                i = 10
        mount_facts = {}
        mount_facts['mounts'] = []
        fstab = get_file_content('/etc/fstab')
        if not fstab:
            return mount_facts
        for line in fstab.splitlines():
            if line.startswith('#') or line.strip() == '':
                continue
            fields = re.sub('\\s+', ' ', line).split()
            mount_statvfs_info = get_mount_size(fields[1])
            mount_info = {'mount': fields[1], 'device': fields[0], 'fstype': fields[2], 'options': fields[3]}
            mount_info.update(mount_statvfs_info)
            mount_facts['mounts'].append(mount_info)
        return mount_facts

    def get_dmi_facts(self):
        if False:
            print('Hello World!')
        dmi_facts = {}
        sysctl_to_dmi = {'machdep.dmi.system-product': 'product_name', 'machdep.dmi.system-version': 'product_version', 'machdep.dmi.system-uuid': 'product_uuid', 'machdep.dmi.system-serial': 'product_serial', 'machdep.dmi.system-vendor': 'system_vendor'}
        for mib in sysctl_to_dmi:
            if mib in self.sysctl:
                dmi_facts[sysctl_to_dmi[mib]] = self.sysctl[mib]
        return dmi_facts

    def get_uptime_facts(self):
        if False:
            for i in range(10):
                print('nop')
        sysctl_cmd = self.module.get_bin_path('sysctl')
        cmd = [sysctl_cmd, '-n', 'kern.boottime']
        (rc, out, err) = self.module.run_command(cmd)
        if rc != 0:
            return {}
        kern_boottime = out.strip()
        if not kern_boottime.isdigit():
            return {}
        return {'uptime_seconds': int(time.time() - int(kern_boottime))}

class NetBSDHardwareCollector(HardwareCollector):
    _fact_class = NetBSDHardware
    _platform = 'NetBSD'