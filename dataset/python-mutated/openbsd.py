from __future__ import annotations
import re
import time
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.facts.hardware.base import Hardware, HardwareCollector
from ansible.module_utils.facts import timeout
from ansible.module_utils.facts.utils import get_file_content, get_mount_size
from ansible.module_utils.facts.sysctl import get_sysctl

class OpenBSDHardware(Hardware):
    """
    OpenBSD-specific subclass of Hardware. Defines memory, CPU and device facts:
    - memfree_mb
    - memtotal_mb
    - swapfree_mb
    - swaptotal_mb
    - processor (a list)
    - processor_cores
    - processor_count
    - processor_speed
    - uptime_seconds

    In addition, it also defines number of DMI facts and device facts.
    """
    platform = 'OpenBSD'

    def populate(self, collected_facts=None):
        if False:
            while True:
                i = 10
        hardware_facts = {}
        self.sysctl = get_sysctl(self.module, ['hw'])
        hardware_facts.update(self.get_processor_facts())
        hardware_facts.update(self.get_memory_facts())
        hardware_facts.update(self.get_device_facts())
        hardware_facts.update(self.get_dmi_facts())
        hardware_facts.update(self.get_uptime_facts())
        try:
            hardware_facts.update(self.get_mount_facts())
        except timeout.TimeoutError:
            pass
        return hardware_facts

    @timeout.timeout()
    def get_mount_facts(self):
        if False:
            i = 10
            return i + 15
        mount_facts = {}
        mount_facts['mounts'] = []
        fstab = get_file_content('/etc/fstab')
        if fstab:
            for line in fstab.splitlines():
                if line.startswith('#') or line.strip() == '':
                    continue
                fields = re.sub('\\s+', ' ', line).split()
                if fields[1] == 'none' or fields[3] == 'xx':
                    continue
                mount_statvfs_info = get_mount_size(fields[1])
                mount_info = {'mount': fields[1], 'device': fields[0], 'fstype': fields[2], 'options': fields[3]}
                mount_info.update(mount_statvfs_info)
                mount_facts['mounts'].append(mount_info)
        return mount_facts

    def get_memory_facts(self):
        if False:
            return 10
        memory_facts = {}
        (rc, out, err) = self.module.run_command('/usr/bin/vmstat')
        if rc == 0:
            memory_facts['memfree_mb'] = int(out.splitlines()[-1].split()[4]) // 1024
            memory_facts['memtotal_mb'] = int(self.sysctl['hw.physmem']) // 1024 // 1024
        (rc, out, err) = self.module.run_command('/sbin/swapctl -sk')
        if rc == 0:
            swaptrans = {ord(u'k'): None, ord(u'm'): None, ord(u'g'): None}
            data = to_text(out, errors='surrogate_or_strict').split()
            memory_facts['swapfree_mb'] = int(data[-2].translate(swaptrans)) // 1024
            memory_facts['swaptotal_mb'] = int(data[1].translate(swaptrans)) // 1024
        return memory_facts

    def get_uptime_facts(self):
        if False:
            i = 10
            return i + 15
        sysctl_cmd = self.module.get_bin_path('sysctl')
        cmd = [sysctl_cmd, '-n', 'kern.boottime']
        (rc, out, err) = self.module.run_command(cmd)
        if rc != 0:
            return {}
        kern_boottime = out.strip()
        if not kern_boottime.isdigit():
            return {}
        return {'uptime_seconds': int(time.time() - int(kern_boottime))}

    def get_processor_facts(self):
        if False:
            i = 10
            return i + 15
        cpu_facts = {}
        processor = []
        for i in range(int(self.sysctl['hw.ncpuonline'])):
            processor.append(self.sysctl['hw.model'])
        cpu_facts['processor'] = processor
        cpu_facts['processor_count'] = self.sysctl['hw.ncpuonline']
        cpu_facts['processor_cores'] = self.sysctl['hw.ncpuonline']
        return cpu_facts

    def get_device_facts(self):
        if False:
            i = 10
            return i + 15
        device_facts = {}
        devices = []
        devices.extend(self.sysctl['hw.disknames'].split(','))
        device_facts['devices'] = devices
        return device_facts

    def get_dmi_facts(self):
        if False:
            for i in range(10):
                print('nop')
        dmi_facts = {}
        sysctl_to_dmi = {'hw.product': 'product_name', 'hw.version': 'product_version', 'hw.uuid': 'product_uuid', 'hw.serialno': 'product_serial', 'hw.vendor': 'system_vendor'}
        for mib in sysctl_to_dmi:
            if mib in self.sysctl:
                dmi_facts[sysctl_to_dmi[mib]] = self.sysctl[mib]
        return dmi_facts

class OpenBSDHardwareCollector(HardwareCollector):
    _fact_class = OpenBSDHardware
    _platform = 'OpenBSD'