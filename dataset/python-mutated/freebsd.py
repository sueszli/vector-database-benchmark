from __future__ import annotations
import os
import json
import re
import struct
import time
from ansible.module_utils.facts.hardware.base import Hardware, HardwareCollector
from ansible.module_utils.facts.timeout import TimeoutError, timeout
from ansible.module_utils.facts.utils import get_file_content, get_mount_size

class FreeBSDHardware(Hardware):
    """
    FreeBSD-specific subclass of Hardware.  Defines memory and CPU facts:
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
    platform = 'FreeBSD'
    DMESG_BOOT = '/var/run/dmesg.boot'

    def populate(self, collected_facts=None):
        if False:
            while True:
                i = 10
        hardware_facts = {}
        cpu_facts = self.get_cpu_facts()
        memory_facts = self.get_memory_facts()
        uptime_facts = self.get_uptime_facts()
        dmi_facts = self.get_dmi_facts()
        device_facts = self.get_device_facts()
        mount_facts = {}
        try:
            mount_facts = self.get_mount_facts()
        except TimeoutError:
            pass
        hardware_facts.update(cpu_facts)
        hardware_facts.update(memory_facts)
        hardware_facts.update(uptime_facts)
        hardware_facts.update(dmi_facts)
        hardware_facts.update(device_facts)
        hardware_facts.update(mount_facts)
        return hardware_facts

    def get_cpu_facts(self):
        if False:
            return 10
        cpu_facts = {}
        cpu_facts['processor'] = []
        sysctl = self.module.get_bin_path('sysctl')
        if sysctl:
            (rc, out, err) = self.module.run_command('%s -n hw.ncpu' % sysctl, check_rc=False)
            cpu_facts['processor_count'] = out.strip()
        dmesg_boot = get_file_content(FreeBSDHardware.DMESG_BOOT)
        if not dmesg_boot:
            try:
                (rc, dmesg_boot, err) = self.module.run_command(self.module.get_bin_path('dmesg'), check_rc=False)
            except Exception:
                dmesg_boot = ''
        for line in dmesg_boot.splitlines():
            if 'CPU:' in line:
                cpu = re.sub('CPU:\\s+', '', line)
                cpu_facts['processor'].append(cpu.strip())
            if 'Logical CPUs per core' in line:
                cpu_facts['processor_cores'] = line.split()[4]
        return cpu_facts

    def get_memory_facts(self):
        if False:
            for i in range(10):
                print('nop')
        memory_facts = {}
        sysctl = self.module.get_bin_path('sysctl')
        if sysctl:
            (rc, out, err) = self.module.run_command('%s vm.stats' % sysctl, check_rc=False)
            for line in out.splitlines():
                data = line.split()
                if 'vm.stats.vm.v_page_size' in line:
                    pagesize = int(data[1])
                if 'vm.stats.vm.v_page_count' in line:
                    pagecount = int(data[1])
                if 'vm.stats.vm.v_free_count' in line:
                    freecount = int(data[1])
            memory_facts['memtotal_mb'] = pagesize * pagecount // 1024 // 1024
            memory_facts['memfree_mb'] = pagesize * freecount // 1024 // 1024
        swapinfo = self.module.get_bin_path('swapinfo')
        if swapinfo:
            (rc, out, err) = self.module.run_command('%s -k' % swapinfo)
            lines = out.splitlines()
            if len(lines[-1]) == 0:
                lines.pop()
            data = lines[-1].split()
            if data[0] != 'Device':
                memory_facts['swaptotal_mb'] = int(data[1]) // 1024
                memory_facts['swapfree_mb'] = int(data[3]) // 1024
        return memory_facts

    def get_uptime_facts(self):
        if False:
            return 10
        sysctl_cmd = self.module.get_bin_path('sysctl')
        cmd = [sysctl_cmd, '-b', 'kern.boottime']
        (rc, out, err) = self.module.run_command(cmd, encoding=None)
        struct_format = '@L'
        struct_size = struct.calcsize(struct_format)
        if rc != 0 or len(out) < struct_size:
            return {}
        (kern_boottime,) = struct.unpack(struct_format, out[:struct_size])
        return {'uptime_seconds': int(time.time() - kern_boottime)}

    @timeout()
    def get_mount_facts(self):
        if False:
            print('Hello World!')
        mount_facts = {}
        mount_facts['mounts'] = []
        fstab = get_file_content('/etc/fstab')
        if fstab:
            for line in fstab.splitlines():
                if line.startswith('#') or line.strip() == '':
                    continue
                fields = re.sub('\\s+', ' ', line).split()
                mount_statvfs_info = get_mount_size(fields[1])
                mount_info = {'mount': fields[1], 'device': fields[0], 'fstype': fields[2], 'options': fields[3]}
                mount_info.update(mount_statvfs_info)
                mount_facts['mounts'].append(mount_info)
        return mount_facts

    def get_device_facts(self):
        if False:
            return 10
        device_facts = {}
        sysdir = '/dev'
        device_facts['devices'] = {}
        drives = re.compile('(ada?\\d+|da\\d+|a?cd\\d+)')
        slices = re.compile('(ada?\\d+s\\d+\\w*|da\\d+s\\d+\\w*)')
        if os.path.isdir(sysdir):
            dirlist = sorted(os.listdir(sysdir))
            for device in dirlist:
                d = drives.match(device)
                if d:
                    device_facts['devices'][d.group(1)] = []
                s = slices.match(device)
                if s:
                    device_facts['devices'][d.group(1)].append(s.group(1))
        return device_facts

    def get_dmi_facts(self):
        if False:
            while True:
                i = 10
        ' learn dmi facts from system\n\n        Use dmidecode executable if available'
        dmi_facts = {}
        dmi_bin = self.module.get_bin_path('dmidecode')
        DMI_DICT = {'bios_date': 'bios-release-date', 'bios_vendor': 'bios-vendor', 'bios_version': 'bios-version', 'board_asset_tag': 'baseboard-asset-tag', 'board_name': 'baseboard-product-name', 'board_serial': 'baseboard-serial-number', 'board_vendor': 'baseboard-manufacturer', 'board_version': 'baseboard-version', 'chassis_asset_tag': 'chassis-asset-tag', 'chassis_serial': 'chassis-serial-number', 'chassis_vendor': 'chassis-manufacturer', 'chassis_version': 'chassis-version', 'form_factor': 'chassis-type', 'product_name': 'system-product-name', 'product_serial': 'system-serial-number', 'product_uuid': 'system-uuid', 'product_version': 'system-version', 'system_vendor': 'system-manufacturer'}
        for (k, v) in DMI_DICT.items():
            if dmi_bin is not None:
                (rc, out, err) = self.module.run_command('%s -s %s' % (dmi_bin, v))
                if rc == 0:
                    dmi_facts[k] = ''.join([line for line in out.splitlines() if not line.startswith('#')])
                    try:
                        json.dumps(dmi_facts[k])
                    except UnicodeDecodeError:
                        dmi_facts[k] = 'NA'
                else:
                    dmi_facts[k] = 'NA'
            else:
                dmi_facts[k] = 'NA'
        return dmi_facts

class FreeBSDHardwareCollector(HardwareCollector):
    _fact_class = FreeBSDHardware
    _platform = 'FreeBSD'