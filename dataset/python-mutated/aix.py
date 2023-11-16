from __future__ import annotations
import re
from ansible.module_utils.facts.hardware.base import Hardware, HardwareCollector
from ansible.module_utils.facts.utils import get_mount_size

class AIXHardware(Hardware):
    """
    AIX-specific subclass of Hardware.  Defines memory and CPU facts:
    - memfree_mb
    - memtotal_mb
    - swapfree_mb
    - swaptotal_mb
    - processor (a list)
    - processor_count
    - processor_cores
    - processor_threads_per_core
    - processor_vcpus
    """
    platform = 'AIX'

    def populate(self, collected_facts=None):
        if False:
            i = 10
            return i + 15
        hardware_facts = {}
        cpu_facts = self.get_cpu_facts()
        memory_facts = self.get_memory_facts()
        dmi_facts = self.get_dmi_facts()
        vgs_facts = self.get_vgs_facts()
        mount_facts = self.get_mount_facts()
        devices_facts = self.get_device_facts()
        hardware_facts.update(cpu_facts)
        hardware_facts.update(memory_facts)
        hardware_facts.update(dmi_facts)
        hardware_facts.update(vgs_facts)
        hardware_facts.update(mount_facts)
        hardware_facts.update(devices_facts)
        return hardware_facts

    def get_cpu_facts(self):
        if False:
            while True:
                i = 10
        cpu_facts = {}
        cpu_facts['processor'] = []
        cpu_facts['processor_count'] = 1
        (rc, out, err) = self.module.run_command('/usr/sbin/lsdev -Cc processor')
        if out:
            i = 0
            for line in out.splitlines():
                if 'Available' in line:
                    if i == 0:
                        data = line.split(' ')
                        cpudev = data[0]
                    i += 1
            cpu_facts['processor_cores'] = int(i)
            (rc, out, err) = self.module.run_command('/usr/sbin/lsattr -El ' + cpudev + ' -a type')
            data = out.split(' ')
            cpu_facts['processor'] = [data[1]]
            cpu_facts['processor_threads_per_core'] = 1
            (rc, out, err) = self.module.run_command('/usr/sbin/lsattr -El ' + cpudev + ' -a smt_threads')
            if out:
                data = out.split(' ')
                cpu_facts['processor_threads_per_core'] = int(data[1])
            cpu_facts['processor_vcpus'] = cpu_facts['processor_cores'] * cpu_facts['processor_threads_per_core']
        return cpu_facts

    def get_memory_facts(self):
        if False:
            while True:
                i = 10
        memory_facts = {}
        pagesize = 4096
        (rc, out, err) = self.module.run_command('/usr/bin/vmstat -v')
        for line in out.splitlines():
            data = line.split()
            if 'memory pages' in line:
                pagecount = int(data[0])
            if 'free pages' in line:
                freecount = int(data[0])
        memory_facts['memtotal_mb'] = pagesize * pagecount // 1024 // 1024
        memory_facts['memfree_mb'] = pagesize * freecount // 1024 // 1024
        (rc, out, err) = self.module.run_command('/usr/sbin/lsps -s')
        if out:
            lines = out.splitlines()
            data = lines[1].split()
            swaptotal_mb = int(data[0].rstrip('MB'))
            percused = int(data[1].rstrip('%'))
            memory_facts['swaptotal_mb'] = swaptotal_mb
            memory_facts['swapfree_mb'] = int(swaptotal_mb * (100 - percused) / 100)
        return memory_facts

    def get_dmi_facts(self):
        if False:
            while True:
                i = 10
        dmi_facts = {}
        (rc, out, err) = self.module.run_command('/usr/sbin/lsattr -El sys0 -a fwversion')
        data = out.split()
        dmi_facts['firmware_version'] = data[1].strip('IBM,')
        lsconf_path = self.module.get_bin_path('lsconf')
        if lsconf_path:
            (rc, out, err) = self.module.run_command(lsconf_path)
            if rc == 0 and out:
                for line in out.splitlines():
                    data = line.split(':')
                    if 'Machine Serial Number' in line:
                        dmi_facts['product_serial'] = data[1].strip()
                    if 'LPAR Info' in line:
                        dmi_facts['lpar_info'] = data[1].strip()
                    if 'System Model' in line:
                        dmi_facts['product_name'] = data[1].strip()
        return dmi_facts

    def get_vgs_facts(self):
        if False:
            i = 10
            return i + 15
        '\n        Get vg and pv Facts\n        rootvg:\n        PV_NAME           PV STATE          TOTAL PPs   FREE PPs    FREE DISTRIBUTION\n        hdisk0            active            546         0           00..00..00..00..00\n        hdisk1            active            546         113         00..00..00..21..92\n        realsyncvg:\n        PV_NAME           PV STATE          TOTAL PPs   FREE PPs    FREE DISTRIBUTION\n        hdisk74           active            1999        6           00..00..00..00..06\n        testvg:\n        PV_NAME           PV STATE          TOTAL PPs   FREE PPs    FREE DISTRIBUTION\n        hdisk105          active            999         838         200..39..199..200..200\n        hdisk106          active            999         599         200..00..00..199..200\n        '
        vgs_facts = {}
        lsvg_path = self.module.get_bin_path('lsvg')
        xargs_path = self.module.get_bin_path('xargs')
        cmd = '%s -o | %s %s -p' % (lsvg_path, xargs_path, lsvg_path)
        if lsvg_path and xargs_path:
            (rc, out, err) = self.module.run_command(cmd, use_unsafe_shell=True)
            if rc == 0 and out:
                vgs_facts['vgs'] = {}
                for m in re.finditer('(\\S+):\\n.*FREE DISTRIBUTION(\\n(\\S+)\\s+(\\w+)\\s+(\\d+)\\s+(\\d+).*)+', out):
                    vgs_facts['vgs'][m.group(1)] = []
                    pp_size = 0
                    cmd = '%s %s' % (lsvg_path, m.group(1))
                    (rc, out, err) = self.module.run_command(cmd)
                    if rc == 0 and out:
                        pp_size = re.search('PP SIZE:\\s+(\\d+\\s+\\S+)', out).group(1)
                        for n in re.finditer('(\\S+)\\s+(\\w+)\\s+(\\d+)\\s+(\\d+).*', m.group(0)):
                            pv_info = {'pv_name': n.group(1), 'pv_state': n.group(2), 'total_pps': n.group(3), 'free_pps': n.group(4), 'pp_size': pp_size}
                            vgs_facts['vgs'][m.group(1)].append(pv_info)
        return vgs_facts

    def get_mount_facts(self):
        if False:
            return 10
        mount_facts = {}
        mount_facts['mounts'] = []
        mounts = []
        mount_path = self.module.get_bin_path('mount')
        (rc, mount_out, err) = self.module.run_command(mount_path)
        if mount_out:
            for line in mount_out.split('\n'):
                fields = line.split()
                if len(fields) != 0 and fields[0] != 'node' and (fields[0][0] != '-') and re.match('^/.*|^[a-zA-Z].*|^[0-9].*', fields[0]):
                    if re.match('^/', fields[0]):
                        mount = fields[1]
                        mount_info = {'mount': mount, 'device': fields[0], 'fstype': fields[2], 'options': fields[6], 'time': '%s %s %s' % (fields[3], fields[4], fields[5])}
                        mount_info.update(get_mount_size(mount))
                    else:
                        if len(fields) < 8:
                            fields.append('')
                        mount_info = {'mount': fields[2], 'device': '%s:%s' % (fields[0], fields[1]), 'fstype': fields[3], 'options': fields[7], 'time': '%s %s %s' % (fields[4], fields[5], fields[6])}
                    mounts.append(mount_info)
        mount_facts['mounts'] = mounts
        return mount_facts

    def get_device_facts(self):
        if False:
            print('Hello World!')
        device_facts = {}
        device_facts['devices'] = {}
        lsdev_cmd = self.module.get_bin_path('lsdev', True)
        lsattr_cmd = self.module.get_bin_path('lsattr', True)
        (rc, out_lsdev, err) = self.module.run_command(lsdev_cmd)
        for line in out_lsdev.splitlines():
            field = line.split()
            device_attrs = {}
            device_name = field[0]
            device_state = field[1]
            device_type = field[2:]
            lsattr_cmd_args = [lsattr_cmd, '-E', '-l', device_name]
            (rc, out_lsattr, err) = self.module.run_command(lsattr_cmd_args)
            for attr in out_lsattr.splitlines():
                attr_fields = attr.split()
                attr_name = attr_fields[0]
                attr_parameter = attr_fields[1]
                device_attrs[attr_name] = attr_parameter
            device_facts['devices'][device_name] = {'state': device_state, 'type': ' '.join(device_type), 'attributes': device_attrs}
        return device_facts

class AIXHardwareCollector(HardwareCollector):
    _platform = 'AIX'
    _fact_class = AIXHardware