from __future__ import annotations
from ansible.module_utils.facts.timeout import TimeoutError
from ansible.module_utils.facts.hardware.base import HardwareCollector
from ansible.module_utils.facts.hardware.linux import LinuxHardware

class HurdHardware(LinuxHardware):
    """
    GNU Hurd specific subclass of Hardware. Define memory and mount facts
    based on procfs compatibility translator mimicking the interface of
    the Linux kernel.
    """
    platform = 'GNU'

    def populate(self, collected_facts=None):
        if False:
            print('Hello World!')
        hardware_facts = {}
        uptime_facts = self.get_uptime_facts()
        memory_facts = self.get_memory_facts()
        mount_facts = {}
        try:
            mount_facts = self.get_mount_facts()
        except TimeoutError:
            pass
        hardware_facts.update(uptime_facts)
        hardware_facts.update(memory_facts)
        hardware_facts.update(mount_facts)
        return hardware_facts

class HurdHardwareCollector(HardwareCollector):
    _fact_class = HurdHardware
    _platform = 'GNU'