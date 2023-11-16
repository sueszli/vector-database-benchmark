import struct
import volatility.utils as utils
import volatility.obj as obj
import volatility.win32.tasks as tasks
import volatility.debug as debug
import volatility.plugins.malware.svcscan as svcscan
import volatility.win32.rawreg as rawreg
import volatility.plugins.registry.hivelist as hivelist

class ServiceDiff(svcscan.SvcScan):
    """List Windows services (ala Plugx)"""

    @staticmethod
    def is_valid_profile(profile):
        if False:
            while True:
                i = 10
        return profile.metadata.get('os', 'unknown') == 'windows' and profile.metadata.get('memory_model', '32bit') == '32bit'

    @staticmethod
    def services_from_registry(addr_space):
        if False:
            for i in range(10):
                print('nop')
        'Enumerate services from the cached registry hive'
        services = {}
        plugin = hivelist.HiveList(addr_space.get_config())
        for hive in plugin.calculate():
            name = hive.get_name()
            if not name.lower().endswith('system'):
                continue
            hive_space = hive.address_space()
            root = rawreg.get_root(hive_space)
            if not root:
                break
            key = rawreg.open_key(root, ['ControlSet001', 'Services'])
            if not key:
                break
            for subkey in rawreg.subkeys(key):
                services[str(subkey.Name).lower()] = subkey
            break
        return services

    @staticmethod
    def services_from_memory_list(addr_space):
        if False:
            i = 10
            return i + 15
        "Enumerate services from walking the SCM's linked list"
        services = {}
        pre_vista = addr_space.profile.metadata.get('major', 0) < 6
        mem_model = addr_space.profile.metadata.get('memory_model', '32bit')
        if mem_model != '32bit':
            return {}
        for process in tasks.pslist(addr_space):
            if str(process.ImageFileName) != 'services.exe':
                continue
            process_space = process.get_process_address_space()
            image_base = process.Peb.ImageBaseAddress
            dos_header = obj.Object('_IMAGE_DOS_HEADER', offset=image_base, vm=process_space)
            if not dos_header:
                debug.warning('Unable to parse DOS header')
                break
            try:
                sections = list(dos_header.get_nt_header().get_sections())
                text_seg = sections[0]
            except ValueError:
                debug.warning('Could not parse the PE header')
                break
            except IndexError:
                debug.warning('No sections were found in the array')
                break
            virtual_address = text_seg.VirtualAddress + image_base
            data = process_space.zread(virtual_address, text_seg.Misc.VirtualSize)
            list_head = None
            for offset in utils.iterfind(data, '£'):
                if not (data[offset + 5] == '£' and data[offset + 10] == '£' and (data[offset + 15] == '£') and (data[offset + 20] == '£') and (data[offset + 25] == 'è')):
                    continue
                list_head = obj.Object('unsigned long', offset=virtual_address + offset + 21, vm=process_space)
            if not list_head:
                debug.warning('Unable to find the signature')
                break
            record = obj.Object('_SERVICE_RECORD', offset=list_head, vm=process_space)
            while record:
                name = str(record.ServiceName.dereference() or '')
                name = name.lower()
                services[name] = record
                record = record.ServiceList.Flink.dereference()
        return services

    @staticmethod
    def compare(reg_list, mem_list):
        if False:
            i = 10
            return i + 15
        'Compare the services found in the registry with those in memory'
        missing = set(reg_list.keys()) - set(mem_list.keys())
        for service in missing:
            has_imagepath = False
            for value in rawreg.values(reg_list[service]):
                if str(value.Name) == 'ImagePath':
                    has_imagepath = True
                    break
            if has_imagepath:
                yield reg_list[service]

    def calculate(self):
        if False:
            print('Hello World!')
        addr_space = utils.load_as(self._config)
        from_memory = ServiceDiff.services_from_memory_list(addr_space)
        if not from_memory:
            debug.error('Could not enumerate services from memory')
        from_registry = ServiceDiff.services_from_registry(addr_space)
        if not from_registry:
            debug.error('Could not enumerate services from the registry')
        return ServiceDiff.compare(from_registry, from_memory)

    def render_text(self, outfd, data):
        if False:
            while True:
                i = 10
        for subkey in data:
            outfd.write('\n{0:<20}: {1}\n'.format('Missing service', subkey.Name))
            for value in rawreg.values(subkey):
                (value_type, value_data) = rawreg.value_data(value)
                outfd.write('{0:<20}: ({1}) {2}\n'.format(value.Name, value_type, value_data))