"""
Interface to SMBIOS/DMI

(Parsing through dmidecode)

External References
-------------------
| `Desktop Management Interface (DMI) <http://www.dmtf.org/standards/dmi>`_
| `System Management BIOS <http://www.dmtf.org/standards/smbios>`_
| `DMIdecode <http://www.nongnu.org/dmidecode/>`_

"""
import logging
import re
import uuid
import salt.modules.cmdmod
import salt.utils.path
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        i = 10
        return i + 15
    '\n    Only work when dmidecode is installed.\n    '
    return (bool(salt.utils.path.which_bin(['dmidecode', 'smbios'])), 'The smbios execution module failed to load: neither dmidecode nor smbios in the path.')

def get(string, clean=True):
    if False:
        for i in range(10):
            print('nop')
    "\n    Get an individual DMI string from SMBIOS info\n\n    string\n        The string to fetch. DMIdecode supports:\n          - ``bios-vendor``\n          - ``bios-version``\n          - ``bios-release-date``\n          - ``system-manufacturer``\n          - ``system-product-name``\n          - ``system-version``\n          - ``system-serial-number``\n          - ``system-uuid``\n          - ``baseboard-manufacturer``\n          - ``baseboard-product-name``\n          - ``baseboard-version``\n          - ``baseboard-serial-number``\n          - ``baseboard-asset-tag``\n          - ``chassis-manufacturer``\n          - ``chassis-type``\n          - ``chassis-version``\n          - ``chassis-serial-number``\n          - ``chassis-asset-tag``\n          - ``processor-family``\n          - ``processor-manufacturer``\n          - ``processor-version``\n          - ``processor-frequency``\n\n    clean\n      | Don't return well-known false information\n      | (invalid UUID's, serial 000000000's, etcetera)\n      | Defaults to ``True``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' smbios.get system-uuid clean=False\n    "
    val = _dmidecoder('-s {}'.format(string)).strip()
    val = '\n'.join([v for v in val.split('\n') if not v.startswith('#')])
    if val.startswith('/dev/mem') or (clean and (not _dmi_isclean(string, val))):
        val = None
    return val

def records(rec_type=None, fields=None, clean=True):
    if False:
        print('Hello World!')
    "\n    Return DMI records from SMBIOS\n\n    type\n        Return only records of type(s)\n        The SMBIOS specification defines the following DMI types:\n\n        ====  ======================================\n        Type  Information\n        ====  ======================================\n         0    BIOS\n         1    System\n         2    Baseboard\n         3    Chassis\n         4    Processor\n         5    Memory Controller\n         6    Memory Module\n         7    Cache\n         8    Port Connector\n         9    System Slots\n        10    On Board Devices\n        11    OEM Strings\n        12    System Configuration Options\n        13    BIOS Language\n        14    Group Associations\n        15    System Event Log\n        16    Physical Memory Array\n        17    Memory Device\n        18    32-bit Memory Error\n        19    Memory Array Mapped Address\n        20    Memory Device Mapped Address\n        21    Built-in Pointing Device\n        22    Portable Battery\n        23    System Reset\n        24    Hardware Security\n        25    System Power Controls\n        26    Voltage Probe\n        27    Cooling Device\n        28    Temperature Probe\n        29    Electrical Current Probe\n        30    Out-of-band Remote Access\n        31    Boot Integrity Services\n        32    System Boot\n        33    64-bit Memory Error\n        34    Management Device\n        35    Management Device Component\n        36    Management Device Threshold Data\n        37    Memory Channel\n        38    IPMI Device\n        39    Power Supply\n        40    Additional Information\n        41    Onboard Devices Extended Information\n        42    Management Controller Host Interface\n        ====  ======================================\n\n    clean\n      | Don't return well-known false information\n      | (invalid UUID's, serial 000000000's, etcetera)\n      | Defaults to ``True``\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' smbios.records clean=False\n        salt '*' smbios.records 14\n        salt '*' smbios.records 4 core_count,thread_count,current_speed\n\n    "
    if rec_type is None:
        smbios = _dmi_parse(_dmidecoder(), clean, fields)
    else:
        smbios = _dmi_parse(_dmidecoder('-t {}'.format(rec_type)), clean, fields)
    return smbios

def _dmi_parse(data, clean=True, fields=None):
    if False:
        return 10
    '\n    Structurize DMI records into a nice list\n    Optionally trash bogus entries and filter output\n    '
    dmi = []
    dmi_split = re.compile('(handle [0-9]x[0-9a-f]+[^\n]+)\n', re.MULTILINE + re.IGNORECASE)
    dmi_raw = iter(re.split(dmi_split, data)[1:])
    for (handle, dmi_raw) in zip(dmi_raw, dmi_raw):
        (handle, htype) = [hline.split()[-1] for hline in handle.split(',')][0:2]
        dmi_raw = dmi_raw.split('\n')
        log.debug('Parsing handle %s', handle)
        record = {'handle': handle, 'description': dmi_raw.pop(0).strip(), 'type': int(htype)}
        if not dmi_raw:
            if not clean:
                dmi.append(record)
            continue
        dmi_data = _dmi_data(dmi_raw, clean, fields)
        if dmi_data:
            record['data'] = dmi_data
            dmi.append(record)
        elif not clean:
            dmi.append(record)
    return dmi

def _dmi_data(dmi_raw, clean, fields):
    if False:
        for i in range(10):
            print('nop')
    '\n    Parse the raw DMIdecode output of a single handle\n    into a nice dict\n    '
    dmi_data = {}
    key = None
    key_data = [None, []]
    for line in dmi_raw:
        if re.match('\\t[^\\s]+', line):
            if key is not None:
                (value, vlist) = key_data
                if vlist:
                    if value is not None:
                        vlist.insert(0, value)
                    dmi_data[key] = vlist
                elif value is not None:
                    dmi_data[key] = value
            (key, val) = line.split(':', 1)
            key = key.strip().lower().replace(' ', '_')
            if clean and key == 'header_and_data' or (fields and key not in fields):
                key = None
                continue
            else:
                key_data = [_dmi_cast(key, val.strip(), clean), []]
        elif key is None:
            continue
        elif re.match('\\t\\t[^\\s]+', line):
            val = _dmi_cast(key, line.strip(), clean)
            if val is not None:
                key_data[1].append(val)
    return dmi_data

def _dmi_cast(key, val, clean=True):
    if False:
        while True:
            i = 10
    '\n    Simple caster thingy for trying to fish out at least ints & lists from strings\n    '
    if clean and (not _dmi_isclean(key, val)):
        return
    elif not re.match('serial|part|asset|product', key, flags=re.IGNORECASE):
        if ',' in val:
            val = [el.strip() for el in val.split(',')]
        else:
            try:
                val = int(val)
            except Exception:
                pass
    return val

def _dmi_isclean(key, val):
    if False:
        print('Hello World!')
    '\n    Clean out well-known bogus values\n    '
    if not val or re.match('none', val, flags=re.IGNORECASE):
        return False
    elif 'uuid' in key:
        for uuidver in range(1, 5):
            try:
                uuid.UUID(val, version=uuidver)
                return True
            except ValueError:
                continue
        log.trace('DMI %s value %s is an invalid UUID', key, val.replace('\n', ' '))
        return False
    elif re.search('serial|part|version', key):
        return not re.match('^[0]+$', val) and (not re.match('[0]?1234567[8]?[9]?[0]?', val)) and (not re.search('sernum|part[_-]?number|specified|filled|applicable', val, flags=re.IGNORECASE))
    elif re.search('asset|manufacturer', key):
        return not re.search('manufacturer|to be filled|available|asset|^no(ne|t)', val, flags=re.IGNORECASE)
    else:
        return not re.search('to be filled', val, flags=re.IGNORECASE) and (not re.search('un(known|specified)|no(t|ne)? (asset|provided|defined|available|present|specified)', val, flags=re.IGNORECASE))

def _dmidecoder(args=None):
    if False:
        print('Hello World!')
    '\n    Call DMIdecode\n    '
    dmidecoder = salt.utils.path.which_bin(['dmidecode', 'smbios'])
    if not args:
        out = salt.modules.cmdmod._run_quiet(dmidecoder)
    else:
        out = salt.modules.cmdmod._run_quiet('{} {}'.format(dmidecoder, args))
    return out