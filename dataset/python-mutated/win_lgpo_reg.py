"""
A Salt Util for working with the Registry.pol file. The Registry.pol file is the
source of truth for registry settings that are configured via LGPO.
"""
import logging
import os
import re
import struct
import salt.modules.win_file
import salt.utils.files
import salt.utils.win_reg
from salt.exceptions import CommandExecutionError
CLASS_INFO = {'User': {'policy_path': os.path.join(os.getenv('WINDIR', 'C:\\Windows'), 'System32', 'GroupPolicy', 'User', 'Registry.pol'), 'hive': 'HKEY_USERS', 'lgpo_section': 'User Configuration', 'gpt_extension_location': 'gPCUserExtensionNames', 'gpt_extension_guid': '[{35378EAC-683F-11D2-A89A-00C04FBBCFA2}{D02B1F73-3407-48AE-BA88-E8213C6761F1}]'}, 'Machine': {'policy_path': os.path.join(os.getenv('WINDIR', 'C:\\Windows'), 'System32', 'GroupPolicy', 'Machine', 'Registry.pol'), 'hive': 'HKEY_LOCAL_MACHINE', 'lgpo_section': 'Computer Configuration', 'gpt_extension_location': 'gPCMachineExtensionNames', 'gpt_extension_guid': '[{35378EAC-683F-11D2-A89A-00C04FBBCFA2}{D02B1F72-3407-48AE-BA88-E8213C6761F1}]'}}
REG_POL_HEADER = '剐来\x01\x00'
GPT_INI_PATH = os.path.join(os.getenv('WINDIR', 'C:\\Windows'), 'System32', 'GroupPolicy', 'gpt.ini')
log = logging.getLogger(__name__)
__virtualname__ = 'lgpo_reg'

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Only works on Windows with the lgpo_reg module\n    '
    if not salt.utils.platform.is_windows():
        return (False, 'LGPO_REG Util: Only available on Windows')
    return __virtualname__

def search_reg_pol(search_string, policy_data):
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper function to do a regex search of a string value in policy_data.\n    This is used to search the policy data from a registry.pol file or from\n    gpt.ini\n\n    Args:\n        search_string (str): The string to search for\n\n        policy_data (str): The data to be searched\n\n    Returns:\n        bool: ``True`` if the regex search_string is found, otherwise ``False``\n    '
    if policy_data:
        if search_string:
            log.debug('LGPO_REG Util: Searching for %s', search_string)
            match = re.search(search_string, policy_data, re.IGNORECASE)
            if match:
                log.debug('LGPO_REG Util: Found')
                return True
    return False

def read_reg_pol_file(reg_pol_path):
    if False:
        i = 10
        return i + 15
    '\n    Helper function to read the content of the Registry.pol file\n\n    Args:\n        reg_pol_path (str): The path to the Registry.pol file\n\n    Returns:\n        bytes: The data as contained in the Registry.pol file\n    '
    return_data = None
    if os.path.exists(reg_pol_path):
        log.debug('LGPO_REG Util: Reading from %s', reg_pol_path)
        with salt.utils.files.fopen(reg_pol_path, 'rb') as pol_file:
            return_data = pol_file.read()
    return return_data

def write_reg_pol_data(data_to_write, policy_file_path, gpt_extension, gpt_extension_guid, gpt_ini_path=GPT_INI_PATH):
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper function to actually write the data to a Registry.pol file\n\n    Also updates/edits the gpt.ini file to include the ADM policy extensions\n    to let the computer know user and/or machine registry policy files need\n    to be processed\n\n    Args:\n        data_to_write (bytes): Data to write into the user/machine registry.pol\n            file\n\n        policy_file_path (str): Path to the registry.pol file\n\n        gpt_extension (str): GPT extension list name from _policy_info class for\n            this registry class gpt_extension_location\n\n        gpt_extension_guid (str): ADMX registry extension guid for the class\n\n        gpt_ini_path (str): The path to the gpt.ini file\n\n    Returns:\n        bool: True if successful\n\n    Raises:\n        CommandExecutionError: On failure\n    '
    if not os.path.exists(policy_file_path):
        log.debug('LGPO_REG Util: Creating parent directories for Registry.pol')
        salt.modules.win_file.makedirs_(policy_file_path)
    if data_to_write is None:
        data_to_write = b''
    try:
        with salt.utils.files.fopen(policy_file_path, 'wb') as pol_file:
            reg_pol_header = REG_POL_HEADER.encode('utf-16-le')
            if not data_to_write.startswith(reg_pol_header):
                log.debug('LGPO_REG Util: Writing header to %s', policy_file_path)
                pol_file.write(reg_pol_header)
            log.debug('LGPO_REG Util: Writing to %s', policy_file_path)
            pol_file.write(data_to_write)
    except Exception as e:
        msg = 'An error occurred attempting to write to {}, the exception was: {}'.format(policy_file_path, e)
        log.exception(msg)
        raise CommandExecutionError(msg)
    gpt_ini_data = ''
    if os.path.exists(gpt_ini_path):
        with salt.utils.files.fopen(gpt_ini_path, 'r') as gpt_file:
            gpt_ini_data = gpt_file.read()
        gpt_ini_data = gpt_ini_data.replace('\r\n', '_|-').replace('\n', '_|-').replace('_|-', '\r\n')
    if not search_reg_pol('\\[General\\]\\r\\n', gpt_ini_data):
        log.debug('LGPO_REG Util: Adding [General] section to gpt.ini')
        gpt_ini_data = '[General]\r\n' + gpt_ini_data
    if search_reg_pol(f'{re.escape(gpt_extension)}=', gpt_ini_data):
        gpt_ext_loc = re.search(f'^{re.escape(gpt_extension)}=.*\\r\\n', gpt_ini_data, re.IGNORECASE | re.MULTILINE)
        gpt_ext_str = gpt_ini_data[gpt_ext_loc.start():gpt_ext_loc.end()]
        if not search_reg_pol(search_string=f'{re.escape(gpt_extension_guid)}', policy_data=gpt_ext_str):
            log.debug('LGPO_REG Util: Inserting gpt extension GUID')
            gpt_ext_str = gpt_ext_str.split('=')
            gpt_ext_str[1] = gpt_extension_guid + gpt_ext_str[1]
            gpt_ext_str = '='.join(gpt_ext_str)
            gpt_ini_data = gpt_ini_data[0:gpt_ext_loc.start()] + gpt_ext_str + gpt_ini_data[gpt_ext_loc.end():]
    else:
        general_location = re.search('^\\[General\\]\\r\\n', gpt_ini_data, re.IGNORECASE | re.MULTILINE)
        gpt_ini_data = '{}{}={}\r\n{}'.format(gpt_ini_data[general_location.start():general_location.end()], gpt_extension, gpt_extension_guid, gpt_ini_data[general_location.end():])
    if search_reg_pol('Version=', gpt_ini_data):
        version_loc = re.search('^Version=.*\\r\\n', gpt_ini_data, re.IGNORECASE | re.MULTILINE)
        version_str = gpt_ini_data[version_loc.start():version_loc.end()]
        version_str = version_str.split('=')
        version_nums = struct.unpack(b'>2H', struct.pack(b'>I', int(version_str[1])))
        if gpt_extension.lower() == 'gPCMachineExtensionNames'.lower():
            version_nums = (version_nums[0], version_nums[1] + 1)
        elif gpt_extension.lower() == 'gPCUserExtensionNames'.lower():
            version_nums = (version_nums[0] + 1, version_nums[1])
        version_num = struct.unpack(b'>I', struct.pack(b'>2H', *version_nums))[0]
        gpt_ini_data = '{}{}={}\r\n{}'.format(gpt_ini_data[0:version_loc.start()], 'Version', version_num, gpt_ini_data[version_loc.end():])
    else:
        general_location = re.search('^\\[General\\]\\r\\n', gpt_ini_data, re.IGNORECASE | re.MULTILINE)
        if gpt_extension.lower() == 'gPCMachineExtensionNames'.lower():
            version_nums = (0, 1)
        elif gpt_extension.lower() == 'gPCUserExtensionNames'.lower():
            version_nums = (1, 0)
        gpt_ini_data = '{}{}={}\r\n{}'.format(gpt_ini_data[general_location.start():general_location.end()], 'Version', int('{}{}'.format(str(version_nums[0]).zfill(4), str(version_nums[1]).zfill(4)), 16), gpt_ini_data[general_location.end():])
    if gpt_ini_data:
        try:
            with salt.utils.files.fopen(gpt_ini_path, 'w') as gpt_file:
                gpt_file.write(gpt_ini_data)
        except Exception as e:
            msg = 'An error occurred attempting to write the gpg.ini file.\npath: {}\nexception: {}'.format(gpt_ini_path, e)
            log.exception(msg)
            raise CommandExecutionError(msg)
    return True

def reg_pol_to_dict(policy_data):
    if False:
        print('Hello World!')
    '\n    Convert the data obtained from a Registry.pol file to a dictionary.\n\n    Args:\n        policy_data (bytes): The data as retrieved from the Registry.pol file\n\n    Raises:\n        SaltInvocationError: Invalid or corrupt policy data\n\n    Returns:\n        dict: A dictionary representation of the Registry.pol data\n    '
    reg_pol_header = REG_POL_HEADER.encode('utf-16-le')
    if policy_data is None:
        policy_data = reg_pol_header
    if not policy_data.startswith(reg_pol_header):
        msg = 'LGPO_REG Util: Invalid Header. Registry.pol may be corrupt'
        raise CommandExecutionError(msg)
    pol_file_data = policy_data.lstrip(reg_pol_header)
    if not pol_file_data:
        log.debug('LGPO_REG Util: No registry.pol data to return')
        return {}

    def strip_field_end(value):
        if False:
            print('Hello World!')
        while value[-2:] == b'\x00\x00':
            value = value[:-2]
        return value
    log.debug('LGPO_REG Util: Unpacking reg pol data')
    reg_pol = {}
    for policy in pol_file_data.split(b']\x00[\x00'):
        policy = policy.replace(b']\x00', b'').replace(b'[\x00', b'')
        (key, v_name, v_type, v_size, v_data) = policy.split(b';\x00', 4)
        key = strip_field_end(key).decode('utf-16-le')
        v_name = strip_field_end(v_name).decode('utf-16-le')
        v_type = struct.unpack('<i', v_type)[0]
        if v_type == 0:
            pass
        elif v_type in (1, 2):
            v_data = strip_field_end(v_data).decode('utf-16-le')
        elif v_type == 4:
            v_data = struct.unpack('<i', v_data)[0]
        elif v_type == 5:
            v_data = struct.unpack('>i', v_data)[0]
        elif v_type == 7:
            v_data = strip_field_end(v_data)
            if not v_data:
                v_data = None
            else:
                v_data = v_data.decode('utf-16-le').split('\x00')
        elif v_type == 11:
            v_data = struct.unpack('<q', v_data)[0]
        else:
            msg = f'LGPO_REG Util: Found unknown registry type: {v_type}'
            raise CommandExecutionError(msg)
        reg = salt.utils.win_reg.Registry()
        v_type = reg.vtype_reverse.get(v_type, 'REG_NONE')
        reg_pol.setdefault(key, {})
        if not v_name:
            reg_pol[key]['*'] = 'CREATEKEY'
        else:
            reg_pol[key][v_name] = {'type': v_type, 'data': v_data}
    return reg_pol

def dict_to_reg_pol(data):
    if False:
        i = 10
        return i + 15
    '\n    Convert a dictionary to the bytes format expected by the Registry.pol file\n\n    Args:\n        data (dict): A dictionary containing the contents to be converted\n\n    Returns:\n        bytes: The data to be written to the Registry.pol file\n    '
    reg = salt.utils.win_reg.Registry()
    pol_enter_delim = '['.encode('utf-16-le')
    pol_exit_delim = ']'.encode('utf-16-le')
    pol_section_delim = ';'.encode('utf-16-le')
    pol_section_term = '\x00'.encode('utf-16-le')
    policies = []
    for (key, value) in data.items():
        for (v_name, d) in value.items():
            if v_name == '*' and d == 'CREATEKEY':
                v_name = ''
                d = {'type': 'REG_NONE'}
            try:
                v_type = reg.vtype[d['type']]
            except KeyError:
                msg = 'LGPO_REG Util: Found unknown registry type: {}'.format(d['type'])
                raise CommandExecutionError(msg)
            policy = [f'{key}'.encode('utf-16-le') + pol_section_term, f'{v_name}'.encode('utf-16-le') + pol_section_term, struct.pack('<i', v_type)]
            if v_type == 0:
                v_data = b''
            elif v_type in (1, 2):
                v_data = d['data'].encode('utf-16-le') + pol_section_term
            elif v_type == 4:
                v_data = struct.pack('<i', int(d['data']))
            elif v_type == 5:
                v_data = struct.pack('>i', int(d['data']))
            elif v_type == 7:
                if d['data'] is None:
                    v_data = pol_section_term
                elif len(d['data']) == 0:
                    v_data = pol_section_term
                elif len(d['data']) == 1 and (not d['data'][0]):
                    v_data = pol_section_term
                else:
                    v_data = '\x00'.join(d['data']).encode('utf-16-le') + pol_section_term + pol_section_term
            elif v_type == 11:
                v_data = struct.pack('<q', int(d['data']))
            if len(v_data) > 65535:
                msg = 'LGPO_REG Util: Size exceeds 65535 bytes'
                raise CommandExecutionError(msg)
            v_size = len(v_data).to_bytes(2, 'little') + pol_section_term
            policy.append(v_size)
            policy.append(v_data)
            policies.append(pol_section_delim.join(policy))
    policy_file_data = REG_POL_HEADER.encode('utf-16-le')
    for policy in policies:
        policy_file_data += pol_enter_delim + policy + pol_exit_delim
    return policy_file_data