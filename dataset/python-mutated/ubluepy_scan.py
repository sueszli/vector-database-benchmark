from ubluepy import Scanner, constants

def bytes_to_str(bytes):
    if False:
        i = 10
        return i + 15
    string = ''
    for b in bytes:
        string += chr(b)
    return string

def get_device_names(scan_entries):
    if False:
        print('Hello World!')
    dev_names = []
    for e in scan_entries:
        scan = e.getScanData()
        if scan:
            for s in scan:
                if s[0] == constants.ad_types.AD_TYPE_COMPLETE_LOCAL_NAME:
                    dev_names.append((e, bytes_to_str(s[2])))
    return dev_names

def find_device_by_name(name):
    if False:
        while True:
            i = 10
    s = Scanner()
    scan_res = s.scan(100)
    device_names = get_device_names(scan_res)
    for dev in device_names:
        if name == dev[1]:
            return dev[0]