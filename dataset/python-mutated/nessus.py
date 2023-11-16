import xmltodict
protocol_dict = {'smb': {'ports': [445, 139], 'services': ['smb', 'cifs']}, 'mssql': {'ports': [1433], 'services': ['mssql']}, 'ssh': {'ports': [22], 'services': ['ssh']}, 'winrm': {'ports': [5986, 5985], 'services': ['www', 'https?']}, 'http': {'ports': [80, 443, 8443, 8008, 8080, 8081], 'services': ['www', 'https?']}}

def parse_nessus_file(nessus_file, protocol):
    if False:
        return 10
    targets = []

    def handle_nessus_file(path, item):
        if False:
            for i in range(10):
                print('nop')
        if any(('ReportHost' and 'ReportItem' in values for values in path)):
            item = dict(path)
            ip = item['ReportHost']['name']
            if ip in targets:
                return True
            port = item['ReportItem']['port']
            svc_name = item['ReportItem']['svc_name']
            if port in protocol_dict[protocol]['ports']:
                targets.append(ip)
            if svc_name in protocol_dict[protocol]['services']:
                targets.append(ip)
            return True
        else:
            return True
    with open(nessus_file, 'r') as file_handle:
        xmltodict.parse(file_handle, item_depth=4, item_callback=handle_nessus_file)
    return targets