sort_programs_key_list = ['cpu_percent', 'memory_percent', 'cpu_times', 'io_counters', 'name']

def create_program_dict(p):
    if False:
        while True:
            i = 10
    'Create a new entry in the dict (new program)'
    return {'time_since_update': p['time_since_update'], 'num_threads': p['num_threads'] or 0, 'cpu_percent': p['cpu_percent'] or 0, 'memory_percent': p['memory_percent'] or 0, 'cpu_times': p['cpu_times'] or (), 'memory_info': p['memory_info'] or (), 'io_counters': p['io_counters'] or (), 'childrens': [p['pid']], 'name': p['name'], 'cmdline': [p['name']], 'pid': '_', 'username': p.get('username', '_'), 'nice': p['nice'], 'status': p['status']}

def update_program_dict(program, p):
    if False:
        print('Hello World!')
    'Update an existing entry in the dict (existing program)'
    program['num_threads'] += p['num_threads'] or 0
    program['cpu_percent'] += p['cpu_percent'] or 0
    program['memory_percent'] += p['memory_percent'] or 0
    program['cpu_times'] += p['cpu_times'] or ()
    program['memory_info'] += p['memory_info'] or ()
    program['io_counters'] += p['io_counters']
    program['childrens'].append(p['pid'])
    program['username'] = p.get('username', '_') if p.get('username') == program['username'] else '_'
    program['nice'] = p['nice'] if p['nice'] == program['nice'] else '_'
    program['status'] = p['status'] if p['status'] == program['status'] else '_'

def processes_to_programs(processes):
    if False:
        while True:
            i = 10
    'Convert a list of processes to a list of programs.'
    programs_dict = {}
    key = 'name'
    for p in processes:
        if p[key] not in programs_dict:
            programs_dict[p[key]] = create_program_dict(p)
        else:
            update_program_dict(programs_dict[p[key]], p)
    return [programs_dict[p] for p in programs_dict]