"""Methods for parsing baseml results files."""
import re
line_floats_re = re.compile('-*\\d+\\.\\d+')

def parse_basics(lines, results):
    if False:
        i = 10
        return i + 15
    'Parse the basics that should be present in most baseml results files.'
    version_re = re.compile('BASEML \\(in paml version (\\d+\\.\\d+[a-z]*).*')
    np_re = re.compile('lnL\\(ntime:\\s+\\d+\\s+np:\\s+(\\d+)\\)')
    num_params = -1
    for line in lines:
        line_floats_res = line_floats_re.findall(line)
        line_floats = [float(val) for val in line_floats_res]
        version_res = version_re.match(line)
        if version_res is not None:
            results['version'] = version_res.group(1)
        if 'ln Lmax' in line and len(line_floats) == 1:
            results['lnL max'] = line_floats[0]
        elif 'lnL(ntime:' in line and line_floats:
            results['lnL'] = line_floats[0]
            np_res = np_re.match(line)
            if np_res is not None:
                num_params = int(np_res.group(1))
        elif 'tree length' in line and len(line_floats) == 1:
            results['tree length'] = line_floats[0]
        elif re.match('\\(+', line) is not None:
            if ':' in line:
                results['tree'] = line.strip()
    return (results, num_params)

def parse_parameters(lines, results, num_params):
    if False:
        i = 10
        return i + 15
    'Parse the various parameters from the file.'
    parameters = {}
    parameters = parse_parameter_list(lines, parameters, num_params)
    parameters = parse_kappas(lines, parameters)
    parameters = parse_rates(lines, parameters)
    parameters = parse_freqs(lines, parameters)
    results['parameters'] = parameters
    return results

def parse_parameter_list(lines, parameters, num_params):
    if False:
        print('Hello World!')
    'Parse the parameters list, which is just an unlabeled list of numeric values.'
    for line_num in range(len(lines)):
        line = lines[line_num]
        line_floats_res = line_floats_re.findall(line)
        line_floats = [float(val) for val in line_floats_res]
        if len(line_floats) == num_params:
            parameters['parameter list'] = line.strip()
            if 'SEs for parameters:' in lines[line_num + 1]:
                SEs_line = lines[line_num + 2]
                parameters['SEs'] = SEs_line.strip()
            break
    return parameters

def parse_kappas(lines, parameters):
    if False:
        print('Hello World!')
    'Parse out the kappa parameters.'
    kappa_found = False
    for line in lines:
        line_floats_res = line_floats_re.findall(line)
        line_floats = [float(val) for val in line_floats_res]
        if 'Parameters (kappa)' in line:
            kappa_found = True
        elif kappa_found and line_floats:
            branch_res = re.match('\\s(\\d+\\.\\.\\d+)', line)
            if branch_res is None:
                if len(line_floats) == 1:
                    parameters['kappa'] = line_floats[0]
                else:
                    parameters['kappa'] = line_floats
                kappa_found = False
            else:
                if parameters.get('branches') is None:
                    parameters['branches'] = {}
                branch = branch_res.group(1)
                if line_floats:
                    parameters['branches'][branch] = {'t': line_floats[0], 'kappa': line_floats[1], 'TS': line_floats[2], 'TV': line_floats[3]}
        elif 'kappa under' in line and line_floats:
            if len(line_floats) == 1:
                parameters['kappa'] = line_floats[0]
            else:
                parameters['kappa'] = line_floats
    return parameters

def parse_rates(lines, parameters):
    if False:
        return 10
    'Parse the rate parameters.'
    Q_mat_found = False
    trans_probs_found = False
    for line in lines:
        line_floats_res = line_floats_re.findall(line)
        line_floats = [float(val) for val in line_floats_res]
        if 'Rate parameters:' in line and line_floats:
            parameters['rate parameters'] = line_floats
        elif 'rate: ' in line and line_floats:
            parameters['rates'] = line_floats
        elif 'matrix Q' in line:
            parameters['Q matrix'] = {'matrix': []}
            if line_floats:
                parameters['Q matrix']['average Ts/Tv'] = line_floats[0]
            Q_mat_found = True
        elif Q_mat_found and line_floats:
            parameters['Q matrix']['matrix'].append(line_floats)
            if len(parameters['Q matrix']['matrix']) == 4:
                Q_mat_found = False
        elif 'alpha' in line and line_floats:
            parameters['alpha'] = line_floats[0]
        elif 'rho' in line and line_floats:
            parameters['rho'] = line_floats[0]
        elif 'transition probabilities' in line:
            parameters['transition probs.'] = []
            trans_probs_found = True
        elif trans_probs_found and line_floats:
            parameters['transition probs.'].append(line_floats)
            if len(parameters['transition probs.']) == len(parameters['rates']):
                trans_probs_found = False
    return parameters

def parse_freqs(lines, parameters):
    if False:
        print('Hello World!')
    'Parse the basepair frequencies.'
    root_re = re.compile('Note: node (\\d+) is root.')
    branch_freqs_found = False
    base_freqs_found = False
    for line in lines:
        line_floats_res = line_floats_re.findall(line)
        line_floats = [float(val) for val in line_floats_res]
        if 'Base frequencies' in line and line_floats:
            base_frequencies = {}
            base_frequencies['T'] = line_floats[0]
            base_frequencies['C'] = line_floats[1]
            base_frequencies['A'] = line_floats[2]
            base_frequencies['G'] = line_floats[3]
            parameters['base frequencies'] = base_frequencies
        elif 'base frequency parameters' in line:
            base_freqs_found = True
        elif 'Base frequencies' in line and (not line_floats):
            base_freqs_found = True
        elif base_freqs_found and line_floats:
            base_frequencies = {}
            base_frequencies['T'] = line_floats[0]
            base_frequencies['C'] = line_floats[1]
            base_frequencies['A'] = line_floats[2]
            base_frequencies['G'] = line_floats[3]
            parameters['base frequencies'] = base_frequencies
            base_freqs_found = False
        elif 'freq: ' in line and line_floats:
            parameters['rate frequencies'] = line_floats
        elif '(frequency parameters for branches)' in line:
            parameters['nodes'] = {}
            branch_freqs_found = True
        elif branch_freqs_found:
            if line_floats:
                node_res = re.match('Node \\#(\\d+)', line)
                node_num = int(node_res.group(1))
                node = {'root': False}
                node['frequency parameters'] = line_floats[:4]
                if len(line_floats) > 4:
                    node['base frequencies'] = {'T': line_floats[4], 'C': line_floats[5], 'A': line_floats[6], 'G': line_floats[7]}
                parameters['nodes'][node_num] = node
            else:
                root_res = root_re.match(line)
                if root_res is not None:
                    root_node = int(root_res.group(1))
                    parameters['nodes'][root_node]['root'] = True
                    branch_freqs_found = False
    return parameters