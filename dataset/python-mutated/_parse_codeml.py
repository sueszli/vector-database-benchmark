"""Methods for parsing codeml results files."""
import re
line_floats_re = re.compile('-*\\d+\\.\\d+')

def parse_basics(lines, results):
    if False:
        i = 10
        return i + 15
    'Parse the basic information that should be present in most codeml output files.'
    multi_models = False
    multi_genes = False
    version_re = re.compile('.+ \\(in paml version (\\d+\\.\\d+[a-z]*).*')
    model_re = re.compile('Model:\\s+(.+)')
    num_genes_re = re.compile('\\(([0-9]+) genes: separate data\\)')
    codon_freq_re = re.compile('Codon frequenc[a-z\\s]{3,7}:\\s+(.+)')
    siteclass_re = re.compile('Site-class models:\\s*([^\\s]*)')
    for line in lines:
        line_floats_res = line_floats_re.findall(line)
        line_floats = [float(val) for val in line_floats_res]
        version_res = version_re.match(line)
        if version_res is not None:
            results['version'] = version_res.group(1)
            continue
        model_res = model_re.match(line)
        if model_res is not None:
            results['model'] = model_res.group(1)
        num_genes_res = num_genes_re.search(line)
        if num_genes_res is not None:
            results['genes'] = []
            num_genes = int(num_genes_res.group(1))
            for n in range(num_genes):
                results['genes'].append({})
            multi_genes = True
            continue
        codon_freq_res = codon_freq_re.match(line)
        if codon_freq_res is not None:
            results['codon model'] = codon_freq_res.group(1)
            continue
        siteclass_res = siteclass_re.match(line)
        if siteclass_res is not None:
            siteclass_model = siteclass_res.group(1)
            if siteclass_model != '':
                results['site-class model'] = siteclass_model
                multi_models = False
            else:
                multi_models = True
        if 'ln Lmax' in line and line_floats:
            results['lnL max'] = line_floats[0]
    return (results, multi_models, multi_genes)

def parse_nssites(lines, results, multi_models, multi_genes):
    if False:
        i = 10
        return i + 15
    'Determine which NSsites models are present and parse them.'
    ns_sites = {}
    model_re = re.compile('Model (\\d+):\\s+(.+)')
    gene_re = re.compile('Gene\\s+([0-9]+)\\s+.+')
    siteclass_model = results.get('site-class model')
    if not multi_models:
        if siteclass_model is None:
            siteclass_model = 'one-ratio'
        current_model = {'one-ratio': 0, 'NearlyNeutral': 1, 'PositiveSelection': 2, 'discrete': 3, 'beta': 7, 'beta&w>1': 8, 'M2a_rel': 22}[siteclass_model]
        if multi_genes:
            genes = results['genes']
            current_gene = None
            gene_start = None
            model_results = None
            for (line_num, line) in enumerate(lines):
                gene_res = gene_re.match(line)
                if gene_res:
                    if current_gene is not None:
                        assert model_results is not None
                        parse_model(lines[gene_start:line_num], model_results)
                        genes[current_gene - 1] = model_results
                    gene_start = line_num
                    current_gene = int(gene_res.group(1))
                    model_results = {'description': siteclass_model}
            if len(genes[current_gene - 1]) == 0:
                model_results = parse_model(lines[gene_start:], model_results)
                genes[current_gene - 1] = model_results
        else:
            model_results = {'description': siteclass_model}
            model_results = parse_model(lines, model_results)
            ns_sites[current_model] = model_results
    else:
        current_model = None
        model_start = None
        for (line_num, line) in enumerate(lines):
            model_res = model_re.match(line)
            if model_res:
                if current_model is not None:
                    parse_model(lines[model_start:line_num], model_results)
                    ns_sites[current_model] = model_results
                model_start = line_num
                current_model = int(model_res.group(1))
                model_results = {'description': model_res.group(2)}
        if ns_sites.get(current_model) is None:
            model_results = parse_model(lines[model_start:], model_results)
            ns_sites[current_model] = model_results
    if len(ns_sites) == 1:
        m0 = ns_sites.get(0)
        if not m0 or len(m0) > 1:
            results['NSsites'] = ns_sites
    elif len(ns_sites) > 1:
        results['NSsites'] = ns_sites
    return results

def parse_model(lines, results):
    if False:
        print('Hello World!')
    "Parse an individual NSsites model's results."
    parameters = {}
    SEs_flag = False
    dS_tree_flag = False
    dN_tree_flag = False
    w_tree_flag = False
    num_params = None
    tree_re = re.compile("^\\([\\w #:',.()]*\\);\\s*$")
    branch_re = re.compile('\\s+(\\d+\\.\\.\\d+)[\\s+\\d+\\.\\d+]+')
    model_params_re = re.compile('(?<!\\S)([a-z]\\d?)\\s*=\\s+(\\d+\\.\\d+)')
    for line in lines:
        line_floats_res = line_floats_re.findall(line)
        line_floats = [float(val) for val in line_floats_res]
        branch_res = branch_re.match(line)
        model_params = model_params_re.findall(line)
        if 'lnL(ntime:' in line and line_floats:
            results['lnL'] = line_floats[0]
            np_res = re.match('lnL\\(ntime:\\s+\\d+\\s+np:\\s+(\\d+)\\)', line)
            if np_res is not None:
                num_params = int(np_res.group(1))
        elif len(line_floats) == num_params and (not SEs_flag):
            parameters['parameter list'] = line.strip()
        elif 'SEs for parameters:' in line:
            SEs_flag = True
        elif SEs_flag and len(line_floats) == num_params:
            parameters['SEs'] = line.strip()
            SEs_flag = False
        elif 'tree length =' in line and line_floats:
            results['tree length'] = line_floats[0]
        elif tree_re.match(line) is not None:
            if ':' in line or '#' in line:
                if dS_tree_flag:
                    results['dS tree'] = line.strip()
                    dS_tree_flag = False
                elif dN_tree_flag:
                    results['dN tree'] = line.strip()
                    dN_tree_flag = False
                elif w_tree_flag:
                    results['omega tree'] = line.strip()
                    w_tree_flag = False
                else:
                    results['tree'] = line.strip()
        elif 'dS tree:' in line:
            dS_tree_flag = True
        elif 'dN tree:' in line:
            dN_tree_flag = True
        elif 'w ratios as labels for TreeView:' in line:
            w_tree_flag = True
        elif 'rates for' in line and line_floats:
            line_floats.insert(0, 1.0)
            parameters['rates'] = line_floats
        elif 'kappa (ts/tv)' in line and line_floats:
            parameters['kappa'] = line_floats[0]
        elif 'omega (dN/dS)' in line and line_floats:
            parameters['omega'] = line_floats[0]
        elif 'w (dN/dS)' in line and line_floats:
            parameters['omega'] = line_floats
        elif 'gene # ' in line:
            gene_num = int(re.match('gene # (\\d+)', line).group(1))
            if parameters.get('genes') is None:
                parameters['genes'] = {}
            parameters['genes'][gene_num] = {'kappa': line_floats[0], 'omega': line_floats[1]}
        elif 'tree length for dN' in line and line_floats:
            parameters['dN'] = line_floats[0]
        elif 'tree length for dS' in line and line_floats:
            parameters['dS'] = line_floats[0]
        elif line[0:2] == 'p:' or line[0:10] == 'proportion':
            site_classes = parse_siteclass_proportions(line_floats)
            parameters['site classes'] = site_classes
        elif line[0:2] == 'w:':
            site_classes = parameters.get('site classes')
            site_classes = parse_siteclass_omegas(line, site_classes)
            parameters['site classes'] = site_classes
        elif 'branch type ' in line:
            branch_type = re.match('branch type (\\d)', line)
            if branch_type:
                site_classes = parameters.get('site classes')
                branch_type_no = int(branch_type.group(1))
                site_classes = parse_clademodelc(branch_type_no, line_floats, site_classes)
                parameters['site classes'] = site_classes
        elif line[0:12] == 'foreground w':
            site_classes = parameters.get('site classes')
            site_classes = parse_branch_site_a(True, line_floats, site_classes)
            parameters['site classes'] = site_classes
        elif line[0:12] == 'background w':
            site_classes = parameters.get('site classes')
            site_classes = parse_branch_site_a(False, line_floats, site_classes)
            parameters['site classes'] = site_classes
        elif branch_res is not None and line_floats:
            branch = branch_res.group(1)
            if parameters.get('branches') is None:
                parameters['branches'] = {}
            params = line.strip().split()[1:]
            parameters['branches'][branch] = {'t': float(params[0].strip()), 'N': float(params[1].strip()), 'S': float(params[2].strip()), 'omega': float(params[3].strip()), 'dN': float(params[4].strip()), 'dS': float(params[5].strip()), 'N*dN': float(params[6].strip()), 'S*dS': float(params[7].strip())}
        elif model_params:
            float_model_params = []
            for param in model_params:
                float_model_params.append((param[0], float(param[1])))
            parameters.update(dict(float_model_params))
    if parameters:
        results['parameters'] = parameters
    return results

def parse_siteclass_proportions(line_floats):
    if False:
        for i in range(10):
            print('nop')
    'Find proportion of alignment assigned to each class.\n\n    For models which have multiple site classes, find the proportion of the\n    alignment assigned to each class.\n    '
    site_classes = {}
    if line_floats:
        for n in range(len(line_floats)):
            site_classes[n] = {'proportion': line_floats[n]}
    return site_classes

def parse_siteclass_omegas(line, site_classes):
    if False:
        i = 10
        return i + 15
    'Find omega estimate for each class.\n\n    For models which have multiple site classes, find the omega estimated\n    for each class.\n    '
    line_floats = re.findall('\\d{1,3}\\.\\d{5}', line)
    if not site_classes or len(line_floats) == 0:
        return
    for n in range(len(line_floats)):
        site_classes[n]['omega'] = line_floats[n]
    return site_classes

def parse_clademodelc(branch_type_no, line_floats, site_classes):
    if False:
        i = 10
        return i + 15
    'Parse results specific to the clade model C.'
    if not site_classes or len(line_floats) == 0:
        return
    for n in range(len(line_floats)):
        if site_classes[n].get('branch types') is None:
            site_classes[n]['branch types'] = {}
        site_classes[n]['branch types'][branch_type_no] = line_floats[n]
    return site_classes

def parse_branch_site_a(foreground, line_floats, site_classes):
    if False:
        print('Hello World!')
    'Parse results specific to the branch site A model.'
    if not site_classes or len(line_floats) == 0:
        return
    for n in range(len(line_floats)):
        if site_classes[n].get('branch types') is None:
            site_classes[n]['branch types'] = {}
        if foreground:
            site_classes[n]['branch types']['foreground'] = line_floats[n]
        else:
            site_classes[n]['branch types']['background'] = line_floats[n]
    return site_classes

def parse_pairwise(lines, results):
    if False:
        print('Hello World!')
    'Parse results from pairwise comparisons.'
    pair_re = re.compile('\\d+ \\((.+)\\) ... \\d+ \\((.+)\\)')
    pairwise = {}
    seq1 = None
    seq2 = None
    for line in lines:
        line_floats_res = line_floats_re.findall(line)
        line_floats = [float(val) for val in line_floats_res]
        pair_res = pair_re.match(line)
        if pair_res:
            seq1 = pair_res.group(1)
            seq2 = pair_res.group(2)
            if seq1 not in pairwise:
                pairwise[seq1] = {}
            if seq2 not in pairwise:
                pairwise[seq2] = {}
        if len(line_floats) == 1 and seq1 is not None and (seq2 is not None):
            pairwise[seq1][seq2] = {'lnL': line_floats[0]}
            pairwise[seq2][seq1] = pairwise[seq1][seq2]
        elif len(line_floats) == 6 and seq1 is not None and (seq2 is not None):
            pairwise[seq1][seq2].update({'t': line_floats[0], 'S': line_floats[1], 'N': line_floats[2], 'omega': line_floats[3], 'dN': line_floats[4], 'dS': line_floats[5]})
            pairwise[seq2][seq1] = pairwise[seq1][seq2]
    if pairwise:
        results['pairwise'] = pairwise
    return results

def parse_distances(lines, results):
    if False:
        return 10
    'Parse amino acid sequence distance results.'
    distances = {}
    sequences = []
    raw_aa_distances_flag = False
    ml_aa_distances_flag = False
    matrix_row_re = re.compile('(.+)\\s{5,15}')
    for line in lines:
        line_floats_res = line_floats_re.findall(line)
        line_floats = [float(val) for val in line_floats_res]
        if 'AA distances' in line:
            raw_aa_distances_flag = True
            ml_aa_distances_flag = False
        elif 'ML distances of aa seqs.' in line:
            ml_aa_distances_flag = True
            raw_aa_distances_flag = False
        matrix_row_res = matrix_row_re.match(line)
        if matrix_row_res and (raw_aa_distances_flag or ml_aa_distances_flag):
            seq_name = matrix_row_res.group(1).strip()
            if seq_name not in sequences:
                sequences.append(seq_name)
            if raw_aa_distances_flag:
                if distances.get('raw') is None:
                    distances['raw'] = {}
                distances['raw'][seq_name] = {}
                for i in range(len(line_floats)):
                    distances['raw'][seq_name][sequences[i]] = line_floats[i]
                    distances['raw'][sequences[i]][seq_name] = line_floats[i]
            else:
                if distances.get('ml') is None:
                    distances['ml'] = {}
                distances['ml'][seq_name] = {}
                for i in range(len(line_floats)):
                    distances['ml'][seq_name][sequences[i]] = line_floats[i]
                    distances['ml'][sequences[i]][seq_name] = line_floats[i]
    if distances:
        results['distances'] = distances
    return results