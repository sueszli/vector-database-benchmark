"""Module to control GenePop."""
import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument

def _gp_float(tok):
    if False:
        print('Hello World!')
    'Get a float from a token, if it fails, returns the string (PRIVATE).'
    try:
        return float(tok)
    except ValueError:
        return str(tok)

def _gp_int(tok):
    if False:
        for i in range(10):
            print('nop')
    'Get a int from a token, if it fails, returns the string (PRIVATE).'
    try:
        return int(tok)
    except ValueError:
        return str(tok)

def _read_allele_freq_table(f):
    if False:
        while True:
            i = 10
    line = f.readline()
    while ' --' not in line:
        if line == '':
            raise StopIteration
        if 'No data' in line:
            return (None, None)
        line = f.readline()
    alleles = [x for x in f.readline().rstrip().split(' ') if x != '']
    alleles = [_gp_int(x) for x in alleles]
    line = f.readline().rstrip()
    table = []
    while line != '':
        parts = [x for x in line.split(' ') if x != '']
        try:
            table.append((parts[0], [_gp_float(x) for x in parts[1:-1]], _gp_int(parts[-1])))
        except ValueError:
            table.append((parts[0], [None] * len(alleles), 0))
        line = f.readline().rstrip()
    return (alleles, table)

def _read_table(f, funs):
    if False:
        i = 10
        return i + 15
    table = []
    line = f.readline().rstrip()
    while '---' not in line:
        line = f.readline().rstrip()
    line = f.readline().rstrip()
    while '===' not in line and '---' not in line and (line != ''):
        toks = [x for x in line.split(' ') if x != '']
        parts = []
        for (i, tok) in enumerate(toks):
            try:
                parts.append(funs[i](tok))
            except ValueError:
                parts.append(tok)
        table.append(tuple(parts))
        line = f.readline().rstrip()
    return table

def _read_triangle_matrix(f):
    if False:
        for i in range(10):
            print('nop')
    matrix = []
    line = f.readline().rstrip()
    while line != '':
        matrix.append([_gp_float(x) for x in [y for y in line.split(' ') if y != '']])
        line = f.readline().rstrip()
    return matrix

def _read_headed_triangle_matrix(f):
    if False:
        print('Hello World!')
    matrix = {}
    header = f.readline().rstrip()
    if '---' in header or '===' in header:
        header = f.readline().rstrip()
    nlines = len([x for x in header.split(' ') if x != '']) - 1
    for line_pop in range(nlines):
        line = f.readline().rstrip()
        vals = [x for x in line.split(' ')[1:] if x != '']
        clean_vals = []
        for val in vals:
            try:
                clean_vals.append(_gp_float(val))
            except ValueError:
                clean_vals.append(None)
        for (col_pop, clean_val) in enumerate(clean_vals):
            matrix[line_pop + 1, col_pop] = clean_val
    return matrix

def _hw_func(stream, is_locus, has_fisher=False):
    if False:
        for i in range(10):
            print('nop')
    line = stream.readline()
    if is_locus:
        hook = 'Locus '
    else:
        hook = 'Pop : '
    while line != '':
        if line.lstrip().startswith(hook):
            stream.readline()
            stream.readline()
            stream.readline()
            table = _read_table(stream, [str, _gp_float, _gp_float, _gp_float, _gp_float, _gp_int, str])
            loci = {}
            for entry in table:
                if len(entry) < 4:
                    loci[entry[0]] = None
                else:
                    (locus, p, se, fis_wc, fis_rh, steps) = entry[:-1]
                    if se == '-':
                        se = None
                    loci[locus] = (p, se, fis_wc, fis_rh, steps)
            return loci
        line = stream.readline()
    raise StopIteration

class _FileIterator:
    """Return an iterator which crawls over a stream of lines with a function (PRIVATE).

    The generator function is expected to yield a tuple, while
    consuming input
    """

    def __init__(self, func, fname, handle=None):
        if False:
            return 10
        self.func = func
        if handle is None:
            self.stream = open(fname)
        else:
            self.stream = handle
        self.fname = fname
        self.done = False

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        if self.done:
            self.done = True
            raise StopIteration
        return self

    def __next__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.func(self)

    def __del__(self):
        if False:
            while True:
                i = 10
        self.stream.close()
        os.remove(self.fname)

class _GenePopCommandline(AbstractCommandline):
    """Return a Command Line Wrapper for GenePop (PRIVATE)."""

    def __init__(self, genepop_dir=None, cmd='Genepop', **kwargs):
        if False:
            print('Hello World!')
        self.parameters = [_Argument(['command'], 'GenePop option to be called', is_required=True), _Argument(['mode'], 'Should always be batch', is_required=True), _Argument(['input'], 'Input file', is_required=True), _Argument(['Dememorization'], 'Dememorization step'), _Argument(['BatchNumber'], 'Number of MCMC batches'), _Argument(['BatchLength'], 'Length of MCMC chains'), _Argument(['HWtests'], 'Enumeration or MCMC'), _Argument(['IsolBDstatistic'], 'IBD statistic (a or e)'), _Argument(['MinimalDistance'], 'Minimal IBD distance'), _Argument(['GeographicScale'], 'Log or Linear')]
        AbstractCommandline.__init__(self, cmd, **kwargs)
        self.set_parameter('mode', 'Mode=Batch')

    def set_menu(self, option_list):
        if False:
            i = 10
            return i + 15
        'Set the menu option.\n\n        Example set_menu([6,1]) = get all F statistics (menu 6.1)\n        '
        self.set_parameter('command', 'MenuOptions=' + '.'.join((str(x) for x in option_list)))

    def set_input(self, fname):
        if False:
            print('Hello World!')
        'Set the input file name.'
        self.set_parameter('input', 'InputFile=' + fname)

class GenePopController:
    """Define a class to interface with the GenePop program."""

    def __init__(self, genepop_dir=None):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the controller.\n\n        genepop_dir is the directory where GenePop is.\n\n        The binary should be called Genepop (capital G)\n        '
        self.controller = _GenePopCommandline(genepop_dir)

    def _get_opts(self, dememorization, batches, iterations, enum_test=None):
        if False:
            return 10
        opts = {}
        opts['Dememorization'] = dememorization
        opts['BatchNumber'] = batches
        opts['BatchLength'] = iterations
        if enum_test is not None:
            if enum_test is True:
                opts['HWtests'] = 'Enumeration'
            else:
                opts['HWtests'] = 'MCMC'
        return opts

    def _run_genepop(self, extensions, option, fname, opts=None):
        if False:
            i = 10
            return i + 15
        if opts is None:
            opts = {}
        cwd = os.getcwd()
        temp_dir = tempfile.mkdtemp()
        os.chdir(temp_dir)
        self.controller.set_menu(option)
        if os.path.isabs(fname):
            self.controller.set_input(fname)
        else:
            self.controller.set_input(cwd + os.sep + fname)
        for opt in opts:
            self.controller.set_parameter(opt, opt + '=' + str(opts[opt]))
        self.controller()
        os.chdir(cwd)
        shutil.rmtree(temp_dir)

    def _test_pop_hz_both(self, fname, type, ext, enum_test=True, dememorization=10000, batches=20, iterations=5000):
        if False:
            for i in range(10):
                print('nop')
        'Use Hardy-Weinberg test for heterozygote deficiency/excess (PRIVATE).\n\n        Returns a population iterator containing a dictionary where\n        dictionary[locus]=(P-val, SE, Fis-WC, Fis-RH, steps).\n\n        Some loci have a None if the info is not available.\n        SE might be none (for enumerations).\n        '
        opts = self._get_opts(dememorization, batches, iterations, enum_test)
        self._run_genepop([ext], [1, type], fname, opts)

        def hw_func(self):
            if False:
                print('Hello World!')
            return _hw_func(self.stream, False)
        return _FileIterator(hw_func, fname + ext)

    def _test_global_hz_both(self, fname, type, ext, enum_test=True, dememorization=10000, batches=20, iterations=5000):
        if False:
            print('Hello World!')
        'Use Global Hardy-Weinberg test for heterozygote deficiency/excess (PRIVATE).\n\n        Returns a triple with:\n         - A list per population containing (pop_name, P-val, SE, switches).\n           Some pops have a None if the info is not available.\n           SE might be none (for enumerations).\n         - A list per loci containing (locus_name, P-val, SE, switches).\n           Some loci have a None if the info is not available.\n           SE might be none (for enumerations).\n         - Overall results (P-val, SE, switches).\n\n        '
        opts = self._get_opts(dememorization, batches, iterations, enum_test)
        self._run_genepop([ext], [1, type], fname, opts)

        def hw_pop_func(self):
            if False:
                while True:
                    i = 10
            return _read_table(self.stream, [str, _gp_float, _gp_float, _gp_float])
        with open(fname + ext) as f1:
            line = f1.readline()
            while 'by population' not in line:
                line = f1.readline()
            pop_p = _read_table(f1, [str, _gp_float, _gp_float, _gp_float])
        with open(fname + ext) as f2:
            line = f2.readline()
            while 'by locus' not in line:
                line = f2.readline()
            loc_p = _read_table(f2, [str, _gp_float, _gp_float, _gp_float])
        with open(fname + ext) as f:
            line = f.readline()
            while 'all locus' not in line:
                line = f.readline()
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            line = f.readline().rstrip()
            (p, se, switches) = tuple((_gp_float(x) for x in [y for y in line.split(' ') if y != '']))
        return (pop_p, loc_p, (p, se, switches))

    def test_pop_hz_deficiency(self, fname, enum_test=True, dememorization=10000, batches=20, iterations=5000):
        if False:
            print('Hello World!')
        'Use Hardy-Weinberg test for heterozygote deficiency.\n\n        Returns a population iterator containing a dictionary wehre\n        dictionary[locus]=(P-val, SE, Fis-WC, Fis-RH, steps).\n\n        Some loci have a None if the info is not available.\n        SE might be none (for enumerations).\n        '
        return self._test_pop_hz_both(fname, 1, '.D', enum_test, dememorization, batches, iterations)

    def test_pop_hz_excess(self, fname, enum_test=True, dememorization=10000, batches=20, iterations=5000):
        if False:
            return 10
        'Use Hardy-Weinberg test for heterozygote deficiency.\n\n        Returns a population iterator containing a dictionary where\n        dictionary[locus]=(P-val, SE, Fis-WC, Fis-RH, steps).\n\n        Some loci have a None if the info is not available.\n        SE might be none (for enumerations).\n        '
        return self._test_pop_hz_both(fname, 2, '.E', enum_test, dememorization, batches, iterations)

    def test_pop_hz_prob(self, fname, ext, enum_test=False, dememorization=10000, batches=20, iterations=5000):
        if False:
            while True:
                i = 10
        "Use Hardy-Weinberg test based on probability.\n\n        Returns 2 iterators and a final tuple:\n\n         1. Returns a loci iterator containing:\n             - A dictionary[pop_pos]=(P-val, SE, Fis-WC, Fis-RH, steps).\n               Some pops have a None if the info is not available.\n               SE might be none (for enumerations).\n             - Result of Fisher's test (Chi2, deg freedom, prob).\n         2. Returns a population iterator containing:\n             - A dictionary[locus]=(P-val, SE, Fis-WC, Fis-RH, steps).\n               Some loci have a None if the info is not available.\n               SE might be none (for enumerations).\n             - Result of Fisher's test (Chi2, deg freedom, prob).\n         3. Final tuple (Chi2, deg freedom, prob).\n\n        "
        opts = self._get_opts(dememorization, batches, iterations, enum_test)
        self._run_genepop([ext], [1, 3], fname, opts)

        def hw_prob_loci_func(self):
            if False:
                i = 10
                return i + 15
            return _hw_func(self.stream, True, True)

        def hw_prob_pop_func(self):
            if False:
                for i in range(10):
                    print('nop')
            return _hw_func(self.stream, False, True)
        shutil.copyfile(fname + '.P', fname + '.P2')
        return (_FileIterator(hw_prob_loci_func, fname + '.P'), _FileIterator(hw_prob_pop_func, fname + '.P2'))

    def test_global_hz_deficiency(self, fname, enum_test=True, dememorization=10000, batches=20, iterations=5000):
        if False:
            for i in range(10):
                print('nop')
        'Use Global Hardy-Weinberg test for heterozygote deficiency.\n\n        Returns a triple with:\n         - An list per population containing (pop_name, P-val, SE, switches).\n           Some pops have a None if the info is not available.\n           SE might be none (for enumerations).\n         - An list per loci containing (locus_name, P-val, SE, switches).\n           Some loci have a None if the info is not available.\n           SE might be none (for enumerations).\n         - Overall results (P-val, SE, switches).\n\n        '
        return self._test_global_hz_both(fname, 4, '.DG', enum_test, dememorization, batches, iterations)

    def test_global_hz_excess(self, fname, enum_test=True, dememorization=10000, batches=20, iterations=5000):
        if False:
            i = 10
            return i + 15
        'Use Global Hardy-Weinberg test for heterozygote excess.\n\n        Returns a triple with:\n         - A list per population containing (pop_name, P-val, SE, switches).\n           Some pops have a None if the info is not available.\n           SE might be none (for enumerations).\n         - A list per loci containing (locus_name, P-val, SE, switches).\n           Some loci have a None if the info is not available.\n           SE might be none (for enumerations).\n         - Overall results (P-val, SE, switches)\n\n        '
        return self._test_global_hz_both(fname, 5, '.EG', enum_test, dememorization, batches, iterations)

    def test_ld(self, fname, dememorization=10000, batches=20, iterations=5000):
        if False:
            i = 10
            return i + 15
        'Test for linkage disequilibrium on each pair of loci in each population.'
        opts = self._get_opts(dememorization, batches, iterations)
        self._run_genepop(['.DIS'], [2, 1], fname, opts)

        def ld_pop_func(self):
            if False:
                print('Hello World!')
            current_pop = None
            line = self.stream.readline().rstrip()
            if line == '':
                self.done = True
                raise StopIteration
            toks = [x for x in line.split(' ') if x != '']
            (pop, locus1, locus2) = (toks[0], toks[1], toks[2])
            if not hasattr(self, 'start_locus1'):
                (start_locus1, start_locus2) = (locus1, locus2)
                current_pop = -1
            if locus1 == start_locus1 and locus2 == start_locus2:
                current_pop += 1
            if toks[3] == 'No':
                return (current_pop, pop, (locus1, locus2), None)
            (p, se, switches) = (_gp_float(toks[3]), _gp_float(toks[4]), _gp_int(toks[5]))
            return (current_pop, pop, (locus1, locus2), (p, se, switches))

        def ld_func(self):
            if False:
                for i in range(10):
                    print('nop')
            line = self.stream.readline().rstrip()
            if line == '':
                self.done = True
                raise StopIteration
            toks = [x for x in line.split(' ') if x != '']
            (locus1, locus2) = (toks[0], toks[2])
            try:
                (chi2, df, p) = (_gp_float(toks[3]), _gp_int(toks[4]), _gp_float(toks[5]))
            except ValueError:
                return ((locus1, locus2), None)
            return ((locus1, locus2), (chi2, df, p))
        f1 = open(fname + '.DIS')
        line = f1.readline()
        while '----' not in line:
            line = f1.readline()
        shutil.copyfile(fname + '.DIS', fname + '.DI2')
        f2 = open(fname + '.DI2')
        line = f2.readline()
        while 'Locus pair' not in line:
            line = f2.readline()
        while '----' not in line:
            line = f2.readline()
        return (_FileIterator(ld_pop_func, fname + '.DIS', f1), _FileIterator(ld_func, fname + '.DI2', f2))

    def create_contingency_tables(self, fname):
        if False:
            for i in range(10):
                print('nop')
        'Provision for creating Genotypic contingency tables.'
        raise NotImplementedError

    def test_genic_diff_all(self, fname, dememorization=10000, batches=20, iterations=5000):
        if False:
            for i in range(10):
                print('nop')
        'Provision for Genic differentiation for all populations.'
        raise NotImplementedError

    def test_genic_diff_pair(self, fname, dememorization=10000, batches=20, iterations=5000):
        if False:
            for i in range(10):
                print('nop')
        'Provision for Genic differentiation for all population pairs.'
        raise NotImplementedError

    def test_genotypic_diff_all(self, fname, dememorization=10000, batches=20, iterations=5000):
        if False:
            for i in range(10):
                print('nop')
        'Provision for Genotypic differentiation for all populations.'
        raise NotImplementedError

    def test_genotypic_diff_pair(self, fname, dememorization=10000, batches=20, iterations=5000):
        if False:
            while True:
                i = 10
        'Provision for Genotypic differentiation for all population pairs.'
        raise NotImplementedError

    def estimate_nm(self, fname):
        if False:
            print('Hello World!')
        'Estimate the Number of Migrants.\n\n        Parameters:\n         - fname - file name\n\n        Returns\n         - Mean sample size\n         - Mean frequency of private alleles\n         - Number of migrants for Ne=10\n         - Number of migrants for Ne=25\n         - Number of migrants for Ne=50\n         - Number of migrants after correcting for expected size\n\n        '
        self._run_genepop(['PRI'], [4], fname)
        with open(fname + '.PRI') as f:
            lines = f.readlines()
        for line in lines:
            m = re.search('Mean sample size: ([.0-9]+)', line)
            if m is not None:
                mean_sample_size = _gp_float(m.group(1))
            m = re.search('Mean frequency of private alleles p\\(1\\)= ([.0-9]+)', line)
            if m is not None:
                mean_priv_alleles = _gp_float(m.group(1))
            m = re.search('N=10: ([.0-9]+)', line)
            if m is not None:
                mig10 = _gp_float(m.group(1))
            m = re.search('N=25: ([.0-9]+)', line)
            if m is not None:
                mig25 = _gp_float(m.group(1))
            m = re.search('N=50: ([.0-9]+)', line)
            if m is not None:
                mig50 = _gp_float(m.group(1))
            m = re.search('for size= ([.0-9]+)', line)
            if m is not None:
                mig_corrected = _gp_float(m.group(1))
        os.remove(fname + '.PRI')
        return (mean_sample_size, mean_priv_alleles, mig10, mig25, mig50, mig_corrected)

    def calc_allele_genotype_freqs(self, fname):
        if False:
            while True:
                i = 10
        'Calculate allele and genotype frequencies per locus and per sample.\n\n        Parameters:\n         - fname - file name\n\n        Returns tuple with 2 elements:\n         - Population iterator with\n\n           - population name\n           - Locus dictionary with key = locus name and content tuple as\n             Genotype List with\n             (Allele1, Allele2, observed, expected)\n             (expected homozygotes, observed hm,\n             expected heterozygotes, observed ht)\n             Allele frequency/Fis dictionary with allele as key and\n             (count, frequency, Fis Weir & Cockerham)\n           - Totals as a pair\n           - count\n           - Fis Weir & Cockerham,\n           - Fis Robertson & Hill\n\n         - Locus iterator with\n\n           - Locus name\n           - allele list\n           - Population list with a triple\n\n             - population name\n             - list of allele frequencies in the same order as allele list above\n             - number of genes\n\n        Will create a file called fname.INF\n\n        '
        self._run_genepop(['INF'], [5, 1], fname)

        def pop_parser(self):
            if False:
                while True:
                    i = 10
            if hasattr(self, 'old_line'):
                line = self.old_line
                del self.old_line
            else:
                line = self.stream.readline()
            loci_content = {}
            while line != '':
                line = line.rstrip()
                if 'Tables of allelic frequencies for each locus' in line:
                    return (self.curr_pop, loci_content)
                match = re.match('.*Pop: (.+) Locus: (.+)', line)
                if match is not None:
                    pop = match.group(1).rstrip()
                    locus = match.group(2)
                    if not hasattr(self, 'first_locus'):
                        self.first_locus = locus
                    if hasattr(self, 'curr_pop'):
                        if self.first_locus == locus:
                            old_pop = self.curr_pop
                            self.old_line = line
                            del self.first_locus
                            del self.curr_pop
                            return (old_pop, loci_content)
                    self.curr_pop = pop
                else:
                    line = self.stream.readline()
                    continue
                geno_list = []
                line = self.stream.readline()
                if 'No data' in line:
                    continue
                while 'Genotypes  Obs.' not in line:
                    line = self.stream.readline()
                while line != '\n':
                    m2 = re.match(' +([0-9]+) , ([0-9]+) *([0-9]+) *(.+)', line)
                    if m2 is not None:
                        geno_list.append((_gp_int(m2.group(1)), _gp_int(m2.group(2)), _gp_int(m2.group(3)), _gp_float(m2.group(4))))
                    else:
                        line = self.stream.readline()
                        continue
                    line = self.stream.readline()
                while 'Expected number of ho' not in line:
                    line = self.stream.readline()
                expHo = _gp_float(line[38:])
                line = self.stream.readline()
                obsHo = _gp_int(line[38:])
                line = self.stream.readline()
                expHe = _gp_float(line[38:])
                line = self.stream.readline()
                obsHe = _gp_int(line[38:])
                line = self.stream.readline()
                while 'Sample count' not in line:
                    line = self.stream.readline()
                line = self.stream.readline()
                freq_fis = {}
                overall_fis = None
                while '----' not in line:
                    vals = [x for x in line.rstrip().split(' ') if x != '']
                    if vals[0] == 'Tot':
                        overall_fis = (_gp_int(vals[1]), _gp_float(vals[2]), _gp_float(vals[3]))
                    else:
                        freq_fis[_gp_int(vals[0])] = (_gp_int(vals[1]), _gp_float(vals[2]), _gp_float(vals[3]))
                    line = self.stream.readline()
                loci_content[locus] = (geno_list, (expHo, obsHo, expHe, obsHe), freq_fis, overall_fis)
            self.done = True
            raise StopIteration

        def locus_parser(self):
            if False:
                i = 10
                return i + 15
            line = self.stream.readline()
            while line != '':
                line = line.rstrip()
                match = re.match(' Locus: (.+)', line)
                if match is not None:
                    locus = match.group(1)
                    (alleles, table) = _read_allele_freq_table(self.stream)
                    return (locus, alleles, table)
                line = self.stream.readline()
            self.done = True
            raise StopIteration
        shutil.copyfile(fname + '.INF', fname + '.IN2')
        pop_iter = _FileIterator(pop_parser, fname + '.INF')
        locus_iter = _FileIterator(locus_parser, fname + '.IN2')
        return (pop_iter, locus_iter)

    def _calc_diversities_fis(self, fname, ext):
        if False:
            return 10
        self._run_genepop([ext], [5, 2], fname)
        with open(fname + ext) as f:
            line = f.readline()
            while line != '':
                line = line.rstrip()
                if line.startswith('Statistics per sample over all loci with at least two individuals typed'):
                    avg_fis = _read_table(f, [str, _gp_float, _gp_float, _gp_float])
                    avg_Qintra = _read_table(f, [str, _gp_float])
                line = f.readline()

        def fis_func(self):
            if False:
                print('Hello World!')
            line = self.stream.readline()
            while line != '':
                line = line.rstrip()
                m = re.search('Locus: (.+)', line)
                if m is not None:
                    locus = m.group(1)
                    self.stream.readline()
                    if 'No complete' in self.stream.readline():
                        return (locus, None)
                    self.stream.readline()
                    fis_table = _read_table(self.stream, [str, _gp_float, _gp_float, _gp_float])
                    self.stream.readline()
                    (avg_qinter, avg_fis) = tuple((_gp_float(x) for x in [y for y in self.stream.readline().split(' ') if y != '']))
                    return (locus, fis_table, avg_qinter, avg_fis)
                line = self.stream.readline()
            self.done = True
            raise StopIteration
        return (_FileIterator(fis_func, fname + ext), avg_fis, avg_Qintra)

    def calc_diversities_fis_with_identity(self, fname):
        if False:
            for i in range(10):
                print('nop')
        'Compute identity-base Gene diversities and Fis.'
        return self._calc_diversities_fis(fname, '.DIV')

    def calc_diversities_fis_with_size(self, fname):
        if False:
            while True:
                i = 10
        'Provision to Computer Allele size-based Gene diversities and Fis.'
        raise NotImplementedError

    def calc_fst_all(self, fname):
        if False:
            while True:
                i = 10
        'Execute GenePop and gets Fst/Fis/Fit (all populations).\n\n        Parameters:\n         - fname - file name\n\n        Returns:\n         - (multiLocusFis, multiLocusFst, multiLocus Fit),\n         - Iterator of tuples\n           (Locus name, Fis, Fst, Fit, Qintra, Qinter)\n\n        Will create a file called ``fname.FST``.\n\n        This does not return the genotype frequencies.\n\n        '
        self._run_genepop(['.FST'], [6, 1], fname)
        with open(fname + '.FST') as f:
            line = f.readline()
            while line != '':
                if line.startswith('           All:'):
                    toks = [x for x in line.rstrip().split(' ') if x != '']
                    try:
                        allFis = _gp_float(toks[1])
                    except ValueError:
                        allFis = None
                    try:
                        allFst = _gp_float(toks[2])
                    except ValueError:
                        allFst = None
                    try:
                        allFit = _gp_float(toks[3])
                    except ValueError:
                        allFit = None
                line = f.readline()

        def proc(self):
            if False:
                return 10
            if hasattr(self, 'last_line'):
                line = self.last_line
                del self.last_line
            else:
                line = self.stream.readline()
            locus = None
            fis = None
            fst = None
            fit = None
            qintra = None
            qinter = None
            while line != '':
                line = line.rstrip()
                if line.startswith('  Locus:'):
                    if locus is not None:
                        self.last_line = line
                        return (locus, fis, fst, fit, qintra, qinter)
                    else:
                        locus = line.split(':')[1].lstrip()
                elif line.startswith('Fis^='):
                    fis = _gp_float(line.split(' ')[1])
                elif line.startswith('Fst^='):
                    fst = _gp_float(line.split(' ')[1])
                elif line.startswith('Fit^='):
                    fit = _gp_float(line.split(' ')[1])
                elif line.startswith('1-Qintra^='):
                    qintra = _gp_float(line.split(' ')[1])
                elif line.startswith('1-Qinter^='):
                    qinter = _gp_float(line.split(' ')[1])
                    return (locus, fis, fst, fit, qintra, qinter)
                line = self.stream.readline()
            if locus is not None:
                return (locus, fis, fst, fit, qintra, qinter)
            self.stream.close()
            self.done = True
            raise StopIteration
        return ((allFis, allFst, allFit), _FileIterator(proc, fname + '.FST'))

    def calc_fst_pair(self, fname):
        if False:
            print('Hello World!')
        'Estimate spatial structure from Allele identity for all population pairs.'
        self._run_genepop(['.ST2', '.MIG'], [6, 2], fname)
        with open(fname + '.ST2') as f:
            line = f.readline()
            while line != '':
                line = line.rstrip()
                if line.startswith('Estimates for all loci'):
                    avg_fst = _read_headed_triangle_matrix(f)
                line = f.readline()

        def loci_func(self):
            if False:
                i = 10
                return i + 15
            line = self.stream.readline()
            while line != '':
                line = line.rstrip()
                m = re.search(' Locus: (.+)', line)
                if m is not None:
                    locus = m.group(1)
                    matrix = _read_headed_triangle_matrix(self.stream)
                    return (locus, matrix)
                line = self.stream.readline()
            self.done = True
            raise StopIteration
        os.remove(fname + '.MIG')
        return (_FileIterator(loci_func, fname + '.ST2'), avg_fst)

    def calc_rho_all(self, fname):
        if False:
            while True:
                i = 10
        'Provision for estimating spatial structure from Allele size for all populations.'
        raise NotImplementedError

    def calc_rho_pair(self, fname):
        if False:
            for i in range(10):
                print('nop')
        'Provision for estimating spatial structure from Allele size for all population pairs.'
        raise NotImplementedError

    def _calc_ibd(self, fname, sub, stat='a', scale='Log', min_dist=1e-05):
        if False:
            print('Hello World!')
        'Calculate isolation by distance statistics (PRIVATE).'
        self._run_genepop(['.GRA', '.MIG', '.ISO'], [6, sub], fname, opts={'MinimalDistance': min_dist, 'GeographicScale': scale, 'IsolBDstatistic': stat})
        with open(fname + '.ISO') as f:
            f.readline()
            f.readline()
            f.readline()
            f.readline()
            estimate = _read_triangle_matrix(f)
            f.readline()
            f.readline()
            distance = _read_triangle_matrix(f)
            f.readline()
            match = re.match('a = (.+), b = (.+)', f.readline().rstrip())
            a = _gp_float(match.group(1))
            b = _gp_float(match.group(2))
            f.readline()
            f.readline()
            match = re.match(' b=(.+)', f.readline().rstrip())
            bb = _gp_float(match.group(1))
            match = re.match('.*\\[(.+)  ;  (.+)\\]', f.readline().rstrip())
            bblow = _gp_float(match.group(1))
            bbhigh = _gp_float(match.group(2))
        os.remove(fname + '.MIG')
        os.remove(fname + '.GRA')
        os.remove(fname + '.ISO')
        return (estimate, distance, (a, b), (bb, bblow, bbhigh))

    def calc_ibd_diplo(self, fname, stat='a', scale='Log', min_dist=1e-05):
        if False:
            print('Hello World!')
        'Calculate isolation by distance statistics for diploid data.\n\n        See _calc_ibd for parameter details.\n\n        Note that each pop can only have a single individual and\n        the individual name has to be the sample coordinates.\n        '
        return self._calc_ibd(fname, 5, stat, scale, min_dist)

    def calc_ibd_haplo(self, fname, stat='a', scale='Log', min_dist=1e-05):
        if False:
            return 10
        'Calculate isolation by distance statistics for haploid data.\n\n        See _calc_ibd for parameter details.\n\n        Note that each pop can only have a single individual and\n        the individual name has to be the sample coordinates.\n        '
        return self._calc_ibd(fname, 6, stat, scale, min_dist)