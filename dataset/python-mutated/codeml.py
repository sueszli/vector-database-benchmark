"""Classes for the support of CODEML.

Maximum likelihood analysis using codon substitution models.
"""
import os.path
from ._paml import Paml
from . import _parse_codeml

class CodemlError(EnvironmentError):
    """CODEML failed. Run with verbose=True to view CODEML's error message."""

class Codeml(Paml):
    """An interface to CODEML, part of the PAML package."""

    def __init__(self, alignment=None, tree=None, working_dir=None, out_file=None):
        if False:
            return 10
        'Initialize the codeml instance.\n\n        The user may optionally pass in strings specifying the locations\n        of the input alignment and tree files, the working directory and\n        the final output file. Other options found in the CODEML control\n        have typical settings by default to run site class models 0, 1 and\n        2 on a nucleotide alignment.\n        '
        Paml.__init__(self, alignment, working_dir, out_file)
        if tree is not None:
            if not os.path.exists(tree):
                raise FileNotFoundError('The specified tree file does not exist.')
        self.tree = tree
        self.ctl_file = 'codeml.ctl'
        self._options = {'noisy': None, 'verbose': None, 'runmode': None, 'seqtype': None, 'CodonFreq': None, 'ndata': None, 'clock': None, 'aaDist': None, 'aaRatefile': None, 'model': None, 'NSsites': None, 'icode': None, 'Mgene': None, 'fix_kappa': None, 'kappa': None, 'fix_omega': None, 'omega': None, 'fix_alpha': None, 'alpha': None, 'Malpha': None, 'ncatG': None, 'getSE': None, 'RateAncestor': None, 'Small_Diff': None, 'cleandata': None, 'fix_blength': None, 'method': None, 'rho': None, 'fix_rho': None}

    def write_ctl_file(self):
        if False:
            i = 10
            return i + 15
        'Dynamically build a CODEML control file from the options.\n\n        The control file is written to the location specified by the\n        ctl_file property of the codeml class.\n        '
        self._set_rel_paths()
        with open(self.ctl_file, 'w') as ctl_handle:
            ctl_handle.write(f'seqfile = {self._rel_alignment}\n')
            ctl_handle.write(f'outfile = {self._rel_out_file}\n')
            ctl_handle.write(f'treefile = {self._rel_tree}\n')
            for option in self._options.items():
                if option[1] is None:
                    continue
                if option[0] == 'NSsites':
                    NSsites = ' '.join((str(site) for site in option[1]))
                    ctl_handle.write(f'{option[0]} = {NSsites}\n')
                else:
                    ctl_handle.write(f'{option[0]} = {option[1]}\n')

    def read_ctl_file(self, ctl_file):
        if False:
            return 10
        'Parse a control file and load the options into the Codeml instance.'
        temp_options = {}
        if not os.path.isfile(ctl_file):
            raise FileNotFoundError(f'File not found: {ctl_file!r}')
        else:
            with open(ctl_file) as ctl_handle:
                for line in ctl_handle:
                    line = line.strip()
                    uncommented = line.split('*', 1)[0]
                    if uncommented != '':
                        if '=' not in uncommented:
                            raise AttributeError(f'Malformed line in control file:\n{line!r}')
                        (option, value) = uncommented.split('=', 1)
                        option = option.strip()
                        value = value.strip()
                        if option == 'seqfile':
                            self.alignment = value
                        elif option == 'treefile':
                            self.tree = value
                        elif option == 'outfile':
                            self.out_file = value
                        elif option == 'NSsites':
                            site_classes = value.split(' ')
                            for n in range(len(site_classes)):
                                try:
                                    site_classes[n] = int(site_classes[n])
                                except ValueError:
                                    raise TypeError(f'Invalid site class: {site_classes[n]}') from None
                            temp_options['NSsites'] = site_classes
                        elif option not in self._options:
                            raise KeyError(f'Invalid option: {option}')
                        else:
                            if '.' in value:
                                try:
                                    converted_value = float(value)
                                except ValueError:
                                    converted_value = value
                            else:
                                try:
                                    converted_value = int(value)
                                except ValueError:
                                    converted_value = value
                            temp_options[option] = converted_value
        for option in self._options:
            if option in temp_options:
                self._options[option] = temp_options[option]
            else:
                self._options[option] = None

    def print_options(self):
        if False:
            i = 10
            return i + 15
        'Print out all of the options and their current settings.'
        for option in self._options.items():
            if option[0] == 'NSsites' and option[1] is not None:
                NSsites = ' '.join((str(site) for site in option[1]))
                print(f'{option[0]} = {NSsites}')
            else:
                print(f'{option[0]} = {option[1]}')

    def _set_rel_paths(self):
        if False:
            i = 10
            return i + 15
        'Make all file/directory paths relative to the PWD (PRIVATE).\n\n        CODEML requires that all paths specified in the control file be\n        relative to the directory from which it is called rather than\n        absolute paths.\n        '
        Paml._set_rel_paths(self)
        if self.tree is not None:
            self._rel_tree = os.path.relpath(self.tree, self.working_dir)

    def run(self, ctl_file=None, verbose=False, command='codeml', parse=True):
        if False:
            print('Hello World!')
        'Run codeml using the current configuration.\n\n        If parse is True then read and return the results, otherwise\n        return None.\n\n        The arguments may be passed as either absolute or relative\n        paths, despite the fact that CODEML requires relative paths.\n        '
        if self.tree is None:
            raise ValueError('Tree file not specified.')
        if not os.path.exists(self.tree):
            raise FileNotFoundError('The specified tree file does not exist.')
        Paml.run(self, ctl_file, verbose, command)
        if parse:
            return read(self.out_file)
        return None

def read(results_file):
    if False:
        for i in range(10):
            print('nop')
    'Parse a CODEML results file.'
    results = {}
    if not os.path.exists(results_file):
        raise FileNotFoundError('Results file does not exist.')
    with open(results_file) as handle:
        lines = handle.readlines()
    if not lines:
        raise ValueError("Empty results file.  Did CODEML exit successfully?  Run 'Codeml.run()' with 'verbose=True'.")
    (results, multi_models, multi_genes) = _parse_codeml.parse_basics(lines, results)
    results = _parse_codeml.parse_nssites(lines, results, multi_models, multi_genes)
    results = _parse_codeml.parse_pairwise(lines, results)
    results = _parse_codeml.parse_distances(lines, results)
    if not results:
        raise ValueError('Invalid results file')
    return results