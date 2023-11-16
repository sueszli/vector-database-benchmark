"""Base class for the support of PAML, Phylogenetic Analysis by Maximum Likelihood."""
import os
import subprocess

class PamlError(EnvironmentError):
    """paml has failed.

    Run with verbose=True to view the error message.
    """

class Paml:
    """Base class for wrapping PAML commands."""

    def __init__(self, alignment=None, working_dir=None, out_file=None):
        if False:
            print('Hello World!')
        'Initialize the class.'
        if working_dir is None:
            self.working_dir = os.getcwd()
        else:
            self.working_dir = working_dir
        if alignment is not None:
            if not os.path.exists(alignment):
                raise FileNotFoundError('The specified alignment file does not exist.')
        self.alignment = alignment
        self.out_file = out_file
        self._options = {}

    def write_ctl_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Write control file.'

    def read_ctl_file(self):
        if False:
            print('Hello World!')
        'Read control file.'

    def print_options(self):
        if False:
            for i in range(10):
                print('nop')
        'Print out all of the options and their current settings.'
        for option in self._options.items():
            print(f'{option[0]} = {option[1]}')

    def set_options(self, **kwargs):
        if False:
            return 10
        'Set the value of an option.\n\n        This function abstracts the options dict to prevent the user from\n        adding options that do not exist or misspelling options.\n        '
        for (option, value) in kwargs.items():
            if option not in self._options:
                raise KeyError('Invalid option: ' + option)
            else:
                self._options[option] = value

    def get_option(self, option):
        if False:
            return 10
        'Return the value of an option.'
        if option not in self._options:
            raise KeyError('Invalid option: ' + option)
        else:
            return self._options.get(option)

    def get_all_options(self):
        if False:
            print('Hello World!')
        'Return the values of all the options.'
        return list(self._options.items())

    def _set_rel_paths(self):
        if False:
            return 10
        'Convert all file/directory locations to paths relative to the current working directory (PRIVATE).\n\n        paml requires that all paths specified in the control file be\n        relative to the directory from which it is called rather than\n        absolute paths.\n        '
        if self.working_dir is not None:
            self._rel_working_dir = os.path.relpath(self.working_dir)
        if self.alignment is not None:
            self._rel_alignment = os.path.relpath(self.alignment, self.working_dir)
        if self.out_file is not None:
            self._rel_out_file = os.path.relpath(self.out_file, self.working_dir)

    def run(self, ctl_file, verbose, command):
        if False:
            for i in range(10):
                print('nop')
        'Run a paml program using the current configuration.\n\n        Check that the class attributes exist and raise an error\n        if not. Then run the command and check if it succeeds with\n        a return code of 0, otherwise raise an error.\n\n        The arguments may be passed as either absolute or relative\n        paths, despite the fact that paml requires relative paths.\n        '
        if self.alignment is None:
            raise ValueError('Alignment file not specified.')
        if not os.path.exists(self.alignment):
            raise FileNotFoundError('The specified alignment file does not exist.')
        if self.out_file is None:
            raise ValueError('Output file not specified.')
        if self.working_dir is None:
            raise ValueError('Working directory not specified.')
        cwd = os.getcwd()
        if not os.path.exists(self.working_dir):
            os.mkdir(self.working_dir)
        os.chdir(self.working_dir)
        if ctl_file is None:
            self.write_ctl_file()
            ctl_file = self.ctl_file
        elif not os.path.exists(ctl_file):
            raise FileNotFoundError('The specified control file does not exist.')
        if verbose:
            result_code = subprocess.call([command, ctl_file])
        else:
            with open(os.devnull) as dn:
                result_code = subprocess.call([command, ctl_file], stdout=dn, stderr=dn)
        os.chdir(cwd)
        if result_code > 0:
            raise PamlError('%s has failed (return code %i). Run with verbose = True to view error message' % (command, result_code))
        if result_code < 0:
            raise OSError('The %s process was killed (return code %i).' % (command, result_code))