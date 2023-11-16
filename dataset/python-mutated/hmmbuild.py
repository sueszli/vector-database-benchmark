"""A Python wrapper for hmmbuild - construct HMM profiles from MSA."""
import os
import re
import subprocess
from absl import logging
from . import utils

class Hmmbuild(object):
    """Python wrapper of the hmmbuild binary."""

    def __init__(self, *, binary_path: str, singlemx: bool=False):
        if False:
            return 10
        'Initializes the Python hmmbuild wrapper.\n\n        Args:\n            binary_path: The path to the hmmbuild executable.\n            singlemx: Whether to use --singlemx flag. If True, it forces HMMBuild to\n                just use a common substitution score matrix.\n\n        Raises:\n            RuntimeError: If hmmbuild binary not found within the path.\n        '
        self.binary_path = binary_path
        self.singlemx = singlemx

    def build_profile_from_sto(self, sto: str, model_construction='fast') -> str:
        if False:
            return 10
        "Builds a HHM for the aligned sequences given as an A3M string.\n\n        Args:\n            sto: A string with the aligned sequences in the Stockholm format.\n            model_construction: Whether to use reference annotation in the msa to\n                determine consensus columns ('hand') or default ('fast').\n\n        Returns:\n            A string with the profile in the HMM format.\n\n        Raises:\n            RuntimeError: If hmmbuild fails.\n        "
        return self._build_profile(sto, model_construction=model_construction)

    def build_profile_from_a3m(self, a3m: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Builds a HHM for the aligned sequences given as an A3M string.\n\n        Args:\n            a3m: A string with the aligned sequences in the A3M format.\n\n        Returns:\n            A string with the profile in the HMM format.\n\n        Raises:\n            RuntimeError: If hmmbuild fails.\n        '
        lines = []
        for line in a3m.splitlines():
            if not line.startswith('>'):
                line = re.sub('[a-z]+', '', line)
            lines.append(line + '\n')
        msa = ''.join(lines)
        return self._build_profile(msa, model_construction='fast')

    def _build_profile(self, msa: str, model_construction: str='fast') -> str:
        if False:
            while True:
                i = 10
        "Builds a HMM for the aligned sequences given as an MSA string.\n\n        Args:\n            msa: A string with the aligned sequences, in A3M or STO format.\n            model_construction: Whether to use reference annotation in the msa to\n                determine consensus columns ('hand') or default ('fast').\n\n        Returns:\n            A string with the profile in the HMM format.\n\n        Raises:\n            RuntimeError: If hmmbuild fails.\n            ValueError: If unspecified arguments are provided.\n        "
        if model_construction not in {'hand', 'fast'}:
            raise ValueError(f'Invalid model_construction {model_construction} - onlyhand and fast supported.')
        with utils.tmpdir_manager() as query_tmp_dir:
            input_query = os.path.join(query_tmp_dir, 'query.msa')
            output_hmm_path = os.path.join(query_tmp_dir, 'output.hmm')
            with open(input_query, 'w') as f:
                f.write(msa)
            cmd = [self.binary_path]
            if model_construction == 'hand':
                cmd.append(f'--{model_construction}')
            if self.singlemx:
                cmd.append('--singlemx')
            cmd.extend(['--amino', output_hmm_path, input_query])
            logging.info('Launching subprocess %s', cmd)
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with utils.timing('hmmbuild query'):
                (stdout, stderr) = process.communicate()
                retcode = process.wait()
                logging.info('hmmbuild stdout:\n%s\n\nstderr:\n%s\n', stdout.decode('utf-8'), stderr.decode('utf-8'))
            if retcode:
                raise RuntimeError('hmmbuild failed\nstdout:\n%s\n\nstderr:\n%s\n' % (stdout.decode('utf-8'), stderr.decode('utf-8')))
            with open(output_hmm_path, encoding='utf-8') as f:
                hmm = f.read()
        return hmm