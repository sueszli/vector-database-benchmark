"""Library to run HHblits from Python."""
import glob
import os
import subprocess
from typing import Any, List, Mapping, Optional, Sequence
from absl import logging
from . import utils
_HHBLITS_DEFAULT_P = 20
_HHBLITS_DEFAULT_Z = 500

class HHBlits:
    """Python wrapper of the HHblits binary."""

    def __init__(self, *, binary_path: str, databases: Sequence[str], n_cpu: int=4, n_iter: int=3, e_value: float=0.001, maxseq: int=1000000, realign_max: int=100000, maxfilt: int=100000, min_prefilter_hits: int=1000, all_seqs: bool=False, alt: Optional[int]=None, p: int=_HHBLITS_DEFAULT_P, z: int=_HHBLITS_DEFAULT_Z):
        if False:
            for i in range(10):
                print('nop')
        'Initializes the Python HHblits wrapper.\n\n        Args:\n            binary_path: The path to the HHblits executable.\n            databases: A sequence of HHblits database paths. This should be the\n                common prefix for the database files (i.e. up to but not including\n                _hhm.ffindex etc.)\n            n_cpu: The number of CPUs to give HHblits.\n            n_iter: The number of HHblits iterations.\n            e_value: The E-value, see HHblits docs for more details.\n            maxseq: The maximum number of rows in an input alignment. Note that this\n                parameter is only supported in HHBlits version 3.1 and higher.\n            realign_max: Max number of HMM-HMM hits to realign. HHblits default: 500.\n            maxfilt: Max number of hits allowed to pass the 2nd prefilter.\n                HHblits default: 20000.\n            min_prefilter_hits: Min number of hits to pass prefilter.\n                HHblits default: 100.\n            all_seqs: Return all sequences in the MSA / Do not filter the result MSA.\n                HHblits default: False.\n            alt: Show up to this many alternative alignments.\n            p: Minimum Prob for a hit to be included in the output hhr file.\n                HHblits default: 20.\n            z: Hard cap on number of hits reported in the hhr file.\n                HHblits default: 500. NB: The relevant HHblits flag is -Z not -z.\n\n        Raises:\n            RuntimeError: If HHblits binary not found within the path.\n        '
        self.binary_path = binary_path
        self.databases = databases
        for database_path in self.databases:
            if not glob.glob(database_path + '_*'):
                logging.error('Could not find HHBlits database %s', database_path)
                raise ValueError(f'Could not find HHBlits database {database_path}')
        self.n_cpu = n_cpu
        self.n_iter = n_iter
        self.e_value = e_value
        self.maxseq = maxseq
        self.realign_max = realign_max
        self.maxfilt = maxfilt
        self.min_prefilter_hits = min_prefilter_hits
        self.all_seqs = all_seqs
        self.alt = alt
        self.p = p
        self.z = z

    def query(self, input_fasta_path: str) -> List[Mapping[str, Any]]:
        if False:
            i = 10
            return i + 15
        'Queries the database using HHblits.'
        with utils.tmpdir_manager() as query_tmp_dir:
            a3m_path = os.path.join(query_tmp_dir, 'output.a3m')
            db_cmd = []
            for db_path in self.databases:
                db_cmd.append('-d')
                db_cmd.append(db_path)
            cmd = [self.binary_path, '-i', input_fasta_path, '-cpu', str(self.n_cpu), '-oa3m', a3m_path, '-o', '/dev/null', '-n', str(self.n_iter), '-e', str(self.e_value), '-maxseq', str(self.maxseq), '-realign_max', str(self.realign_max), '-maxfilt', str(self.maxfilt), '-min_prefilter_hits', str(self.min_prefilter_hits)]
            if self.all_seqs:
                cmd += ['-all']
            if self.alt:
                cmd += ['-alt', str(self.alt)]
            if self.p != _HHBLITS_DEFAULT_P:
                cmd += ['-p', str(self.p)]
            if self.z != _HHBLITS_DEFAULT_Z:
                cmd += ['-Z', str(self.z)]
            cmd += db_cmd
            logging.info('Launching subprocess "%s"', ' '.join(cmd))
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with utils.timing('HHblits query'):
                (stdout, stderr) = process.communicate()
                retcode = process.wait()
            if retcode:
                logging.error('HHblits failed. HHblits stderr begin:')
                for error_line in stderr.decode('utf-8').splitlines():
                    if error_line.strip():
                        logging.error(error_line.strip())
                logging.error('HHblits stderr end')
                raise RuntimeError('HHblits failed\nstdout:\n%s\n\nstderr:\n%s\n' % (stdout.decode('utf-8'), stderr[:500000].decode('utf-8')))
            with open(a3m_path) as f:
                a3m = f.read()
        raw_output = dict(a3m=a3m, output=stdout, stderr=stderr, n_iter=self.n_iter, e_value=self.e_value)
        return [raw_output]