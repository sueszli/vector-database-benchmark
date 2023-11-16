"""A Python wrapper for hmmsearch - search profile against a sequence db."""
import os
import subprocess
from typing import Optional, Sequence
from absl import logging
from modelscope.models.science.unifold.msa import parsers
from . import hmmbuild, utils

class Hmmsearch(object):
    """Python wrapper of the hmmsearch binary."""

    def __init__(self, *, binary_path: str, hmmbuild_binary_path: str, database_path: str, flags: Optional[Sequence[str]]=None):
        if False:
            i = 10
            return i + 15
        'Initializes the Python hmmsearch wrapper.\n\n        Args:\n            binary_path: The path to the hmmsearch executable.\n            hmmbuild_binary_path: The path to the hmmbuild executable. Used to build\n                an hmm from an input a3m.\n            database_path: The path to the hmmsearch database (FASTA format).\n            flags: List of flags to be used by hmmsearch.\n\n        Raises:\n            RuntimeError: If hmmsearch binary not found within the path.\n        '
        self.binary_path = binary_path
        self.hmmbuild_runner = hmmbuild.Hmmbuild(binary_path=hmmbuild_binary_path)
        self.database_path = database_path
        if flags is None:
            flags = ['--F1', '0.1', '--F2', '0.1', '--F3', '0.1', '--incE', '100', '-E', '100', '--domE', '100', '--incdomE', '100']
        self.flags = flags
        if not os.path.exists(self.database_path):
            logging.error('Could not find hmmsearch database %s', database_path)
            raise ValueError(f'Could not find hmmsearch database {database_path}')

    @property
    def output_format(self) -> str:
        if False:
            while True:
                i = 10
        return 'sto'

    @property
    def input_format(self) -> str:
        if False:
            i = 10
            return i + 15
        return 'sto'

    def query(self, msa_sto: str) -> str:
        if False:
            return 10
        'Queries the database using hmmsearch using a given stockholm msa.'
        hmm = self.hmmbuild_runner.build_profile_from_sto(msa_sto, model_construction='hand')
        return self.query_with_hmm(hmm)

    def query_with_hmm(self, hmm: str) -> str:
        if False:
            while True:
                i = 10
        'Queries the database using hmmsearch using a given hmm.'
        with utils.tmpdir_manager() as query_tmp_dir:
            hmm_input_path = os.path.join(query_tmp_dir, 'query.hmm')
            out_path = os.path.join(query_tmp_dir, 'output.sto')
            with open(hmm_input_path, 'w') as f:
                f.write(hmm)
            cmd = [self.binary_path, '--noali', '--cpu', '8']
            if self.flags:
                cmd.extend(self.flags)
            cmd.extend(['-A', out_path, hmm_input_path, self.database_path])
            logging.info('Launching sub-process %s', cmd)
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            with utils.timing(f'hmmsearch ({os.path.basename(self.database_path)}) query'):
                (stdout, stderr) = process.communicate()
                retcode = process.wait()
            if retcode:
                raise RuntimeError('hmmsearch failed:\nstdout:\n%s\n\nstderr:\n%s\n' % (stdout.decode('utf-8'), stderr.decode('utf-8')))
            with open(out_path) as f:
                out_msa = f.read()
        return out_msa

    def get_template_hits(self, output_string: str, input_sequence: str) -> Sequence[parsers.TemplateHit]:
        if False:
            for i in range(10):
                print('nop')
        'Gets parsed template hits from the raw string output by the tool.'
        a3m_string = parsers.convert_stockholm_to_a3m(output_string, remove_first_row_gaps=False)
        template_hits = parsers.parse_hmmsearch_a3m(query_sequence=input_sequence, a3m_string=a3m_string, skip_first=False)
        return template_hits