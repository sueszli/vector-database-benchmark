"""Runs a few EMBOSS tools to check our wrappers and parsers."""
import os
import sys
import unittest
import subprocess
from io import StringIO
from Bio.Emboss.Applications import WaterCommandline, NeedleCommandline
from Bio.Emboss.Applications import SeqretCommandline, SeqmatchallCommandline
from Bio import SeqIO
from Bio import AlignIO
from Bio import MissingExternalDependencyError
from Bio.Application import _escape_filename
from Bio.Seq import Seq, translate
from Bio.SeqRecord import SeqRecord
os.environ['LANG'] = 'C'
exes_wanted = ['water', 'needle', 'seqret', 'transeq', 'seqmatchall', 'embossversion']
exes = {}
if 'EMBOSS_ROOT' in os.environ:
    path = os.environ['EMBOSS_ROOT']
    if os.path.isdir(path):
        for name in exes_wanted:
            if os.path.isfile(os.path.join(path, name + '.exe')):
                exes[name] = os.path.join(path, name + '.exe')
        del name
    else:
        raise MissingExternalDependencyError(f'$EMBOSS_ROOT={path!r} which does not exist!')
    del path
if sys.platform != 'win32':
    from subprocess import getoutput
    for name in exes_wanted:
        output = getoutput(f'{name} -help')
        if 'not found' not in output and 'not recognized' not in output:
            exes[name] = name
        del output
    del name
if len(exes) < len(exes_wanted):
    raise MissingExternalDependencyError('Install EMBOSS if you want to use Bio.Emboss.')

def get_emboss_version():
    if False:
        while True:
            i = 10
    'Return a tuple of three ints, e.g. (6,1,0).'
    child = subprocess.Popen(_escape_filename(exes['embossversion']), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, shell=sys.platform != 'win32')
    (stdout, stderr) = child.communicate()
    child.stdout.close()
    del child
    assert stderr is None
    for line in stdout.split('\n'):
        if line.strip() == 'Report the current EMBOSS version number':
            pass
        elif line.strip() == 'Reports the current EMBOSS version number':
            pass
        elif line.startswith('Writes the current EMBOSS version number'):
            pass
        elif line.count('.') == 2:
            return tuple((int(v) for v in line.strip().split('.')))
        elif line.count('.') == 3:
            return tuple((int(v) for v in line.strip().split('.')))[:3]
        else:
            raise MissingExternalDependencyError(f'Install EMBOSS if you want to use Bio.Emboss ({line}).')
    raise MissingExternalDependencyError('Could not get EMBOSS version')
emboss_version = get_emboss_version()
if emboss_version < (6, 1, 0):
    raise MissingExternalDependencyError('Test requires EMBOSS 6.1.0 patch 3 or later.')

def emboss_piped_SeqIO_convert(records, old_format, new_format):
    if False:
        print('Hello World!')
    'Run seqret, returns records (as a generator).'
    cline = SeqretCommandline(exes['seqret'], sformat=old_format, osformat=new_format, auto=True, filter=True)
    child = subprocess.Popen(str(cline), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=sys.platform != 'win32')
    SeqIO.write(records, child.stdin, old_format)
    child.stdin.close()
    child.stderr.close()
    records = SeqIO.parse(child.stdout, new_format)
    yield from records
    child.stdout.close()

def emboss_piped_AlignIO_convert(alignments, old_format, new_format):
    if False:
        return 10
    'Run seqret, returns alignments (as a generator).'
    cline = SeqretCommandline(exes['seqret'], sformat=old_format, osformat=new_format, auto=True, filter=True)
    with subprocess.Popen(str(cline), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=sys.platform != 'win32') as child:
        AlignIO.write(alignments, child.stdin, old_format)
        child.stdin.close()
        aligns = list(AlignIO.parse(child.stdout, new_format))
    return aligns

class SeqRetTests(unittest.TestCase):
    """Base class providing SeqRecord comparison method."""

    def compare_records(self, old_records, new_records, msg=None):
        if False:
            while True:
                i = 10
        self.assertEqual(len(old_records), len(new_records), msg)
        for (old, new) in zip(old_records, new_records):
            self.assertTrue(old.id in new.id or new.id in old.id or old.id.replace(' ', '_') == new.id.replace(' ', '_') or (old.name == new.name), msg)
            self.assertEqual(len(old.seq), len(new.seq), msg)
            if old.seq.upper() != new.seq.upper():
                raise Exception
                if str(old.seq).replace('X', 'N') == str(new.seq):
                    self.fail(f'{msg}: X -> N (protein forced into nucleotide?)')
                else:
                    self.assertEqual(old.seq, new.seq, msg)
            if old.features and new.features:
                self.assertEqual(len(old.features), len(new.features), msg)

class SeqRetSeqIOTests(SeqRetTests):
    """Check EMBOSS seqret against Bio.SeqIO for converting files."""

    def tearDown(self):
        if False:
            return 10
        clean_up()

    def check_SeqIO_to_EMBOSS(self, in_filename, in_format, skip_formats=()):
        if False:
            return 10
        'Check SeqIO writes files seqret can read back.'
        records = list(SeqIO.parse(in_filename, in_format))
        for temp_format in ['genbank', 'embl', 'fasta']:
            if temp_format in skip_formats:
                continue
            new_records = list(emboss_piped_SeqIO_convert(records, temp_format, 'fasta'))
            msg = f'converting {in_filename} from {in_format} to {temp_format}'
            self.compare_records(records, new_records, msg)

    def check_EMBOSS_to_SeqIO(self, filename, old_format, skip_formats=()):
        if False:
            return 10
        "Check SeqIO can read read seqret's conversion output."
        self.assertTrue(os.path.isfile(filename))
        old_records = list(SeqIO.parse(filename, old_format))
        for new_format in ['genbank', 'fasta', 'pir', 'embl', 'ig']:
            if new_format in skip_formats:
                continue
            cline = SeqretCommandline(exes['seqret'], sequence=filename, sformat=old_format, osformat=new_format, auto=True, stdout=True)
            with subprocess.Popen(str(cline), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=sys.platform != 'win32') as child:
                child.stdin.close()
                new_records = list(SeqIO.parse(child.stdout, new_format))
            msg = f'converting {filename} from {old_format} to {new_format}'
            self.compare_records(old_records, new_records, msg)

    def check_SeqIO_with_EMBOSS(self, filename, old_format, skip_formats=()):
        if False:
            i = 10
            return i + 15
        self.check_SeqIO_to_EMBOSS(filename, old_format, skip_formats)
        self.check_EMBOSS_to_SeqIO(filename, old_format, skip_formats)

    def test_abi(self):
        if False:
            while True:
                i = 10
        "Check SeqIO agrees with EMBOSS' Abi to FASTQ conversion."
        for filename in ['Abi/3730.ab1', 'Abi/empty.ab1']:
            old = SeqIO.read(filename, 'abi')
            cline = SeqretCommandline(exes['seqret'], sequence=filename, sformat='abi', osformat='fastq-sanger', auto=True, stdout=True)
            with subprocess.Popen(str(cline), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=sys.platform != 'win32') as child:
                child.stdin.close()
                new = SeqIO.read(child.stdout, 'fastq-sanger')
            if emboss_version == (6, 4, 0) and new.id == 'EMBOSS_001':
                pass
            else:
                self.assertEqual(old.id, new.id)
            self.assertEqual(old.seq, new.seq)
            if emboss_version < (6, 3, 0) and new.letter_annotations['phred_quality'] == [1] * len(old):
                pass
            else:
                self.assertEqual(old.letter_annotations, new.letter_annotations)

    def test_genbank(self):
        if False:
            while True:
                i = 10
        "Check SeqIO & EMBOSS reading each other's conversions of a GenBank file."
        self.check_SeqIO_with_EMBOSS('GenBank/cor6_6.gb', 'genbank')

    def test_genbank2(self):
        if False:
            while True:
                i = 10
        "Check SeqIO & EMBOSS reading each other's conversions of another GenBank file."
        self.check_SeqIO_with_EMBOSS('GenBank/NC_000932.gb', 'genbank')

    def test_embl(self):
        if False:
            while True:
                i = 10
        "Check SeqIO & EMBOSS reading each other's conversions of an EMBL file."
        self.check_SeqIO_with_EMBOSS('EMBL/U87107.embl', 'embl')

    def test_ig(self):
        if False:
            print('Hello World!')
        "Check SeqIO & EMBOSS reading each other's conversions of an ig file."
        self.check_SeqIO_to_EMBOSS('IntelliGenetics/VIF_mase-pro.txt', 'ig', skip_formats=['genbank', 'embl'])

    def test_pir(self):
        if False:
            for i in range(10):
                print('nop')
        "Check SeqIO & EMBOSS reading each other's conversions of a PIR file."
        self.check_SeqIO_with_EMBOSS('NBRF/clustalw.pir', 'pir', skip_formats=['genbank'])
        self.check_SeqIO_with_EMBOSS('NBRF/DMB_prot.pir', 'pir', skip_formats=['embl', 'genbank'])

    def test_clustalw(self):
        if False:
            while True:
                i = 10
        "Check SeqIO & EMBOSS reading each other's conversions of a Clustalw file."
        self.check_SeqIO_with_EMBOSS('Clustalw/hedgehog.aln', 'clustal', skip_formats=['embl', 'genbank'])
        self.check_SeqIO_with_EMBOSS('Clustalw/opuntia.aln', 'clustal', skip_formats=['embl', 'genbank'])

class SeqRetAlignIOTests(SeqRetTests):
    """Check EMBOSS seqret against Bio.AlignIO for converting files."""

    def tearDown(self):
        if False:
            return 10
        clean_up()

    def compare_alignments(self, old_list, new_list, msg=None):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(len(old_list), len(new_list), msg)
        for (old, new) in zip(old_list, new_list):
            self.compare_records(old, new, msg)

    def check_EMBOSS_to_AlignIO(self, filename, old_format, skip_formats=()):
        if False:
            return 10
        "Check AlignIO can read seqret's conversion of the file."
        self.assertTrue(os.path.isfile(filename), filename)
        old_aligns = list(AlignIO.parse(filename, old_format))
        formats = ['clustal', 'phylip', 'ig', 'msf']
        if len(old_aligns) == 1:
            formats.extend(['fasta', 'nexus'])
        for new_format in formats:
            if new_format in skip_formats:
                continue
            cline = SeqretCommandline(exes['seqret'], sequence=filename, sformat=old_format, osformat=new_format, auto=True, stdout=True)
            with subprocess.Popen(str(cline), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=sys.platform != 'win32') as child:
                child.stdin.close()
                new_aligns = list(AlignIO.parse(child.stdout, new_format))
            msg = f'converting {filename} from {old_format} to {new_format}'
            self.compare_alignments(old_aligns, new_aligns, msg)

    def check_AlignIO_to_EMBOSS(self, in_filename, in_format, skip_formats=()):
        if False:
            for i in range(10):
                print('nop')
        'Check Bio.AlignIO can write files seqret can read.'
        old_aligns = list(AlignIO.parse(in_filename, in_format))
        formats = ['clustal', 'phylip']
        if len(old_aligns) == 1:
            formats.extend(['fasta', 'nexus'])
        for temp_format in formats:
            if temp_format in skip_formats:
                continue
            try:
                new_aligns = list(emboss_piped_AlignIO_convert(old_aligns, temp_format, 'phylip'))
            except ValueError as e:
                self.assertIn(str(e), ('Need the molecule type to be defined', "Repeated name 'AT3G20900.' (originally 'AT3G20900.1-SEQ'), possibly due to truncation", "Repeated name 'gi|1377497' (originally 'gi|13774975|gb|AAK39115.1|AF35'), possibly due to truncation", "Repeated name 'gi_1393639' (originally 'gi_13936397_dbj_BAB47195.'), possibly due to truncation"))
                continue
            msg = f'converting {in_filename} from {in_format} to {temp_format}'
            self.compare_alignments(old_aligns, new_aligns, msg)

    def check_AlignIO_with_EMBOSS(self, filename, old_format, skip_formats=()):
        if False:
            return 10
        self.check_AlignIO_to_EMBOSS(filename, old_format, skip_formats)
        self.check_EMBOSS_to_AlignIO(filename, old_format, skip_formats)

    def test_align_clustalw(self):
        if False:
            print('Hello World!')
        "Check AlignIO & EMBOSS reading each other's conversions of a ClustalW file."
        self.check_AlignIO_with_EMBOSS('Clustalw/hedgehog.aln', 'clustal')
        self.check_AlignIO_with_EMBOSS('Clustalw/opuntia.aln', 'clustal')
        self.check_AlignIO_with_EMBOSS('Clustalw/odd_consensus.aln', 'clustal', skip_formats=['nexus'])
        self.check_AlignIO_with_EMBOSS('Clustalw/protein.aln', 'clustal')
        self.check_AlignIO_with_EMBOSS('Clustalw/promals3d.aln', 'clustal')

    def test_clustalw(self):
        if False:
            i = 10
            return i + 15
        "Check AlignIO & EMBOSS reading each other's conversions of a PHYLIP file."
        self.check_AlignIO_with_EMBOSS('Phylip/horses.phy', 'phylip')
        self.check_AlignIO_with_EMBOSS('Phylip/hennigian.phy', 'phylip')
        self.check_AlignIO_with_EMBOSS('Phylip/reference_dna.phy', 'phylip')
        self.check_AlignIO_with_EMBOSS('Phylip/reference_dna2.phy', 'phylip')
        self.check_AlignIO_with_EMBOSS('Phylip/interlaced.phy', 'phylip')
        self.check_AlignIO_with_EMBOSS('Phylip/interlaced2.phy', 'phylip')
        self.check_AlignIO_with_EMBOSS('Phylip/random.phy', 'phylip')

class PairwiseAlignmentTests(unittest.TestCase):
    """Run pairwise alignments with water and needle, and parse them."""

    def tearDown(self):
        if False:
            while True:
                i = 10
        clean_up()

    def pairwise_alignment_check(self, query_seq, targets, alignments, local=True):
        if False:
            while True:
                i = 10
        'Check pairwise alignment data is sane.'
        targets = list(targets)
        alignments = list(alignments)
        self.assertEqual(len(targets), len(alignments))
        for (target, alignment) in zip(targets, alignments):
            self.assertEqual(len(alignment), 2)
            msg = f'{alignment[1].id} vs {target.id} or {target.name}'
            self.assertTrue(alignment[1].id in target.id or alignment[1].id in target.name, msg=msg)
            if local:
                self.assertIn(str(alignment[0].seq).replace('-', ''), query_seq)
                self.assertIn(str(alignment[1].seq).replace('-', '').upper(), target.seq.upper())
            else:
                self.assertEqual(query_seq, str(alignment[0].seq).replace('-', ''))
                self.assertEqual(target.seq.upper(), str(alignment[1].seq).replace('-', '').upper())
        return True

    def run_water(self, cline):
        if False:
            for i in range(10):
                print('nop')
        (stdout, stderr) = cline()
        self.assertTrue(stderr.strip().startswith('Smith-Waterman local alignment'), stderr)
        if cline.outfile:
            self.assertEqual(stdout.strip(), '')
            self.assertTrue(os.path.isfile(cline.outfile), f'Missing output file {cline.outfile!r} from:\n{cline}')
        else:
            return stdout

    def test_water_file(self):
        if False:
            print('Hello World!')
        'Run water with the asis trick, output to a file.'
        cline = WaterCommandline(cmd=exes['water'], gapopen='10', gapextend='0.5')
        cline.set_parameter('asequence', 'asis:ACCCGGGCGCGGT')
        cline.set_parameter('-bsequence', 'asis:ACCCGAGCGCGGT')
        cline.outfile = 'Emboss/temp with space.water'
        self.assertEqual(str(eval(repr(cline))), str(cline))
        self.run_water(cline)
        align = AlignIO.read(cline.outfile, 'emboss')
        self.assertEqual(len(align), 2)
        self.assertEqual(align[0].seq, 'ACCCGGGCGCGGT')
        self.assertEqual(align[1].seq, 'ACCCGAGCGCGGT')
        os.remove(cline.outfile)

    def test_water_piped(self):
        if False:
            while True:
                i = 10
        'Run water with asis trick, output piped to stdout.'
        cline = WaterCommandline(cmd=exes['water'], asequence='asis:ACCCGGGCGCGGT', bsequence='asis:ACCCGAGCGCGGT', gapopen=10, gapextend=0.5, auto=True, filter=True)
        self.assertEqual(str(cline), exes['water'] + ' -auto -filter' + ' -asequence=asis:ACCCGGGCGCGGT' + ' -bsequence=asis:ACCCGAGCGCGGT' + ' -gapopen=10 -gapextend=0.5')
        child = subprocess.Popen(str(cline), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=sys.platform != 'win32')
        child.stdin.close()
        align = AlignIO.read(child.stdout, 'emboss')
        self.assertEqual(len(align), 2)
        self.assertEqual(align[0].seq, 'ACCCGGGCGCGGT')
        self.assertEqual(align[1].seq, 'ACCCGAGCGCGGT')
        self.assertEqual(child.stderr.read(), '')
        self.assertEqual(0, child.wait())
        child.stdout.close()
        child.stderr.close()

    def test_needle_file(self):
        if False:
            print('Hello World!')
        'Run needle with the asis trick, output to a file.'
        cline = NeedleCommandline(cmd=exes['needle'])
        cline.set_parameter('-asequence', 'asis:ACCCGGGCGCGGT')
        cline.set_parameter('-bsequence', 'asis:ACCCGAGCGCGGT')
        cline.set_parameter('-gapopen', '10')
        cline.set_parameter('-gapextend', '0.5')
        cline.set_parameter('-snucleotide', 'True')
        cline.set_parameter('-outfile', 'Emboss/temp with space.needle')
        self.assertEqual(str(eval(repr(cline))), str(cline))
        (stdout, stderr) = cline()
        self.assertTrue(stderr.strip().startswith('Needleman-Wunsch global alignment'), stderr)
        self.assertEqual(stdout.strip(), '')
        filename = cline.outfile
        self.assertTrue(os.path.isfile(filename), f'Missing output file {filename!r} from:\n{cline}')
        align = AlignIO.read(filename, 'emboss')
        self.assertEqual(len(align), 2)
        self.assertEqual(align[0].seq, 'ACCCGGGCGCGGT')
        self.assertEqual(align[1].seq, 'ACCCGAGCGCGGT')
        os.remove(filename)

    def test_needle_piped(self):
        if False:
            while True:
                i = 10
        'Run needle with asis trick, output piped to stdout.'
        cline = NeedleCommandline(cmd=exes['needle'], asequence='asis:ACCCGGGCGCGGT', bsequence='asis:ACCCGAGCGCGGT', gapopen=10, gapextend=0.5, auto=True, filter=True)
        self.assertEqual(str(cline), exes['needle'] + ' -auto -filter' + ' -asequence=asis:ACCCGGGCGCGGT' + ' -bsequence=asis:ACCCGAGCGCGGT' + ' -gapopen=10 -gapextend=0.5')
        child = subprocess.Popen(str(cline), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=sys.platform != 'win32')
        child.stdin.close()
        align = AlignIO.read(child.stdout, 'emboss')
        self.assertEqual(len(align), 2)
        self.assertEqual(align[0].seq, 'ACCCGGGCGCGGT')
        self.assertEqual(align[1].seq, 'ACCCGAGCGCGGT')
        self.assertEqual(child.stderr.read(), '')
        self.assertEqual(0, child.wait())
        child.stdout.close()
        child.stderr.close()

    def test_water_file2(self):
        if False:
            while True:
                i = 10
        'Run water with the asis trick and nucleotide FASTA file, output to a file.'
        query = 'ACACACTCACACACACTTGGTCAGAGATGCTGTGCTTCTTGGAAGCAAGGNCTCAAAGGCAAGGTGCACGCAGAGGGACGTTTGAGTCTGGGATGAAGCATGTNCGTATTATTTATATGATGGAATTTCACGTTTTTATG'
        out_file = 'Emboss/temp_test2.water'
        in_file = 'Fasta/f002'
        self.assertTrue(os.path.isfile(in_file))
        if os.path.isfile(out_file):
            os.remove(out_file)
        cline = WaterCommandline(cmd=exes['water'])
        cline.set_parameter('-asequence', f'asis:{query}')
        cline.set_parameter('-bsequence', in_file)
        cline.set_parameter('-gapopen', '10')
        cline.set_parameter('-gapextend', '0.5')
        cline.set_parameter('-outfile', out_file)
        self.assertEqual(str(eval(repr(cline))), str(cline))
        self.run_water(cline)
        self.pairwise_alignment_check(query, SeqIO.parse(in_file, 'fasta'), AlignIO.parse(out_file, 'emboss'), local=True)
        os.remove(out_file)

    def test_water_file3(self):
        if False:
            return 10
        'Run water with the asis trick and GenBank file, output to a file.'
        query = 'TGTTGTAATGTTTTAATGTTTCTTCTCCCTTTAGATGTACTACGTTTGGA'
        out_file = 'Emboss/temp_test3.water'
        in_file = 'GenBank/cor6_6.gb'
        self.assertTrue(os.path.isfile(in_file))
        if os.path.isfile(out_file):
            os.remove(out_file)
        cline = WaterCommandline(cmd=exes['water'])
        cline.set_parameter('asequence', f'asis:{query}')
        cline.set_parameter('bsequence', in_file)
        cline.set_parameter('gapopen', '1')
        cline.set_parameter('gapextend', '0.5')
        cline.set_parameter('outfile', out_file)
        self.assertEqual(str(eval(repr(cline))), str(cline))
        self.run_water(cline)
        self.pairwise_alignment_check(query, SeqIO.parse(in_file, 'genbank'), AlignIO.parse(out_file, 'emboss'), local=True)
        os.remove(out_file)

    def test_water_file4(self):
        if False:
            for i in range(10):
                print('nop')
        'Run water with the asis trick and SwissProt file, output to a file.'
        query = 'DVCTGKALCDPVTQNIKTYPVKIENLRVMI'
        out_file = 'Emboss/temp_test4.water'
        in_file = 'SwissProt/P0A186.txt'
        self.assertTrue(os.path.isfile(in_file))
        if os.path.isfile(out_file):
            os.remove(out_file)
        cline = WaterCommandline(cmd=exes['water'])
        cline.set_parameter('-asequence', f'asis:{query}')
        cline.set_parameter('-bsequence', in_file)
        cline.set_parameter('-sprotein', True)
        cline.set_parameter('-gapopen', '20')
        cline.set_parameter('-gapextend', '5')
        cline.set_parameter('-outfile', out_file)
        self.assertEqual(str(eval(repr(cline))), str(cline))
        self.run_water(cline)
        self.pairwise_alignment_check(query, SeqIO.parse(in_file, 'swiss'), AlignIO.parse(out_file, 'emboss'), local=True)
        os.remove(out_file)

    def test_needle_piped2(self):
        if False:
            return 10
        'Run needle with asis trick, and nucleotide FASTA file, output piped to stdout.'
        query = 'ACACACTCACACACACTTGGTCAGAGATGCTGTGCTTCTTGGAA'
        cline = exes['needle']
        cline += ' -asequence asis:' + query
        cline += ' -bsequence Fasta/f002'
        cline += ' -auto'
        cline += ' -filter'
        child = subprocess.Popen(str(cline), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=sys.platform != 'win32')
        child.stdin.close()
        self.pairwise_alignment_check(query, SeqIO.parse('Fasta/f002', 'fasta'), AlignIO.parse(child.stdout, 'emboss'), local=False)
        self.assertEqual(child.stderr.read(), '')
        self.assertEqual(0, child.wait())
        child.stdout.close()
        child.stderr.close()

    def test_water_needs_output(self):
        if False:
            while True:
                i = 10
        'Run water without output file or stdout/filter should give error.'
        cline = WaterCommandline(cmd=exes['water'], asequence='asis:ACCCGGGCGCGGT', bsequence='asis:ACCCGAGCGCGGT', gapopen=10, gapextend=0.5, auto=True)
        self.assertTrue(cline.auto)
        self.assertTrue(not cline.stdout)
        self.assertTrue(not cline.filter)
        self.assertIsNone(cline.outfile)
        self.assertRaises(ValueError, str, cline)

    def test_needle_needs_output(self):
        if False:
            while True:
                i = 10
        'Run needle without output file or stdout/filter should give error.'
        cline = NeedleCommandline(cmd=exes['needle'], asequence='asis:ACCCGGGCGCGGT', bsequence='asis:ACCCGAGCGCGGT', gapopen=10, gapextend=0.5, auto=True)
        self.assertTrue(cline.auto)
        self.assertTrue(not cline.stdout)
        self.assertTrue(not cline.filter)
        self.assertIsNone(cline.outfile)
        self.assertRaises(ValueError, str, cline)

    def test_seqtmatchall_piped(self):
        if False:
            print('Hello World!')
        'Run seqmatchall with pair output piped to stdout.'
        cline = SeqmatchallCommandline(cmd=exes['seqmatchall'], sequence='Fasta/f002', aformat='pair', wordsize=9, auto=True, stdout=True)
        self.assertEqual(str(cline), exes['seqmatchall'] + ' -auto -stdout' + ' -sequence=Fasta/f002' + ' -wordsize=9 -aformat=pair')
        child = subprocess.Popen(str(cline), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=sys.platform != 'win32')
        child.stdin.close()
        for align in AlignIO.parse(child.stdout, 'emboss'):
            self.assertEqual(len(align), 2)
            self.assertEqual(align.get_alignment_length(), 9)
        self.assertEqual(child.stderr.read(), '')
        self.assertEqual(0, child.wait())
        child.stdout.close()
        child.stderr.close()

class TranslationTests(unittest.TestCase):
    """Run pairwise alignments with water and needle, and parse them."""

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        clean_up()

    def test_simple(self):
        if False:
            while True:
                i = 10
        'Run transeq vs Bio.Seq for simple translations (including alt tables).'
        examples = [Seq('ACGTGACTGACGTAGCATGCCACTAGG'), Seq('TAATACTATTAG'), Seq('TANTARTAYTAMTAKTAHTABTADTAV'), Seq('ACGGGGGGGGTAAGTGGTGTGTGTGTAGT')]
        for sequence in examples:
            if len(sequence) % 3 != 0:
                sequence = sequence[:-(len(sequence) % 3)]
            self.assertEqual(len(sequence) % 3, 0)
            self.assertGreater(len(sequence), 0)
            self.check(sequence)

    def check_emboss_translate(self, sequence, table=None, frame=None):
        if False:
            while True:
                i = 10
        'Call transeq, returns protein sequence as string.'
        cline = exes['transeq']
        if len(sequence) < 100:
            filename = None
            cline += f' -sequence asis:{sequence}'
        else:
            filename = 'Emboss/temp_transeq.txt'
            SeqIO.write(SeqRecord(sequence, id='Test'), filename, 'fasta')
            cline += f' -sequence {filename}'
        cline += ' -auto'
        cline += ' -filter'
        if table is not None:
            cline += f' -table {table!s}'
        if frame is not None:
            cline += f' -frame {frame!s}'
        child = subprocess.Popen(str(cline), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=sys.platform != 'win32')
        (out, err) = child.communicate()
        msg = f"cline='{cline}'"
        self.assertEqual(err, '', msg=msg)
        record = SeqIO.read(StringIO(out), 'fasta')
        result = child.wait()
        self.assertEqual(result, 0, msg=msg)
        if filename:
            os.remove(filename)
            self.assertTrue(record.id.startswith('Test'), msg=msg)
        else:
            self.assertTrue(record.id.startswith('asis'), msg=msg)
        translation = record.seq
        if table is None:
            table = 1
        self.assertEqual(translation, sequence.translate(table))
        self.assertEqual(translation, translate(sequence, table))
        self.assertEqual(translation, translate(str(sequence), table))
        for (i, amino) in enumerate(translation):
            codon = sequence[i * 3:i * 3 + 3]
            msg = f'codon {codon}, table {table}'
            self.assertEqual(amino, codon.translate(table), msg=msg)

    def check(self, sequence):
        if False:
            while True:
                i = 10
        "Compare our translation to EMBOSS's using all tables.\n\n        Takes a Seq object (and a filename containing it).\n        "
        self.check_emboss_translate(sequence)
        for table in [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 21, 22, 23]:
            self.check_emboss_translate(sequence, table)

    def translate_all_codons(self, letters):
        if False:
            return 10
        sequence = Seq(''.join((c1 + c3 + c3 for c1 in letters for c2 in letters for c3 in letters)))
        self.check(sequence)

    def test_all_unambig_dna_codons(self):
        if False:
            for i in range(10):
                print('nop')
        'Run transeq vs Bio.Seq on unambiguous DNA codons (inc. alt tables).'
        self.translate_all_codons('ATCGatcg')

    def test_all_unambig_rna_codons(self):
        if False:
            print('Hello World!')
        'Run transeq vs Bio.Seq on unambiguous RNA codons (inc. alt tables).'
        self.translate_all_codons('AUCGaucg')

    def test_mixed_unambig_rna_codons(self):
        if False:
            for i in range(10):
                print('nop')
        'Run transeq vs Bio.Seq on unambiguous DNA/RNA codons (inc. alt tables).'
        self.translate_all_codons('ATUCGatucg')

def clean_up():
    if False:
        print('Hello World!')
    'Fallback clean up method to remove temp files.'
    for filename in os.listdir('Emboss'):
        if filename.startswith('temp_'):
            try:
                os.remove(filename)
            except Exception:
                pass
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)
    clean_up()