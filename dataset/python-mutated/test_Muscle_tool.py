"""Tests for Muscle tool."""
import os
import sys
import subprocess
import unittest
from Bio.Application import _escape_filename
from Bio import MissingExternalDependencyError
from Bio.Align.Applications import MuscleCommandline
from Bio import SeqIO
from Bio import AlignIO
os.environ['LANG'] = 'C'
muscle_exe = None
if sys.platform == 'win32':
    try:
        prog_files = os.environ['PROGRAMFILES']
    except KeyError:
        prog_files = 'C:\\Program Files'
    likely_dirs = ['', prog_files, os.path.join(prog_files, 'Muscle3.6'), os.path.join(prog_files, 'Muscle3.7'), os.path.join(prog_files, 'Muscle3.8'), os.path.join(prog_files, 'Muscle3.9'), os.path.join(prog_files, 'Muscle')] + sys.path
    for folder in likely_dirs:
        if os.path.isdir(folder):
            if os.path.isfile(os.path.join(folder, 'muscle.exe')):
                muscle_exe = os.path.join(folder, 'muscle.exe')
                break
        if muscle_exe:
            break
else:
    from subprocess import getoutput
    output = getoutput('muscle -version')
    if 'not found' not in output and 'not recognized' not in output:
        if 'MUSCLE' in output and 'Edgar' in output:
            muscle_exe = 'muscle'
if not muscle_exe:
    raise MissingExternalDependencyError('Install MUSCLE if you want to use the Bio.Align.Applications wrapper.')

class MuscleApplication(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.infile1 = 'Fasta/f002'
        self.infile2 = 'Fasta/fa01'
        self.infile3 = 'Fasta/f001'
        self.outfile1 = 'Fasta/temp align out1.fa'
        self.outfile2 = 'Fasta/temp_align_out2.fa'
        self.outfile3 = 'Fasta/temp_align_out3.fa'
        self.outfile4 = 'Fasta/temp_align_out4.fa'

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        if os.path.isfile(self.outfile1):
            os.remove(self.outfile1)
        if os.path.isfile(self.outfile2):
            os.remove(self.outfile2)
        if os.path.isfile(self.outfile3):
            os.remove(self.outfile3)
        if os.path.isfile(self.outfile4):
            os.remove(self.outfile4)

    def test_Muscle_simple(self):
        if False:
            i = 10
            return i + 15
        'Simple round-trip through app just infile and outfile.'
        cmdline = MuscleCommandline(muscle_exe, input=self.infile1, out=self.outfile1)
        self.assertEqual(str(cmdline), _escape_filename(muscle_exe) + ' -in Fasta/f002 -out "Fasta/temp align out1.fa"')
        self.assertEqual(str(eval(repr(cmdline))), str(cmdline))
        (output, error) = cmdline()
        self.assertEqual(output, '')
        self.assertNotIn('ERROR', error)

    def test_Muscle_with_options(self):
        if False:
            return 10
        'Round-trip through app with a switch and valued option.'
        cmdline = MuscleCommandline(muscle_exe)
        cmdline.set_parameter('input', self.infile1)
        cmdline.set_parameter('out', self.outfile2)
        cmdline.objscore = 'sp'
        cmdline.noanchors = True
        self.assertEqual(str(cmdline), _escape_filename(muscle_exe) + ' -in Fasta/f002 -out Fasta/temp_align_out2.fa -objscore sp -noanchors')
        self.assertEqual(str(eval(repr(cmdline))), str(cmdline))
        (output, error) = cmdline()
        self.assertEqual(output, '')
        self.assertNotIn('ERROR', error)
        self.assertTrue(error.strip().startswith('MUSCLE'), output)

    def test_Muscle_profile_simple(self):
        if False:
            for i in range(10):
                print('nop')
        'Simple round-trip through app doing a profile alignment.'
        cmdline = MuscleCommandline(muscle_exe)
        cmdline.set_parameter('out', self.outfile3)
        cmdline.set_parameter('profile', True)
        cmdline.set_parameter('in1', self.infile2)
        cmdline.set_parameter('in2', self.infile3)
        self.assertEqual(str(cmdline), _escape_filename(muscle_exe) + ' -out Fasta/temp_align_out3.fa -profile -in1 Fasta/fa01 -in2 Fasta/f001')
        self.assertEqual(str(eval(repr(cmdline))), str(cmdline))
        (output, error) = cmdline()
        self.assertEqual(output, '')
        self.assertNotIn('ERROR', error)
        self.assertTrue(error.strip().startswith('MUSCLE'), output)

    def test_Muscle_profile_with_options(self):
        if False:
            while True:
                i = 10
        'Profile alignment, and switch and valued options.'
        cmdline = MuscleCommandline(muscle_exe, out=self.outfile4, in1=self.infile2, in2=self.infile3, profile=True, stable=True, cluster1='neighborjoining')
        self.assertEqual(str(cmdline), _escape_filename(muscle_exe) + ' -out Fasta/temp_align_out4.fa -profile -in1 Fasta/fa01 -in2 Fasta/f001' + ' -cluster1 neighborjoining -stable')
        self.assertEqual(str(eval(repr(cmdline))), str(cmdline))
        '\n        #TODO - Why doesn\'t this work with MUSCLE 3.6 on the Mac?\n        #It may be another bug fixed in MUSCLE 3.7 ...\n        result, stdout, stderr = generic_run(cmdline)\n        #NOTE: generic_run has been removed from Biopython\n        self.assertEqual(result.return_code, 0)\n        self.assertEqual(stdout.read(), "")\n        self.assertNotIn("ERROR", stderr.read())\n        self.assertEqual(str(result._cl), str(cmdline))\n        '

class SimpleAlignTest(unittest.TestCase):
    """Simple MUSCLE tests."""
    '\n    #FASTA output seems broken on Muscle 3.6 (on the Mac).\n    def test_simple_fasta(self):\n        input_file = "Fasta/f002"\n        self.assertTrue(os.path.isfile(input_file))\n        records = list(SeqIO.parse(input_file,"fasta"))\n        #Prepare the command...\n        cmdline = MuscleCommandline(muscle_exe)\n        cmdline.set_parameter("in", input_file)\n        #Preserve input record order (makes checking output easier)\n        cmdline.set_parameter("stable")\n        #Set some others options just to test them\n        cmdline.set_parameter("maxiters", 2)\n        self.assertEqual(str(cmdline).rstrip(), "muscle -in Fasta/f002 -maxiters 2 -stable")\n        result, out_handle, err_handle = generic_run(cmdline)\n        #NOTE: generic_run has been removed from Biopython\n        print(err_handle.read())\n        print(out_handle.read())\n        align = AlignIO.read(out_handle, "fasta")\n        self.assertEqual(len(records),len(align))\n        for old, new in zip(records, align):\n            self.assertEqual(old.id, new.id)\n            self.assertEqual(str(new.seq).replace("-",""), old.seq)\n    '

    def test_simple_msf(self):
        if False:
            print('Hello World!')
        'Simple muscle call using MSF output.'
        input_file = 'Fasta/f002'
        self.assertTrue(os.path.isfile(input_file))
        records = list(SeqIO.parse(input_file, 'fasta'))
        records.sort(key=lambda rec: rec.id)
        cmdline = MuscleCommandline(muscle_exe, input=input_file, msf=True)
        self.assertEqual(str(cmdline).rstrip(), _escape_filename(muscle_exe) + ' -in Fasta/f002 -msf')
        self.assertEqual(str(eval(repr(cmdline))), str(cmdline))
        child = subprocess.Popen(str(cmdline), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=sys.platform != 'win32')
        align = AlignIO.read(child.stdout, 'msf')
        align.sort()
        self.assertTrue(child.stderr.read().strip().startswith('MUSCLE'))
        return_code = child.wait()
        self.assertEqual(return_code, 0)
        child.stdout.close()
        child.stderr.close()
        del child
        self.assertEqual(len(records), len(align))
        for (old, new) in zip(records, align):
            self.assertEqual(old.id, new.id)
            self.assertEqual(str(new.seq).replace('-', ''), old.seq)

    def test_simple_clustal(self):
        if False:
            print('Hello World!')
        'Simple muscle call using Clustal output with a MUSCLE header.'
        input_file = 'Fasta/f002'
        self.assertTrue(os.path.isfile(input_file))
        records = list(SeqIO.parse(input_file, 'fasta'))
        records.sort(key=lambda rec: rec.id)
        cmdline = MuscleCommandline(muscle_exe, input=input_file, clw=True)
        self.assertEqual(str(cmdline).rstrip(), _escape_filename(muscle_exe) + ' -in Fasta/f002 -clw')
        self.assertEqual(str(eval(repr(cmdline))), str(cmdline))
        child = subprocess.Popen(str(cmdline), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=sys.platform != 'win32')
        align = AlignIO.read(child.stdout, 'clustal')
        align.sort()
        self.assertTrue(child.stderr.read().strip().startswith('MUSCLE'))
        return_code = child.wait()
        self.assertEqual(return_code, 0)
        child.stdout.close()
        child.stderr.close()
        del child
        self.assertEqual(len(records), len(align))
        for (old, new) in zip(records, align):
            self.assertEqual(old.id, new.id)
            self.assertEqual(str(new.seq).replace('-', ''), old.seq)

    def test_simple_clustal_strict(self):
        if False:
            while True:
                i = 10
        'Simple muscle call using strict Clustal output.'
        input_file = 'Fasta/f002'
        self.assertTrue(os.path.isfile(input_file))
        records = list(SeqIO.parse(input_file, 'fasta'))
        records.sort(key=lambda rec: rec.id)
        cmdline = MuscleCommandline(muscle_exe)
        cmdline.set_parameter('in', input_file)
        cmdline.set_parameter('clwstrict', True)
        self.assertEqual(str(cmdline).rstrip(), _escape_filename(muscle_exe) + ' -in Fasta/f002 -clwstrict')
        self.assertEqual(str(eval(repr(cmdline))), str(cmdline))
        child = subprocess.Popen(str(cmdline), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=sys.platform != 'win32')
        align = AlignIO.read(child.stdout, 'clustal')
        align.sort()
        self.assertTrue(child.stderr.read().strip().startswith('MUSCLE'))
        self.assertEqual(len(records), len(align))
        for (old, new) in zip(records, align):
            self.assertEqual(old.id, new.id)
            self.assertEqual(str(new.seq).replace('-', ''), old.seq)
        return_code = child.wait()
        self.assertEqual(return_code, 0)
        child.stdout.close()
        child.stderr.close()
        del child

    def test_long(self):
        if False:
            i = 10
            return i + 15
        'Simple muscle call using long file.'
        temp_large_fasta_file = 'temp_cw_prot.fasta'
        records = list(SeqIO.parse('NBRF/Cw_prot.pir', 'pir'))[:40]
        SeqIO.write(records, temp_large_fasta_file, 'fasta')
        cmdline = MuscleCommandline(muscle_exe)
        cmdline.set_parameter('in', temp_large_fasta_file)
        cmdline.set_parameter('maxiters', 1)
        cmdline.set_parameter('diags', True)
        cmdline.set_parameter('clwstrict', True)
        cmdline.set_parameter('maxhours', 0.1)
        cmdline.set_parameter('quiet', True)
        self.assertEqual(str(cmdline).rstrip(), _escape_filename(muscle_exe) + ' -in temp_cw_prot.fasta -diags -maxhours 0.1' + ' -maxiters 1 -clwstrict -quiet')
        self.assertEqual(str(eval(repr(cmdline))), str(cmdline))
        child = subprocess.Popen(str(cmdline), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=sys.platform != 'win32')
        align = AlignIO.read(child.stdout, 'clustal')
        align.sort()
        records.sort(key=lambda rec: rec.id)
        self.assertEqual(len(records), len(align))
        for (old, new) in zip(records, align):
            self.assertEqual(old.id, new.id)
            self.assertEqual(str(new.seq).replace('-', ''), old.seq)
        self.assertEqual('', child.stderr.read().strip())
        return_code = child.wait()
        self.assertEqual(return_code, 0)
        child.stdout.close()
        child.stderr.close()
        del child
        os.remove(temp_large_fasta_file)

    def test_using_stdin(self):
        if False:
            for i in range(10):
                print('nop')
        'Simple alignment using stdin.'
        input_file = 'Fasta/f002'
        self.assertTrue(os.path.isfile(input_file))
        records = list(SeqIO.parse(input_file, 'fasta'))
        cline = MuscleCommandline(muscle_exe, clw=True)
        self.assertEqual(str(cline).rstrip(), _escape_filename(muscle_exe) + ' -clw')
        self.assertEqual(str(eval(repr(cline))), str(cline))
        child = subprocess.Popen(str(cline), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=sys.platform != 'win32')
        SeqIO.write(records, child.stdin, 'fasta')
        child.stdin.close()
        align = AlignIO.read(child.stdout, 'clustal')
        align.sort()
        records.sort(key=lambda rec: rec.id)
        self.assertEqual(len(records), len(align))
        for (old, new) in zip(records, align):
            self.assertEqual(old.id, new.id)
            self.assertEqual(str(new.seq).replace('-', ''), old.seq)
        self.assertEqual(0, child.wait())
        child.stdout.close()
        child.stderr.close()
        del child

    def test_with_multiple_output_formats(self):
        if False:
            i = 10
            return i + 15
        'Simple muscle call with multiple output formats.'
        input_file = 'Fasta/f002'
        output_html = 'temp_f002.html'
        output_clwstrict = 'temp_f002.clw'
        self.assertTrue(os.path.isfile(input_file))
        records = list(SeqIO.parse(input_file, 'fasta'))
        records.sort(key=lambda rec: rec.id)
        cmdline = MuscleCommandline(muscle_exe, input=input_file, clw=True, htmlout=output_html, clwstrictout=output_clwstrict)
        self.assertEqual(str(cmdline).rstrip(), _escape_filename(muscle_exe) + ' -in Fasta/f002 -clw -htmlout temp_f002.html' + ' -clwstrictout temp_f002.clw')
        self.assertEqual(str(eval(repr(cmdline))), str(cmdline))
        child = subprocess.Popen(str(cmdline), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=sys.platform != 'win32')
        align = AlignIO.read(child.stdout, 'clustal')
        align.sort()
        self.assertTrue(child.stderr.read().strip().startswith('MUSCLE'))
        return_code = child.wait()
        self.assertEqual(return_code, 0)
        self.assertEqual(len(records), len(align))
        for (old, new) in zip(records, align):
            self.assertEqual(old.id, new.id)
        child.stdout.close()
        child.stderr.close()
        del child
        handle = open(output_html)
        html = handle.read().strip().upper()
        handle.close()
        self.assertTrue(html.startswith('<HTML'))
        self.assertTrue(html.endswith('</HTML>'))
        align = AlignIO.read(output_clwstrict, 'clustal')
        align.sort()
        self.assertEqual(len(records), len(align))
        for (old, new) in zip(records, align):
            self.assertEqual(old.id, new.id)
        os.remove(output_html)
        os.remove(output_clwstrict)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)