"""Tests for ClustalOmega tool."""
import os
import unittest
from subprocess import getoutput
from Bio import MissingExternalDependencyError
from Bio import SeqIO
from Bio import Align
from Bio.Align.Applications import ClustalOmegaCommandline
from Bio.Application import ApplicationError
os.environ['LANG'] = 'C'
clustalo_exe = None
try:
    output = getoutput('clustalo --help')
    if output.startswith('Clustal Omega'):
        clustalo_exe = 'clustalo'
except FileNotFoundError:
    pass
if not clustalo_exe:
    raise MissingExternalDependencyError('Install clustalo if you want to use Clustal Omega from Biopython.')

class ClustalOmegaTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.files_to_clean = set()

    def tearDown(self):
        if False:
            print('Hello World!')
        for filename in self.files_to_clean:
            if os.path.isfile(filename):
                os.remove(filename)

    def standard_test_procedure(self, cline):
        if False:
            for i in range(10):
                print('nop')
        'Shared test procedure used by all tests.'
        cline.force = True
        self.add_file_to_clean(cline.outfile)
        if cline.guidetree_out:
            self.add_file_to_clean(cline.guidetree_out)
        input_records = SeqIO.to_dict(SeqIO.parse(cline.infile, 'fasta'))
        self.assertEqual(str(eval(repr(cline))), str(cline))
        (output, error) = cline()
        self.assertTrue(not output or output.strip().startswith('CLUSTAL'))
        self.assertTrue(error.strip() == '' or error.startswith(('WARNING: Sequence type is DNA.', 'WARNING: DNA alignment is still experimental.')))
        if cline.guidetree_out:
            self.assertTrue(os.path.isfile(cline.guidetree_out))

    def add_file_to_clean(self, filename):
        if False:
            print('Hello World!')
        'Add a file for deferred removal by the tearDown routine.'
        self.files_to_clean.add(filename)

class ClustalOmegaTestErrorConditions(ClustalOmegaTestCase):

    def test_empty_file(self):
        if False:
            while True:
                i = 10
        'Test an empty file.'
        input_file = 'does_not_exist.fasta'
        self.assertFalse(os.path.isfile(input_file))
        cline = ClustalOmegaCommandline(clustalo_exe, infile=input_file)
        try:
            (stdout, stderr) = cline()
        except ApplicationError as err:
            message = str(err)
            self.assertTrue('Cannot open sequence file' in message or 'Cannot open input file' in message or 'Non-zero return code' in message, message)
        else:
            self.fail(f'Should have failed, returned:\n{stdout}\n{stderr}')

    def test_single_sequence(self):
        if False:
            return 10
        'Test an input file containing a single sequence.'
        input_file = 'Fasta/f001'
        self.assertTrue(os.path.isfile(input_file))
        self.assertEqual(len(list(SeqIO.parse(input_file, 'fasta'))), 1)
        cline = ClustalOmegaCommandline(clustalo_exe, infile=input_file)
        try:
            (stdout, stderr) = cline()
        except ApplicationError as err:
            self.assertIn('contains 1 sequence, nothing to align', str(err))
        else:
            self.fail(f'Should have failed, returned:\n{stdout}\n{stderr}')

    def test_invalid_format(self):
        if False:
            print('Hello World!')
        'Test an input file in an invalid format.'
        input_file = 'Medline/pubmed_result1.txt'
        self.assertTrue(os.path.isfile(input_file))
        cline = ClustalOmegaCommandline(clustalo_exe, infile=input_file)
        with self.assertRaises(ApplicationError) as cm:
            (stdout, stderr) = cline()
            self.fail(f'Should have failed, returned:\n{stdout}\n{stderr}')
        err = str(cm.exception)
        self.assertIn("Can't determine format of sequence file", err)

class ClustalOmegaTestNormalConditions(ClustalOmegaTestCase):

    def test_simple_fasta(self):
        if False:
            i = 10
            return i + 15
        'Test a simple fasta file.'
        input_file = 'Registry/seqs.fasta'
        output_file = 'temp_test.aln'
        cline = ClustalOmegaCommandline(clustalo_exe, infile=input_file, outfile=output_file, outfmt='clustal')
        self.standard_test_procedure(cline)
        alignment = Align.read(cline.outfile, 'clustal')
        self.assertEqual(str(alignment), 'gi|134891         0 GATCCCTACCCTTNCCGTTGGTCTCTNTCGCTGACTCGAGGCACCTAACATCCATTCACA\n                  0 ---------..-........|......|....|......|..............|.----\ngi|129628         0 ---------MP-VVVVASSKGGAGKSTTAVVLGTELAHKGVPVTMLDCDPNRSLTI----\n\ngi|134891        60 CCCAACACAGGCCAGCGACTTCTGGGGCTCAGCCACAGACATGGTTTGTNACTNTTGAGC\n                 60 -----.|.||.......|....|-------------------......|.......||..\ngi|129628        46 -----WANAGEVPENITALSDVT-------------------ESSIVKTIKQHDVDGAVV\n\ngi|134891       120 TTCTGTTCCTAGAGAATCCTAGAGGCTTGATTGGCCCAGGCTGCTGTNTGTNCTGGAGG-\n                120 ...--------..|.|......|..............|...|..............|..-\ngi|129628        82 IVD--------LEGVASRMVSRAISQADLVLIPMRPKALDATIGAQSLQLIAEEEEAIDR\n\ngi|134891       179 -CAAAGAATCCCTACCTCCTAGGGGTGAAAGGAAATNAAAATGGAAAGTTCTTGTAGCGC\n                180 -.|.|...|....|.......|........|------------...........||....\ngi|129628       134 KIAHAVVFTMVSPAIRSHEYTGIKASLIENG------------VEIIEPPLVERTAYSAL\n\ngi|134891       238 AAGGCCTGACATGGGTAGCTGCTCAATAAATGCTAGTNTGTTATTTC 285\n                240 ...|..........|..........|.|-----.|.....|.|..-- 287\ngi|129628       182 FQFGGNLHSMKSKQGNMAAAIENAEAFA-----MAIFKKLTEALR-- 222\n')
        self.assertEqual(alignment.column_annotations['clustal_consensus'], '                    *      *    *      *              *           * **       *    *                         *       **               * *      *              *   *              *     * *   *    *       *        *                       **       *          *          * *      *     * *    ')

    def test_properties(self):
        if False:
            for i in range(10):
                print('nop')
        'Test setting options via properties.'
        input_file = 'Registry/seqs.fasta'
        output_file = 'temp_test.aln'
        cline = ClustalOmegaCommandline(clustalo_exe)
        cline.infile = input_file
        cline.outfile = output_file
        cline.outfmt = 'clustal'
        self.standard_test_procedure(cline)
        alignment = Align.read(cline.outfile, 'clustal')
        self.assertEqual(str(alignment), 'gi|134891         0 GATCCCTACCCTTNCCGTTGGTCTCTNTCGCTGACTCGAGGCACCTAACATCCATTCACA\n                  0 ---------..-........|......|....|......|..............|.----\ngi|129628         0 ---------MP-VVVVASSKGGAGKSTTAVVLGTELAHKGVPVTMLDCDPNRSLTI----\n\ngi|134891        60 CCCAACACAGGCCAGCGACTTCTGGGGCTCAGCCACAGACATGGTTTGTNACTNTTGAGC\n                 60 -----.|.||.......|....|-------------------......|.......||..\ngi|129628        46 -----WANAGEVPENITALSDVT-------------------ESSIVKTIKQHDVDGAVV\n\ngi|134891       120 TTCTGTTCCTAGAGAATCCTAGAGGCTTGATTGGCCCAGGCTGCTGTNTGTNCTGGAGG-\n                120 ...--------..|.|......|..............|...|..............|..-\ngi|129628        82 IVD--------LEGVASRMVSRAISQADLVLIPMRPKALDATIGAQSLQLIAEEEEAIDR\n\ngi|134891       179 -CAAAGAATCCCTACCTCCTAGGGGTGAAAGGAAATNAAAATGGAAAGTTCTTGTAGCGC\n                180 -.|.|...|....|.......|........|------------...........||....\ngi|129628       134 KIAHAVVFTMVSPAIRSHEYTGIKASLIENG------------VEIIEPPLVERTAYSAL\n\ngi|134891       238 AAGGCCTGACATGGGTAGCTGCTCAATAAATGCTAGTNTGTTATTTC 285\n                240 ...|..........|..........|.|-----.|.....|.|..-- 287\ngi|129628       182 FQFGGNLHSMKSKQGNMAAAIENAEAFA-----MAIFKKLTEALR-- 222\n')
        self.assertEqual(alignment.column_annotations['clustal_consensus'], '                    *      *    *      *              *           * **       *    *                         *       **               * *      *              *   *              *     * *   *    *       *        *                       **       *          *          * *      *     * *    ')

    def test_input_filename_with_space(self):
        if False:
            for i in range(10):
                print('nop')
        'Test an input filename containing a space.'
        input_file = 'Clustalw/temp horses.fasta'
        with open(input_file, 'w') as handle:
            SeqIO.write(SeqIO.parse('Phylip/hennigian.phy', 'phylip'), handle, 'fasta')
        output_file = 'temp_test.aln'
        cline = ClustalOmegaCommandline(clustalo_exe, infile=input_file, outfile=output_file, outfmt='clustal')
        self.add_file_to_clean(input_file)
        self.standard_test_procedure(cline)
        alignment = Align.read(cline.outfile, 'clustal')
        self.assertEqual(str(alignment), 'A                 0 -CACACACAAAAAAAAAAACAAAAAAAAAAAAAAAAAAAAA 40\nB                 0 -CACACAACAAAAAAAAAACAAAAAAAAAAAAAAAAAAAAA 40\nC                 0 -CACAACAAAAAAAAAAAACAAAAAAAAAAAAAAAAAAAAA 40\nD                 0 -CAACAAAACAAAAAAAAACAAAAAAAAAAAAAAAAAAAAA 40\nE                 0 -CAACAAAAACAAAAAAAACAAAAAAAAAAAAAAAAAAAAA 40\nF                 0 ACAAAAAAAACACACAAAACAAAAAAAAAAAAAAAAAAAA- 40\nG                 0 ACAAAAAAAACACAACAAACAAAAAAAAAAAAAAAAAAAA- 40\nH                 0 ACAAAAAAAACAACAAAAACAAAAAAAAAAAAAAAAAAAA- 40\nI                 0 ACAAAAAAAAACAAAACAACAAAAAAAAAAAAAAAAAAAA- 40\nJ                 0 ACAAAAAAAAACAAAAACACAAAAAAAAAAAAAAAAAAAA- 40\n')
        self.assertEqual(alignment.column_annotations['clustal_consensus'], ' **               ********************** ')

    def test_output_filename_with_spaces(self):
        if False:
            for i in range(10):
                print('nop')
        'Test an output filename containing spaces.'
        input_file = 'Registry/seqs.fasta'
        output_file = 'temp with spaces.aln'
        cline = ClustalOmegaCommandline(clustalo_exe, infile=input_file, outfile=output_file, outfmt='clustal')
        self.standard_test_procedure(cline)
        alignment = Align.read(cline.outfile, 'clustal')
        self.assertEqual(str(alignment), 'gi|134891         0 GATCCCTACCCTTNCCGTTGGTCTCTNTCGCTGACTCGAGGCACCTAACATCCATTCACA\n                  0 ---------..-........|......|....|......|..............|.----\ngi|129628         0 ---------MP-VVVVASSKGGAGKSTTAVVLGTELAHKGVPVTMLDCDPNRSLTI----\n\ngi|134891        60 CCCAACACAGGCCAGCGACTTCTGGGGCTCAGCCACAGACATGGTTTGTNACTNTTGAGC\n                 60 -----.|.||.......|....|-------------------......|.......||..\ngi|129628        46 -----WANAGEVPENITALSDVT-------------------ESSIVKTIKQHDVDGAVV\n\ngi|134891       120 TTCTGTTCCTAGAGAATCCTAGAGGCTTGATTGGCCCAGGCTGCTGTNTGTNCTGGAGG-\n                120 ...--------..|.|......|..............|...|..............|..-\ngi|129628        82 IVD--------LEGVASRMVSRAISQADLVLIPMRPKALDATIGAQSLQLIAEEEEAIDR\n\ngi|134891       179 -CAAAGAATCCCTACCTCCTAGGGGTGAAAGGAAATNAAAATGGAAAGTTCTTGTAGCGC\n                180 -.|.|...|....|.......|........|------------...........||....\ngi|129628       134 KIAHAVVFTMVSPAIRSHEYTGIKASLIENG------------VEIIEPPLVERTAYSAL\n\ngi|134891       238 AAGGCCTGACATGGGTAGCTGCTCAATAAATGCTAGTNTGTTATTTC 285\n                240 ...|..........|..........|.|-----.|.....|.|..-- 287\ngi|129628       182 FQFGGNLHSMKSKQGNMAAAIENAEAFA-----MAIFKKLTEALR-- 222\n')
        self.assertEqual(alignment.column_annotations['clustal_consensus'], '                    *      *    *      *              *           * **       *    *                         *       **               * *      *              *   *              *     * *   *    *       *        *                       **       *          *          * *      *     * *    ')

    def test_large_fasta_file(self):
        if False:
            while True:
                i = 10
        'Test a large fasta input file.'
        input_file = 'temp_cw_prot.fasta'
        records = list(SeqIO.parse('NBRF/Cw_prot.pir', 'pir'))[:40]
        with open(input_file, 'w') as handle:
            SeqIO.write(records, handle, 'fasta')
        del handle, records
        output_file = 'temp_cw_prot.aln'
        cline = ClustalOmegaCommandline(clustalo_exe, infile=input_file, outfile=output_file, outfmt='clustal')
        self.add_file_to_clean(input_file)
        self.standard_test_procedure(cline)
        alignment = Align.read(cline.outfile, 'clustal')

    def test_newtree_files(self):
        if False:
            while True:
                i = 10
        'Test requesting a guide tree.'
        input_file = 'Fasta/f002'
        output_file = 'temp_test.aln'
        newtree_file = 'temp_test.dnd'
        alignment_text = 'gi|134891         0 CGGACCAGACGGACACAGGGAGAAGCTAGTTTCTTTCATGTGATTGANATNATGACTCTA\ngi|134891         0 ---------CGGAGCCAGCGAGCATAT---------------------------------\ngi|159293         0 ------------------------------------------------------------\n\ngi|134891        60 CTCCTAAAAGGGAAAAANCAATATCCTTGTTTACAGAAGAGAAACAAACAAGCCCCACTC\ngi|134891        18 ----------------------------------------------------GCTGCATG\ngi|159293         0 --------------------------------------------GATCAAATCTGCACTG\n\ngi|134891       120 AGCTCAGTCACAGGAGAGANCACAGAAAGTCTTAGGATCATGANCTCTGAA-AAAAAGAG\ngi|134891        26 -------------------------AGGACCTTTCTATCTTACATTATGGC-TGGGAATC\ngi|159293        16 TGTCTACATATAGGAAAGGTCCTGGTGTGTGCTAATGTTCCCAATGCAGGACTTGAGGAA\n\ngi|134891       179 AAACCTTATCTTTNCTTTGTGGTTCCTTTAAACACACTCACACACACTTGGTCAGAGATG\ngi|134891        60 TTACTCTTTCATCTG-------ATACCTTGTTCAGATTTCAAAATAGTTGTAGCCTTATC\ngi|159293        76 GAGCTCTGTTATATGTTTCCATTTCTCTTTATCAAAGATAACCAAACCTTATGGCCCTT-\n\ngi|134891       239 CTGTGCTTCTTGGAAGCAAGGNCTCAAAGGCAAGGTGCACGC----------AGAGGGAC\ngi|134891       113 CTGGTTTTACAGATGTGAAACTT----TCAAGAGATTTACTGACTTTCCTAGAATA----\ngi|159293       135 ---ATAACAATGGAGGCACTGGCTGCCTCTTAATTTTCAATCATGGACCTAAAGAAGTAC\n\ngi|134891       289 GTTTGA--GTCTGGGATGAAGCATGTNCGTATTATTTATATGATGGAATTTCACGTTTTT\ngi|134891       165 --------GT--------------TTCTCTACTGGAAACCTGATGCTTTTATAAGCCATT\ngi|159293       192 TCTGAAGGGTCTCAACAATGCCAGGTGGGGACAGATATACTCAGAGATTATCCAGGTCTG\n\ngi|134891       347 ATGTNAAGCNTGACAACACCAGGCAGGTATGAGAGGA-AAGCAAGGCCCGTCCATNGCTG\ngi|134891       203 GTGATTAGGATGACTGTTACAGGCTTAGCTTTGTGTGAAANCCAGTCACCTTT------C\ngi|159293       252 CCTCCCAGCGAGCC-----------TGGA------GT-ACACCAGACCCTCCTAGAGAAA\n\ngi|134891       406 TCCGTACNCTTACGGNTTGCTTGTNGGAGNCATTTNGGTATTGTTTGTTGTAANANCCAA\ngi|134891       257 TCCTAGGTAATGAGTAGTGCTGTTCATATTACTNT-------AAGTTCTATAGCATACTT\ngi|159293       294 TCTGTT------------------------------------ATAATTTACCACCCACTT\n\ngi|134891       466 AANGGGCTTTGGNNTGGNAAAA----GGGCAGANNGGGGGGGTTGGTGTNGTTTTTTGG-\ngi|134891       310 GCNATCCTTTANCCATGCTTATCATANGTACCATTTGAGGAATTGNTT-----TGCCCTT\ngi|159293       318 ATCCACCTTTAAACTTGGGGAA----GGNNGCN------TTTCAAATTAAATTTAATCNT\n\ngi|134891       521 GGGGANNNTTTNGATTTGG-------TNCCGGGNTTTNGTTTNCCNCGGNACCGGNTTTT\ngi|134891       365 TTG-GGTTTNTTNTTGGTAA--ANNNTTCCCGGGTGGGGGNGGTNNNGAAA---------\ngi|159293       368 NGGGGGNTTTTAAACTTTAACCCTTTTNCCNTTNTNGGGGTNGGNANTTGNCCCCNTTAA\n\ngi|134891       574 GGTTGGGGNCCATTTNTGNGGGGCNTTGGNGTTNCNTTNCCCNNNTNNGANTGGTTTNA\ngi|134891       413 -----------------------------------------------------------\ngi|159293       428 AGGGGGNNCCCCT-NCNNGGGGGAATAA-AACAA----------NTTNNTTT--TTT--\n\ngi|134891       633\ngi|134891       413\ngi|159293       471\n'
        clustal_consensus = '                                                                                                                      *                                 *    *          *              *  * *  *           *   **   ** *       * *  *         *            *     *              *  *  *             *               **               *    *         * *     *     *   *       **   * *                        *  * ** * *           **                                              *        *        ****      *   *      *                  *      *        *     * *               * **    *   *                                                                                '
        cline = ClustalOmegaCommandline(clustalo_exe, infile=input_file, outfile=output_file, guidetree_out=newtree_file, outfmt='clustal')
        self.standard_test_procedure(cline)
        alignment = Align.read(cline.outfile, 'clustal')
        self.assertEqual(str(alignment), alignment_text)
        self.assertEqual(alignment.column_annotations['clustal_consensus'], clustal_consensus)
        cline.guidetree_out = 'temp with space.dnd'
        self.standard_test_procedure(cline)
        alignment = Align.read(cline.outfile, 'clustal')
        self.assertEqual(str(alignment), alignment_text)
        self.assertEqual(alignment.column_annotations['clustal_consensus'], clustal_consensus)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)