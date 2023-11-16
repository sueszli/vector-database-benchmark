"""Command line wrapper for the multiple alignment programme MAFFT."""
from Bio.Application import _Option, _Switch, _Argument, AbstractCommandline

class MafftCommandline(AbstractCommandline):
    """Command line wrapper for the multiple alignment program MAFFT.

    http://align.bmr.kyushu-u.ac.jp/mafft/software/

    Notes
    -----
    Last checked against version: MAFFT v6.717b (2009/12/03)

    References
    ----------
    Katoh, Toh (BMC Bioinformatics 9:212, 2008) Improved accuracy of
    multiple ncRNA alignment by incorporating structural information into
    a MAFFT-based framework (describes RNA structural alignment methods)

    Katoh, Toh (Briefings in Bioinformatics 9:286-298, 2008) Recent
    developments in the MAFFT multiple sequence alignment program
    (outlines version 6)

    Katoh, Toh (Bioinformatics 23:372-374, 2007)  Errata PartTree: an
    algorithm to build an approximate tree from a large number of
    unaligned sequences (describes the PartTree algorithm)

    Katoh, Kuma, Toh, Miyata (Nucleic Acids Res. 33:511-518, 2005) MAFFT
    version 5: improvement in accuracy of multiple sequence alignment
    (describes [ancestral versions of] the G-INS-i, L-INS-i and E-INS-i
    strategies)

    Katoh, Misawa, Kuma, Miyata (Nucleic Acids Res. 30:3059-3066, 2002)

    Examples
    --------
    >>> from Bio.Align.Applications import MafftCommandline
    >>> mafft_exe = "/opt/local/mafft"
    >>> in_file = "../Doc/examples/opuntia.fasta"
    >>> mafft_cline = MafftCommandline(mafft_exe, input=in_file)
    >>> print(mafft_cline)
    /opt/local/mafft ../Doc/examples/opuntia.fasta

    If the mafft binary is on the path (typically the case on a Unix style
    operating system) then you don't need to supply the executable location:

    >>> from Bio.Align.Applications import MafftCommandline
    >>> in_file = "../Doc/examples/opuntia.fasta"
    >>> mafft_cline = MafftCommandline(input=in_file)
    >>> print(mafft_cline)
    mafft ../Doc/examples/opuntia.fasta

    You would typically run the command line with mafft_cline() or via
    the Python subprocess module, as described in the Biopython tutorial.

    Note that MAFFT will write the alignment to stdout, which you may
    want to save to a file and then parse, e.g.::

        stdout, stderr = mafft_cline()
        with open("aligned.fasta", "w") as handle:
            handle.write(stdout)
        from Bio import AlignIO
        align = AlignIO.read("aligned.fasta", "fasta")

    Alternatively, to parse the output with AlignIO directly you can
    use StringIO to turn the string into a handle::

        stdout, stderr = mafft_cline()
        from io import StringIO
        from Bio import AlignIO
        align = AlignIO.read(StringIO(stdout), "fasta")

    """

    def __init__(self, cmd='mafft', **kwargs):
        if False:
            return 10
        'Initialize the class.'
        BLOSUM_MATRICES = ['30', '45', '62', '80']
        self.parameters = [_Switch(['--auto', 'auto'], 'Automatically select strategy. Default off.'), _Switch(['--6merpair', '6merpair', 'sixmerpair'], 'Distance is calculated based on the number of shared 6mers. Default: on'), _Switch(['--globalpair', 'globalpair'], 'All pairwise alignments are computed with the Needleman-Wunsch algorithm. Default: off'), _Switch(['--localpair', 'localpair'], 'All pairwise alignments are computed with the Smith-Waterman algorithm. Default: off'), _Switch(['--genafpair', 'genafpair'], 'All pairwise alignments are computed with a local algorithm with the generalized affine gap cost (Altschul 1998). Default: off'), _Switch(['--fastapair', 'fastapair'], 'All pairwise alignments are computed with FASTA (Pearson and Lipman 1988). Default: off'), _Option(['--weighti', 'weighti'], 'Weighting factor for the consistency term calculated from pairwise alignments. Default: 2.7', checker_function=lambda x: isinstance(x, float), equate=False), _Option(['--retree', 'retree'], 'Guide tree is built number times in the progressive stage. Valid with 6mer distance. Default: 2', checker_function=lambda x: isinstance(x, int), equate=False), _Option(['--maxiterate', 'maxiterate'], 'Number cycles of iterative refinement are performed. Default: 0', checker_function=lambda x: isinstance(x, int), equate=False), _Option(['--thread', 'thread'], 'Number of threads to use. Default: 1', checker_function=lambda x: isinstance(x, int), equate=False), _Switch(['--fft', 'fft'], 'Use FFT approximation in group-to-group alignment. Default: on'), _Switch(['--nofft', 'nofft'], 'Do not use FFT approximation in group-to-group alignment. Default: off'), _Switch(['--noscore', 'noscore'], 'Alignment score is not checked in the iterative refinement stage. Default: off (score is checked)'), _Switch(['--memsave', 'memsave'], 'Use the Myers-Miller (1988) algorithm. Default: automatically turned on when the alignment length exceeds 10,000 (aa/nt).'), _Switch(['--parttree', 'parttree'], 'Use a fast tree-building method with the 6mer distance. Default: off'), _Switch(['--dpparttree', 'dpparttree'], 'The PartTree algorithm is used with distances based on DP. Default: off'), _Switch(['--fastaparttree', 'fastaparttree'], 'The PartTree algorithm is used with distances based on FASTA. Default: off'), _Option(['--partsize', 'partsize'], 'The number of partitions in the PartTree algorithm. Default: 50', checker_function=lambda x: isinstance(x, int), equate=False), _Switch(['--groupsize', 'groupsize'], 'Do not make alignment larger than number sequences. Default: the number of input sequences'), _Switch(['--adjustdirection', 'adjustdirection'], 'Adjust direction according to the first sequence. Default off.'), _Switch(['--adjustdirectionaccurately', 'adjustdirectionaccurately'], 'Adjust direction according to the first sequence,for highly diverged data; very slowDefault off.'), _Option(['--op', 'op'], 'Gap opening penalty at group-to-group alignment. Default: 1.53', checker_function=lambda x: isinstance(x, float), equate=False), _Option(['--ep', 'ep'], 'Offset value, which works like gap extension penalty, for group-to- group alignment. Default: 0.123', checker_function=lambda x: isinstance(x, float), equate=False), _Option(['--lop', 'lop'], 'Gap opening penalty at local pairwise alignment. Default: 0.123', checker_function=lambda x: isinstance(x, float), equate=False), _Option(['--lep', 'lep'], 'Offset value at local pairwise alignment. Default: 0.1', checker_function=lambda x: isinstance(x, float), equate=False), _Option(['--lexp', 'lexp'], 'Gap extension penalty at local pairwise alignment. Default: -0.1', checker_function=lambda x: isinstance(x, float), equate=False), _Option(['--LOP', 'LOP'], 'Gap opening penalty to skip the alignment. Default: -6.00', checker_function=lambda x: isinstance(x, float), equate=False), _Option(['--LEXP', 'LEXP'], 'Gap extension penalty to skip the alignment. Default: 0.00', checker_function=lambda x: isinstance(x, float), equate=False), _Option(['--bl', 'bl'], 'BLOSUM number matrix is used. Default: 62', checker_function=lambda x: x in BLOSUM_MATRICES, equate=False), _Option(['--jtt', 'jtt'], 'JTT PAM number (Jones et al. 1992) matrix is used. number>0. Default: BLOSUM62', equate=False), _Option(['--tm', 'tm'], 'Transmembrane PAM number (Jones et al. 1994) matrix is used. number>0. Default: BLOSUM62', filename=True, equate=False), _Option(['--aamatrix', 'aamatrix'], 'Use a user-defined AA scoring matrix. Default: BLOSUM62', filename=True, equate=False), _Switch(['--fmodel', 'fmodel'], 'Incorporate the AA/nuc composition information into the scoring matrix (True) or not (False, default)'), _Option(['--namelength', 'namelength'], 'Name length in CLUSTAL and PHYLIP output.\n\n                    MAFFT v6.847 (2011) added --namelength for use with\n                    the --clustalout option for CLUSTAL output.\n\n                    MAFFT v7.024 (2013) added support for this with the\n                    --phylipout option for PHYLIP output (default 10).\n                    ', checker_function=lambda x: isinstance(x, int), equate=False), _Switch(['--clustalout', 'clustalout'], 'Output format: clustal (True) or fasta (False, default)'), _Switch(['--phylipout', 'phylipout'], 'Output format: phylip (True), or fasta (False, default)'), _Switch(['--inputorder', 'inputorder'], 'Output order: same as input (True, default) or alignment based (False)'), _Switch(['--reorder', 'reorder'], 'Output order: aligned (True) or in input order (False, default)'), _Switch(['--treeout', 'treeout'], 'Guide tree is output to the input.tree file (True) or not (False, default)'), _Switch(['--quiet', 'quiet'], 'Do not report progress (True) or not (False, default).'), _Switch(['--nuc', 'nuc'], 'Assume the sequences are nucleotide (True/False). Default: auto'), _Switch(['--amino', 'amino'], 'Assume the sequences are amino acid (True/False). Default: auto'), _Option(['--seed', 'seed'], 'Seed alignments given in alignment_n (fasta format) are aligned with sequences in input.', filename=True, equate=False), _Argument(['input'], 'Input file name', filename=True, is_required=True), _Argument(['input1'], 'Second input file name for the mafft-profile command', filename=True)]
        AbstractCommandline.__init__(self, cmd, **kwargs)
if __name__ == '__main__':
    from Bio._utils import run_doctest
    run_doctest()