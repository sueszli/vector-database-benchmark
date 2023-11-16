"""Command line wrapper for the multiple alignment program PRANK."""
from Bio.Application import _Option, _Switch, AbstractCommandline

class PrankCommandline(AbstractCommandline):
    """Command line wrapper for the multiple alignment program PRANK.

    http://www.ebi.ac.uk/goldman-srv/prank/prank/

    Notes
    -----
    Last checked against version: 081202

    References
    ----------
    Loytynoja, A. and Goldman, N. 2005. An algorithm for progressive
    multiple alignment of sequences with insertions. Proceedings of
    the National Academy of Sciences, 102: 10557--10562.

    Loytynoja, A. and Goldman, N. 2008. Phylogeny-aware gap placement
    prevents errors in sequence alignment and evolutionary analysis.
    Science, 320: 1632.

    Examples
    --------
    To align a FASTA file (unaligned.fasta) with the output in aligned
    FASTA format with the output filename starting with "aligned" (you
    can't pick the filename explicitly), no tree output and no XML output,
    use:

    >>> from Bio.Align.Applications import PrankCommandline
    >>> prank_cline = PrankCommandline(d="unaligned.fasta",
    ...                                o="aligned", # prefix only!
    ...                                f=8, # FASTA output
    ...                                notree=True, noxml=True)
    >>> print(prank_cline)
    prank -d=unaligned.fasta -o=aligned -f=8 -noxml -notree

    You would typically run the command line with prank_cline() or via
    the Python subprocess module, as described in the Biopython tutorial.

    """

    def __init__(self, cmd='prank', **kwargs):
        if False:
            return 10
        'Initialize the class.'
        OUTPUT_FORMAT_VALUES = list(range(1, 18))
        self.parameters = [_Option(['-d', 'd'], 'Input filename', filename=True, is_required=True), _Option(['-t', 't'], 'Input guide tree filename', filename=True), _Option(['-tree', 'tree'], 'Input guide tree as Newick string'), _Option(['-m', 'm'], 'User-defined alignment model filename. Default: HKY2/WAG'), _Option(['-o', 'o'], "Output filenames prefix. Default: 'output'\n Will write: output.?.fas (depending on requested format), output.?.xml and output.?.dnd", filename=True), _Option(['-f', 'f'], 'Output alignment format. Default: 8 FASTA\nOption are:\n1. IG/Stanford\t8. Pearson/Fasta\n2. GenBank/GB \t11. Phylip3.2\n3. NBRF       \t12. Phylip\n4. EMBL       \t14. PIR/CODATA\n6. DNAStrider \t15. MSF\n7. Fitch      \t17. PAUP/NEXUS', checker_function=lambda x: x in OUTPUT_FORMAT_VALUES), _Switch(['-noxml', 'noxml'], 'Do not output XML files (PRANK versions earlier than v.120626)'), _Switch(['-notree', 'notree'], 'Do not output dnd tree files (PRANK versions earlier than v.120626)'), _Switch(['-showxml', 'showxml'], 'Output XML files (PRANK v.120626 and later)'), _Switch(['-showtree', 'showtree'], 'Output dnd tree files (PRANK v.120626 and later)'), _Switch(['-shortnames', 'shortnames'], 'Truncate names at first space'), _Switch(['-quiet', 'quiet'], 'Reduce verbosity'), _Switch(['-F', '+F', 'F'], 'Force insertions to be always skipped: same as +F'), _Switch(['-dots', 'dots'], 'Show insertion gaps as dots'), _Option(['-gaprate', 'gaprate'], 'Gap opening rate. Default: dna 0.025 prot 0.0025', checker_function=lambda x: isinstance(x, float)), _Option(['-gapext', 'gapext'], 'Gap extension probability. Default: dna 0.5 / prot 0.5', checker_function=lambda x: isinstance(x, float)), _Option(['-dnafreqs', 'dnafreqs'], "DNA frequencies - 'A,C,G,T'. eg '25,25,25,25' as a quote surrounded string value. Default: empirical", checker_function=lambda x: isinstance(x, bytes)), _Option(['-kappa', 'kappa'], 'Transition/transversion ratio. Default: 2', checker_function=lambda x: isinstance(x, int)), _Option(['-rho', 'rho'], 'Purine/pyrimidine ratio. Default: 1', checker_function=lambda x: isinstance(x, int)), _Switch(['-codon', 'codon'], 'Codon aware alignment or not'), _Switch(['-termgap', 'termgap'], 'Penalise terminal gaps normally'), _Switch(['-nopost', 'nopost'], 'Do not compute posterior support. Default: compute'), _Option(['-pwdist', 'pwdist'], 'Expected pairwise distance for computing guidetree. Default: dna 0.25 / prot 0.5', checker_function=lambda x: isinstance(x, float)), _Switch(['-once', 'once'], 'Run only once. Default: twice if no guidetree given'), _Switch(['-twice', 'twice'], 'Always run twice'), _Switch(['-skipins', 'skipins'], 'Skip insertions in posterior support'), _Switch(['-uselogs', 'uselogs'], 'Slower but should work for a greater number of sequences'), _Switch(['-writeanc', 'writeanc'], 'Output ancestral sequences'), _Switch(['-printnodes', 'printnodes'], 'Output each node; mostly for debugging'), _Option(['-matresize', 'matresize'], 'Matrix resizing multiplier', checker_function=lambda x: isinstance(x, (float, int))), _Option(['-matinitsize', 'matinitsize'], 'Matrix initial size multiplier', checker_function=lambda x: isinstance(x, (float, int))), _Switch(['-longseq', 'longseq'], 'Save space in pairwise alignments'), _Switch(['-pwgenomic', 'pwgenomic'], 'Do pairwise alignment, no guidetree'), _Option(['-pwgenomicdist', 'pwgenomicdist'], 'Distance for pairwise alignment. Default: 0.3', checker_function=lambda x: isinstance(x, float)), _Option(['-scalebranches', 'scalebranches'], 'Scale branch lengths. Default: dna 1 / prot 2', checker_function=lambda x: isinstance(x, int)), _Option(['-fixedbranches', 'fixedbranches'], 'Use fixed branch lengths of input value', checker_function=lambda x: isinstance(x, float)), _Option(['-maxbranches', 'maxbranches'], 'Use maximum branch lengths of input value', checker_function=lambda x: isinstance(x, float)), _Switch(['-realbranches', 'realbranches'], 'Disable branch length truncation'), _Switch(['-translate', 'translate'], 'Translate to protein'), _Switch(['-mttranslate', 'mttranslate'], 'Translate to protein using mt table'), _Switch(['-convert', 'convert'], 'Convert input alignment to new format. Do not perform alignment')]
        AbstractCommandline.__init__(self, cmd, **kwargs)
if __name__ == '__main__':
    from Bio._utils import run_doctest
    run_doctest()