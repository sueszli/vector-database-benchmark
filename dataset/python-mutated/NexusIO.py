"""Bio.AlignIO support for the "nexus" file format.

You are expected to use this module via the Bio.AlignIO functions (or the
Bio.SeqIO functions if you want to work directly with the gapped sequences).

See also the Bio.Nexus module (which this code calls internally),
as this offers more than just accessing the alignment or its
sequences as SeqRecord objects.
"""
from typing import IO, Iterator, Optional
from Bio.Align import MultipleSeqAlignment
from Bio.AlignIO.Interfaces import AlignmentWriter
from Bio.Nexus import Nexus
from Bio.SeqRecord import SeqRecord

def NexusIterator(handle: IO[str], seq_count: Optional[int]=None) -> Iterator[MultipleSeqAlignment]:
    if False:
        print('Hello World!')
    'Return SeqRecord objects from a Nexus file.\n\n    Thus uses the Bio.Nexus module to do the hard work.\n\n    You are expected to call this function via Bio.SeqIO or Bio.AlignIO\n    (and not use it directly).\n\n    NOTE - We only expect ONE alignment matrix per Nexus file,\n    meaning this iterator will only yield one MultipleSeqAlignment.\n    '
    n = Nexus.Nexus(handle)
    if not n.matrix:
        return
    assert len(n.unaltered_taxlabels) == len(n.taxlabels)
    if seq_count and seq_count != len(n.unaltered_taxlabels):
        raise ValueError('Found %i sequences, but seq_count=%i' % (len(n.unaltered_taxlabels), seq_count))
    annotations: Optional[SeqRecord._AnnotationsDict]
    if n.datatype in ('dna', 'nucleotide'):
        annotations = {'molecule_type': 'DNA'}
    elif n.datatype == 'rna':
        annotations = {'molecule_type': 'RNA'}
    elif n.datatype == 'protein':
        annotations = {'molecule_type': 'protein'}
    else:
        annotations = None
    records = (SeqRecord(n.matrix[new_name], id=new_name, name=old_name, description='', annotations=annotations) for (old_name, new_name) in zip(n.unaltered_taxlabels, n.taxlabels))
    yield MultipleSeqAlignment(records)

class NexusWriter(AlignmentWriter):
    """Nexus alignment writer.

    Note that Nexus files are only expected to hold ONE alignment
    matrix.

    You are expected to call this class via the Bio.AlignIO.write() or
    Bio.SeqIO.write() functions.
    """

    def write_file(self, alignments):
        if False:
            i = 10
            return i + 15
        'Use this to write an entire file containing the given alignments.\n\n        Arguments:\n         - alignments - A list or iterator returning MultipleSeqAlignment objects.\n           This should hold ONE and only one alignment.\n\n        '
        align_iter = iter(alignments)
        try:
            alignment = next(align_iter)
        except StopIteration:
            return 0
        try:
            next(align_iter)
            raise ValueError('We can only write one Alignment to a Nexus file.')
        except StopIteration:
            pass
        self.write_alignment(alignment)
        return 1

    def write_alignment(self, alignment, interleave=None):
        if False:
            for i in range(10):
                print('nop')
        'Write an alignment to file.\n\n        Creates an empty Nexus object, adds the sequences\n        and then gets Nexus to prepare the output.\n        Default interleave behaviour: Interleave if columns > 1000\n        --> Override with interleave=[True/False]\n        '
        if len(alignment) == 0:
            raise ValueError('Must have at least one sequence')
        columns = alignment.get_alignment_length()
        if columns == 0:
            raise ValueError('Non-empty sequences are required')
        datatype = self._classify_mol_type_for_nexus(alignment)
        minimal_record = '#NEXUS\nbegin data; dimensions ntax=0 nchar=0; format datatype=%s; end;' % datatype
        n = Nexus.Nexus(minimal_record)
        for record in alignment:
            if datatype == 'dna' and 'U' in record.seq:
                raise ValueError(f'{record.id} contains U, but DNA alignment')
            elif datatype == 'rna' and 'T' in record.seq:
                raise ValueError(f'{record.id} contains T, but RNA alignment')
            n.add_sequence(record.id, str(record.seq))
        if interleave is None:
            interleave = columns > 1000
        n.write_nexus_data(self.handle, interleave=interleave)

    def _classify_mol_type_for_nexus(self, alignment):
        if False:
            for i in range(10):
                print('nop')
        "Return 'protein', 'dna', or 'rna' based on records' molecule type (PRIVATE).\n\n        All the records must have a molecule_type annotation, and they must\n        agree.\n\n        Raises an exception if this is not possible.\n        "
        values = {_.annotations.get('molecule_type', None) for _ in alignment}
        if all((_ and 'DNA' in _ for _ in values)):
            return 'dna'
        elif all((_ and 'RNA' in _ for _ in values)):
            return 'rna'
        elif all((_ and 'protein' in _ for _ in values)):
            return 'protein'
        else:
            raise ValueError('Need the molecule type to be defined')
if __name__ == '__main__':
    from Bio._utils import run_doctest
    run_doctest(verbose=0)