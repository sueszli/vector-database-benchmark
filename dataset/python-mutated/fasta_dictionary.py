"""Example using Bio.SeqIO to load a FASTA file as a dictionary.

An example function (get_accession_num) is defined to demonstrate
a non-trivial naming scheme where the dictionary key is based on
the record identifier.

The first version uses Bio.SeqIO.parse() and loads the entire
FASTA file into memory as a Python dictionary of SeqRecord
objects. This is *not* suitable for large files.

The second version used Bio.SeqIO.index() which is suitable
for FASTA files with millions of records.

See also Bio.SeqIO.index_db() and the examples in the main tutorial.
"""
from Bio import SeqIO

def get_accession_num(seq_record):
    if False:
        print('Hello World!')
    'Extract accession number from sequence id.'
    accession_atoms = seq_record.id.split('|')
    gb_name = accession_atoms[3]
    return gb_name[:-2]
rec_iterator = SeqIO.parse('ls_orchid.fasta', 'fasta')
orchid_dict = SeqIO.to_dict(rec_iterator, get_accession_num)
for id_num in orchid_dict:
    print('id number: %s' % id_num)
    print('description: %s' % orchid_dict[id_num].description)
    print('sequence: %s' % orchid_dict[id_num].seq)
orchid_dict = SeqIO.index('ls_orchid.fasta', 'fasta')
for id_num in orchid_dict:
    print('id number: %s' % id_num)
    print('description: %s' % orchid_dict[id_num].description)
    print('sequence: %s' % orchid_dict[id_num].seq)