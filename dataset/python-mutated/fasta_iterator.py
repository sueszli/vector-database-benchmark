"""Example using Bio.SeqIO to parse a FASTA file."""
from Bio import SeqIO

def extract_organisms(file_to_parse, fmt):
    if False:
        i = 10
        return i + 15
    'Extract species names from sequence description line.'
    all_species = set()
    for cur_record in SeqIO.parse(open(file_to_parse), fmt):
        new_species = cur_record.description.split()[1]
        all_species.add(new_species)
    all_species = sorted(all_species)
    return all_species
if __name__ == '__main__':
    print('Using Bio.SeqIO on a FASTA file')
    all_species = extract_organisms('ls_orchid.fasta', 'fasta')
    print('number of species: %i' % len(all_species))
    print('species names: %s' % all_species)