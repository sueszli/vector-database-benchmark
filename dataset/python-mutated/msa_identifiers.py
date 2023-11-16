"""Utilities for extracting identifiers from MSA sequence descriptions."""
import dataclasses
import re
from typing import Optional
_UNIPROT_PATTERN = re.compile('\n        ^\n        # UniProtKB/TrEMBL or UniProtKB/Swiss-Prot\n        (?:tr|sp)\n        \\|\n        # A primary accession number of the UniProtKB entry.\n        (?P<AccessionIdentifier>[A-Za-z0-9]{6,10})\n        # Occasionally there is a _0 or _1 isoform suffix, which we ignore.\n        (?:_\\d)?\n        \\|\n        # TREMBL repeats the accession ID here. Swiss-Prot has a mnemonic\n        # protein ID code.\n        (?:[A-Za-z0-9]+)\n        _\n        # A mnemonic species identification code.\n        (?P<SpeciesIdentifier>([A-Za-z0-9]){1,5})\n        # Small BFD uses a final value after an underscore, which we ignore.\n        (?:_\\d+)?\n        $\n        ', re.VERBOSE)

@dataclasses.dataclass(frozen=True)
class Identifiers:
    species_id: str = ''

def _parse_sequence_identifier(msa_sequence_identifier: str) -> Identifiers:
    if False:
        print('Hello World!')
    'Gets accession id and species from an msa sequence identifier.\n\n    The sequence identifier has the format specified by\n    _UNIPROT_TREMBL_ENTRY_NAME_PATTERN or _UNIPROT_SWISSPROT_ENTRY_NAME_PATTERN.\n    An example of a sequence identifier: `tr|A0A146SKV9|A0A146SKV9_FUNHE`\n\n    Args:\n        msa_sequence_identifier: a sequence identifier.\n\n    Returns:\n        An `Identifiers` instance with a species_id. These\n        can be empty in the case where no identifier was found.\n    '
    matches = re.search(_UNIPROT_PATTERN, msa_sequence_identifier.strip())
    if matches:
        return Identifiers(species_id=matches.group('SpeciesIdentifier'))
    return Identifiers()

def _extract_sequence_identifier(description: str) -> Optional[str]:
    if False:
        return 10
    'Extracts sequence identifier from description. Returns None if no match.'
    split_description = description.split()
    if split_description:
        return split_description[0].partition('/')[0]
    else:
        return None

def get_identifiers(description: str) -> Identifiers:
    if False:
        for i in range(10):
            print('nop')
    'Computes extra MSA features from the description.'
    sequence_identifier = _extract_sequence_identifier(description)
    if sequence_identifier is None:
        return Identifiers()
    else:
        return _parse_sequence_identifier(sequence_identifier)