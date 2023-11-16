"""Models for application claims."""
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass

@dataclass
class LedgerEntryClaim:
    """
    LedgerEntryClaim represents an Application Claim derived from ledger entry data.

    :keyword protocol: The protocol used to compute the claim.
    :paramtype protocol: str

    :keyword collectionId: The collection ID of the ledger entry.
    :paramtype collectionId: str

    :keyword contents: The contents of the ledger entry.
    :paramtype contents: str

    :keyword secretKey: The secret key used to compute the claim digest.
    :paramtype secretKey: str
    """
    protocol: str
    collectionId: str
    contents: str
    secretKey: str

    @classmethod
    def from_dict(cls, ledger_entry_claim_dict: Dict[str, Any]):
        if False:
            i = 10
            return i + 15
        'Create a new instance of this class from a dictionary.\n\n        :param dict[str, any] ledger_entry_claim_dict: The dictionary representation of the ledger entry claim.\n        :return: A new instance of this class corresponding to the provided dictionary.\n        :rtype: LedgerEntryClaim\n        '
        return cls(**ledger_entry_claim_dict)

@dataclass
class ClaimDigest:
    """
    ClaimDigest represents an Application Claim in digested form.

    :keyword protocol: The protocol used to compute the claim.
    :paramtype protocol: str

    :keyword value: The digest of the claim.
    :paramtype value: str
    """
    protocol: str
    value: str

    @classmethod
    def from_dict(cls, ledger_entry_claim_dict: Dict[str, Any]):
        if False:
            print('Hello World!')
        'Create a new instance of this class from a dictionary.\n\n        :param dict[str, any] ledger_entry_claim_dict: The dictionary representation of the claim digest.\n        :return: A new instance of this class corresponding to the provided dictionary.\n        :rtype: ClaimDigest\n        '
        return cls(**ledger_entry_claim_dict)

@dataclass
class ApplicationClaim:
    """
    ApplicationClaim represents a claim of a ledger application.

    :keyword kind: The kind of the claim.
    :paramtype kind: str

    :keyword ledgerEntry: The ledger entry claim.
    :paramtype ledgerEntry: Optional[Union[Dict[str, Any], LedgerEntryClaim]]

    :keyword digest: The claim digest object.
    :paramtype digest: Optional[Union[Dict[str, Any], ClaimDigest]]
    """
    kind: str
    ledgerEntry: Optional[LedgerEntryClaim] = None
    digest: Optional[ClaimDigest] = None

    def __init__(self, kind: str, ledgerEntry: Optional[Union[Dict[str, Any], LedgerEntryClaim]]=None, digest: Optional[Union[Dict[str, Any], ClaimDigest]]=None, **kwargs: Any):
        if False:
            for i in range(10):
                print('nop')
        '\n        :keyword kind: The kind of the claim.\n        :paramtype kind: str\n\n        :keyword ledgerEntry: The ledger entry claim.\n        :paramtype ledgerEntry: Optional[Union[Dict[str, Any], LedgerEntryClaim]]\n\n        :keyword digest: The claim digest object.\n        :paramtype digest: Optional[Union[Dict[str, Any], ClaimDigest]]\n        '
        self.kind = kind
        if ledgerEntry:
            if isinstance(ledgerEntry, LedgerEntryClaim):
                self.ledgerEntry = ledgerEntry
            else:
                self.ledgerEntry = LedgerEntryClaim.from_dict(ledgerEntry)
        else:
            self.ledgerEntry = None
        if digest:
            if isinstance(digest, ClaimDigest):
                self.digest = digest
            else:
                self.digest = ClaimDigest.from_dict(digest)
        else:
            self.digest = None
        self.kwargs = kwargs

    @classmethod
    def from_dict(cls, claim_dict: Dict[str, Any]):
        if False:
            return 10
        'Create a new instance of this class from a dictionary.\n\n        :param dict[str, any] claim_dict: The dictionary representation of the application claim.\n        :return: A new instance of this class corresponding to the provided dictionary.\n        :rtype: ApplicationClaim\n        '
        return cls(**claim_dict)