"""Domain objects for Wipeout."""
from __future__ import annotations
from core import feconf
from core import utils
from typing import Dict, Optional
USER_DELETION_SUCCESS = 'SUCCESS'
USER_DELETION_ALREADY_DONE = 'ALREADY DONE'
USER_VERIFICATION_NOT_DELETED = 'NOT DELETED'
USER_VERIFICATION_SUCCESS = 'SUCCESS'
USER_VERIFICATION_FAILURE = 'FAILURE'

class PendingDeletionRequest:
    """Domain object for a PendingDeletionRequest."""

    def __init__(self, user_id: str, username: Optional[str], email: str, normalized_long_term_username: Optional[str], deletion_complete: bool, pseudonymizable_entity_mappings: Dict[str, Dict[str, str]]) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Constructs a PendingDeletionRequest domain object.\n\n        Args:\n            user_id: str. The ID of the user who is being deleted.\n            username: str. The username of the  user who is being deleted.\n            email: str. The email of the user who is being deleted.\n            normalized_long_term_username: str|None. The normalized username of\n                the user who is being deleted. Can be None when the user was on\n                the Oppia site only for a short time and thus the username\n                hasn't been well-established yet.\n            deletion_complete: bool. Whether the deletion is completed.\n            pseudonymizable_entity_mappings: dict(str, dict(str, str)).\n                Mapping between the entity IDs and pseudonymized user IDs.\n        "
        self.user_id = user_id
        self.username = username
        self.email = email
        self.normalized_long_term_username = normalized_long_term_username
        self.deletion_complete = deletion_complete
        self.pseudonymizable_entity_mappings = pseudonymizable_entity_mappings

    @classmethod
    def create_default(cls, user_id: str, username: Optional[str], email: str, normalized_long_term_username: Optional[str]=None) -> PendingDeletionRequest:
        if False:
            print('Hello World!')
        "Creates a PendingDeletionRequest object with default values.\n\n        Args:\n            user_id: str. The ID of the user who is being deleted.\n            username: str. The username of the  user who is being deleted.\n            email: str. The email of the user who is being deleted.\n            normalized_long_term_username: str|None. The normalized username of\n                the user who is being deleted. Can be None when the user was on\n                the Oppia site only for a short time and thus the username\n                hasn't been well-established yet.\n\n        Returns:\n            PendingDeletionRequest. The default pending deletion request\n            domain object.\n        "
        return cls(user_id, username, email, normalized_long_term_username, False, {})

    def validate(self) -> None:
        if False:
            while True:
                i = 10
        'Checks that the domain object is valid.\n\n        Raises:\n            ValidationError. The field pseudonymizable_entity_mappings\n                contains wrong key.\n        '
        for key in self.pseudonymizable_entity_mappings.keys():
            if key not in [name.value for name in feconf.ValidModelNames]:
                raise utils.ValidationError('pseudonymizable_entity_mappings contain wrong key')