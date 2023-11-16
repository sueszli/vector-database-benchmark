"""File containing a fake CPIX client for demonstration purposes."""
import secrets
from typing import Dict, List
from . import cpix_client

class FakeClient(cpix_client.CpixClient):
    """Fake CPIX client, for demonstration purposes only."""

    def fetch_keys(self, media_id: str, key_ids: List[str]) -> Dict[str, object]:
        if False:
            print('Hello World!')
        'Generates random key information.\n\n        Args:\n            media_id (string): Name for your asset, sometimes used by DRM providers to\n            show usage and reports.\n            key_ids (list[string]): List of IDs of any keys to fetch and prepare.\n\n        Returns:\n            Dictionary mapping key IDs to JSON-structured object containing key\n            information to be written to Secret Manager.\n        '
        key_info = dict()
        key_info['encryptionKeys'] = []
        for key_id in key_ids:
            fake_key = secrets.token_hex(16)
            key_info['encryptionKeys'].append({'keyId': key_id.replace('-', ''), 'key': fake_key, 'keyUri': f'https://storage.googleapis.com/bucket-name/{fake_key}.bin', 'iv': secrets.token_hex(16)})
        return key_info

    def required_env_vars(self) -> List[str]:
        if False:
            print('Hello World!')
        return []