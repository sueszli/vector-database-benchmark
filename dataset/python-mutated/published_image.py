from __future__ import annotations
from dataclasses import dataclass
import semver

@dataclass
class PublishedImage:
    registry: str
    repository: str
    tag: str
    sha: str

    @property
    def address(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'{self.registry}/{self.repository}:{self.tag}@sha256:{self.sha}'

    @classmethod
    def from_address(cls, address: str) -> PublishedImage:
        if False:
            print('Hello World!')
        'Creates a PublishedImage instance from a docker image address.\n        A docker image address is a string of the form:\n        registry/repository:tag@sha256:sha\n\n        Args:\n            address (str): _description_\n\n        Returns:\n            PublishedImage: _description_\n        '
        parts = address.split('/')
        registry = parts.pop(0)
        without_registry = '/'.join(parts)
        (repository, tag, sha) = without_registry.replace('@sha256', '').split(':')
        return cls(registry, repository, tag, sha)

    @property
    def name_with_tag(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return f'{self.repository}:{self.tag}'

    @property
    def version(self) -> semver.VersionInfo:
        if False:
            while True:
                i = 10
        return semver.VersionInfo.parse(self.tag)