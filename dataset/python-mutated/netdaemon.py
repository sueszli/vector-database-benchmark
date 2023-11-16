"""Class for netdaemon apps in HACS."""
from __future__ import annotations
from typing import TYPE_CHECKING
from ..enums import HacsCategory, HacsDispatchEvent
from ..exceptions import HacsException
from ..utils import filters
from ..utils.decorator import concurrent
from .base import HacsRepository
if TYPE_CHECKING:
    from ..base import HacsBase

class HacsNetdaemonRepository(HacsRepository):
    """Netdaemon apps in HACS."""

    def __init__(self, hacs: HacsBase, full_name: str):
        if False:
            return 10
        'Initialize.'
        super().__init__(hacs=hacs)
        self.data.full_name = full_name
        self.data.full_name_lower = full_name.lower()
        self.data.category = HacsCategory.NETDAEMON
        self.content.path.local = self.localpath
        self.content.path.remote = 'apps'

    @property
    def localpath(self):
        if False:
            return 10
        'Return localpath.'
        return f'{self.hacs.core.config_path}/netdaemon/apps/{self.data.name}'

    async def validate_repository(self):
        """Validate."""
        await self.common_validate()
        if self.repository_manifest:
            if self.repository_manifest.content_in_root:
                self.content.path.remote = ''
        if self.content.path.remote == 'apps':
            self.data.domain = filters.get_first_directory_in_directory(self.tree, self.content.path.remote)
            self.content.path.remote = f'apps/{self.data.name}'
        compliant = False
        for treefile in self.treefiles:
            if treefile.startswith(f'{self.content.path.remote}') and treefile.endswith('.cs'):
                compliant = True
                break
        if not compliant:
            raise HacsException(f"{self.string} Repository structure for {self.ref.replace('tags/', '')} is not compliant")
        if self.validate.errors:
            for error in self.validate.errors:
                if not self.hacs.status.startup:
                    self.logger.error('%s %s', self.string, error)
        return self.validate.success

    @concurrent(concurrenttasks=10, backoff_time=5)
    async def update_repository(self, ignore_issues=False, force=False):
        """Update."""
        if not await self.common_update(ignore_issues, force) and (not force):
            return
        if self.repository_manifest:
            if self.repository_manifest.content_in_root:
                self.content.path.remote = ''
        if self.content.path.remote == 'apps':
            self.data.domain = filters.get_first_directory_in_directory(self.tree, self.content.path.remote)
            self.content.path.remote = f'apps/{self.data.name}'
        self.content.path.local = self.localpath
        if self.data.installed:
            self.hacs.async_dispatch(HacsDispatchEvent.REPOSITORY, {'id': 1337, 'action': 'update', 'repository': self.data.full_name, 'repository_id': self.data.id})

    async def async_post_installation(self):
        """Run post installation steps."""
        try:
            await self.hacs.hass.services.async_call('hassio', 'addon_restart', {'addon': 'c6a2317c_netdaemon'})
        except BaseException:
            pass