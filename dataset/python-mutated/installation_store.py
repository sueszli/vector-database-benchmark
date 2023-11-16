"""Slack installation data store

Refer to https://slack.dev/python-slack-sdk/oauth/ for details.
"""
from logging import Logger
from typing import Optional
from .models.bot import Bot
from .models.installation import Installation

class InstallationStore:
    """The installation store interface.

    The minimum required methods are:

    * save(installation)
    * find_installation(enterprise_id, team_id, user_id, is_enterprise_install)

    If you would like to properly handle app uninstallations and token revocations,
    the following methods should be implemented.

    * delete_installation(enterprise_id, team_id, user_id)
    * delete_all(enterprise_id, team_id)

    If your app needs only bot scope installations, the simpler way to implement would be:

    * save(installation)
    * find_bot(enterprise_id, team_id, is_enterprise_install)
    * delete_bot(enterprise_id, team_id)
    * delete_all(enterprise_id, team_id)
    """

    @property
    def logger(self) -> Logger:
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError()

    def save(self, installation: Installation):
        if False:
            while True:
                i = 10
        'Saves an installation data'
        raise NotImplementedError()

    def save_bot(self, bot: Bot):
        if False:
            print('Hello World!')
        'Saves a bot installation data'
        raise NotImplementedError()

    def find_bot(self, *, enterprise_id: Optional[str], team_id: Optional[str], is_enterprise_install: Optional[bool]=False) -> Optional[Bot]:
        if False:
            return 10
        'Finds a bot scope installation per workspace / org'
        raise NotImplementedError()

    def find_installation(self, *, enterprise_id: Optional[str], team_id: Optional[str], user_id: Optional[str]=None, is_enterprise_install: Optional[bool]=False) -> Optional[Installation]:
        if False:
            return 10
        'Finds a relevant installation for the given IDs.\n        If the user_id is absent, this method may return the latest installation in the workspace / org.\n        '
        raise NotImplementedError()

    def delete_bot(self, *, enterprise_id: Optional[str], team_id: Optional[str]) -> None:
        if False:
            i = 10
            return i + 15
        'Deletes a bot scope installation per workspace / org'
        raise NotImplementedError()

    def delete_installation(self, *, enterprise_id: Optional[str], team_id: Optional[str], user_id: Optional[str]=None) -> None:
        if False:
            print('Hello World!')
        'Deletes an installation that matches the given IDs'
        raise NotImplementedError()

    def delete_all(self, *, enterprise_id: Optional[str], team_id: Optional[str]):
        if False:
            for i in range(10):
                print('nop')
        'Deletes all installation data for the given workspace / org'
        self.delete_bot(enterprise_id=enterprise_id, team_id=team_id)
        self.delete_installation(enterprise_id=enterprise_id, team_id=team_id)