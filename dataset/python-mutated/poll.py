"""Provide poll-related classes."""
from __future__ import annotations
from typing import Any
from ...util import cachedproperty
from ..base import PRAWBase

class PollOption(PRAWBase):
    """Class to represent one option of a poll.

    If ``submission`` is a poll :class:`.Submission`, access the poll's options like so:

    .. code-block:: python

        poll_data = submission.poll_data

        # By index -- print the first option
        print(poll_data.options[0])

        # By ID -- print the option with ID "576797"
        print(poll_data.option("576797"))

    .. include:: ../../typical_attributes.rst

    ============== =================================================
    Attribute      Description
    ============== =================================================
    ``id``         ID of the poll option.
    ``text``       The text of the poll option.
    ``vote_count`` The number of votes the poll option has received.
    ============== =================================================

    """

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Return an object initialization representation of the instance.'
        return f'PollOption(id={self.id!r})'

    def __str__(self) -> str:
        if False:
            i = 10
            return i + 15
        'Return a string version of the PollData, its text.'
        return self.text

class PollData(PRAWBase):
    """Class to represent poll data on a poll submission.

    If ``submission`` is a poll :class:`.Submission`, access the poll data like so:

    .. code-block:: python

        poll_data = submission.poll_data
        print(f"There are {poll_data.total_vote_count} votes total.")
        print("The options are:")
        for option in poll_data.options:
            print(f"{option} ({option.vote_count} votes)")
        print(f"I voted for {poll_data.user_selection}.")

    .. include:: ../../typical_attributes.rst

    ======================== =========================================================
    Attribute                Description
    ======================== =========================================================
    ``options``              A list of :class:`.PollOption` of the poll.
    ``total_vote_count``     The total number of votes cast in the poll.
    ``user_selection``       The poll option selected by the authenticated user
                             (possibly ``None``).
    ``voting_end_timestamp`` Time the poll voting closes, represented in `Unix Time`_.
    ======================== =========================================================

    .. _unix time: https://en.wikipedia.org/wiki/Unix_time

    """

    @cachedproperty
    def user_selection(self) -> PollOption | None:
        if False:
            while True:
                i = 10
        "Get the user's selection in this poll, if any.\n\n        :returns: The user's selection as a :class:`.PollOption`, or ``None`` if there\n            is no choice.\n\n        "
        if self._user_selection is None:
            return None
        return self.option(self._user_selection)

    def __setattr__(self, attribute: str, value: Any) -> None:
        if False:
            print('Hello World!')
        'Objectify the options attribute, and save user_selection.'
        if attribute == 'options' and isinstance(value, list):
            value = [PollOption(self._reddit, option) for option in value]
        elif attribute == 'user_selection':
            attribute = '_user_selection'
        super().__setattr__(attribute, value)

    def option(self, option_id: str) -> PollOption:
        if False:
            for i in range(10):
                print('nop')
        'Get the option with the specified ID.\n\n        :param option_id: The ID of a poll option, as a ``str``.\n\n        :returns: The specified :class:`.PollOption`.\n\n        :raises: :py:class:`KeyError` if no option exists with the specified ID.\n\n        '
        for option in self.options:
            if option.id == option_id:
                return option
        msg = f'No poll option with ID {option_id!r}.'
        raise KeyError(msg)