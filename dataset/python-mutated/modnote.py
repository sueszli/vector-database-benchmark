"""Provide the ModNoteMixin class."""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Generator
if TYPE_CHECKING:
    import praw.models

class ModNoteMixin:
    """Interface for classes that can have a moderator note set on them."""

    def author_notes(self, **generator_kwargs: Any) -> Generator[praw.models.ModNote, None, None]:
        if False:
            while True:
                i = 10
        'Get the moderator notes for the author of this object in the subreddit it\'s posted in.\n\n        :param generator_kwargs: Additional keyword arguments are passed in the\n            initialization of the moderator note generator.\n\n        :returns: A generator of :class:`.ModNote`.\n\n        For example, to list all notes the author of a submission, try:\n\n        .. code-block:: python\n\n            for note in reddit.submission("92dd8").mod.author_notes():\n                print(f"{note.label}: {note.note}")\n\n        '
        return self.thing.subreddit.mod.notes.redditors(self.thing.author, **generator_kwargs)

    def create_note(self, *, label: str | None=None, note: str, **other_settings: Any) -> praw.models.ModNote:
        if False:
            while True:
                i = 10
        'Create a moderator note on the author of this object in the subreddit it\'s posted in.\n\n        :param label: The label for the note. As of this writing, this can be one of the\n            following: ``"ABUSE_WARNING"``, ``"BAN"``, ``"BOT_BAN"``,\n            ``"HELPFUL_USER"``, ``"PERMA_BAN"``, ``"SOLID_CONTRIBUTOR"``,\n            ``"SPAM_WARNING"``, ``"SPAM_WATCH"``, or ``None`` (default: ``None``).\n        :param note: The content of the note. As of this writing, this is limited to 250\n            characters.\n        :param other_settings: Additional keyword arguments are passed to\n            :meth:`~.BaseModNotes.create`.\n\n        :returns: The new :class:`.ModNote` object.\n\n        For example, to create a note on a :class:`.Submission`, try:\n\n        .. code-block:: python\n\n            reddit.submission("92dd8").mod.create_note(label="HELPFUL_USER", note="Test note")\n\n        '
        return self.thing.subreddit.mod.notes.create(label=label, note=note, thing=self.thing, **other_settings)