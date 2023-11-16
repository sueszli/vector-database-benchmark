"""Provides classes for interacting with moderator notes."""
from itertools import islice
from typing import TYPE_CHECKING, Any, Generator, List, Optional, Tuple, Union
from ..const import API_PATH
from .base import PRAWBase
from .listing.generator import ListingGenerator
from .reddit.comment import Comment
from .reddit.redditor import Redditor
from .reddit.submission import Submission
if TYPE_CHECKING:
    import praw.models
RedditorType = Union[Redditor, str]
SubredditType = Union['praw.models.Subreddit', str]
ThingType = Union[Comment, Submission]

class BaseModNotes:
    """Provides base methods to interact with moderator notes."""

    def __init__(self, reddit: 'praw.Reddit'):
        if False:
            i = 10
            return i + 15
        'Initialize a :class:`.BaseModNotes` instance.\n\n        :param reddit: An instance of :class:`.Reddit`.\n\n        '
        self._reddit = reddit

    def _all_generator(self, redditor: RedditorType, subreddit: SubredditType, **generator_kwargs: Any):
        if False:
            return 10
        PRAWBase._safely_add_arguments(arguments=generator_kwargs, key='params', subreddit=subreddit, user=redditor)
        return ListingGenerator(self._reddit, API_PATH['mod_notes'], **generator_kwargs)

    def _bulk_generator(self, redditors: List[RedditorType], subreddits: List[SubredditType]) -> Generator['praw.models.ModNote', None, None]:
        if False:
            return 10
        subreddits_iter = iter(subreddits)
        redditors_iter = iter(redditors)
        while True:
            subreddits_chunk = list(islice(subreddits_iter, 500))
            users_chunk = list(islice(redditors_iter, 500))
            if not any([subreddits_chunk, users_chunk]):
                break
            params = {'subreddits': ','.join(map(str, subreddits_chunk)), 'users': ','.join(map(str, users_chunk))}
            response = self._reddit.get(API_PATH['mod_notes_bulk'], params=params)
            for note_dict in response['mod_notes']:
                yield self._reddit._objector.objectify(note_dict)

    def _ensure_attribute(self, error_message: str, **attributes: Any) -> Any:
        if False:
            i = 10
            return i + 15
        (attribute, _value) = attributes.popitem()
        value = _value or getattr(self, attribute, None)
        if value is None:
            raise TypeError(error_message)
        return value

    def _notes(self, all_notes: bool, redditors: List[RedditorType], subreddits: List[SubredditType], **generator_kwargs: Any) -> Generator['praw.models.ModNote', None, None]:
        if False:
            print('Hello World!')
        if all_notes:
            for subreddit in subreddits:
                for redditor in redditors:
                    yield from self._all_generator(redditor, subreddit, **generator_kwargs)
        else:
            yield from self._bulk_generator(redditors, subreddits)

    def create(self, *, label: Optional[str]=None, note: str, redditor: Optional[RedditorType]=None, subreddit: Optional[SubredditType]=None, thing: Optional[Union[Comment, Submission, str]]=None, **other_settings: Any) -> 'praw.models.ModNote':
        if False:
            for i in range(10):
                print('nop')
        'Create a :class:`.ModNote` for a redditor in the specified subreddit.\n\n        :param label: The label for the note. As of this writing, this can be one of the\n            following: ``"ABUSE_WARNING"``, ``"BAN"``, ``"BOT_BAN"``,\n            ``"HELPFUL_USER"``, ``"PERMA_BAN"``, ``"SOLID_CONTRIBUTOR"``,\n            ``"SPAM_WARNING"``, ``"SPAM_WATCH"``, or ``None`` (default: ``None``).\n        :param note: The content of the note. As of this writing, this is limited to 250\n            characters.\n        :param redditor: The redditor to create the note for (default: ``None``).\n\n            .. note::\n\n                This parameter is required if ``thing`` is not provided or this is not\n                called from a :class:`.Redditor` instance (e.g.,\n                ``reddit.redditor.notes``).\n\n        :param subreddit: The subreddit associated with the note (default: ``None``).\n\n            .. note::\n\n                This parameter is required if ``thing`` is not provided or this is not\n                called from a :class:`.Subreddit` instance (e.g.,\n                ``reddit.subreddit.mod``).\n\n        :param thing: Either the fullname of a comment/submission, a :class:`.Comment`,\n            or a :class:`.Submission` to associate with the note.\n        :param other_settings: Additional keyword arguments can be provided to handle\n            new parameters as Reddit introduces them.\n\n        :returns: The new :class:`.ModNote` object.\n\n        For example, to create a note for u/spez in r/test:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").mod.notes.create(\n                label="HELPFUL_USER", note="Test note", redditor="spez"\n            )\n            # or\n            reddit.redditor("spez").mod.notes.create(\n                label="HELPFUL_USER", note="Test note", subreddit="test"\n            )\n            # or\n            reddit.notes.create(\n                label="HELPFUL_USER", note="Test note", redditor="spez", subreddit="test"\n            )\n\n        '
        reddit_id = None
        if thing:
            if isinstance(thing, str):
                reddit_id = thing
                if not (getattr(self, 'redditor', redditor) and getattr(self, 'subreddit', subreddit)):
                    thing = next(self._reddit.info(fullnames=[thing]))
            else:
                reddit_id = thing.fullname
            redditor = getattr(self, 'redditor', redditor) or thing.author
            subreddit = getattr(self, 'subreddit', subreddit) or thing.subreddit
        redditor = self._ensure_attribute("Either the 'redditor' or 'thing' parameters must be provided or this method must be called from a Redditor instance (e.g., 'redditor.notes').", redditor=redditor)
        subreddit = self._ensure_attribute("Either the 'subreddit' or 'thing' parameters must be provided or this method must be called from a Subreddit instance (e.g., 'subreddit.mod.notes').", subreddit=subreddit)
        data = {'user': str(redditor), 'subreddit': str(subreddit), 'note': note}
        if label:
            data['label'] = label
        if reddit_id:
            data['reddit_id'] = reddit_id
        data.update(other_settings)
        return self._reddit.post(API_PATH['mod_notes'], data=data)

    def delete(self, *, delete_all: bool=False, note_id: Optional[str]=None, redditor: Optional[RedditorType]=None, subreddit: Optional[SubredditType]=None):
        if False:
            for i in range(10):
                print('nop')
        'Delete note(s) for a redditor.\n\n        :param delete_all: When ``True``, delete all notes for the specified redditor in\n            the specified subreddit (default: ``False``).\n\n            .. note::\n\n                This will make a request for each note.\n\n        :param note_id: The ID of the note to delete. This parameter is ignored if\n            ``delete_all`` is ``True``.\n        :param redditor: The redditor to delete the note(s) for (default: ``None``). Can\n            be a :class:`.Redditor` instance or a redditor name.\n\n            .. note::\n\n                This parameter is required if this method is **not** called from a\n                :class:`.Redditor` instance (e.g., ``redditor.notes``).\n\n        :param subreddit: The subreddit to delete the note(s) from (default: ``None``).\n            Can be a :class:`.Subreddit` instance or a subreddit name.\n\n            .. note::\n\n                This parameter is required if this method is **not** called from a\n                :class:`.Subreddit` instance (e.g., ``reddit.subreddit.mod``).\n\n\n        For example, to delete a note with the ID\n        ``"ModNote_d324b280-5ecc-435d-8159-3e259e84e339"``, try:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").mod.notes.delete(\n                note_id="ModNote_d324b280-5ecc-435d-8159-3e259e84e339", redditor="spez"\n            )\n            # or\n            reddit.redditor("spez").notes.delete(\n                note_id="ModNote_d324b280-5ecc-435d-8159-3e259e84e339", subreddit="test"\n            )\n            # or\n            reddit.notes.delete(\n                note_id="ModNote_d324b280-5ecc-435d-8159-3e259e84e339",\n                subreddit="test",\n                redditor="spez",\n            )\n\n        To delete all notes for u/spez, try:\n\n        .. code-block:: python\n\n            reddit.subreddit("test").mod.notes.delete(delete_all=True, redditor="spez")\n            # or\n            reddit.redditor("spez").notes.delete(delete_all=True, subreddit="test")\n            # or\n            reddit.notes.delete(delete_all=True, subreddit="test", redditor="spez")\n\n        '
        redditor = self._ensure_attribute("Either the 'redditor' parameter must be provided or this method must be called from a Redditor instance (e.g., 'redditor.notes').", redditor=redditor)
        subreddit = self._ensure_attribute("Either the 'subreddit' parameter must be provided or this method must be called from a Subreddit instance (e.g., 'subreddit.mod.notes').", subreddit=subreddit)
        if not delete_all and note_id is None:
            msg = "Either 'note_id' or 'delete_all' must be provided."
            raise TypeError(msg)
        if delete_all:
            for note in self._notes(True, [redditor], [subreddit]):
                note.delete()
        else:
            params = {'user': str(redditor), 'subreddit': str(subreddit), 'note_id': note_id}
            self._reddit.delete(API_PATH['mod_notes'], params=params)

class RedditorModNotes(BaseModNotes):
    """Provides methods to interact with moderator notes at the redditor level.

    .. note::

        The authenticated user must be a moderator of the provided subreddit(s).

    For example, all the notes for u/spez in r/test can be iterated through like so:

    .. code-block:: python

        redditor = reddit.redditor("spez")

        for note in redditor.notes.subreddits("test"):
            print(f"{note.label}: {note.note}")

    """

    def __init__(self, reddit: 'praw.Reddit', redditor: RedditorType):
        if False:
            for i in range(10):
                print('nop')
        'Initialize a :class:`.RedditorModNotes` instance.\n\n        :param reddit: An instance of :class:`.Reddit`.\n        :param redditor: An instance of :class:`.Redditor`.\n\n        '
        super().__init__(reddit)
        self.redditor = redditor

    def subreddits(self, *subreddits: SubredditType, all_notes: Optional[bool]=None, **generator_kwargs: Any) -> Generator['praw.models.ModNote', None, None]:
        if False:
            print('Hello World!')
        'Return notes for this :class:`.Redditor` from one or more subreddits.\n\n        :param subreddits: One or more subreddits to retrieve the notes from. Must be\n            either a :class:`.Subreddit` or a subreddit name.\n        :param all_notes: Whether to return all notes or only the latest note (default:\n            ``True`` if only one subreddit is provided otherwise ``False``).\n\n            .. note::\n\n                Setting this to ``True`` will result in a request for each subreddit.\n\n\n        :returns: A generator that yields the most recent :class:`.ModNote` (or ``None``\n            if this redditor doesn\'t have any notes) per subreddit in their relative\n            order. If ``all_notes`` is ``True``, this will yield all notes or ``None``\n            from each subreddit for this redditor.\n\n        For example, all the notes for u/spez in r/test can be iterated through like so:\n\n        .. code-block:: python\n\n            redditor = reddit.redditor("spez")\n\n            for note in redditor.notes.subreddits("test"):\n                print(f"{note.label}: {note.note}")\n\n        For example, the latest note for u/spez from r/test and r/redditdev can be\n        iterated through like so:\n\n        .. code-block:: python\n\n            redditor = reddit.redditor("spez")\n            subreddit = reddit.subreddit("redditdev")\n\n            for note in redditor.notes.subreddits("test", subreddit):\n                print(f"{note.label}: {note.note}")\n\n        For example, **all** the notes for u/spez in r/test and r/redditdev can be\n        iterated through like so:\n\n        .. code-block:: python\n\n            redditor = reddit.redditor("spez")\n            subreddit = reddit.subreddit("redditdev")\n\n            for note in redditor.notes.subreddits("test", subreddit, all_notes=True):\n                print(f"{note.label}: {note.note}")\n\n        '
        if len(subreddits) == 0:
            msg = 'At least 1 subreddit must be provided.'
            raise ValueError(msg)
        if all_notes is None:
            all_notes = len(subreddits) == 1
        return self._notes(all_notes, [self.redditor] * len(subreddits), list(subreddits), **generator_kwargs)

class SubredditModNotes(BaseModNotes):
    """Provides methods to interact with moderator notes at the subreddit level.

    .. note::

        The authenticated user must be a moderator of this subreddit.

    For example, all the notes for u/spez in r/test can be iterated through like so:

    .. code-block:: python

        subreddit = reddit.subreddit("test")

        for note in subreddit.mod.notes.redditors("spez"):
            print(f"{note.label}: {note.note}")

    """

    def __init__(self, reddit: 'praw.Reddit', subreddit: SubredditType):
        if False:
            while True:
                i = 10
        'Initialize a :class:`.SubredditModNotes` instance.\n\n        :param reddit: An instance of :class:`.Reddit`.\n        :param subreddit: An instance of :class:`.Subreddit`.\n\n        '
        super().__init__(reddit)
        self.subreddit = subreddit

    def redditors(self, *redditors: RedditorType, all_notes: Optional[bool]=None, **generator_kwargs: Any) -> Generator['praw.models.ModNote', None, None]:
        if False:
            return 10
        'Return notes from this :class:`.Subreddit` for one or more redditors.\n\n        :param redditors: One or more redditors to retrieve notes for. Must be either a\n            :class:`.Redditor` or a redditor name.\n        :param all_notes: Whether to return all notes or only the latest note (default:\n            ``True`` if only one redditor is provided otherwise ``False``).\n\n            .. note::\n\n                Setting this to ``True`` will result in a request for each redditor.\n\n\n        :returns: A generator that yields the most recent :class:`.ModNote` (or ``None``\n            if the user doesn\'t have any notes in this subreddit) per redditor in their\n            relative order. If ``all_notes`` is ``True``, this will yield all notes for\n            each redditor.\n\n        For example, all the notes for u/spez in r/test can be iterated through like so:\n\n        .. code-block:: python\n\n            subreddit = reddit.subreddit("test")\n\n            for note in subreddit.mod.notes.redditors("spez"):\n                print(f"{note.label}: {note.note}")\n\n        For example, the latest note for u/spez and u/bboe from r/test can be iterated\n        through like so:\n\n        .. code-block:: python\n\n            subreddit = reddit.subreddit("test")\n            redditor = reddit.redditor("bboe")\n\n            for note in subreddit.mod.notes.redditors("spez", redditor):\n                print(f"{note.label}: {note.note}")\n\n        For example, **all** the notes for both u/spez and u/bboe in r/test can be\n        iterated through like so:\n\n        .. code-block:: python\n\n            subreddit = reddit.subreddit("test")\n            redditor = reddit.redditor("bboe")\n\n            for note in subreddit.mod.notes.redditors("spez", redditor, all_notes=True):\n                print(f"{note.label}: {note.note}")\n\n        '
        if len(redditors) == 0:
            msg = 'At least 1 redditor must be provided.'
            raise ValueError(msg)
        if all_notes is None:
            all_notes = len(redditors) == 1
        return self._notes(all_notes, list(redditors), [self.subreddit] * len(redditors), **generator_kwargs)

class RedditModNotes(BaseModNotes):
    """Provides methods to interact with moderator notes at a global level.

    .. note::

        The authenticated user must be a moderator of the provided subreddit(s).

    For example, the latest note for u/spez in r/redditdev and r/test, and for u/bboe in
    r/redditdev can be iterated through like so:

    .. code-block:: python

        redditor = reddit.redditor("bboe")
        subreddit = reddit.subreddit("redditdev")

        pairs = [(subreddit, "spez"), ("test", "spez"), (subreddit, redditor)]

        for note in reddit.notes(pairs=pairs):
            print(f"{note.label}: {note.note}")

    """

    def __call__(self, *, all_notes: bool=False, pairs: Optional[List[Tuple[SubredditType, RedditorType]]]=None, redditors: Optional[List[RedditorType]]=None, subreddits: Optional[List[SubredditType]]=None, things: Optional[List[ThingType]]=None, **generator_kwargs: Any) -> Generator['praw.models.ModNote', None, None]:
        if False:
            i = 10
            return i + 15
        'Get note(s) for each subreddit/user pair, or ``None`` if they don\'t have any.\n\n        :param all_notes: Whether to return all notes or only the latest note for each\n            subreddit/redditor pair (default: ``False``).\n\n            .. note::\n\n                Setting this to ``True`` will result in a request for each unique\n                subreddit/redditor pair. If ``subreddits`` and ``redditors`` are\n                provided, this will make a request equivalent to number of redditors\n                multiplied by the number of subreddits.\n\n        :param pairs: A list of subreddit/redditor tuples.\n\n            .. note::\n\n                Required if ``subreddits``, ``redditors``, nor ``things`` are provided.\n\n        :param redditors: A list of redditors to return notes for. This parameter is\n            used in tandem with ``subreddits`` to get notes from multiple subreddits for\n            each of the provided redditors.\n\n            .. note::\n\n                Required if ``items`` or ``things`` is not provided or if ``subreddits``\n                **is** provided.\n\n        :param subreddits: A list of subreddits to return notes for. This parameter is\n            used in tandem with ``redditors`` to get notes for multiple redditors from\n            each of the provided subreddits.\n\n            .. note::\n\n                Required if ``items`` or ``things`` is not provided or if ``redditors``\n                **is** provided.\n\n        :param things: A list of comments and/or submissions to return notes for.\n        :param generator_kwargs: Additional keyword arguments passed to the generator.\n            This parameter is ignored when ``all_notes`` is ``False``.\n\n        :returns: A generator that yields the most recent :class:`.ModNote` (or ``None``\n            if the user doesn\'t have any notes) per entry in their relative order. If\n            ``all_notes`` is ``True``, this will yield all notes for each entry.\n\n        .. note::\n\n            This method will merge the subreddits and redditors provided from ``pairs``,\n            ``redditors``, ``subreddits``, and ``things``.\n\n        .. note::\n\n            This method accepts :class:`.Redditor` instances or redditor names and\n            :class:`.Subreddit` instances or subreddit names where applicable.\n\n        For example, the latest note for u/spez in r/redditdev and r/test, and for\n        u/bboe in r/redditdev can be iterated through like so:\n\n        .. code-block:: python\n\n            redditor = reddit.redditor("bboe")\n            subreddit = reddit.subreddit("redditdev")\n\n            pairs = [(subreddit, "spez"), ("test", "spez"), (subreddit, redditor)]\n\n            for note in reddit.notes(pairs=pairs):\n                print(f"{note.label}: {note.note}")\n\n        For example, **all** the notes for u/spez and u/bboe in r/announcements,\n        r/redditdev, and r/test can be iterated through like so:\n\n        .. code-block:: python\n\n            redditor = reddit.redditor("bboe")\n            subreddit = reddit.subreddit("redditdev")\n\n            for note in reddit.notes(\n                redditors=["spez", redditor],\n                subreddits=["announcements", subreddit, "test"],\n                all_notes=True,\n            ):\n                print(f"{note.label}: {note.note}")\n\n        For example, the latest note for the authors of the last 5 comments and\n        submissions from r/test can be iterated through like so:\n\n        .. code-block:: python\n\n            submissions = list(reddit.subreddit("test").new(limit=5))\n            comments = list(reddit.subreddit("test").comments(limit=5))\n\n            for note in reddit.notes(things=submissions + comments):\n                print(f"{note.label}: {note.note}")\n\n        .. note::\n\n            Setting ``all_notes`` to ``True`` will make a request for each redditor and\n            subreddit combination. The previous example will make 6 requests.\n\n        '
        if pairs is None:
            pairs = []
        if redditors is None:
            redditors = []
        if subreddits is None:
            subreddits = []
        if things is None:
            things = []
        if not pairs + redditors + subreddits + things:
            msg = "Either the 'pairs', 'redditors', 'subreddits', or 'things' parameters must be provided."
            raise TypeError(msg)
        if len(redditors) * len(subreddits) == 0 and len(redditors) + len(subreddits) > 0:
            raise TypeError("'redditors' must be non-empty if 'subreddits' is not empty." if len(subreddits) > 0 else "'subreddits' must be non-empty if 'redditors' is not empty.")
        merged_redditors = []
        merged_subreddits = []
        items = pairs + [(subreddit, redditor) for redditor in set(redditors) for subreddit in set(subreddits)] + things
        for item in items:
            if isinstance(item, (Comment, Submission)):
                merged_redditors.append(item.author.name)
                merged_subreddits.append(item.subreddit.display_name)
            elif isinstance(item, Tuple):
                (subreddit, redditor) = item
                merged_redditors.append(redditor)
                merged_subreddits.append(subreddit)
            else:
                msg = f'Cannot get subreddit and author fields from type {type(item)}'
                raise ValueError(msg)
        return self._notes(all_notes, merged_redditors, merged_subreddits, **generator_kwargs)

    def things(self, *things: ThingType, all_notes: Optional[bool]=None, **generator_kwargs: Any) -> Generator['praw.models.ModNote', None, None]:
        if False:
            i = 10
            return i + 15
        'Return notes associated with the author of a :class:`.Comment` or :class:`.Submission`.\n\n        :param things: One or more things to return notes on. Must be a\n            :class:`.Comment` or :class:`.Submission`.\n        :param all_notes: Whether to return all notes, or only the latest (default:\n            ``True`` if only one thing is provided otherwise ``False``).\n\n            .. note::\n\n                Setting this to ``True`` will result in a request for each thing.\n\n\n        :returns: A generator that yields the most recent :class:`.ModNote` (or ``None``\n            if the user doesn\'t have any notes) per entry in their relative order. If\n            ``all_notes`` is ``True``, this will yield all notes for each entry.\n\n        For example, to get the latest note for the authors of the top 25 submissions in\n        r/test:\n\n        .. code-block:: python\n\n            submissions = reddit.subreddit("test").top(limit=25)\n            for note in reddit.notes.things(*submissions):\n                print(f"{note.label}: {note.note}")\n\n        For example, to get the latest note for the authors of the last 25 comments in\n        r/test:\n\n        .. code-block:: python\n\n            comments = reddit.subreddit("test").comments(limit=25)\n            for note in reddit.notes.things(*comments):\n                print(f"{note.label}: {note.note}")\n\n        '
        subreddits = []
        redditors = []
        for thing in things:
            subreddits.append(thing.subreddit)
            redditors.append(thing.author)
        if all_notes is None:
            all_notes = len(things) == 1
        return self._notes(all_notes, redditors, subreddits, **generator_kwargs)