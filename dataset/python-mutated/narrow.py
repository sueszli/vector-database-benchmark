import os
import re
from dataclasses import dataclass
from typing import Any, Callable, Collection, Dict, Generic, Iterable, List, Optional, Protocol, Sequence, Set, Tuple, TypeVar, Union
import orjson
from django.conf import settings
from django.core.exceptions import ValidationError
from django.db import connection
from django.utils.translation import gettext as _
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Connection, Row
from sqlalchemy.sql import ClauseElement, ColumnElement, Select, and_, column, func, join, literal, literal_column, not_, or_, select, table, union_all
from sqlalchemy.sql.selectable import SelectBase
from sqlalchemy.types import ARRAY, Boolean, Integer, Text
from typing_extensions import TypeAlias, override
from zerver.lib.addressee import get_user_profiles, get_user_profiles_by_ids
from zerver.lib.exceptions import ErrorCode, JsonableError
from zerver.lib.message import get_first_visible_message_id
from zerver.lib.narrow_helpers import NarrowTerm
from zerver.lib.recipient_users import recipient_for_user_profiles
from zerver.lib.sqlalchemy_utils import get_sqlalchemy_connection
from zerver.lib.streams import can_access_stream_history_by_id, can_access_stream_history_by_name, get_public_streams_queryset, get_stream_by_narrow_operand_access_unchecked, get_web_public_streams_queryset
from zerver.lib.topic import RESOLVED_TOPIC_PREFIX, get_resolved_topic_condition_sa, get_topic_from_message_info, topic_column_sa, topic_match_sa
from zerver.lib.types import Validator
from zerver.lib.user_topics import exclude_topic_mutes
from zerver.lib.validator import check_bool, check_dict, check_required_string, check_string, check_string_or_int, check_string_or_int_list
from zerver.models import Realm, Recipient, Stream, Subscription, UserMessage, UserProfile, get_active_streams, get_user_by_id_in_realm_including_cross_realm, get_user_including_cross_realm
stop_words_list: Optional[List[str]] = None

def read_stop_words() -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    global stop_words_list
    if stop_words_list is None:
        file_path = os.path.join(settings.DEPLOY_ROOT, 'puppet/zulip/files/postgresql/zulip_english.stop')
        with open(file_path) as f:
            stop_words_list = f.read().splitlines()
    return stop_words_list

def check_narrow_for_events(narrow: Collection[NarrowTerm]) -> None:
    if False:
        for i in range(10):
            print('nop')
    for narrow_term in narrow:
        operator = narrow_term.operator
        if operator not in ['stream', 'topic', 'sender', 'is']:
            raise JsonableError(_('Operator {operator} not supported.').format(operator=operator))

def is_spectator_compatible(narrow: Iterable[Dict[str, Any]]) -> bool:
    if False:
        i = 10
        return i + 15
    for element in narrow:
        operator = element['operator']
        if 'operand' not in element:
            return False
        if operator not in ['streams', 'stream', 'topic', 'sender', 'has', 'search', 'near', 'id']:
            return False
    return True

def is_web_public_narrow(narrow: Optional[Iterable[Dict[str, Any]]]) -> bool:
    if False:
        i = 10
        return i + 15
    if narrow is None:
        return False
    return any((term['operator'] == 'streams' and term['operand'] == 'web-public' and (term['negated'] is False) for term in narrow))

class NarrowPredicate(Protocol):

    def __call__(self, *, message: Dict[str, Any], flags: List[str]) -> bool:
        if False:
            for i in range(10):
                print('nop')
        ...

def build_narrow_predicate(narrow: Collection[NarrowTerm]) -> NarrowPredicate:
    if False:
        return 10
    'Changes to this function should come with corresponding changes to\n    NarrowLibraryTest.'
    check_narrow_for_events(narrow)

    def narrow_predicate(*, message: Dict[str, Any], flags: List[str]) -> bool:
        if False:
            i = 10
            return i + 15

        def satisfies_operator(*, operator: str, operand: str) -> bool:
            if False:
                while True:
                    i = 10
            if operator == 'stream':
                if message['type'] != 'stream':
                    return False
                if operand.lower() != message['display_recipient'].lower():
                    return False
            elif operator == 'topic':
                if message['type'] != 'stream':
                    return False
                topic_name = get_topic_from_message_info(message)
                if operand.lower() != topic_name.lower():
                    return False
            elif operator == 'sender':
                if operand.lower() != message['sender_email'].lower():
                    return False
            elif operator == 'is' and operand in ['dm', 'private']:
                if message['type'] != 'private':
                    return False
            elif operator == 'is' and operand in ['starred']:
                if operand not in flags:
                    return False
            elif operator == 'is' and operand == 'unread':
                if 'read' in flags:
                    return False
            elif operator == 'is' and operand in ['alerted', 'mentioned']:
                if 'mentioned' not in flags:
                    return False
            elif operator == 'is' and operand == 'resolved':
                if message['type'] != 'stream':
                    return False
                topic_name = get_topic_from_message_info(message)
                if not topic_name.startswith(RESOLVED_TOPIC_PREFIX):
                    return False
            return True
        for narrow_term in narrow:
            if not satisfies_operator(operator=narrow_term.operator, operand=narrow_term.operand):
                return False
        return True
    return narrow_predicate
LARGER_THAN_MAX_MESSAGE_ID = 10000000000000000

class BadNarrowOperatorError(JsonableError):
    code = ErrorCode.BAD_NARROW
    data_fields = ['desc']

    def __init__(self, desc: str) -> None:
        if False:
            print('Hello World!')
        self.desc: str = desc

    @staticmethod
    @override
    def msg_format() -> str:
        if False:
            i = 10
            return i + 15
        return _('Invalid narrow operator: {desc}')
ConditionTransform: TypeAlias = Callable[[ClauseElement], ClauseElement]
OptionalNarrowListT: TypeAlias = Optional[List[Dict[str, Any]]]
TS_START = '<ts-match>'
TS_STOP = '</ts-match>'

def ts_locs_array(config: ColumnElement[Text], text: ColumnElement[Text], tsquery: ColumnElement[Any]) -> ColumnElement[ARRAY[Integer]]:
    if False:
        i = 10
        return i + 15
    options = f'HighlightAll = TRUE, StartSel = {TS_START}, StopSel = {TS_STOP}'
    delimited = func.ts_headline(config, text, tsquery, options, type_=Text)
    part = func.unnest(func.string_to_array(delimited, TS_START, type_=ARRAY(Text)), type_=Text).column_valued()
    part_len = func.length(part, type_=Integer) - len(TS_STOP)
    match_pos = func.sum(part_len, type_=Integer).over(rows=(None, -1)) + len(TS_STOP)
    match_len = func.strpos(part, TS_STOP, type_=Integer) - 1
    return func.array(select(postgresql.array([match_pos, match_len])).offset(1).scalar_subquery(), type_=ARRAY(Integer))

class NarrowBuilder:
    """
    Build up a SQLAlchemy query to find messages matching a narrow.
    """

    def __init__(self, user_profile: Optional[UserProfile], msg_id_column: ColumnElement[Integer], realm: Realm, is_web_public_query: bool=False) -> None:
        if False:
            print('Hello World!')
        self.user_profile = user_profile
        self.msg_id_column = msg_id_column
        self.realm = realm
        self.is_web_public_query = is_web_public_query
        self.by_method_map = {'has': self.by_has, 'in': self.by_in, 'is': self.by_is, 'stream': self.by_stream, 'streams': self.by_streams, 'topic': self.by_topic, 'sender': self.by_sender, 'near': self.by_near, 'id': self.by_id, 'search': self.by_search, 'dm': self.by_dm, 'pm-with': self.by_dm, 'dm-including': self.by_dm_including, 'group-pm-with': self.by_group_pm_with, 'pm_with': self.by_dm, 'group_pm_with': self.by_group_pm_with}

    def add_term(self, query: Select, term: Dict[str, Any]) -> Select:
        if False:
            print('Hello World!')
        "\n        Extend the given query to one narrowed by the given term, and return the result.\n\n        This method satisfies an important security property: the returned\n        query never includes a message that the given query didn't.  In\n        particular, if the given query will only find messages that a given\n        user can legitimately see, then so will the returned query.\n        "
        operator = term['operator']
        operand = term['operand']
        negated = term.get('negated', False)
        if operator in self.by_method_map:
            method = self.by_method_map[operator]
        else:
            raise BadNarrowOperatorError('unknown operator ' + operator)
        if negated:
            maybe_negate = not_
        else:
            maybe_negate = lambda cond: cond
        return method(query, operand, maybe_negate)

    def by_has(self, query: Select, operand: str, maybe_negate: ConditionTransform) -> Select:
        if False:
            i = 10
            return i + 15
        if operand not in ['attachment', 'image', 'link']:
            raise BadNarrowOperatorError("unknown 'has' operand " + operand)
        col_name = 'has_' + operand
        cond = column(col_name, Boolean)
        return query.where(maybe_negate(cond))

    def by_in(self, query: Select, operand: str, maybe_negate: ConditionTransform) -> Select:
        if False:
            while True:
                i = 10
        assert not self.is_web_public_query
        assert self.user_profile is not None
        if operand == 'home':
            conditions = exclude_muting_conditions(self.user_profile, [])
            return query.where(and_(*conditions))
        elif operand == 'all':
            return query
        raise BadNarrowOperatorError("unknown 'in' operand " + operand)

    def by_is(self, query: Select, operand: str, maybe_negate: ConditionTransform) -> Select:
        if False:
            while True:
                i = 10
        assert not self.is_web_public_query
        assert self.user_profile is not None
        if operand in ['dm', 'private']:
            cond = column('flags', Integer).op('&')(UserMessage.flags.is_private.mask) != 0
            return query.where(maybe_negate(cond))
        elif operand == 'starred':
            cond = column('flags', Integer).op('&')(UserMessage.flags.starred.mask) != 0
            return query.where(maybe_negate(cond))
        elif operand == 'unread':
            cond = column('flags', Integer).op('&')(UserMessage.flags.read.mask) == 0
            return query.where(maybe_negate(cond))
        elif operand == 'mentioned':
            mention_flags_mask = UserMessage.flags.mentioned.mask | UserMessage.flags.stream_wildcard_mentioned.mask | UserMessage.flags.topic_wildcard_mentioned.mask | UserMessage.flags.group_mentioned.mask
            cond = column('flags', Integer).op('&')(mention_flags_mask) != 0
            return query.where(maybe_negate(cond))
        elif operand == 'alerted':
            cond = column('flags', Integer).op('&')(UserMessage.flags.has_alert_word.mask) != 0
            return query.where(maybe_negate(cond))
        elif operand == 'resolved':
            cond = get_resolved_topic_condition_sa()
            return query.where(maybe_negate(cond))
        raise BadNarrowOperatorError("unknown 'is' operand " + operand)
    _alphanum = frozenset('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

    def _pg_re_escape(self, pattern: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        "\n        Escape user input to place in a regex\n\n        Python's re.escape escapes Unicode characters in a way which PostgreSQL\n        fails on, 'λ' to '\\λ'. This function will correctly escape\n        them for PostgreSQL, 'λ' to '\\u03bb'.\n        "
        s = list(pattern)
        for (i, c) in enumerate(s):
            if c not in self._alphanum:
                if ord(c) >= 128:
                    s[i] = f'\\u{ord(c):0>4x}'
                else:
                    s[i] = '\\' + c
        return ''.join(s)

    def by_stream(self, query: Select, operand: Union[str, int], maybe_negate: ConditionTransform) -> Select:
        if False:
            return 10
        try:
            stream = get_stream_by_narrow_operand_access_unchecked(operand, self.realm)
            if self.is_web_public_query and (not stream.is_web_public):
                raise BadNarrowOperatorError('unknown web-public stream ' + str(operand))
        except Stream.DoesNotExist:
            raise BadNarrowOperatorError('unknown stream ' + str(operand))
        if self.realm.is_zephyr_mirror_realm:
            assert not stream.is_public()
            m = re.search('^(?:un)*(.+?)(?:\\.d)*$', stream.name, re.IGNORECASE)
            assert m is not None
            base_stream_name = m.group(1)
            matching_streams = get_active_streams(self.realm).filter(name__iregex=f'^(un)*{self._pg_re_escape(base_stream_name)}(\\.d)*$')
            recipient_ids = [matching_stream.recipient_id for matching_stream in matching_streams]
            cond = column('recipient_id', Integer).in_(recipient_ids)
            return query.where(maybe_negate(cond))
        recipient_id = stream.recipient_id
        assert recipient_id is not None
        cond = column('recipient_id', Integer) == recipient_id
        return query.where(maybe_negate(cond))

    def by_streams(self, query: Select, operand: str, maybe_negate: ConditionTransform) -> Select:
        if False:
            return 10
        if operand == 'public':
            recipient_queryset = get_public_streams_queryset(self.realm)
        elif operand == 'web-public':
            recipient_queryset = get_web_public_streams_queryset(self.realm)
        else:
            raise BadNarrowOperatorError('unknown streams operand ' + operand)
        recipient_ids = recipient_queryset.values_list('recipient_id', flat=True).order_by('id')
        cond = column('recipient_id', Integer).in_(recipient_ids)
        return query.where(maybe_negate(cond))

    def by_topic(self, query: Select, operand: str, maybe_negate: ConditionTransform) -> Select:
        if False:
            return 10
        if self.realm.is_zephyr_mirror_realm:
            m = re.search('^(.*?)(?:\\.d)*$', operand, re.IGNORECASE)
            assert m is not None
            base_topic = m.group(1)
            if base_topic in ('', 'personal', '(instance "")'):
                cond: ClauseElement = or_(topic_match_sa(''), topic_match_sa('.d'), topic_match_sa('.d.d'), topic_match_sa('.d.d.d'), topic_match_sa('.d.d.d.d'), topic_match_sa('personal'), topic_match_sa('personal.d'), topic_match_sa('personal.d.d'), topic_match_sa('personal.d.d.d'), topic_match_sa('personal.d.d.d.d'), topic_match_sa('(instance "")'), topic_match_sa('(instance "").d'), topic_match_sa('(instance "").d.d'), topic_match_sa('(instance "").d.d.d'), topic_match_sa('(instance "").d.d.d.d'))
            else:
                cond = or_(topic_match_sa(base_topic), topic_match_sa(base_topic + '.d'), topic_match_sa(base_topic + '.d.d'), topic_match_sa(base_topic + '.d.d.d'), topic_match_sa(base_topic + '.d.d.d.d'))
            return query.where(maybe_negate(cond))
        cond = topic_match_sa(operand)
        return query.where(maybe_negate(cond))

    def by_sender(self, query: Select, operand: Union[str, int], maybe_negate: ConditionTransform) -> Select:
        if False:
            return 10
        try:
            if isinstance(operand, str):
                sender = get_user_including_cross_realm(operand, self.realm)
            else:
                sender = get_user_by_id_in_realm_including_cross_realm(operand, self.realm)
        except UserProfile.DoesNotExist:
            raise BadNarrowOperatorError('unknown user ' + str(operand))
        cond = column('sender_id', Integer) == literal(sender.id)
        return query.where(maybe_negate(cond))

    def by_near(self, query: Select, operand: str, maybe_negate: ConditionTransform) -> Select:
        if False:
            return 10
        return query

    def by_id(self, query: Select, operand: Union[int, str], maybe_negate: ConditionTransform) -> Select:
        if False:
            i = 10
            return i + 15
        if not str(operand).isdigit():
            raise BadNarrowOperatorError('Invalid message ID')
        cond = self.msg_id_column == literal(operand)
        return query.where(maybe_negate(cond))

    def by_dm(self, query: Select, operand: Union[str, Iterable[int]], maybe_negate: ConditionTransform) -> Select:
        if False:
            while True:
                i = 10
        assert not self.is_web_public_query
        assert self.user_profile is not None
        try:
            if isinstance(operand, str):
                email_list = operand.split(',')
                user_profiles = get_user_profiles(emails=email_list, realm=self.realm)
            else:
                '\n                This is where we handle passing a list of user IDs for the narrow, which is the\n                preferred/cleaner API.\n                '
                user_profiles = get_user_profiles_by_ids(user_ids=operand, realm=self.realm)
            recipient = recipient_for_user_profiles(user_profiles=user_profiles, forwarded_mirror_message=False, forwarder_user_profile=None, sender=self.user_profile, allow_deactivated=True)
        except (JsonableError, ValidationError):
            raise BadNarrowOperatorError('unknown user in ' + str(operand))
        if recipient.type == Recipient.HUDDLE:
            cond = column('recipient_id', Integer) == recipient.id
            return query.where(maybe_negate(cond))
        other_participant = None
        for user in user_profiles:
            if user.id != self.user_profile.id:
                other_participant = user
        if other_participant:
            self_recipient_id = self.user_profile.recipient_id
            cond = and_(column('flags', Integer).op('&')(UserMessage.flags.is_private.mask) != 0, column('realm_id', Integer) == self.realm.id, or_(and_(column('sender_id', Integer) == other_participant.id, column('recipient_id', Integer) == self_recipient_id), and_(column('sender_id', Integer) == self.user_profile.id, column('recipient_id', Integer) == recipient.id)))
            return query.where(maybe_negate(cond))
        cond = and_(column('flags', Integer).op('&')(UserMessage.flags.is_private.mask) != 0, column('realm_id', Integer) == self.realm.id, column('sender_id', Integer) == self.user_profile.id, column('recipient_id', Integer) == recipient.id)
        return query.where(maybe_negate(cond))

    def _get_huddle_recipients(self, other_user: UserProfile) -> Set[int]:
        if False:
            i = 10
            return i + 15
        self_recipient_ids = [recipient_tuple['recipient_id'] for recipient_tuple in Subscription.objects.filter(user_profile=self.user_profile, recipient__type=Recipient.HUDDLE).values('recipient_id')]
        narrow_recipient_ids = [recipient_tuple['recipient_id'] for recipient_tuple in Subscription.objects.filter(user_profile=other_user, recipient__type=Recipient.HUDDLE).values('recipient_id')]
        return set(self_recipient_ids) & set(narrow_recipient_ids)

    def by_dm_including(self, query: Select, operand: Union[str, int], maybe_negate: ConditionTransform) -> Select:
        if False:
            while True:
                i = 10
        assert not self.is_web_public_query
        assert self.user_profile is not None
        try:
            if isinstance(operand, str):
                narrow_user_profile = get_user_including_cross_realm(operand, self.realm)
            else:
                narrow_user_profile = get_user_by_id_in_realm_including_cross_realm(operand, self.realm)
        except UserProfile.DoesNotExist:
            raise BadNarrowOperatorError('unknown user ' + str(operand))
        if narrow_user_profile.id == self.user_profile.id:
            cond = column('flags', Integer).op('&')(UserMessage.flags.is_private.mask) != 0
            return query.where(maybe_negate(cond))
        huddle_recipient_ids = self._get_huddle_recipients(narrow_user_profile)
        self_recipient_id = self.user_profile.recipient_id
        cond = and_(column('flags', Integer).op('&')(UserMessage.flags.is_private.mask) != 0, column('realm_id', Integer) == self.realm.id, or_(and_(column('sender_id', Integer) == narrow_user_profile.id, column('recipient_id', Integer) == self_recipient_id), and_(column('sender_id', Integer) == self.user_profile.id, column('recipient_id', Integer) == narrow_user_profile.recipient_id), and_(column('recipient_id', Integer).in_(huddle_recipient_ids))))
        return query.where(maybe_negate(cond))

    def by_group_pm_with(self, query: Select, operand: Union[str, int], maybe_negate: ConditionTransform) -> Select:
        if False:
            print('Hello World!')
        assert not self.is_web_public_query
        assert self.user_profile is not None
        try:
            if isinstance(operand, str):
                narrow_profile = get_user_including_cross_realm(operand, self.realm)
            else:
                narrow_profile = get_user_by_id_in_realm_including_cross_realm(operand, self.realm)
        except UserProfile.DoesNotExist:
            raise BadNarrowOperatorError('unknown user ' + str(operand))
        recipient_ids = self._get_huddle_recipients(narrow_profile)
        cond = and_(column('flags', Integer).op('&')(UserMessage.flags.is_private.mask) != 0, column('realm_id', Integer) == self.realm.id, column('recipient_id', Integer).in_(recipient_ids))
        return query.where(maybe_negate(cond))

    def by_search(self, query: Select, operand: str, maybe_negate: ConditionTransform) -> Select:
        if False:
            print('Hello World!')
        if settings.USING_PGROONGA:
            return self._by_search_pgroonga(query, operand, maybe_negate)
        else:
            return self._by_search_tsearch(query, operand, maybe_negate)

    def _by_search_pgroonga(self, query: Select, operand: str, maybe_negate: ConditionTransform) -> Select:
        if False:
            while True:
                i = 10
        match_positions_character = func.pgroonga_match_positions_character
        query_extract_keywords = func.pgroonga_query_extract_keywords
        operand_escaped = func.escape_html(operand, type_=Text)
        keywords = query_extract_keywords(operand_escaped)
        query = query.add_columns(match_positions_character(column('rendered_content', Text), keywords).label('content_matches'), match_positions_character(func.escape_html(topic_column_sa(), type_=Text), keywords).label('topic_matches'))
        condition = column('search_pgroonga', Text).op('&@~')(operand_escaped)
        return query.where(maybe_negate(condition))

    def _by_search_tsearch(self, query: Select, operand: str, maybe_negate: ConditionTransform) -> Select:
        if False:
            for i in range(10):
                print('nop')
        tsquery = func.plainto_tsquery(literal('zulip.english_us_search'), literal(operand))
        query = query.add_columns(ts_locs_array(literal('zulip.english_us_search', Text), column('rendered_content', Text), tsquery).label('content_matches'), ts_locs_array(literal('zulip.english_us_search', Text), func.escape_html(topic_column_sa(), type_=Text), tsquery).label('topic_matches'))
        for term in re.findall('"[^"]+"|\\S+', operand):
            if term[0] == '"' and term[-1] == '"':
                term = term[1:-1]
                term = '%' + connection.ops.prep_for_like_query(term) + '%'
                cond: ClauseElement = or_(column('content', Text).ilike(term), topic_column_sa().ilike(term))
                query = query.where(maybe_negate(cond))
        cond = column('search_tsvector', postgresql.TSVECTOR).op('@@')(tsquery)
        return query.where(maybe_negate(cond))

def narrow_parameter(var_name: str, json: str) -> OptionalNarrowListT:
    if False:
        for i in range(10):
            print('nop')
    data = orjson.loads(json)
    if not isinstance(data, list):
        raise ValueError('argument is not a list')
    if len(data) == 0:
        return None

    def convert_term(elem: Union[Dict[str, Any], List[str]]) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        if isinstance(elem, list):
            if len(elem) != 2 or any((not isinstance(x, str) for x in elem)):
                raise ValueError('element is not a string pair')
            return dict(operator=elem[0], operand=elem[1])
        if isinstance(elem, dict):
            operators_supporting_id = ['id', 'stream', 'sender', 'group-pm-with', 'dm-including']
            operators_supporting_ids = ['pm-with', 'dm']
            operators_non_empty_operand = {'search'}
            operator = elem.get('operator', '')
            if operator in operators_supporting_id:
                operand_validator: Validator[object] = check_string_or_int
            elif operator in operators_supporting_ids:
                operand_validator = check_string_or_int_list
            elif operator in operators_non_empty_operand:
                operand_validator = check_required_string
            else:
                operand_validator = check_string
            validator = check_dict(required_keys=[('operator', check_string), ('operand', operand_validator)], optional_keys=[('negated', check_bool)])
            try:
                validator('elem', elem)
            except ValidationError as error:
                raise JsonableError(error.message)
            return dict(operator=elem['operator'], operand=elem['operand'], negated=elem.get('negated', False))
        raise ValueError('element is not a dictionary')
    return list(map(convert_term, data))

def ok_to_include_history(narrow: OptionalNarrowListT, user_profile: Optional[UserProfile], is_web_public_query: bool) -> bool:
    if False:
        return 10
    if is_web_public_query:
        assert user_profile is None
        return True
    assert user_profile is not None
    include_history = False
    if narrow is not None:
        for term in narrow:
            if term['operator'] == 'stream' and (not term.get('negated', False)):
                operand: Union[str, int] = term['operand']
                if isinstance(operand, str):
                    include_history = can_access_stream_history_by_name(user_profile, operand)
                else:
                    include_history = can_access_stream_history_by_id(user_profile, operand)
            elif term['operator'] == 'streams' and term['operand'] == 'public' and (not term.get('negated', False)) and user_profile.can_access_public_streams():
                include_history = True
        for term in narrow:
            if term['operator'] == 'is':
                include_history = False
    return include_history

def get_stream_from_narrow_access_unchecked(narrow: OptionalNarrowListT, realm: Realm) -> Optional[Stream]:
    if False:
        for i in range(10):
            print('nop')
    if narrow is not None:
        for term in narrow:
            if term['operator'] == 'stream':
                return get_stream_by_narrow_operand_access_unchecked(term['operand'], realm)
    return None

def exclude_muting_conditions(user_profile: UserProfile, narrow: OptionalNarrowListT) -> List[ClauseElement]:
    if False:
        i = 10
        return i + 15
    conditions: List[ClauseElement] = []
    stream_id = None
    try:
        stream = get_stream_from_narrow_access_unchecked(narrow, user_profile.realm)
        if stream is not None:
            stream_id = stream.id
    except Stream.DoesNotExist:
        pass
    if stream_id is None:
        rows = Subscription.objects.filter(user_profile=user_profile, active=True, is_muted=True, recipient__type=Recipient.STREAM).values('recipient_id')
        muted_recipient_ids = [row['recipient_id'] for row in rows]
        if len(muted_recipient_ids) > 0:
            condition = not_(column('recipient_id', Integer).in_(muted_recipient_ids))
            conditions.append(condition)
    conditions = exclude_topic_mutes(conditions, user_profile, stream_id)
    return conditions

def get_base_query_for_search(realm_id: int, user_profile: Optional[UserProfile], need_message: bool, need_user_message: bool) -> Tuple[Select, ColumnElement[Integer]]:
    if False:
        i = 10
        return i + 15
    if not need_user_message:
        assert need_message
        query = select(column('id', Integer).label('message_id')).select_from(table('zerver_message')).where(column('realm_id', Integer) == literal(realm_id))
        inner_msg_id_col = literal_column('zerver_message.id', Integer)
        return (query, inner_msg_id_col)
    assert user_profile is not None
    if need_message:
        query = select(column('message_id', Integer), column('flags', Integer)).where(column('user_profile_id', Integer) == literal(user_profile.id)).select_from(join(table('zerver_usermessage'), table('zerver_message'), literal_column('zerver_usermessage.message_id', Integer) == literal_column('zerver_message.id', Integer)))
        inner_msg_id_col = column('message_id', Integer)
        return (query, inner_msg_id_col)
    query = select(column('message_id', Integer), column('flags', Integer)).where(column('user_profile_id', Integer) == literal(user_profile.id)).select_from(table('zerver_usermessage'))
    inner_msg_id_col = column('message_id', Integer)
    return (query, inner_msg_id_col)

def add_narrow_conditions(user_profile: Optional[UserProfile], inner_msg_id_col: ColumnElement[Integer], query: Select, narrow: OptionalNarrowListT, is_web_public_query: bool, realm: Realm) -> Tuple[Select, bool]:
    if False:
        print('Hello World!')
    is_search = False
    if narrow is None:
        return (query, is_search)
    builder = NarrowBuilder(user_profile, inner_msg_id_col, realm, is_web_public_query)
    search_operands = []
    for term in narrow:
        if term['operator'] == 'search':
            search_operands.append(term['operand'])
        else:
            query = builder.add_term(query, term)
    if search_operands:
        is_search = True
        query = query.add_columns(topic_column_sa(), column('rendered_content', Text))
        search_term = dict(operator='search', operand=' '.join(search_operands))
        query = builder.add_term(query, search_term)
    return (query, is_search)

def find_first_unread_anchor(sa_conn: Connection, user_profile: Optional[UserProfile], narrow: OptionalNarrowListT) -> int:
    if False:
        for i in range(10):
            print('nop')
    if user_profile is None:
        return LARGER_THAN_MAX_MESSAGE_ID
    need_user_message = True
    need_message = True
    (query, inner_msg_id_col) = get_base_query_for_search(realm_id=user_profile.realm_id, user_profile=user_profile, need_message=need_message, need_user_message=need_user_message)
    (query, is_search) = add_narrow_conditions(user_profile=user_profile, inner_msg_id_col=inner_msg_id_col, query=query, narrow=narrow, is_web_public_query=False, realm=user_profile.realm)
    condition = column('flags', Integer).op('&')(UserMessage.flags.read.mask) == 0
    muting_conditions = exclude_muting_conditions(user_profile, narrow)
    if muting_conditions:
        condition = and_(condition, *muting_conditions)
    first_unread_query = query.where(condition)
    first_unread_query = first_unread_query.order_by(inner_msg_id_col.asc()).limit(1)
    first_unread_result = list(sa_conn.execute(first_unread_query).fetchall())
    if len(first_unread_result) > 0:
        anchor = first_unread_result[0][0]
    else:
        anchor = LARGER_THAN_MAX_MESSAGE_ID
    return anchor

def parse_anchor_value(anchor_val: Optional[str], use_first_unread_anchor: bool) -> Optional[int]:
    if False:
        while True:
            i = 10
    'Given the anchor and use_first_unread_anchor parameters passed by\n    the client, computes what anchor value the client requested,\n    handling backwards-compatibility and the various string-valued\n    fields.  We encode use_first_unread_anchor as anchor=None.\n    '
    if use_first_unread_anchor:
        return None
    if anchor_val is None:
        raise JsonableError(_("Missing 'anchor' argument."))
    if anchor_val == 'oldest':
        return 0
    if anchor_val == 'newest':
        return LARGER_THAN_MAX_MESSAGE_ID
    if anchor_val == 'first_unread':
        return None
    try:
        anchor = int(anchor_val)
        if anchor < 0:
            return 0
        elif anchor > LARGER_THAN_MAX_MESSAGE_ID:
            return LARGER_THAN_MAX_MESSAGE_ID
        return anchor
    except ValueError:
        raise JsonableError(_('Invalid anchor'))

def limit_query_to_range(query: Select, num_before: int, num_after: int, anchor: int, include_anchor: bool, anchored_to_left: bool, anchored_to_right: bool, id_col: ColumnElement[Integer], first_visible_message_id: int) -> SelectBase:
    if False:
        i = 10
        return i + 15
    '\n    This code is actually generic enough that we could move it to a\n    library, but our only caller for now is message search.\n    '
    need_before_query = not anchored_to_left and num_before > 0
    need_after_query = not anchored_to_right and num_after > 0
    need_both_sides = need_before_query and need_after_query
    if need_both_sides:
        before_anchor = anchor - 1
        after_anchor = max(anchor, first_visible_message_id)
        before_limit = num_before
        after_limit = num_after + 1
    elif need_before_query:
        before_anchor = anchor - (not include_anchor)
        before_limit = num_before
        if not anchored_to_right:
            before_limit += include_anchor
    elif need_after_query:
        after_anchor = max(anchor + (not include_anchor), first_visible_message_id)
        after_limit = num_after + include_anchor
    if need_before_query:
        before_query = query
        if not anchored_to_right:
            before_query = before_query.where(id_col <= before_anchor)
        before_query = before_query.order_by(id_col.desc())
        before_query = before_query.limit(before_limit)
    if need_after_query:
        after_query = query
        if not anchored_to_left:
            after_query = after_query.where(id_col >= after_anchor)
        after_query = after_query.order_by(id_col.asc())
        after_query = after_query.limit(after_limit)
    if need_both_sides:
        return union_all(before_query.self_group(), after_query.self_group())
    elif need_before_query:
        return before_query
    elif need_after_query:
        return after_query
    else:
        return query.where(id_col == anchor)
MessageRowT = TypeVar('MessageRowT', bound=Sequence[Any])

@dataclass
class LimitedMessages(Generic[MessageRowT]):
    rows: List[MessageRowT]
    found_anchor: bool
    found_newest: bool
    found_oldest: bool
    history_limited: bool

def post_process_limited_query(rows: Sequence[MessageRowT], num_before: int, num_after: int, anchor: int, anchored_to_left: bool, anchored_to_right: bool, first_visible_message_id: int) -> LimitedMessages[MessageRowT]:
    if False:
        print('Hello World!')
    if first_visible_message_id > 0:
        visible_rows: Sequence[MessageRowT] = [r for r in rows if r[0] >= first_visible_message_id]
    else:
        visible_rows = rows
    rows_limited = len(visible_rows) != len(rows)
    if anchored_to_right:
        num_after = 0
        before_rows = visible_rows[:]
        anchor_rows = []
        after_rows = []
    else:
        before_rows = [r for r in visible_rows if r[0] < anchor]
        anchor_rows = [r for r in visible_rows if r[0] == anchor]
        after_rows = [r for r in visible_rows if r[0] > anchor]
    if num_before:
        before_rows = before_rows[-1 * num_before:]
    if num_after:
        after_rows = after_rows[:num_after]
    limited_rows = [*before_rows, *anchor_rows, *after_rows]
    found_anchor = len(anchor_rows) == 1
    found_oldest = anchored_to_left or len(before_rows) < num_before
    found_newest = anchored_to_right or len(after_rows) < num_after
    history_limited = rows_limited and found_oldest
    return LimitedMessages(rows=limited_rows, found_anchor=found_anchor, found_newest=found_newest, found_oldest=found_oldest, history_limited=history_limited)

@dataclass
class FetchedMessages(LimitedMessages[Row]):
    anchor: int
    include_history: bool
    is_search: bool

def fetch_messages(*, narrow: OptionalNarrowListT, user_profile: Optional[UserProfile], realm: Realm, is_web_public_query: bool, anchor: Optional[int], include_anchor: bool, num_before: int, num_after: int) -> FetchedMessages:
    if False:
        return 10
    include_history = ok_to_include_history(narrow, user_profile, is_web_public_query)
    if include_history:
        need_message = True
        need_user_message = False
    elif narrow is None:
        need_message = False
        need_user_message = True
    else:
        need_message = True
        need_user_message = True
    query: SelectBase
    (query, inner_msg_id_col) = get_base_query_for_search(realm_id=realm.id, user_profile=user_profile, need_message=need_message, need_user_message=need_user_message)
    (query, is_search) = add_narrow_conditions(user_profile=user_profile, inner_msg_id_col=inner_msg_id_col, query=query, narrow=narrow, realm=realm, is_web_public_query=is_web_public_query)
    with get_sqlalchemy_connection() as sa_conn:
        if anchor is None:
            anchor = find_first_unread_anchor(sa_conn, user_profile, narrow)
        anchored_to_left = anchor == 0
        anchored_to_right = anchor >= LARGER_THAN_MAX_MESSAGE_ID
        if anchored_to_right:
            num_after = 0
        first_visible_message_id = get_first_visible_message_id(realm)
        query = limit_query_to_range(query=query, num_before=num_before, num_after=num_after, anchor=anchor, include_anchor=include_anchor, anchored_to_left=anchored_to_left, anchored_to_right=anchored_to_right, id_col=inner_msg_id_col, first_visible_message_id=first_visible_message_id)
        main_query = query.subquery()
        query = select(*main_query.c).select_from(main_query).order_by(column('message_id', Integer).asc())
        query = query.prefix_with('/* get_messages */')
        rows = list(sa_conn.execute(query).fetchall())
    query_info = post_process_limited_query(rows=rows, num_before=num_before, num_after=num_after, anchor=anchor, anchored_to_left=anchored_to_left, anchored_to_right=anchored_to_right, first_visible_message_id=first_visible_message_id)
    return FetchedMessages(rows=query_info.rows, found_anchor=query_info.found_anchor, found_newest=query_info.found_newest, found_oldest=query_info.found_oldest, history_limited=query_info.history_limited, anchor=anchor, include_history=include_history, is_search=is_search)