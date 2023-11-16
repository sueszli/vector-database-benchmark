"""Wraps third-party bridge bots to make them usable in OpenSpiel.

This code enables OpenSpiel interoperation for bots which implement the BlueChip
bridge protocol. This is widely used, e.g. in the World computer bridge
championships. For a rough outline of the protocol, see:
http://www.bluechipbridge.co.uk/protocol.htm

No formal specification is available. This implementation has been verified
to work correctly with WBridge5.

This bot controls a single player in the game of uncontested bridge bidding. It
chooses its actions by invoking an external bot which plays the full game of
bridge. This means that each time the bot is asked for an action, it sends up to
three actions (forced passes from both opponents, plus partner's most recent
action) to the external bridge bot, and obtains an action in return.

Since we are restricting ourselves to the uncontested bidding game, we have
no support for Doubling, Redoubling, or the play of the cards.
"""
import re
import pyspiel
_CONNECT = 'Connecting "(?P<client_name>.*)" as ANYPL using protocol version 18'
_SELF_BID_OR_PASS = '{seat} ((?P<pass>PASSES)|bids (?P<bid>[^ ]*))( Alert.)?'
_READY_FOR_TEAMS = '{seat} ready for teams'
_READY_TO_START = '{seat} ready to start'
_READY_FOR_DEAL = '{seat} ready for deal'
_READY_FOR_CARDS = '{seat} ready for cards'
_READY_FOR_BID = "{seat} ready for {other}'s bid"
_SEATED = '{seat} ("{client_name}") seated'
_TEAMS = 'Teams: N/S "opponents" E/W "bidders"'
_START_BOARD = 'start of board'
_DEAL = 'Board number 8. Dealer WEST. Neither vulnerable.'
_CARDS = "{seat}'s cards: {hand}"
_OTHER_PLAYER_PASS = '{player} PASSES'
_OTHER_PLAYER_BID = '{player} bids {bid}'
_SEATS = ['WEST', 'EAST']
_OPPONENTS = ['NORTH', 'SOUTH']
_TRUMP_SUIT = ['C', 'D', 'H', 'S', 'NT']
_NUMBER_TRUMP_SUITS = len(_TRUMP_SUIT)
_RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
_PASS_ACTION = 0

def _string_to_action(call_str):
    if False:
        print('Hello World!')
    'Converts a BlueChip bid string to an OpenSpiel action id (an integer).\n\n  Args:\n    call_str: string representing a bid in the BlueChip format, i.e. "[level]\n      (as a digit) + [trump suit (S, H, D, C or NT)]", e.g. "1C".\n\n  Returns:\n    An integer action id - see `bridge_uncontested_bidding.cc`, functions\n    `Denomination` and `Level`.\n    0 is reserved for Pass, so bids are in order from 1 upwards: 1 = 1C,\n    2 = 1D, etc.\n  '
    level = int(call_str[0])
    trumps = _TRUMP_SUIT.index(call_str[1:])
    return (level - 1) * _NUMBER_TRUMP_SUITS + trumps + 1

def _action_to_string(action):
    if False:
        i = 10
        return i + 15
    'Converts OpenSpiel action id (an integer) to a BlueChip bid string.\n\n  Args:\n    action: an integer action id corresponding to a bid.\n\n  Returns:\n    A string in BlueChip format.\n\n  Inverse of `_string_to_action`. See documentation there.\n  '
    level = str((action - 1) // _NUMBER_TRUMP_SUITS + 1)
    trumps = _TRUMP_SUIT[(action - 1) % _NUMBER_TRUMP_SUITS]
    return level + trumps

def _expect_regex(client, regex):
    if False:
        while True:
            i = 10
    'Reads a line from the client, parses it using the regular expression.'
    line = client.read_line()
    match = re.match(regex, line)
    if not match:
        raise ValueError("Received '{}' which does not match regex '{}'".format(line, regex))
    return match.groupdict()

def _expect(client, expected):
    if False:
        i = 10
        return i + 15
    'Reads a line from the client, checks it matches expected line exactly.'
    line = client.read_line()
    if expected != line:
        raise ValueError("Received '{}' but expected '{}'".format(line, expected))

def _hand_string(state_vec):
    if False:
        print('Hello World!')
    'Returns the hand of the to-play player in the state in BlueChip format.'
    suits = []
    for suit in reversed(range(4)):
        cards = []
        for rank in reversed(range(13)):
            if state_vec[rank * 4 + suit]:
                cards.append(_RANKS[rank])
        suits.append(_TRUMP_SUIT[suit] + ' ' + (' '.join(cards) if cards else '-') + '.')
    return ' '.join(suits)

def _actions(state_vec):
    if False:
        return 10
    'Returns the player actions that have been taken in the game so far.'
    actions = state_vec[52:-2]
    return [index // 2 for (index, value) in enumerate(actions) if value]

def _connect(client, seat, state_vec):
    if False:
        print('Hello World!')
    'Performs the initial handshake with a BlueChip bot.'
    client.start()
    client_name = _expect_regex(client, _CONNECT)['client_name']
    client.send_line(_SEATED.format(seat=seat, client_name=client_name))
    _expect(client, _READY_FOR_TEAMS.format(seat=seat))
    client.send_line(_TEAMS)
    _expect(client, _READY_TO_START.format(seat=seat))
    client.send_line(_START_BOARD)
    _expect(client, _READY_FOR_DEAL.format(seat=seat))
    client.send_line(_DEAL)
    _expect(client, _READY_FOR_CARDS.format(seat=seat))
    client.send_line(_CARDS.format(seat=seat, hand=_hand_string(state_vec)))

class BlueChipBridgeBot(pyspiel.Bot):
    """An OpenSpiel bot, wrapping a BlueChip bridge bot implementation."""

    def __init__(self, game, player_id, client):
        if False:
            print('Hello World!')
        'Initializes an OpenSpiel `Bot` wrapping a BlueChip-compatible bot.\n\n    Args:\n      game: The OpenSpiel game object, should be an instance of\n        bridge_uncontested_bidding, without forced actions.\n      player_id: The id of the player the bot will act as, 0 = West (dealer), 1\n        = East.\n      client: The BlueChip bot; must support methods `start`, `read_line`, and\n        `send_line`.\n    '
        pyspiel.Bot.__init__(self)
        self._game = game
        self._player_id = player_id
        self._client = client
        self._seat = _SEATS[player_id]
        self._partner = _SEATS[1 - player_id]
        self._left_hand_opponent = _OPPONENTS[player_id]
        self._right_hand_opponent = _OPPONENTS[1 - player_id]
        self._connected = False

    def player_id(self):
        if False:
            while True:
                i = 10
        return self._player_id

    def restart(self):
        if False:
            print('Hello World!')
        'Indicates that the next step may be from a non-sequential state.'
        self._connected = False

    def restart_at(self, state):
        if False:
            return 10
        'Indicates that the next step may be from a non-sequential state.'
        self._connected = False

    def step(self, state):
        if False:
            print('Hello World!')
        'Returns the action and policy for the bot in this state.'
        state_vec = state.information_state_tensor(self.player_id())
        if not self._connected:
            _connect(self._client, self._seat, state_vec)
            self._connected = True
        actions = _actions(state_vec)
        if len(actions) > 1:
            _expect(self._client, _READY_FOR_BID.format(seat=self._seat, other=self._left_hand_opponent))
            self._client.send_line(_OTHER_PLAYER_PASS.format(player=self._left_hand_opponent))
        if actions:
            _expect(self._client, _READY_FOR_BID.format(seat=self._seat, other=self._partner))
            if actions[-1] == _PASS_ACTION:
                self._client.send_line(_OTHER_PLAYER_PASS.format(player=self._partner))
            else:
                self._client.send_line(_OTHER_PLAYER_BID.format(player=self._partner, bid=_action_to_string(actions[-1])))
        if actions:
            _expect(self._client, _READY_FOR_BID.format(seat=self._seat, other=self._right_hand_opponent))
            self._client.send_line(_OTHER_PLAYER_PASS.format(player=self._right_hand_opponent))
        our_action = _expect_regex(self._client, _SELF_BID_OR_PASS.format(seat=self._seat))
        action = 0 if our_action['pass'] else _string_to_action(our_action['bid'])
        return ((action, 1.0), action)