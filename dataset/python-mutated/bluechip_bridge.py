"""Wraps third-party bridge bots to make them usable in OpenSpiel.

This code enables OpenSpiel interoperation for bots which implement the BlueChip
bridge protocol. This is widely used, e.g. in the World computer bridge
championships. For a rough outline of the protocol, see:
http://www.bluechipbridge.co.uk/protocol.htm

No formal specification is available. This implementation has been verified
to work correctly with WBridge5.

This bot controls a single player in the full game of bridge, including both the
bidding and play phase. It chooses its actions by invoking an external bot which
plays the full game of bridge. This means that each time the bot is asked for an
action, it sends up to three actions (one for each other player) to the external
bridge bot, and obtains an action in return.
"""
import re
import pyspiel
GAME_STR = 'bridge(use_double_dummy_result=False)'
_CONNECT = 'Connecting "(?P<client_name>.*)" as ANYPL using protocol version 18'
_PLAYER_ACTION = '(?P<seat>NORTH|SOUTH|EAST|WEST) ((?P<pass>PASSES)|(?P<dbl>DOUBLES)|(?P<rdbl>REDOUBLES)|bids (?P<bid>[^ ]*)|(plays (?P<play>[23456789tjqka][cdhs])))(?P<alert> Alert.)?'
_READY_FOR_OTHER = "{seat} ready for (((?P<other>[^']*)'s ((bid)|(card to trick \\d+)))|(?P<dummy>dummy))"
_READY_FOR_TEAMS = '{seat} ready for teams'
_READY_TO_START = '{seat} ready to start'
_READY_FOR_DEAL = '{seat} ready for deal'
_READY_FOR_CARDS = '{seat} ready for cards'
_READY_FOR_BID = "{seat} ready for {other}'s bid"
_SEATED = '{seat} ("{client_name}") seated'
_TEAMS = 'Teams: N/S "north-south" E/W "east-west"'
_START_BOARD = 'start of board'
_DEAL = 'Board number {board}. Dealer NORTH. Neither vulnerable.'
_CARDS = "{seat}'s cards: {hand}"
_OTHER_PLAYER_ACTION = '{player} {action}'
_PLAYER_TO_LEAD = '{seat} to lead'
_DUMMY_CARDS = "Dummy's cards: {}"
_SEATS = ['NORTH', 'EAST', 'SOUTH', 'WEST']
_TRUMP_SUIT = ['C', 'D', 'H', 'S', 'NT']
_NUMBER_TRUMP_SUITS = len(_TRUMP_SUIT)
_SUIT = _TRUMP_SUIT[:4]
_NUMBER_SUITS = len(_SUIT)
_RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
_LSUIT = [x.lower() for x in _SUIT]
_LRANKS = [x.lower() for x in _RANKS]
_ACTION_PASS = 52
_ACTION_DBL = 53
_ACTION_RDBL = 54
_ACTION_BID = 55

def _bid_to_action(action_str):
    if False:
        i = 10
        return i + 15
    'Returns an OpenSpiel action id (an integer) from a BlueChip bid string.'
    level = int(action_str[0])
    trumps = _TRUMP_SUIT.index(action_str[1:])
    return _ACTION_BID + (level - 1) * _NUMBER_TRUMP_SUITS + trumps

def _play_to_action(action_str):
    if False:
        for i in range(10):
            print('nop')
    'Returns an OpenSpiel action id (an integer) from a BlueChip card string.'
    rank = _LRANKS.index(action_str[0])
    suit = _LSUIT.index(action_str[1])
    return rank * _NUMBER_SUITS + suit

def _action_to_string(action):
    if False:
        print('Hello World!')
    "Converts OpenSpiel action id (an integer) to a BlueChip action string.\n\n  Args:\n    action: an integer action id corresponding to a bid.\n\n  Returns:\n    A string in BlueChip format, e.g. 'PASSES' or 'bids 1H', or 'plays ck'.\n  "
    if action == _ACTION_PASS:
        return 'PASSES'
    elif action == _ACTION_DBL:
        return 'DOUBLES'
    elif action == _ACTION_RDBL:
        return 'REDOUBLES'
    elif action >= _ACTION_BID:
        level = str((action - _ACTION_BID) // _NUMBER_TRUMP_SUITS + 1)
        trumps = _TRUMP_SUIT[(action - _ACTION_BID) % _NUMBER_TRUMP_SUITS]
        return 'bids ' + level + trumps
    else:
        rank = action // _NUMBER_SUITS
        suit = action % _NUMBER_SUITS
        return 'plays ' + _LRANKS[rank] + _LSUIT[suit]

def _expect_regex(controller, regex):
    if False:
        i = 10
        return i + 15
    'Reads a line from the controller, parses it using the regular expression.'
    line = controller.read_line()
    match = re.match(regex, line)
    if not match:
        raise ValueError("Received '{}' which does not match regex '{}'".format(line, regex))
    return match.groupdict()

def _expect(controller, expected):
    if False:
        return 10
    'Reads a line from the controller, checks it matches expected line exactly.'
    line = controller.read_line()
    if expected != line:
        raise ValueError("Received '{}' but expected '{}'".format(line, expected))

def _hand_string(cards):
    if False:
        print('Hello World!')
    'Returns the hand of the to-play player in the state in BlueChip format.'
    if len(cards) != 13:
        raise ValueError('Must have 13 cards')
    suits = [[] for _ in range(4)]
    for card in reversed(sorted(cards)):
        suit = card % 4
        rank = card // 4
        suits[suit].append(_RANKS[rank])
    for i in range(4):
        if suits[i]:
            suits[i] = _TRUMP_SUIT[i] + ' ' + ' '.join(suits[i]) + '.'
        else:
            suits[i] = _TRUMP_SUIT[i] + ' -.'
    return ' '.join(suits)

def _connect(controller, seat):
    if False:
        print('Hello World!')
    'Performs the initial handshake with a BlueChip bot.'
    client_name = _expect_regex(controller, _CONNECT)['client_name']
    controller.send_line(_SEATED.format(seat=seat, client_name=client_name))
    _expect(controller, _READY_FOR_TEAMS.format(seat=seat))
    controller.send_line(_TEAMS)
    _expect(controller, _READY_TO_START.format(seat=seat))

def _new_deal(controller, seat, hand, board):
    if False:
        return 10
    'Informs a BlueChip bots that there is a new deal.'
    controller.send_line(_START_BOARD)
    _expect(controller, _READY_FOR_DEAL.format(seat=seat))
    controller.send_line(_DEAL.format(board=board))
    _expect(controller, _READY_FOR_CARDS.format(seat=seat))
    controller.send_line(_CARDS.format(seat=seat, hand=hand))

class BlueChipBridgeBot(pyspiel.Bot):
    """An OpenSpiel bot, wrapping a BlueChip bridge bot implementation."""

    def __init__(self, game, player_id, controller_factory):
        if False:
            return 10
        'Initializes an OpenSpiel `Bot` wrapping a BlueChip-compatible bot.\n\n    Args:\n      game: The OpenSpiel game object, should be an instance of\n        `bridge(use_double_dummy_result=false)`.\n      player_id: The id of the player the bot will act as, 0 = North (dealer), 1\n        = East, 2 = South, 3 = West.\n      controller_factory: Callable that returns new BlueChip controllers which\n        must support methods `read_line` and `send_line`, and `terminate`.\n    '
        pyspiel.Bot.__init__(self)
        if str(game) != GAME_STR:
            raise ValueError(f'BlueChipBridgeBot invoked with {game}')
        self._game = game
        self._player_id = player_id
        self._controller_factory = controller_factory
        self._seat = _SEATS[player_id]
        self._num_actions = 52
        self.dummy = None
        self.is_play_phase = False
        self.cards_played = 0
        self._board = 0
        self._state = self._game.new_initial_state()
        self._controller = None

    def player_id(self):
        if False:
            i = 10
            return i + 15
        return self._player_id

    def restart(self):
        if False:
            for i in range(10):
                print('nop')
        'Indicates that we are starting a new episode.'
        if not self._state.history():
            return
        self._num_actions = 52
        self.dummy = None
        self.is_play_phase = False
        self.cards_played = 0
        if not self._state.is_terminal():
            state = self._state.clone()
            while not state.is_terminal() and state.current_player() != self._player_id:
                legal_actions = state.legal_actions()
                if _ACTION_PASS in legal_actions:
                    state.apply(_ACTION_PASS)
                elif len(legal_actions) == 1:
                    state.apply_action(legal_actions[0])
            if state.is_terminal():
                self.inform_state(state)
        if not self._state.is_terminal():
            self._controller.terminate()
            self._controller = None
        self._state = self._game.new_initial_state()

    def _update_for_state(self):
        if False:
            while True:
                i = 10
        'Called for all non-chance nodes, whether or not we have to act.'
        actions = self._state.history()
        self.is_play_phase = not self._state.is_terminal() and max(self._state.legal_actions()) < 52
        self.cards_played = sum((1 if a < 52 else 0 for a in actions)) - 52
        if len(actions) == 52:
            self._board += 1
            _new_deal(self._controller, self._seat, _hand_string(actions[self._player_id:52:4]), self._board)
        for other_player_action in actions[self._num_actions:]:
            other = _expect_regex(self._controller, _READY_FOR_OTHER.format(seat=self._seat))
            other_player = other['other']
            if other_player == 'Dummy':
                other_player = _SEATS[self.dummy]
            self._controller.send_line(_OTHER_PLAYER_ACTION.format(player=other_player, action=_action_to_string(other_player_action)))
        self._num_actions = len(actions)
        if self.is_play_phase and self.cards_played == 1:
            self.dummy = self._state.current_player() ^ 2
            if self._player_id != self.dummy:
                other = _expect_regex(self._controller, _READY_FOR_OTHER.format(seat=self._seat))
                dummy_cards = _hand_string(actions[self.dummy:52:4])
                self._controller.send_line(_DUMMY_CARDS.format(dummy_cards))
        if self._state.is_terminal():
            self._controller.send_line('Timing - N/S : this board  [1:15],  total  [0:11:23].  E/W : this board  [1:18],  total  [0:10:23]')
            self.dummy = None
            self.is_play_phase = False
            self.cards_played = 0

    def inform_action(self, state, player, action):
        if False:
            i = 10
            return i + 15
        del player, action
        self.inform_state(state)

    def inform_state(self, state):
        if False:
            print('Hello World!')
        if self._controller is None:
            self._controller = self._controller_factory()
            _connect(self._controller, self._seat)
        full_history = state.history()
        known_history = self._state.history()
        if full_history[:len(known_history)] != known_history:
            raise ValueError(f"Supplied state is inconsistent with bot's internal state\nSupplied state:\n{state}\nInternal state:\n{self._state}\n")
        for action in full_history[len(known_history):]:
            self._state.apply_action(action)
            if not self._state.is_chance_node():
                self._update_for_state()

    def step(self, state):
        if False:
            i = 10
            return i + 15
        'Returns an action for the given state.'
        self.inform_state(state)
        if self.is_play_phase and self.cards_played % 4 == 0:
            self._controller.send_line(_PLAYER_TO_LEAD.format(seat=self._seat))
        our_action = _expect_regex(self._controller, _PLAYER_ACTION)
        self._num_actions += 1
        if our_action['pass']:
            return _ACTION_PASS
        elif our_action['dbl']:
            return _ACTION_DBL
        elif our_action['rdbl']:
            return _ACTION_RDBL
        elif our_action['bid']:
            return _bid_to_action(our_action['bid'])
        elif our_action['play']:
            return _play_to_action(our_action['play'])

    def terminate(self):
        if False:
            for i in range(10):
                print('nop')
        self._controller.terminate()
        self._controller = None