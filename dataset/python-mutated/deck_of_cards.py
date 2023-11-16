from abc import ABCMeta, abstractmethod
from enum import Enum
import sys

class Suit(Enum):
    HEART = 0
    DIAMOND = 1
    CLUBS = 2
    SPADE = 3

class Card(metaclass=ABCMeta):

    def __init__(self, value, suit):
        if False:
            for i in range(10):
                print('nop')
        self.value = value
        self.suit = suit
        self.is_available = True

    @property
    @abstractmethod
    def value(self):
        if False:
            return 10
        pass

    @value.setter
    @abstractmethod
    def value(self, other):
        if False:
            for i in range(10):
                print('nop')
        pass

class BlackJackCard(Card):

    def __init__(self, value, suit):
        if False:
            for i in range(10):
                print('nop')
        super(BlackJackCard, self).__init__(value, suit)

    def is_ace(self):
        if False:
            return 10
        return True if self._value == 1 else False

    def is_face_card(self):
        if False:
            i = 10
            return i + 15
        'Jack = 11, Queen = 12, King = 13'
        return True if 10 < self._value <= 13 else False

    @property
    def value(self):
        if False:
            print('Hello World!')
        if self.is_ace() == 1:
            return 1
        elif self.is_face_card():
            return 10
        else:
            return self._value

    @value.setter
    def value(self, new_value):
        if False:
            i = 10
            return i + 15
        if 1 <= new_value <= 13:
            self._value = new_value
        else:
            raise ValueError('Invalid card value: {}'.format(new_value))

class Hand(object):

    def __init__(self, cards):
        if False:
            for i in range(10):
                print('nop')
        self.cards = cards

    def add_card(self, card):
        if False:
            i = 10
            return i + 15
        self.cards.append(card)

    def score(self):
        if False:
            return 10
        total_value = 0
        for card in self.cards:
            total_value += card.value
        return total_value

class BlackJackHand(Hand):
    BLACKJACK = 21

    def __init__(self, cards):
        if False:
            i = 10
            return i + 15
        super(BlackJackHand, self).__init__(cards)

    def score(self):
        if False:
            return 10
        min_over = sys.MAXSIZE
        max_under = -sys.MAXSIZE
        for score in self.possible_scores():
            if self.BLACKJACK < score < min_over:
                min_over = score
            elif max_under < score <= self.BLACKJACK:
                max_under = score
        return max_under if max_under != -sys.MAXSIZE else min_over

    def possible_scores(self):
        if False:
            return 10
        'Return a list of possible scores, taking Aces into account.'
        pass

class Deck(object):

    def __init__(self, cards):
        if False:
            i = 10
            return i + 15
        self.cards = cards
        self.deal_index = 0

    def remaining_cards(self):
        if False:
            while True:
                i = 10
        return len(self.cards) - self.deal_index

    def deal_card(self):
        if False:
            print('Hello World!')
        try:
            card = self.cards[self.deal_index]
            card.is_available = False
            self.deal_index += 1
        except IndexError:
            return None
        return card

    def shuffle(self):
        if False:
            return 10
        pass