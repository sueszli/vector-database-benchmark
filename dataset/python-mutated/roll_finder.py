from abc import ABCMeta, abstractmethod
from six import with_metaclass
ROLL_DAYS_FOR_CURRENT_CONTRACT = 90

class RollFinder(with_metaclass(ABCMeta, object)):
    """
    Abstract base class for calculating when futures contracts are the active
    contract.
    """

    @abstractmethod
    def _active_contract(self, oc, front, back, dt):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def _get_active_contract_at_offset(self, root_symbol, dt, offset):
        if False:
            for i in range(10):
                print('nop')
        '\n        For the given root symbol, find the contract that is considered active\n        on a specific date at a specific offset.\n        '
        oc = self.asset_finder.get_ordered_contracts(root_symbol)
        session = self.trading_calendar.minute_to_session_label(dt)
        front = oc.contract_before_auto_close(session.value)
        back = oc.contract_at_offset(front, 1, dt.value)
        if back is None:
            return front
        primary = self._active_contract(oc, front, back, session)
        return oc.contract_at_offset(primary, offset, session.value)

    def get_contract_center(self, root_symbol, dt, offset):
        if False:
            i = 10
            return i + 15
        '\n        Parameters\n        ----------\n        root_symbol : str\n            The root symbol for the contract chain.\n        dt : Timestamp\n            The datetime for which to retrieve the current contract.\n        offset : int\n            The offset from the primary contract.\n            0 is the primary, 1 is the secondary, etc.\n\n        Returns\n        -------\n        Future\n            The active future contract at the given dt.\n        '
        return self._get_active_contract_at_offset(root_symbol, dt, offset)

    def get_rolls(self, root_symbol, start, end, offset):
        if False:
            print('Hello World!')
        '\n        Get the rolls, i.e. the session at which to hop from contract to\n        contract in the chain.\n\n        Parameters\n        ----------\n        root_symbol : str\n            The root symbol for which to calculate rolls.\n        start : Timestamp\n            Start of the date range.\n        end : Timestamp\n            End of the date range.\n        offset : int\n            Offset from the primary.\n\n        Returns\n        -------\n        rolls - list[tuple(sid, roll_date)]\n            A list of rolls, where first value is the first active `sid`,\n        and the `roll_date` on which to hop to the next contract.\n            The last pair in the chain has a value of `None` since the roll\n            is after the range.\n        '
        oc = self.asset_finder.get_ordered_contracts(root_symbol)
        front = self._get_active_contract_at_offset(root_symbol, end, 0)
        back = oc.contract_at_offset(front, 1, end.value)
        if back is not None:
            end_session = self.trading_calendar.minute_to_session_label(end)
            first = self._active_contract(oc, front, back, end_session)
        else:
            first = front
        first_contract = oc.sid_to_contract[first]
        rolls = [((first_contract >> offset).contract.sid, None)]
        tc = self.trading_calendar
        sessions = tc.sessions_in_range(tc.minute_to_session_label(start), tc.minute_to_session_label(end))
        freq = sessions.freq
        if first == front:
            curr = first_contract << 1
        else:
            curr = first_contract << 2
        session = sessions[-1]
        while session > start and curr is not None:
            front = curr.contract.sid
            back = rolls[0][0]
            prev_c = curr.prev
            while session > start:
                prev = session - freq
                if prev_c is not None:
                    if prev < prev_c.contract.auto_close_date:
                        break
                if back != self._active_contract(oc, front, back, prev):
                    rolls.insert(0, ((curr >> offset).contract.sid, session))
                    break
                session = prev
            curr = curr.prev
            if curr is not None:
                session = min(session, curr.contract.auto_close_date + freq)
        return rolls

class CalendarRollFinder(RollFinder):
    """
    The CalendarRollFinder calculates contract rolls based purely on the
    contract's auto close date.
    """

    def __init__(self, trading_calendar, asset_finder):
        if False:
            while True:
                i = 10
        self.trading_calendar = trading_calendar
        self.asset_finder = asset_finder

    def _active_contract(self, oc, front, back, dt):
        if False:
            for i in range(10):
                print('nop')
        contract = oc.sid_to_contract[front].contract
        auto_close_date = contract.auto_close_date
        auto_closed = dt >= auto_close_date
        return back if auto_closed else front

class VolumeRollFinder(RollFinder):
    """
    The VolumeRollFinder calculates contract rolls based on when
    volume activity transfers from one contract to another.
    """
    GRACE_DAYS = 7

    def __init__(self, trading_calendar, asset_finder, session_reader):
        if False:
            return 10
        self.trading_calendar = trading_calendar
        self.asset_finder = asset_finder
        self.session_reader = session_reader

    def _active_contract(self, oc, front, back, dt):
        if False:
            i = 10
            return i + 15
        "\n        Return the active contract based on the previous trading day's volume.\n\n        In the rare case that a double volume switch occurs we treat the first\n        switch as the roll. Take the following case for example:\n\n        | +++++             _____\n        |      +   __      /       <--- 'G'\n        |       ++/++\\++++/++\n        |       _/    \\__/   +\n        |      /              +\n        | ____/                +   <--- 'F'\n        |_________|__|___|________\n                  a  b   c         <--- Switches\n\n        We should treat 'a' as the roll date rather than 'c' because from the\n        perspective of 'a', if a switch happens and we are pretty close to the\n        auto-close date, we would probably assume it is time to roll. This\n        means that for every date after 'a', `data.current(cf, 'contract')`\n        should return the 'G' contract.\n        "
        front_contract = oc.sid_to_contract[front].contract
        back_contract = oc.sid_to_contract[back].contract
        tc = self.trading_calendar
        trading_day = tc.day
        prev = dt - trading_day
        get_value = self.session_reader.get_value
        if dt > min(front_contract.auto_close_date, front_contract.end_date):
            return back
        elif front_contract.start_date > prev:
            return back
        elif dt > min(back_contract.auto_close_date, back_contract.end_date):
            return front
        elif back_contract.start_date > prev:
            return front
        front_vol = get_value(front, prev, 'volume')
        back_vol = get_value(back, prev, 'volume')
        if back_vol > front_vol:
            return back
        gap_start = max(back_contract.start_date, front_contract.auto_close_date - trading_day * self.GRACE_DAYS)
        gap_end = prev - trading_day
        if dt < gap_start:
            return front
        sessions = tc.sessions_in_range(tc.minute_to_session_label(gap_start), tc.minute_to_session_label(gap_end))
        for session in sessions:
            front_vol = get_value(front, session, 'volume')
            back_vol = get_value(back, session, 'volume')
            if back_vol > front_vol:
                return back
        return front

    def get_contract_center(self, root_symbol, dt, offset):
        if False:
            i = 10
            return i + 15
        '\n        Parameters\n        ----------\n        root_symbol : str\n            The root symbol for the contract chain.\n        dt : Timestamp\n            The datetime for which to retrieve the current contract.\n        offset : int\n            The offset from the primary contract.\n            0 is the primary, 1 is the secondary, etc.\n\n        Returns\n        -------\n        Future\n            The active future contract at the given dt.\n        '
        day = self.trading_calendar.day
        end_date = min(dt + ROLL_DAYS_FOR_CURRENT_CONTRACT * day, self.session_reader.last_available_dt)
        rolls = self.get_rolls(root_symbol=root_symbol, start=dt, end=end_date, offset=offset)
        (sid, acd) = rolls[0]
        return self.asset_finder.retrieve_asset(sid)