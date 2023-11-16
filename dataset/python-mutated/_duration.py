import re
_match_duration = re.compile('^(-?\\d+\\.?\\d*)(s|ms)$').match

class DurationError(Exception):
    """
    Exception indicating a general issue with a CSS duration.
    """

class DurationParseError(DurationError):
    """
    Indicates a malformed duration string that could not be parsed.
    """

def _duration_as_seconds(duration: str) -> float:
    if False:
        while True:
            i = 10
    '\n    Args:\n        duration: A string of the form ``"2s"`` or ``"300ms"``, representing 2 seconds and\n            300 milliseconds respectively. If no unit is supplied, e.g. ``"2"``, then the duration is\n            assumed to be in seconds.\n    Raises:\n        DurationParseError: If the argument ``duration`` is not a valid duration string.\n    Returns:\n        The duration in seconds.\n    '
    match = _match_duration(duration)
    if match:
        (value, unit_name) = match.groups()
        value = float(value)
        if unit_name == 'ms':
            duration_secs = value / 1000
        else:
            duration_secs = value
    else:
        try:
            duration_secs = float(duration)
        except ValueError:
            raise DurationParseError(f'{duration!r} is not a valid duration.') from ValueError
    return duration_secs